import hashlib
import io
import json
import tarfile

import pytest

from SpiralInterventionLab.examples import summarize_structural_table as report
from SpiralInterventionLab.runtime import reach_datasheet

B0_TOKEN, COMMA, WHILE = 1603, 11, 1418


def branch(token, ok, ok_v1=None, output=None):
    return {"first_token_id": token, "joint_success": ok, "output": output or f"out{token}", "constraints_ok": ok,
            "role_status": "pass", "evaluation_v1": {"joint_success": ok if ok_v1 is None else ok_v1},
            "baseline_logits_sha256": ["a", "a"]}


def push(case_id, token, cap, gap, closure, first, piece=","):
    return {"recipe_id": f"{case_id}:{token}@{cap}", "kind": "readout", "component": "readout", "cap": cap,
            "sources": [{"token_id": token, "piece": piece}], "first_token_id": first,
            "edited_winner_matches_first_token": True,
            "readout": {"edited_winner": first, "rows": [{"token_id": token, "baseline_gap_to_fixed_winner": gap,
                                                          "edited_gap_to_fixed_winner": gap - closure}]}}


def case(case_id, *, b0_ok, comma_gap, comma_ok, split="rescue", index=0, while_gap=None):
    """B0 emits ' before'; a ',' push closes 20 logits per unit of cap, a ' while' push 50."""
    caps = (0.04, 0.08, 0.12)
    pushes = [push(case_id, COMMA, c, comma_gap, 20 * c, COMMA if 20 * c > comma_gap else B0_TOKEN) for c in caps]
    readout_rows = [{"token_id": COMMA, "gap_to_top1": comma_gap, "is_greedy_token": False}]
    eligible = [{"token_id": COMMA, "piece": ","}]
    outcomes = [{**branch(COMMA, comma_ok), "push_first_token_id": COMMA, "role": "outcome", "cap": 0.08}]
    if while_gap is not None:
        pushes += [push(case_id, WHILE, c, while_gap, 50 * c, WHILE if 50 * c > while_gap else B0_TOKEN, " while")
                   for c in caps]
        readout_rows.append({"token_id": WHILE, "gap_to_top1": while_gap, "is_greedy_token": False})
        eligible.append({"token_id": WHILE, "piece": " while"})
        outcomes.append({**branch(WHILE, True), "push_first_token_id": WHILE, "role": "outcome", "cap": 0.08})
    return {"case_id": case_id, "split": split, "pool_index": index, "status": "ok", "boundary_found": True,
            "b0": branch(B0_TOKEN, b0_ok), "b0_winner_id": B0_TOKEN, "pushes": pushes, "outcomes": outcomes,
            "structure_readout": readout_rows, "eligible_sources": eligible, "decomposition_mismatches": [],
            "state_restored": True, "packets": {}, "source": "In order to finish before dusk, A should x."}


def pool():
    return [case("fail_near", b0_ok=False, comma_gap=1.0, comma_ok=True, index=0),
            case("fail_far", b0_ok=False, comma_gap=2.0, comma_ok=True, index=1),
            case("ok_far", b0_ok=True, comma_gap=2.2, comma_ok=False, index=2),
            case("fail_while", b0_ok=False, comma_gap=3.0, comma_ok=False, index=3, while_gap=3.5)]


def test_r_star_keeps_0_08_unless_another_cap_wins_by_the_margin():
    assert report.r_star({0.04: 0, 0.08: 1, 0.10: 2})["r_star"] == 0.08
    assert report.r_star({0.04: 0, 0.08: 1, 0.10: 3}) == {"r_star": 0.10, "reason": "net_beats_0.08_by_at_least_2",
                                                         "best_net": 3}
    assert report.r_star({0.04: -1, 0.08: 0, 0.10: 0})["reason"] == "fallback_max_net_not_positive"
    # 0.11 and 0.12 are measured but never chosen.
    assert report.r_star({0.04: 0, 0.08: 1, 0.10: 1, 0.12: 9})["r_star"] == 0.08


def test_arms_are_lookups_and_the_rule_reaches_past_an_unreachable_comma():
    rows = reach_datasheet.table_rows(report.measured(pool()))
    # Five calibration cases per stratum are required; replicate the pool for the sheet.
    many = [dict(r, case_id=r["case_id"] + str(k)) for k in range(5) for r in rows]
    sheet = reach_datasheet.build(many, cap=0.08, provenance={})
    arms = report.arm_table(pool(), 0.08, {"*": sheet})
    assert arms["fail_near"]["fixed_readout"]["joint_success"] and arms["fail_near"]["datasheet_rule"]["joint_success"]
    # ',' is out of reach at 0.08 (closure 1.6 < gap 3.0) but ' while' (closure 4.0 > 3.5) is not.
    assert not arms["fail_while"]["fixed_readout"]["first_token_changed"]
    assert arms["fail_while"]["datasheet_rule"]["first_token_id"] == WHILE
    assert arms["fail_while"]["oracle"]["joint_success"]
    counts = report.tally(pool(), arms, "datasheet_rule")
    assert counts["rescues"] == 2 and counts["losses"] == 0
    pair = report.compare(arms, "datasheet_rule", "fixed_readout")
    assert pair["datasheet_rule_only"] == 1 and pair["discordant"][0]["case_id"] == "fail_while"


def test_sweep_counts_rescues_and_losses_per_cap_under_both_evaluators():
    manifest = {"version": "t", "caps": [0.04, 0.08, 0.12], "patch_caps": [], "cases": []}
    summary = report.sweep_summary(manifest, pool())
    nets = {cap: counts["fixed_readout_v2"]["net"] for cap, counts in summary["per_cap"].items()}
    assert nets == {"0.04": 0, "0.08": 1, "0.12": 1}
    assert summary["r_star"]["r_star"] == 0.08
    assert summary["reach"][",|other"]["0.12"]["closure"]["n"] == 4


def test_controller_samples_are_scored_by_lookup_with_errors_as_noop():
    cases = pool()
    choices = [{"case_id": "fail_while", "arm": "controller_datasheet", "sample": 0, "status": "selected",
                "recipe_id": f"fail_while:{WHILE}@0.08"},
               {"case_id": "fail_while", "arm": "controller", "sample": 0, "status": "error", "recipe_id": None},
               {"case_id": "fail_while", "arm": "controller", "sample": 1, "status": "selected",
                "recipe_id": f"fail_while:{COMMA}@0.08"}]
    summary = report.controller_summary(cases, choices, 0.08, {"*": None})
    assert summary["arms"]["controller"]["errors"] == 1 and summary["arms"]["controller"]["noop"] == 1
    assert summary["arms"]["controller"]["unchanged_push"] == 1
    assert summary["mean_joint_success_per_case"]["fail_while"] == {"controller_datasheet": 1.0, "controller": 0.0}
    contrasts = report.controller_comparisons(summary["mean_joint_success_per_case"], {}, ["fail_while"])
    assert contrasts["controller_datasheet_vs_controller"]["left_higher"] == 1


def test_invariants_and_reproduction_flag_differences(tmp_path):
    cases = pool()
    cases[0]["structure_readout"][0]["gap_to_top1"] = 0.5
    cases[1]["decomposition_mismatches"] = [COMMA]
    issues = report.invariants(cases)
    assert issues["gap_mismatch"] == [["fail_near", COMMA]] and issues["decomposition_mismatch"][0][0] == "fail_far"
    old = {"cases": [{"case_id": "fail_near", "policies": {
        "b0": {"status": "ok", "baseline_logits_sha256": ["a", "a"], "output": "out1603"},
        "fixed_readout": {"status": "ok", "first_token_id": COMMA, "output": "different",
                          "command": {"edits": [{"id": f"fail_near:{COMMA}@0.08"}]},
                          "readout": cases[0]["pushes"][1]["readout"]}}}]}
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    checks = report.reproduce(cases, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()})
    assert checks["b0_logits"] == 1 and checks["pushes"] == 1
    assert checks["mismatches"] == [["fail_near", "output", f"fail_near:{COMMA}@0.08"]]
    with pytest.raises(ValueError, match="changed"):
        report.reproduce(cases, {str(path): "0" * 64})


def write_part(root, name, cases, *, code_sha="x", status="complete"):
    part = root / name
    part.mkdir(parents=True)
    data = b"print('hi')\n"
    with tarfile.open(part / "source_snapshot.tgz", "w:gz") as archive:
        member = tarfile.TarInfo("SpiralInterventionLab/a.py")
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    digest = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    (part / "results.json").write_text(json.dumps({"manifest_sha256": digest, "status": status, "cases": cases,
                                                   "checkpoint_sha256_verified": {"m": "1"}}))
    (part / "execution_plan.json").write_text(json.dumps({
        "manifest_sha256": digest, "code_sha256": {"SpiralInterventionLab/a.py": hashlib.sha256(data).hexdigest()},
        "device": "mps", "dtype": "float16", "mps_memory_fraction": 0.85, "evaluator_sha256": code_sha}))
    (part / "environment.json").write_text(json.dumps({"torch": "2"}))


def test_parts_must_share_code_and_environment_and_errored_cases_stay_listed(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"cases": [{}, {}], "evaluator_sha256": "x", "caps": [0.08],
                                                        "references": {}}))
    first, second = case("a", b0_ok=True, comma_gap=1, comma_ok=True), case("b", b0_ok=True, comma_gap=1,
                                                                              comma_ok=True, index=1)
    write_part(tmp_path, "part_000_001_r1", [first])
    write_part(tmp_path, "part_001_002_r1", [{**second, "status": "error", "error": "oom"}], status="error")
    write_part(tmp_path, "part_001_002_r2", [second])
    manifest, cases, binding = report.load_table(tmp_path)
    assert [c["case_id"] for c in cases] == ["a", "b"] and binding["missing_pool_indices"] == []
    assert binding["case_errors"] == [{"pool_index": 1, "case_id": "b", "part": "part_001_002_r1", "error": "oom"}]
    write_part(tmp_path, "part_000_001_r2", [first], code_sha="y")
    with pytest.raises(ValueError, match="different"):
        report.load_table(tmp_path)
