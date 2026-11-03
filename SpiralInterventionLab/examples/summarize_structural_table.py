"""Read-only analysis of first-token tables and controller samples.

Verifies every part (manifest, executed sources, and that all parts ran the same
code, environment, device, dtype, MPS fraction and checkpoint), checks the
table's own invariants and its bitwise reproduction of earlier receipts, then
scores every arm by lookup: deterministic rules, controller samples and the
oracle. No model or provider calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

from ..runtime import reach_datasheet
from .summarize_structural_rescue import failure_type, snapshot_member


R_STAR_CANDIDATES = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
R_STAR_DEFAULT = 0.08
R_STAR_MARGIN = 2
SAME_ACROSS_PARTS = ("code_sha256", "device", "dtype", "mps_memory_fraction", "evaluator_sha256")


# ---------------------------------------------------------------- loading


def load_table(table_dir: Path) -> tuple[dict, list[dict], dict]:
    manifest_bytes = (table_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    attempts = []
    for part in sorted(p for p in table_dir.iterdir() if p.is_dir() and p.name.startswith("part_")):
        raw = (part / "results.json").read_bytes()
        report = json.loads(raw)
        execution = json.loads((part / "execution_plan.json").read_text())
        if report["manifest_sha256"] != digest or execution["manifest_sha256"] != digest:
            raise ValueError("frozen_manifest_hash_mismatch:" + part.name)
        with tarfile.open(part / "source_snapshot.tgz", "r:gz") as snapshot:
            for path, expected in execution["code_sha256"].items():
                item = snapshot.extractfile(snapshot_member(path))
                if item is None or hashlib.sha256(item.read()).hexdigest() != expected:
                    raise ValueError("executed_source_snapshot_mismatch:" + path)
        attempts.append({"part": part.name, "report": report, "execution": execution,
                         "environment": json.loads((part / "environment.json").read_text()),
                         "results_sha256": hashlib.sha256(raw).hexdigest()})
    reference = None
    for attempt in attempts:
        identity = {k: attempt["execution"].get(k) for k in SAME_ACROSS_PARTS}
        identity.update(environment=attempt["environment"],
                        checkpoint=attempt["report"].get("checkpoint_sha256_verified"))
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("parts_ran_different_code_or_environment:" + attempt["part"])
    if reference and reference["evaluator_sha256"] != manifest["evaluator_sha256"]:
        raise ValueError("evaluator_differs_from_the_frozen_manifest")
    # The last attempt of a range that completed each case wins; earlier ones are listed.
    chosen: dict[int, tuple[dict, str]] = {}
    for attempt in attempts:
        for case in attempt["report"]["cases"]:
            if case.get("status") == "ok":
                chosen[case["pool_index"]] = (case, attempt["part"])
    missing = [i for i in range(len(manifest["cases"])) if i not in chosen]
    cases = [chosen[i][0] for i in sorted(chosen)]
    errors = [{"pool_index": c["pool_index"], "case_id": c["case_id"], "part": a["part"], "error": c.get("error")}
              for a in attempts for c in a["report"]["cases"] if c.get("status") == "error"]
    return manifest, cases, {"manifest_sha256": digest, "missing_pool_indices": missing, "case_errors": errors,
                             "attempts": [{"part": a["part"], "status": a["report"]["status"],
                                           "results_sha256": a["results_sha256"],
                                           "elapsed_seconds": a["report"].get("elapsed_seconds")} for a in attempts]}


# ---------------------------------------------------------------- invariants and reproduction


def invariants(cases: list[dict]) -> dict[str, list]:
    issues = {"gap_mismatch": [], "edited_winner_not_first_token": [], "decomposition_mismatch": [],
              "state_not_restored": []}
    for case in measured(cases):
        b0_rows = {}
        for push in case["pushes"]:
            for row in push["readout"]["rows"]:
                b0_rows[row["token_id"]] = row["baseline_gap_to_fixed_winner"]
            if not push["edited_winner_matches_first_token"]:
                issues["edited_winner_not_first_token"].append([case["case_id"], push["recipe_id"]])
        for row in case["structure_readout"]:
            if row["token_id"] in b0_rows and row.get("gap_to_top1") != b0_rows[row["token_id"]]:
                issues["gap_mismatch"].append([case["case_id"], row["token_id"]])
        if case["decomposition_mismatches"]:
            issues["decomposition_mismatch"].append([case["case_id"], case["decomposition_mismatches"]])
        if not case.get("state_restored"):
            issues["state_not_restored"].append(case["case_id"])
    return issues


def reproduce(cases: list[dict], references: dict[str, str]) -> dict[str, Any]:
    """Bitwise agreement with earlier receipts on every recipe both measured."""
    by_id = {c["case_id"]: c for c in measured(cases)}
    checks = {"b0_logits": 0, "b0_output": 0, "pushes": 0, "outputs": 0, "mismatches": []}
    for path, expected in references.items():
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("reference_receipt_changed:" + path)
        report = json.loads(raw)
        for old in report.get("cases", []):
            case = by_id.get(old["case_id"])
            if case is None:
                continue
            b0 = old["policies"].get("b0", {})
            if b0.get("status") == "ok":
                for key, ours, theirs in (("b0_logits", case["b0"]["baseline_logits_sha256"], b0["baseline_logits_sha256"]),
                                          ("b0_output", case["b0"]["output"], b0["output"])):
                    checks[key] += 1
                    if ours != theirs:
                        checks["mismatches"].append([case["case_id"], key])
            pushes = {p["recipe_id"]: p for p in case["pushes"]}
            for row in list(old["policies"].values()) + old.get("calibration_matrix", []):
                command = row.get("command")
                if row.get("status") != "ok" or not command or command["edits"][0]["id"] not in pushes:
                    continue
                mine = pushes[command["edits"][0]["id"]]
                checks["pushes"] += 1
                if mine["first_token_id"] != row["first_token_id"] or mine["readout"]["rows"] != row["readout"]["rows"]:
                    checks["mismatches"].append([case["case_id"], "push", mine["recipe_id"]])
                checks["outputs"] += 1
                if outcome(case, mine["first_token_id"])["output"] != row["output"]:
                    checks["mismatches"].append([case["case_id"], "output", mine["recipe_id"]])
        for old in report.get("rows", []):
            case = by_id.get(old["case_id"])
            if case is not None and old.get("boundary_found"):
                checks["b0_output"] += 1
                if case["b0"]["output"] != old["output"]:
                    checks["mismatches"].append([case["case_id"], "screen_b0_output"])
    return checks


# ---------------------------------------------------------------- lookups


def measured(cases: list[dict]) -> list[dict]:
    return [c for c in cases if c.get("status") == "ok" and c.get("boundary_found") and "b0" in c]


def outcome(case: dict, token: int) -> dict:
    if token == case["b0"]["first_token_id"]:
        return case["b0"]
    return next(o for o in case["outcomes"] if o["push_first_token_id"] == token and o["role"] == "outcome")


def family(source: str) -> str:
    if source.startswith("In order to"):
        return "purpose_first"
    return "purpose_last" if ", in order to" in source else "deadline_first"


def strata(case: dict) -> dict[str, str]:
    return {"b0_failure": failure_type(case["b0"]) or "success", "top1_class": reach_datasheet.top1_class(case),
            "family": family(case.get("source", ""))}


def catalog(case: dict, cap: float) -> list[dict]:
    """The v2 catalog order at ``cap`` when the table stored its packet; else the measured readouts."""
    packet = case.get("packets", {}).get(f"{cap:.2f}")
    if packet is not None:
        return packet["recipes"]
    return [{"recipe_id": p["recipe_id"], "kind": p["kind"], "component": p["component"], "sources": p["sources"]}
            for p in case["pushes"] if p["kind"] == "readout" and abs(p["cap"] - cap) < 1e-12]


def deterministic_choices(case: dict, cap: float, sheet: dict | None) -> dict[str, str | None]:
    recipes = catalog(case, cap)
    readouts = [r for r in recipes if r["kind"] == "readout"]
    patches = [r for r in recipes if r["kind"] == "patch"]
    digest = int.from_bytes(hashlib.sha256((case["case_id"] + ":17").encode()).digest(), "big")
    choices = {"b0": None, "fixed_readout": readouts[0]["recipe_id"] if readouts else None,
               "random_readout": readouts[digest % len(readouts)]["recipe_id"] if readouts else None}
    if patches:
        orthogonal = next((r for r in patches if r["component"] == "common_orthogonal"), None)
        choices.update(fixed=orthogonal and orthogonal["recipe_id"], random=recipes[digest % len(recipes)]["recipe_id"])
    if sheet is not None:
        klass = reach_datasheet.top1_class(case)
        entries = [reach_datasheet.recipe_entry(r, case["structure_readout"], sheet, case_top1_class=klass)
                   for r in readouts]
        choices["datasheet_rule"] = reach_datasheet.rule_choice(readouts, entries)
    return choices


def judged(case: dict, recipe_id: str | None) -> dict[str, Any]:
    """The table's outcome for one choice; None is noop."""
    if recipe_id is None:
        branch, token, measured_push = case["b0"], case["b0"]["first_token_id"], True
    else:
        push = next((p for p in case["pushes"] if p["recipe_id"] == recipe_id), None)
        if push is None:
            return {"recipe_id": recipe_id, "measured": False}
        token, branch, measured_push = push["first_token_id"], outcome(case, push["first_token_id"]), True
    return {"recipe_id": recipe_id, "measured": measured_push, "first_token_id": token,
            "first_token_changed": token != case["b0"]["first_token_id"], "joint_success": branch["joint_success"],
            "joint_success_v1": branch["evaluation_v1"]["joint_success"], "output": branch["output"]}


def oracle(case: dict, cap: float) -> dict[str, Any]:
    """Best single push at ``cap`` (B0 if none succeeds); headroom, never shown to a policy."""
    for recipe in catalog(case, cap):
        result = judged(case, recipe["recipe_id"])
        if result.get("joint_success") and not case["b0"]["joint_success"]:
            return result
    return judged(case, None)


# ---------------------------------------------------------------- datasheets


def pool_sheets(manifest: dict, cases: list[dict], *, provenance: dict, patch_cap: float | None,
                exclude_case: str | None = None) -> dict[float, dict]:
    rows = reach_datasheet.table_rows(measured(cases))
    return {cap: reach_datasheet.build(rows, cap=cap, provenance=provenance, patch_rows=rows, patch_cap=patch_cap,
                                       exclude_case=exclude_case) for cap in manifest["caps"]}


def loo_sheets(manifest: dict, cases: list[dict], cap: float, provenance: dict) -> dict[str, dict]:
    rows = reach_datasheet.table_rows(measured(cases))
    return {c["case_id"]: reach_datasheet.build(rows, cap=cap, provenance=provenance, patch_rows=rows, patch_cap=cap,
                                                exclude_case=c["case_id"]) for c in measured(cases)}


# ---------------------------------------------------------------- arms and comparisons


def arm_table(cases: list[dict], cap: float, sheets: dict[str, dict] | dict) -> dict[str, dict[str, dict]]:
    """Per case: every deterministic arm's choice and outcome, plus the oracle."""
    table = {}
    for case in measured(cases):
        sheet = sheets.get(case["case_id"], sheets.get("*"))
        choices = deterministic_choices(case, cap, sheet)
        table[case["case_id"]] = {arm: judged(case, recipe) for arm, recipe in choices.items()}
        table[case["case_id"]]["oracle"] = oracle(case, cap)
    return table


def tally(cases: list[dict], per_case: dict[str, dict[str, dict]], arm: str, key: str = "joint_success") -> dict:
    by_id = {c["case_id"]: c for c in measured(cases)}
    out = {"cases": 0, "joint_successes": 0, "rescues": 0, "losses": 0, "edits": 0, "first_token_changes": 0}
    for case_id, arms in per_case.items():
        if arm not in arms or not arms[arm].get("measured"):
            continue
        result = arms[arm]
        b0 = by_id[case_id]["b0"]
        base = b0["joint_success"] if key == "joint_success" else b0["evaluation_v1"]["joint_success"]
        out["cases"] += 1
        out["joint_successes"] += bool(result[key])
        out["rescues"] += bool(result[key] and not base)
        out["losses"] += bool(base and not result[key])
        out["edits"] += result["recipe_id"] is not None
        out["first_token_changes"] += bool(result["first_token_changed"])
    out["net"] = out["rescues"] - out["losses"]
    return out


def compare(per_case: dict[str, dict[str, dict]], left: str, right: str) -> dict[str, Any]:
    pairs = [(cid, arms[left], arms[right]) for cid, arms in per_case.items() if left in arms and right in arms]
    discordant = [{"case_id": cid, left: [a["recipe_id"], a.get("first_token_id"), a["joint_success"]],
                   right: [b["recipe_id"], b.get("first_token_id"), b["joint_success"]]}
                  for cid, a, b in pairs if a.get("joint_success") != b.get("joint_success")]
    return {"pairs": len(pairs), f"{left}_only": sum(bool(d[left][2]) for d in discordant),
            f"{right}_only": sum(bool(d[right][2]) for d in discordant), "discordant": discordant}


# ---------------------------------------------------------------- sweep (step 2)


def r_star(per_cap_net: dict[float, int]) -> dict[str, Any]:
    candidates = {c: per_cap_net[c] for c in R_STAR_CANDIDATES if c in per_cap_net}
    best = max(candidates.values())
    if best <= 0:
        return {"r_star": R_STAR_DEFAULT, "reason": "fallback_max_net_not_positive", "best_net": best}
    c_best = min(c for c, net in candidates.items() if net == best)
    if c_best != R_STAR_DEFAULT and best - candidates[R_STAR_DEFAULT] >= R_STAR_MARGIN:
        return {"r_star": c_best, "reason": f"net_beats_0.08_by_at_least_{R_STAR_MARGIN}", "best_net": best}
    return {"r_star": R_STAR_DEFAULT, "reason": "0.08_kept_margin_below_threshold_or_best", "best_net": best,
            "best_cap": c_best}


def loo_flip_prediction(cases: list[dict]) -> dict[str, Any]:
    rows = [r for r in reach_datasheet.table_rows(measured(cases)) if r["kind"] == "readout" and not r["already_top"]]
    out: dict[str, dict] = {}
    for row in rows:
        others = [o["closure"] for o in rows if o["case_id"] != row["case_id"] and o["token_id"] == row["token_id"]
                  and o["top1_class"] == row["top1_class"] and abs(o["cap"] - row["cap"]) < 1e-12]
        key = f"{row['piece']}|{row['top1_class']}"
        bucket = out.setdefault(key, {"n": 0, "uncalibrated": 0, "hits": 0, "misses": 0, "ties": 0,
                                      "third_token": 0})
        bucket["n"] += 1
        bucket["ties"] += row["tie"]
        bucket["third_token"] += bool(row["third_token"])
        if len(others) < reach_datasheet.MIN_N:
            bucket["uncalibrated"] += 1
            continue
        predicted = row["baseline_gap"] < max(others)
        bucket["hits" if predicted == row["target_became_next"] else "misses"] += 1
    return out


def sweep_summary(manifest: dict, cases: list[dict]) -> dict[str, Any]:
    caps = manifest["caps"]
    provenance = {"table_manifest_version": manifest["version"], "pool_cases": [c["case_id"] for c in measured(cases)]}
    patch_cap = (manifest["patch_caps"] or [None])[0]
    sheets = pool_sheets(manifest, cases, provenance=provenance, patch_cap=patch_cap)
    per_cap, by_stratum = {}, {}
    for cap in caps:
        arms = arm_table(cases, cap, {"*": sheets[cap]})
        per_cap[f"{cap:.2f}"] = {f"{arm}_{ev}": tally(cases, arms, arm, key)
                                 for arm in ("fixed_readout", "datasheet_rule", "oracle")
                                 for ev, key in (("v2", "joint_success"), ("v1", "joint_success_v1"))}
        for case in measured(cases):
            s = strata(case)
            for name in ("b0_failure", "top1_class", "family"):
                cell = by_stratum.setdefault(f"{cap:.2f}", {}).setdefault(name, {}).setdefault(
                    s[name], {"cases": 0, "fixed_readout_success": 0, "oracle_success": 0})
                cell["cases"] += 1
                cell["fixed_readout_success"] += bool(arms[case["case_id"]]["fixed_readout"]["joint_success"])
                cell["oracle_success"] += bool(arms[case["case_id"]]["oracle"]["joint_success"])
    rule = r_star({cap: per_cap[f"{cap:.2f}"]["fixed_readout_v2"]["net"] for cap in caps})
    reach = {}
    for cap, sheet in sheets.items():
        for key, entry in sheet["readout"].items():
            reach.setdefault(f"{entry['piece']}|{entry['top1_class']}", {})[f"{cap:.2f}"] = {
                "closure": entry["closure_logits"], "already_top": entry["already_top"],
                "target_became_next": entry["target_became_next"], "third_token": entry["third_token_changes"]}
    return {"per_cap": per_cap, "strata": by_stratum, "r_star": rule, "reach": reach,
            "loo_flip_prediction": loo_flip_prediction(cases),
            "patch_control": sheets[caps[0]]["patch"], "_sheets": sheets}


# ---------------------------------------------------------------- controller samples


def load_sampling(sampling_dir: Path, table_digest: str) -> tuple[dict, list[dict], str]:
    manifest_bytes = (sampling_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    raw = (sampling_dir / "results.json").read_bytes()
    report = json.loads(raw)
    if report["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest() or report["status"] != "complete":
        raise ValueError("sampling_incomplete_or_unbound")
    if manifest["table"]["table_manifest_sha256"] != table_digest:
        raise ValueError("sampling_ran_on_another_table")
    if len(report.get("resolved_models", [])) > 1:
        raise ValueError("resolved_controller_model_changed")
    return manifest, report["choices"], hashlib.sha256(raw).hexdigest()


def controller_summary(cases: list[dict], choices: list[dict], cap: float, sheets: dict) -> dict[str, Any]:
    by_id = {c["case_id"]: c for c in measured(cases)}
    determined = arm_table(cases, cap, sheets)
    arms: dict[str, dict] = {}
    per_case: dict[str, dict[str, list]] = {}
    for row in choices:
        case = by_id.get(row["case_id"])
        if case is None:
            continue
        # Intention to treat: an error or an invalid choice is a noop.
        recipe = row.get("recipe_id") if row["status"] == "selected" else None
        result = judged(case, recipe)
        kind = next((p["kind"] for p in case["pushes"] if p["recipe_id"] == recipe), None)
        source = next((p["sources"][0]["piece"] for p in case["pushes"] if p["recipe_id"] == recipe), None)
        stats = arms.setdefault(row["arm"], {"samples": 0, "noop": 0, "errors": 0, "invalid": 0, "patch": 0,
                                             "readout": 0, "ends_output_target": 0, "changed_first_token": 0,
                                             "unchanged_push": 0, "unmeasured": 0, "joint_successes": 0,
                                             "agrees_with_datasheet_rule": 0, "helpful_available": 0,
                                             "helpful_chosen": 0, "targets": {}})
        stats["samples"] += 1
        stats["noop"] += recipe is None
        stats["errors"] += row["status"] == "error"
        stats["invalid"] += row["status"] == "invalid"
        if not result.get("measured"):
            stats["unmeasured"] += 1
            continue
        if kind:
            stats[kind] += 1
        if source is not None:
            stats["targets"][source] = stats["targets"].get(source, 0) + 1
            stats["ends_output_target"] += source.strip() in reach_datasheet.ENDS_OUTPUT
        stats["changed_first_token"] += bool(result["first_token_changed"])
        stats["unchanged_push"] += recipe is not None and not result["first_token_changed"]
        stats["joint_successes"] += bool(result["joint_success"])
        stats["agrees_with_datasheet_rule"] += recipe == determined[case["case_id"]].get("datasheet_rule", {}).get("recipe_id")
        if determined[case["case_id"]]["oracle"]["joint_success"] and not case["b0"]["joint_success"]:
            stats["helpful_available"] += 1
            stats["helpful_chosen"] += bool(result["joint_success"])
        per_case.setdefault(case["case_id"], {}).setdefault(row["arm"], []).append(result)
    mean = {cid: {arm: sum(bool(r["joint_success"]) for r in results) / len(results) for arm, results in arms_.items()}
            for cid, arms_ in per_case.items()}
    return {"arms": arms, "mean_joint_success_per_case": mean, "samples": per_case}


def controller_comparisons(mean: dict[str, dict[str, float]], determined: dict[str, dict[str, dict]],
                           case_ids: list[str]) -> dict[str, Any]:
    """Pre-declared paired contrasts on per-case mean joint success over the k samples."""
    def value(case_id: str, arm: str) -> float | None:
        if arm in mean.get(case_id, {}):
            return mean[case_id][arm]
        result = determined.get(case_id, {}).get(arm)
        return None if result is None or not result.get("measured") else float(bool(result["joint_success"]))

    out = {}
    for left, right in (("controller_datasheet", "controller"), ("controller_datasheet", "datasheet_rule"),
                        ("controller", "fixed_readout"), ("controller_datasheet", "fixed_readout"),
                        ("controller_mechanics", "controller")):
        rows = [(cid, value(cid, left), value(cid, right)) for cid in case_ids]
        rows = [(cid, a, b) for cid, a, b in rows if a is not None and b is not None]
        out[f"{left}_vs_{right}"] = {"cases": len(rows), "left_higher": sum(a > b for _, a, b in rows),
                                     "right_higher": sum(b > a for _, a, b in rows),
                                     "left_mean_total": round(sum(a for _, a, _ in rows), 3),
                                     "right_mean_total": round(sum(b for _, _, b in rows), 3),
                                     "discordant": [[cid, a, b] for cid, a, b in rows if a != b]}
    return out


def retest(choices: list[dict], reference_paths: list[str]) -> dict[str, Any]:
    """Agreement of the v2 arm with the PR #39 choice on a byte-identical packet."""
    earlier = {}
    for path in reference_paths:
        report = json.loads(Path(path).read_text())
        for case in report.get("cases", []):
            trace = case["policies"].get("controller", {}).get("selection", {})
            if trace.get("packet_sha256"):
                chosen = json.loads(trace["response"]).get("recipe_id") if trace.get("response") else None
                earlier[case["case_id"]] = (trace["packet_sha256"], chosen)
    rows = [r for r in choices if r["arm"] == "controller" and r["case_id"] in earlier]
    same_packet = [r for r in rows if r["packet_sha256"] == earlier[r["case_id"]][0]]
    return {"samples": len(rows), "byte_identical_packets": len(same_packet),
            "same_choice": sum(r.get("recipe_id") == earlier[r["case_id"]][1] for r in same_packet)}


# ---------------------------------------------------------------- CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("table_dir", type=Path)
    parser.add_argument("--mode", choices=("known16", "sweep", "rescue2"), required=True)
    parser.add_argument("--cap", type=float, help="known16 and rescue2: the cap that arms are scored at")
    parser.add_argument("--control-cap", type=float, help="rescue2: the 0.08 cap-control arms")
    parser.add_argument("--datasheet", type=Path, help="rescue2: datasheet v1 at --cap")
    parser.add_argument("--control-datasheet", type=Path, help="rescue2: datasheet at --control-cap")
    parser.add_argument("--sampling-dir", type=Path)
    parser.add_argument("--write-loo-datasheets", type=Path, help="known16: write leave-one-out sheets and stop")
    args = parser.parse_args(argv)
    manifest, cases, binding = load_table(args.table_dir)
    summary: dict[str, Any] = {"mode": args.mode, **binding, "invariants": invariants(cases),
                               "reproduction": reproduce(cases, manifest["references"]),
                               "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for case in cases:
        case["source"] = next(c["source"] for c in manifest["cases"] if c["case_id"] == case["case_id"])
    provenance = {"table_manifest_sha256": binding["manifest_sha256"],
                  "table_parts": {a["part"]: a["results_sha256"] for a in binding["attempts"]}}
    if args.mode == "known16":
        sheets = loo_sheets(manifest, cases, args.cap, provenance)
        if args.write_loo_datasheets:
            args.write_loo_datasheets.write_text(json.dumps(sheets, indent=2, sort_keys=True) + "\n")
            print(json.dumps({"datasheets": len(sheets), "path": str(args.write_loo_datasheets)}))
            return 0
        determined = arm_table(cases, args.cap, sheets)
        summary["arms"] = {arm: tally(cases, determined, arm) for arm in
                           ("b0", "fixed", "fixed_readout", "random", "random_readout", "datasheet_rule", "oracle")}
        if args.sampling_dir:
            sampling, choices, sha = load_sampling(args.sampling_dir, binding["manifest_sha256"])
            summary["sampling_results_sha256"] = sha
            summary["controllers"] = controller_summary(cases, choices, args.cap, sheets)
            summary["retest"] = retest(choices, list(manifest["references"]))
    elif args.mode == "sweep":
        result = sweep_summary(manifest, cases)
        for cap, sheet in result.pop("_sheets").items():
            (args.table_dir / f"datasheet_cap_{cap:.2f}.json").write_text(json.dumps(sheet, indent=2, sort_keys=True) + "\n")
            result.setdefault("datasheets", {})[f"{cap:.2f}"] = sheet["datasheet_sha256"]
        summary.update(result)
    else:
        sheet = json.loads(args.datasheet.read_text())
        control = json.loads(args.control_datasheet.read_text())
        reach_datasheet.verify(sheet, cap=args.cap)
        reach_datasheet.verify(control, cap=args.control_cap)
        determined = arm_table(cases, args.cap, {"*": sheet})
        controls = arm_table(cases, args.control_cap, {"*": control})
        for case_id, arms in controls.items():
            determined[case_id]["fixed_readout_at_control_cap"] = arms["fixed_readout"]
            determined[case_id]["datasheet_rule_at_control_cap"] = arms["datasheet_rule"]
        for split in ("rescue", "harm", "confirmation"):
            subset = [c for c in cases if c["split"] == split]
            rows = {c["case_id"]: determined[c["case_id"]] for c in measured(subset)}
            summary.setdefault("splits", {})[split] = {arm: tally(subset, rows, arm) for arm in next(iter(rows.values()), {})}
        if args.sampling_dir:
            sampling, choices, sha = load_sampling(args.sampling_dir, binding["manifest_sha256"])
            summary["sampling_results_sha256"] = sha
            summary["controllers"] = controller_summary(cases, choices, args.cap, {"*": sheet})
            summary["controller_comparisons"] = controller_comparisons(
                summary["controllers"]["mean_joint_success_per_case"], determined,
                [c["case_id"] for c in cases if c["split"] == "rescue"])
        summary["comparisons"] = {f"{a}_vs_{b}": compare(determined, a, b) for a, b in (
            ("datasheet_rule", "fixed_readout"), ("fixed_readout", "fixed_readout_at_control_cap"),
            ("datasheet_rule", "datasheet_rule_at_control_cap"))}
        summary["fresh_datasheet_check"] = fresh_check(cases, args.cap, sheet)
    (args.table_dir / f"analysis_{args.mode}.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"mode": args.mode, "missing": binding["missing_pool_indices"],
                      "mismatches": len(summary["reproduction"]["mismatches"]),
                      **({"r_star": summary["r_star"]} if "r_star" in summary else {})}))
    return 0


def fresh_check(cases: list[dict], cap: float, sheet: dict) -> dict[str, Any]:
    """Out-of-sample: fresh closures inside the sheet's [min, max]; the rule's flip predictions."""
    out = {"inside": 0, "outside": 0, "uncalibrated": 0, "rule_predicted_change": 0, "rule_change_happened": 0}
    for row in reach_datasheet.table_rows(measured(cases)):
        if row["kind"] != "readout" or row["already_top"] or abs(row["cap"] - cap) > 1e-12:
            continue
        entry = sheet["readout"].get(f"{row['token_id']}|{row['top1_class']}")
        closure = entry and entry["closure_logits"]
        if not closure or closure["n"] < reach_datasheet.MIN_N:
            out["uncalibrated"] += 1
            continue
        out["inside" if closure["min"] <= row["closure"] <= closure["max"] else "outside"] += 1
        if row["baseline_gap"] < closure["max"]:
            out["rule_predicted_change"] += 1
            out["rule_change_happened"] += row["target_became_next"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
