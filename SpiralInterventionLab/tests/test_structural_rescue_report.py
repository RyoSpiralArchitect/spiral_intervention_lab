import hashlib
import io
import json
import tarfile
import subprocess
import sys
from pathlib import Path

import pytest

from SpiralInterventionLab.examples.structural_rescue_benchmark import aggregate
from SpiralInterventionLab.examples.structural_rescue_benchmark import capture_sources
from SpiralInterventionLab.examples.summarize_structural_rescue import main, markdown, summarize
from SpiralInterventionLab.examples.structural_rescue_selection import constraint_consistency


def test_report_counts_shared_work_once_and_retains_role_failures_and_costs():
    baseline = {"status": "ok", "joint_success": False, "output": "role swap.",
                "role_status": "fail", "legacy_score": 1.0, "command": None,
                "forwards": {"diagnostic": {"attempted": 3, "completed": 3}},
                "selection": {"status": "noop", "provider_calls": 0}}
    controller = {**baseline, "selection": {"status": "selected", "provider_calls": 1,
                                              "usage": {"input_tokens": 12, "output_tokens": 5}}}
    case = {"case_id": "example", "split": "confirmation", "comparison_eligible": True,
            "shared_setup_forwards": {"setup": {"attempted": 8, "completed": 8}},
            "policies": {"b0": baseline, "controller": controller}, "calibration_matrix": []}
    report = {"status": "complete", "policies_requested": ["b0", "controller"],
              "cases": [case], "aggregate": aggregate([case]), "manifest_sha256": "frozen"}
    summary = summarize(report)
    assert summary["shared_setup_forwards"] == 8
    assert summary["policy_actual_forward_totals"] == {"b0": 3, "controller": 3}
    assert summary["total_provider_calls"] == 1
    assert summary["provider_token_totals"] == {"input_tokens": 12, "output_tokens": 5}
    assert summary["aggregate"]["confirmation"]["controller"]["joint_successes"] == 0
    assert summary["aggregate"]["confirmation"]["controller"]["joint_success_delta_vs_b0"] == 0
    assert "1.0" not in markdown(summary)  # No lexical-only success is promoted into the task table.


@pytest.mark.parametrize("tamper", ["none", "manifest", "source"])
def test_offline_report_requires_frozen_manifest_and_executed_source(tmp_path, tamper):
    frozen, code = b"frozen manifest", b"executed code"
    digest = hashlib.sha256(frozen).hexdigest()
    (tmp_path / "manifest.json").write_bytes(frozen if tamper != "manifest" else b"changed")
    (tmp_path / "execution_plan.json").write_text(json.dumps({"manifest_sha256": digest,
        "code_sha256": {"/workspace/SpiralInterventionLab/unit.py": hashlib.sha256(code).hexdigest()}}))
    report = {"status": "complete", "policies_requested": [], "cases": [], "aggregate": {},
              "manifest_sha256": digest}
    (tmp_path / "results.json").write_text(json.dumps(report))
    with tarfile.open(tmp_path / "source_snapshot.tgz", "w:gz") as snapshot:
        value = code if tamper != "source" else b"modified"
        member = tarfile.TarInfo("SpiralInterventionLab/unit.py")
        member.size = len(value)
        snapshot.addfile(member, io.BytesIO(value))
    if tamper != "none":
        with pytest.raises(ValueError, match="mismatch"):
            main([str(tmp_path)])
        assert not (tmp_path / "analysis.json").exists()
    else:
        assert main([str(tmp_path)]) == 0
        assert json.loads((tmp_path / "analysis.json").read_text())["manifest_and_executed_source_verified"]


def test_runner_captures_sources_that_offline_analyzer_accepts(tmp_path):
    frozen = b"frozen"
    digest = hashlib.sha256(frozen).hexdigest()
    hashes = capture_sources(tmp_path)
    assert "SpiralInterventionLab/runtime/worker.py" in hashes
    assert "pyproject.toml" in hashes
    assert all(not key.startswith("/") for key in hashes)
    (tmp_path / "manifest.json").write_bytes(frozen)
    (tmp_path / "execution_plan.json").write_text(json.dumps({"manifest_sha256": digest, "code_sha256": hashes}))
    (tmp_path / "results.json").write_text(json.dumps({"status": "complete", "policies_requested": [],
        "cases": [], "aggregate": {}, "manifest_sha256": digest}))
    assert main([str(tmp_path)]) == 0
    analyzer = Path(__file__).parents[1] / "examples" / "summarize_structural_rescue.py"
    standalone = subprocess.run([sys.executable, "-S", str(analyzer), str(tmp_path)],
                                capture_output=True, text=True)
    assert standalone.returncode == 0, standalone.stderr
    with pytest.raises(FileExistsError):
        capture_sources(tmp_path)


def test_failed_conditions_still_count_actual_work_without_entering_efficacy():
    failure = {"status": "error", "forwards": {"diagnostic": {"attempted": 3, "completed": 2}},
               "selection": {"provider_calls": 1}}
    case = {"case_id": "case", "split": "confirmation", "policies": {"controller": failure},
            "calibration_matrix": [failure], "shared_fallback_forwards": {"continuation": {"attempted": 5}}}
    summary = summarize({"status": "error", "manifest_sha256": "hash", "cases": [case],
                         "policies_requested": ["controller"], "aggregate": aggregate([case])})
    assert summary["policy_actual_forward_totals"]["controller"] == 3
    assert summary["calibration_matrix_actual_forwards"] == 3
    assert summary["shared_fallback_forwards"] == 5
    assert summary["total_provider_calls"] == 1
    assert summary["conditions"] == [] and summary["calibration_matrix"] == []


def test_selector_report_distinguishes_rescue_regression_noop_and_missing_pairs():
    def row(ok, recipe=None, selection="selected", provider_calls=1):
        return {"status": "ok", "joint_success": ok, "role_status": "pass" if ok else "fail",
                "constraints_ok": True, "legacy_score": 1.0, "output": "Observed.",
                "selected_recipe_id": recipe,
                "selection": {"status": selection, "provider_calls": provider_calls}}

    def case(key, b0, fixed, controller, eligible=True):
        return {"case_id": key, "split": "heldout", "comparison_eligible": eligible,
                "policies": {"b0": b0, "fixed_routed": fixed, "controller": controller},
                "calibration_matrix": []}

    cases = [
        case("avoid", row(True), row(False, "comma"), row(True, "while")),
        case("rescue", row(False), row(False, "same"), row(True, "other")),
        case("abstain", row(True), row(True, "comma"), row(True, selection="noop")),
        case("no_catalog", row(True), row(True), row(True, selection="noop", provider_calls=0)),
        case("no_boundary", row(True), row(True), row(True), eligible=False),
        case("provider_error", row(True), row(True), row(True, selection="error")),
    ]
    summary = summarize({"status": "running", "manifest_sha256": "frozen", "cases": cases,
                         "case_ids_requested": [c["case_id"] for c in cases] + ["pending"],
                         "policies_requested": ["b0", "fixed_routed", "controller"], "aggregate": {}})
    pair, = summary["selector_pairs"]
    assert pair["expected"] == 7
    assert pair["unpaired"] == ["provider_error", "pending"]
    assert pair["unavailable_boundary"] == ["no_boundary"]
    assert pair["strata"]["b0_pass"]["challenger_only"] == 1
    assert pair["strata"]["b0_pass"]["challenger_provider_noop"] == 1
    assert pair["strata"]["b0_fail"]["challenger_only"] == 1
    assert "| b0_pass | 3 | 2 | 0 | 1 |" in markdown(summary)


def test_inconsistent_constraint_labels_are_flagged_without_rescoring():
    arm = {"status": "ok", "constraints_ok": False, "constraint_violations": [],
           "single_sentence_ok": True, "joint_success": False,
           "surface_notes": ["required_action_covered_by_inflection:give"]}
    report = {"cases": [{"case_id": "inflection", "policies": {"b0": arm,
                          "real_failure": {**arm, "constraint_violations": ["over_word_budget"]}}}]}
    findings = constraint_consistency(report)
    assert len(findings) == 1 and findings[0]["policy"] == "b0"
    assert findings[0]["joint_success_as_recorded"] is False
    assert arm["joint_success"] is False
