import hashlib
import io
import json
import tarfile

import pytest

from SpiralInterventionLab.examples.structural_rescue_benchmark import aggregate
from SpiralInterventionLab.examples.structural_rescue_benchmark import capture_sources
from SpiralInterventionLab.examples.summarize_structural_rescue import main, markdown, summarize


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
