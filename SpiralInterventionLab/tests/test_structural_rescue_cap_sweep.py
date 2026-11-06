import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch
import time

import pytest

from SpiralInterventionLab.examples import structural_rescue_benchmark_v2 as v2
from SpiralInterventionLab.examples import structural_rescue_cap_sweep as sweep
from SpiralInterventionLab.examples.structural_rescue_benchmark import Meter
from SpiralInterventionLab.runtime import structural_contract
from SpiralInterventionLab.tasks import structural_rescue
from SpiralInterventionLab.tests.test_structural_contract import contract_worker
from SpiralInterventionLab.tests.test_structural_rescue_benchmark_v2 import TERMS, run_branch


def test_manifest_is_separate_stable_and_calibration_only():
    assert sweep.manifest_bytes() == sweep.manifest_bytes()
    frozen = sweep.manifest()
    assert frozen["version"] not in {structural_rescue.benchmark_manifest()["version"],
                                     v2.manifest()["version"]}
    v1_ids = {c["case_id"]: c for c in structural_rescue.benchmark_manifest()["cases"]}
    assert [c["case_id"] for c in frozen["cases"]] == list(sweep.CALIBRATION_CASES)
    assert all(v1_ids[c["case_id"]] == c for c in frozen["cases"])
    assert all(v1_ids[c["case_id"]]["split"] == "calibration" for c in frozen["cases"])
    assert frozen["caps"] == list(sweep.CAPS) == sorted(sweep.CAPS)
    assert frozen["disk_floor_gib"] == 6 and frozen["protocol_deviations"]
    assert "policies" not in frozen and "controller" not in json.dumps(frozen)


def test_sweep_recipes_vary_only_the_cap():
    worker = contract_worker()
    task = SimpleNamespace(_SCENARIOS=[{"required_terms": list(TERMS)}])
    where = structural_contract.surface(worker, 11)
    by_cap = {}
    for cap in sweep.CAPS:
        recipes, policy = sweep.sweep_recipes(worker, task, 11, cap)
        assert policy.write_contract.relative_write_cap == cap
        assert all(r["kind"] == "readout" and r["surface"] == where for r in recipes)
        assert all(r["relative_write_cap"] == r["requested_relative_write"] == cap for r in recipes)
        by_cap[cap] = recipes
    pieces = [r["sources"][0]["piece"] for r in by_cap[sweep.CAPS[0]]]
    assert pieces == [",", " while"]
    for cap in sweep.CAPS[1:]:
        assert [r["sources"][0]["piece"] for r in by_cap[cap]] == pieces
        for low, high in zip(by_cap[sweep.CAPS[0]], by_cap[cap]):
            assert high["alpha"] == pytest.approx(low["alpha"] * cap / sweep.CAPS[0])
            assert high["recipe_id"] != low["recipe_id"]


def test_branch_enforces_the_branch_cap_not_the_default(tmp_path):
    worker = contract_worker()
    sources = structural_contract.eligible_sources(worker, TERMS)
    where = structural_contract.surface(worker, 11)
    policy = structural_contract.policy_for(worker, TERMS, cap=0.08)
    recipe = structural_contract.readout_recipes(worker, sources, where, cap=0.08)[0]
    result, _ = run_branch(worker, recipe, policy, tmp_path)
    realized = result["runtime_telemetry"][0]["relative_write_norm"]
    assert realized <= 0.08 + v2.CAST_TOLERANCE
    # The same realized write must fail closed against a smaller declared cap.
    if realized > 0.04 + v2.CAST_TOLERANCE:
        worker2 = contract_worker()
        worker2.stop_checker = lambda text: len(text) > 4
        meter = Meter(worker2, deadline=time.monotonic() + 30, disk_path=tmp_path)
        meter.reset("diagnostic")
        meter.boundary_length = worker2._current_token_tensor().shape[-1]
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=20 * 2**30)):
            with meter.install():
                with pytest.raises(RuntimeError, match="exceeded_cap"):
                    v2.branch(worker2, SimpleNamespace(evaluate=lambda t: {}), recipe, meter, policy, cap=0.04)


def test_summarize_keeps_error_rows_and_joins_b0():
    cases = [{"case_id": "c1", "branches": {
        "b0": {"first_token_id": 5},
        "sweep": [
            {"cap": 0.04, "source_piece": ",", "source_token_id": 3, "recipe_id": "r1",
             "status": "ok", "first_token_id": 3, "joint_success": True,
             "readout": {"rows": [{"token_id": 3, "baseline_gap_to_fixed_winner": 1.0,
                                   "gap_closure_fraction": 0.5}]},
             "runtime_telemetry": [{"relative_write_norm": 0.04}]},
            {"cap": 0.12, "source_piece": ".", "source_token_id": 2, "recipe_id": "r2",
             "status": "error"}]}}]
    rows = sweep.summarize(cases)["reach_rows"]
    assert rows[0]["first_token_changed"] is True and rows[0]["gap_closure"] == 0.5
    assert rows[0]["relative_write_norm"] == 0.04
    assert rows[1] == {"cap": 0.12, "source_piece": ".", "recipe_id": "r2",
                       "case_id": "c1", "status": "error"}


def test_pre_model_failure_is_persisted_without_any_provider(tmp_path):
    output = tmp_path / "run"
    assert sweep.main(["--prepare", "--output-dir", str(output)]) == 0
    digest = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"checkpoint_sha256": {"config.json": "hash"}}))
    with patch.object(sweep, "verify_checkpoint", side_effect=ValueError("checkpoint_mismatch")):
        with pytest.raises(ValueError, match="checkpoint_mismatch"):
            sweep.main(["--run", "--output-dir", str(output), "--manifest-sha256", digest,
                        "--worker-model-path", str(tmp_path), "--checkpoint-manifest", str(checkpoint)])
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "error" and report["cases"] == []
    with pytest.raises(ValueError, match="manifest"):
        sweep.main(["--run", "--output-dir", str(output), "--manifest-sha256", "0" * 64,
                    "--worker-model-path", str(tmp_path), "--checkpoint-manifest", str(checkpoint)])
