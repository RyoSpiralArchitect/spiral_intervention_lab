from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import time

import pytest
import torch

from SpiralInterventionLab.examples.structural_rescue_benchmark import (
    ForwardLimit, Meter, aggregate, branch, choose, compact_packet, main, readout, record_branch, unavailable_policies,
)
from SpiralInterventionLab.runtime import clause_radial
from SpiralInterventionLab.tests.test_clause_components import component_worker


@pytest.mark.parametrize("reply,status", [
    ('{"decision":"probe","recipe_id":"r2"}', "selected"),
    ('{"decision":"noop"}', "noop"),
    ('{"decision":"probe","recipe_id":"stale"}', "error"),
    ('{"decision":"probe","recipe_id":"r2","answer":"secret"}', "error"),
    ('not json', "error"),
])
def test_choice_schema_rejects_answer_channels_and_unoffered_ids(reply, status):
    packet = {"case_id": "test", "recipes": [
        {"recipe_id": "r1", "component": "common"},
        {"recipe_id": "r2", "component": "common_orthogonal"}]}
    provider = SimpleNamespace(provider_name="fake", model_name="fake", complete=Mock(return_value=
        SimpleNamespace(text=reply, usage={"tokens": 3}, metadata={})))
    chosen, trace = choose("controller", packet, provider=provider)
    assert trace["status"] == status
    assert trace["provider_calls"] == 1 and provider.complete.call_count == 1
    assert chosen == ("r2" if status == "selected" else None)
    assert "latency_seconds" in trace
    assert choose("fixed", packet, fixed_recipe_id="r2")[0] == "r2"
    assert choose("random", packet) == choose("random", packet)


def test_packet_has_no_role_answers_or_tensors_and_preserves_recipe_identity():
    worker = component_worker()
    recipes = clause_radial.offer(worker)["recipes"]
    packet = compact_packet(worker, recipes, "case", {"structure_readout_rows": []})
    assert set(packet) == {"version", "case_id", "generated_tail", "structure_readout", "recipes", "budget", "production_apply_allowed"}
    assert "source_expressions" not in packet["recipes"][0]
    assert packet["recipes"][0]["recipe_id"] == recipes[0]["component_recipe_id"]


def test_raw_readout_gap_is_anchored_to_baseline_winner_even_after_flip():
    row = readout(torch.tensor([3., 2., 0.]), torch.tensor([2., 4., 0.]), [1])["rows"][0]
    assert row["baseline_gap_to_fixed_winner"] == 1
    assert row["edited_gap_to_fixed_winner"] == -2
    assert row["gap_closure_fraction"] == 3
    assert readout(torch.ones(2), torch.ones(2), [0])["rows"][0]["gap_closure_fraction"] is None


def test_meter_refuses_forward_25_and_restores_wrapper(tmp_path):
    runtime = SimpleNamespace(run_with_cache=Mock(return_value=(torch.zeros(1, 1, 3), {})))
    worker = SimpleNamespace(runtime_state=runtime, _collect_active_edits=lambda: [])
    original = runtime.run_with_cache
    meter = Meter(worker, deadline=time.monotonic()+10, disk_path=tmp_path)
    meter.reset("diagnostic")
    with patch("shutil.disk_usage", return_value=SimpleNamespace(free=20*2**30)):
        with meter.install():
            for _ in range(24):
                runtime.run_with_cache(torch.ones(1, 2))
            with pytest.raises(ForwardLimit):
                runtime.run_with_cache(torch.ones(1, 2))
    assert runtime.run_with_cache is original
    assert original.call_count == 24
    assert meter.counts["diagnostic"] == {"attempted": 24, "completed": 24}


def test_normal_cap_branch_expires_after_one_token_and_restores_prefix(tmp_path):
    worker = component_worker()
    worker.stop_checker = lambda text: len(text) > 4
    recipe = next(r for r in clause_radial.offer(worker)["recipes"] if r["component"] == "common_orthogonal")
    task = SimpleNamespace(evaluate=lambda text: {"output": text})
    before = worker.final_text()
    meter = Meter(worker, deadline=time.monotonic()+30, disk_path=tmp_path)
    meter.reset("diagnostic")
    meter.boundary_length = worker._current_token_tensor().shape[-1]
    with patch("shutil.disk_usage", return_value=SimpleNamespace(free=20*2**30)):
        with meter.install():
            result, raw = branch(worker, task, recipe, meter)
    assert raw is not None
    assert result["state_restored"]
    assert result["runtime_telemetry"][0]["hook_call_count"] == 1
    assert not worker._collect_active_edits() and worker.final_text() == before
    assert result["command"]["edits"][0]["budget"]["step_size"] == 0.08


def test_provider_failure_is_excluded_from_efficacy_pairs():
    ok = {"status": "ok", "joint_success": False, "output": "bad", "role_status": "fail"}
    failed = {**ok, "selection": {"status": "error"}}
    result = aggregate([{"split": "confirmation", "policies": {"b0": ok, "controller": failed}}])
    assert result["confirmation"]["controller"]["completed_pairs"] == 0
    assert result["confirmation"]["controller"]["selection_errors"] == 1


def test_no_boundary_reuses_baseline_without_duplicating_forward_cost():
    baseline = {"status": "ok", "joint_success": False, "output": "bad", "role_status": "fail",
                "forwards": {"continuation": {"attempted": 5, "completed": 5}}}
    policies = unavailable_policies(baseline, ["b0", "fixed", "random", "controller"])
    assert baseline["forwards"]["continuation"]["attempted"] == 5
    assert all(row["forwards"] == {} and row["reused_baseline"] for row in policies.values())
    result = aggregate([{"split": "confirmation", "comparison_eligible": False, "policies": policies}])
    assert result["confirmation"]["controller"]["completed_pairs"] == 1
    assert result["confirmation"]["controller"]["eligible_pairs"] == 0
    assert result["confirmation"]["controller"]["unavailable_boundary_cases"] == 1


def test_failed_branch_keeps_attempts_and_selection_metadata():
    meter = SimpleNamespace(counts={"diagnostic": {"attempted": 3, "completed": 2}})
    outcome = {"selection": {"status": "selected", "provider_calls": 1}}
    with patch("SpiralInterventionLab.examples.structural_rescue_benchmark.branch", side_effect=ForwardLimit("cap")):
        with pytest.raises(ForwardLimit):
            record_branch(None, None, None, meter, outcome)
    assert outcome["status"] == "error"
    assert outcome["forwards"] == meter.counts
    assert outcome["selection"]["provider_calls"] == 1


def test_pre_model_failure_is_persisted_without_any_provider_call(tmp_path):
    output = tmp_path / "run"
    assert main(["--prepare", "--output-dir", str(output)]) == 0
    digest = hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest()
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"checkpoint_sha256": {"config.json": "hash"}}))
    with patch("SpiralInterventionLab.examples.structural_rescue_benchmark.verify_checkpoint",
               side_effect=ValueError("checkpoint_mismatch")), patch(
               "SpiralInterventionLab.examples.structural_rescue_benchmark.create_controller_provider") as provider:
        with pytest.raises(ValueError, match="checkpoint_mismatch"):
            main(["--run", "--output-dir", str(output), "--manifest-sha256", digest,
                  "--worker-model-path", str(tmp_path), "--checkpoint-manifest", str(checkpoint)])
    assert provider.call_count == 0
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "error" and report["cases"] == []
    assert (output / "source_snapshot.tgz").exists()


def test_fixed_policy_does_not_fall_through_to_another_available_pair():
    packet = {"recipes": [{"recipe_id": "second_pair", "component": "common_orthogonal"}]}
    selected, trace = choose("fixed", packet, fixed_recipe_id="unavailable_first_pair")
    assert selected is None and trace["status"] == "fixed_recipe_unavailable"
