from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.examples.replay_prefix_response import plan_from_events
from SpiralInterventionLab.runtime import diagnostic_reuse
from SpiralInterventionLab.runtime.prefix_probe import FrozenCandidate, measure_prefix, prefix_plan
from SpiralInterventionLab.runtime.response_probe import tensor_identity
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime


def review_worker():
    worker = object.__new__(HookedTransformerWorkerRuntime)
    worker.max_diagnostic_calls_per_run = 6
    worker.diagnostic_result_window = 12
    worker._diagnostic_results = []
    worker._diagnostic_review_cache = {}
    worker._evidence_inspection_count = 0
    worker._pending_diagnostic_events = []
    worker._last_packet = {}
    worker._steps = 5
    worker._segments = [SimpleNamespace(kind="output", token_ids=[1])]
    worker._collect_active_edits = lambda: []
    return worker


def closed_result():
    return {"diagnostic": "activation_patch_candidate_review", "status": "ok", "objective_bundle_key": "a",
            "target_piece_binding_seed_matrix_summary": {"status": "dose_matched_response_complete",
                "state_restored": True, "measurement_context_id": "c1"},
            "activation_patch_compile_preview_created": False,
            "activation_patch_compile_preview_blocked_reason": "rank_carrier_not_target_actuator",
            "evidence_rows": [{"objective_bundle_key": "a", "operator_recipe_id": "recipe",
                "actual_delta_class": "rank_carrier", "execution_id": "e1", "observable_id": "o1"}]}


def test_closed_review_is_reused_without_budget_and_reports_changed_prefix():
    worker = review_worker()
    request = {"diagnostic": "activation_patch_candidate_review", "objective_bundle_key": "a"}
    with patch.object(worker, "_execute_controller_diagnostic_request", return_value=closed_result()) as execute:
        first = worker.request_controller_diagnostics(request)[0]
        assert first["budget_after"]["diagnostic_calls_left"] == 5
        worker._steps += 1
        worker._segments[0].token_ids.append(2)
        again = worker.request_controller_diagnostics({**request, "reason": "different wording"})[0]
        assert execute.call_count == 1
    assert again["status"] == "review_reused" and again["prefix_changed_since_review"]
    assert again["new_measurement_count"] == again["physical_replay_count"] == 0
    assert again["budget_before"] == again["budget_after"]
    assert len(worker._diagnostic_results) == 1
    assert not again["certified_for_apply"] and not again["production_apply_allowed"]
    assert _compact_diagnostic_result(again)["evidence_scope"] == "historical_review_not_current_measurement"
    hints = diagnostic_reuse.reuse_hints(worker)
    assert _compact_controller_payload({"strategy_hints": hints})["strategy_hints"]["diagnostic_review_reuse"]


def test_new_execution_or_intent_reopens_review_but_never_promotion_cache():
    worker = review_worker()
    request = {"diagnostic": "activation_patch_candidate_review", "objective_bundle_key": "a"}
    with patch.object(worker, "_execute_controller_diagnostic_request", return_value=closed_result()) as execute:
        worker.request_controller_diagnostics(request)
        worker._diagnostic_results.append({"evidence_rows": [{"objective_bundle_key": "a",
            "operator_recipe_id": "recipe", "execution_id": "new", "observable_id": "o2"}]})
        worker.request_controller_diagnostics(request)
        worker.request_controller_diagnostics({**request, "dose_grid": [0.16]})
        assert execute.call_count == 3
    for name in ("matched_response_probe", "activation_patch_production_shadow_replay",
                 "activation_patch_promotion_gate_review", "activation_patch_production_trial_gate_review"):
        assert diagnostic_reuse.review_key({**request, "diagnostic": name}, {}, worker._diagnostic_results) is None


def test_incomplete_review_is_not_reused_and_reuse_works_after_budget_exhaustion():
    worker = review_worker()
    request = {"diagnostic": "activation_patch_candidate_review", "objective_bundle_key": "a"}
    result = closed_result()
    result["target_piece_binding_seed_matrix_summary"]["status"] = "unavailable"
    with patch.object(worker, "_execute_controller_diagnostic_request", return_value=result) as execute:
        worker.request_controller_diagnostics(request)
        worker.request_controller_diagnostics(request)
        assert execute.call_count == 2
    with patch.object(worker, "_execute_controller_diagnostic_request", return_value=closed_result()) as execute:
        worker.request_controller_diagnostics(request)
        worker.max_diagnostic_calls_per_run = len(worker._diagnostic_results)
        cached = worker.request_controller_diagnostics(request)[0]
        assert cached["budget_after"]["diagnostic_calls_left"] == 0 and execute.call_count == 1


def test_closed_review_does_not_cache_unsafe_or_positive_permission_results():
    result = closed_result()
    for change in ({"production_trial_allowed": True}, {"activation_patch_compile_preview_created": True},
                   {"production_apply_allowed": True}, {"status": "error"}):
        assert not diagnostic_reuse.closed_review({**result, **change})


def test_prefix_plan_uses_connective_positions_not_effects_or_objective_names():
    rows = [{"step": i + 1, "generated_tail": text} for i, text in enumerate(
        [" Nora", " Nora takes", " Nora takes sample", " Nora takes sample to Ivo",
         " Nora takes sample to Ivo before", " Nora takes sample to Ivo before sending"])]
    assert [r["worker_step"] for r in prefix_plan(rows, 3)] == [3, 4, 5]
    for row in rows:
        row["target_mass_delta"] = 999
    assert [r["worker_step"] for r in prefix_plan(rows, 3)] == [3, 4, 5]
    with pytest.raises(ValueError):
        prefix_plan(rows[:4], 3)


def prefix_worker():
    source = torch.ones(4, device="cpu")
    frozen = FrozenCandidate("f1", {"op": {"kind": "activation_patch"}}, {}, source,
                             tensor_identity(source), {}, "f1:source", "hook", "a", "a", 30, "a")
    worker = SimpleNamespace(_steps=1, _segments=[SimpleNamespace(kind="output", token_ids=[1])],
        _last_packet={"original": True}, max_generated_tokens=64, stop_token_ids=(), stop_checker=None,
        _collect_active_edits=lambda: [], done=lambda: False, _replay_policy=lambda **kw: "diagnostic",
        runtime_state=SimpleNamespace(trace_caches={}, trace_sequences={}, trace_alignment_step=1),
        codec=SimpleNamespace(decode=lambda ids: "".join("a" if i == 30 else "b" for i in ids)))
    worker.final_text = lambda: worker.codec.decode(worker._segments[0].token_ids)
    worker._append_output_token = lambda i: worker._segments[0].token_ids.append(i)
    worker.runtime_state.put_trace_cache = lambda key, cache: worker.runtime_state.trace_caches.update({key: cache})
    calls = []
    before = torch.linspace(5, -5, 40, device="cpu")
    def simulate(**kw):
        calls.append(kw)
        logits = before.clone()
        if kw.get("command"):
            logits[30] += 0.02
        return {"first_logits": logits, "continuation_token_ids": [30] * kw["max_new_tokens"],
                "edit_runtime_telemetry": [{"op": "activation_patch", "hook_call_count": 1}] if kw.get("command") else []}
    worker._simulate_decode = simulate
    return worker, frozen, calls


def test_prefix_probe_preserves_binding_source_and_removes_edit_before_suffix():
    worker, frozen, calls = prefix_worker()
    segments = deepcopy(worker._segments)
    with patch("SpiralInterventionLab.runtime.prefix_probe.state_identity", return_value="c1"):
        result = measure_prefix(worker, frozen, horizon=8)
    assert result["status"] == "complete", result
    assert result["physical_replay_count"] == 4 and result["model_forward_token_count"] == 32
    assert result["completed_replay_count"] == 4
    assert sum(bool(c.get("command")) for c in calls) == 2
    assert len(calls) == 32 and all(c["max_new_tokens"] == 1 for c in calls)
    assert result["target_piece_token_id"] == 30 and result["source_tensor_identity"] == frozen.source_identity
    assert worker._segments == segments and worker._last_packet == {"original": True}
    assert worker.runtime_state.trace_caches == {}
    assert result["state_restored"] and not result["production_apply_allowed"]


def test_prefix_probe_fails_closed_on_drift_or_replay_error():
    worker, frozen, calls = prefix_worker()
    with patch("SpiralInterventionLab.runtime.prefix_probe.state_identity", side_effect=["before", "after"]):
        result = measure_prefix(worker, frozen)
    assert result["status"] == "incomplete" and not result["state_restored"]
    worker._simulate_decode = lambda **kw: None
    worker._last_simulate_decode_error = "test"
    with patch("SpiralInterventionLab.runtime.prefix_probe.state_identity", return_value="same"):
        result = measure_prefix(worker, frozen)
    assert result["status"] == "incomplete" and worker.runtime_state.trace_caches == {}
    assert result["physical_replay_count"] == 1 and result["completed_replay_count"] == 0
    assert result["model_forward_token_count_is_lower_bound"]
    frozen.source_tensor[0] = 2
    with pytest.raises(ValueError, match="frozen_source_mutated"):
        measure_prefix(worker, frozen)


def recorded_plan_inputs():
    row = {"activation_patch_step_size": 0.16, "target_piece_binding_variant": "canonical",
           "measurement_context_id": "c1", "target_piece_token_id": 30}
    observations = [{"event": "controller_observation", "step": i + 1, "generated_tail": text}
                    for i, text in enumerate(["A", "A walks", "A walks before", "A walks before sunrise"])]
    return [
        {"event": "episode_start", "prompt": "prompt"},
        {"event": "controller_command", "command": {"decision": "noop"}},
        *observations,
        {"event": "controller_diagnostic_result", "step": 0, "target_piece_binding_seed_matrix_executed": True,
         "target_piece_binding_seed_matrix_summary": {"status": "dose_matched_response_complete",
             "state_restored": True, "measurement_context_id": "c1"},
         "target_piece_binding_seed_matrix_rows": [row]},
    ], {"fixture": {"prompt": "prompt"}}


def test_recorded_plan_fixes_binding_dose_and_refuses_ambiguous_selection():
    events, manifest = recorded_plan_inputs()
    plan = plan_from_events(events, manifest, controller_step=0, dose=0.16)
    assert [p["worker_step"] for p in plan["points"]] == [1, 2, 3]
    assert plan["planned_physical_replays"] == 12 and not plan["production_apply_allowed"]
    events[-1]["target_piece_binding_seed_matrix_rows"].append(
        {**plan["candidate"], "target_piece_binding_variant": "alternate", "target_piece_logit_delta": 999})
    assert plan_from_events(events, manifest, controller_step=0, dose=0.16)["candidate"] == plan["candidate"]
    events[-1]["target_piece_binding_seed_matrix_rows"].append(dict(plan["candidate"]))
    with pytest.raises(ValueError, match="ambiguous"):
        plan_from_events(events, manifest, controller_step=0, dose=0.16)


@pytest.mark.parametrize("change", ["apply", "active", "prompt", "state"])
def test_recorded_plan_rejects_uncontrolled_context(change):
    events, manifest = recorded_plan_inputs()
    if change == "apply":
        events[1]["command"]["decision"] = "apply"
    elif change == "active":
        events[2]["active_edit_ids"] = ["e1"]
    elif change == "prompt":
        manifest["fixture"]["prompt"] = "different"
    else:
        events[-1]["target_piece_binding_seed_matrix_summary"]["state_restored"] = False
    with pytest.raises(ValueError):
        plan_from_events(events, manifest, controller_step=0, dose=0.16)
