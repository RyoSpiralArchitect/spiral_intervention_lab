from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.runtime import clause_direction_controls as controls, clause_trajectory, diagnostic_budget
from SpiralInterventionLab.runtime.response_probe import state_identity
from SpiralInterventionLab.tests.test_clause_components import STAGES, component_worker
from SpiralInterventionLab.examples.replay_source_clause_trajectory import main


def worker_with_traces():
    worker = component_worker()
    original = worker.adapter.read_ref

    def read(ref, ctx):
        if ref["scope"] == "trace":
            return ctx.runtime_state.get_cache("trace", ref["trace_id"])[f"{ref['tensor']}:{ref['layer']}"][0, -1].clone()
        return original(ref, ctx)

    worker.adapter.read_ref = read
    return worker


def test_fixed_permutation_preserves_coordinates_and_does_not_draw_or_change_rng():
    state = torch.random.get_rng_state()
    common = torch.tensor([3., 8., 1., 7.], device="cpu")
    target = torch.tensor([1., 2., 3., 4.], device="cpu")
    positive, negative = controls.permuted_sources(common, target)
    delta = common - target
    assert sorted(controls.permutation(4).tolist()) == [0, 1, 2, 3]
    assert torch.equal(positive - target, -(negative - target))
    assert torch.equal((positive - target).sort().values, delta.sort().values)
    assert float((positive - target).norm()) == pytest.approx(float(delta.norm()))
    assert not torch.equal(positive, common)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("bad", ["zero", "constant", "shape", "nonfinite"])
def test_degenerate_controls_fail_closed_without_resampling(bad):
    target = torch.tensor([1., 2., 3., 4.], device="cpu")
    common = target.clone()
    if bad == "constant":
        common += 1
    elif bad == "shape":
        common = common[:2]
    elif bad == "nonfinite":
        common[0] = float("nan")
    with pytest.raises(ValueError):
        controls.permuted_sources(common, target)


def test_nine_conditions_use_existing_policy_temporary_traces_and_no_production_cards():
    worker = worker_with_traces()
    state = worker.runtime_state
    before, used, packet = state_identity(worker), diagnostic_budget.used(worker), worker._last_packet
    traces, alignment = dict(state.trace_caches), state.trace_alignment_step
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        records = controls.measure(worker, [1, 3], stages=STAGES)
    assert simulate.call_count == 36
    assert all("policy_override" not in call.kwargs for call in simulate.call_args_list)
    assert diagnostic_budget.used(worker) == used + 9
    assert state_identity(worker) == before and state.trace_caches.keys() == traces.keys()
    assert state.trace_alignment_step == alignment
    assert worker._last_packet is packet
    assert all(r["result"]["temporary_trace_removed"] for r in records[7:])
    assert not worker._frozen_diagnostic_candidates and not worker._collect_active_edits()
    result = controls.summarize(records)
    assert not result["production_apply_allowed"] and not result["target_readout_effect_certified"]
    assert result["plan"]["condition_order"] == [r["label"] for r in records]
    assert len(result["surfaces"][0]["controls"]) == 2
    forged = deepcopy(records)
    forged[-1]["component_definition"]["permutation_spec"] = "a different direction"
    with pytest.raises(ValueError, match="identity_or_restoration"):
        controls.summarize(forged)
    with pytest.raises(ValueError):
        clause_trajectory.summarize(records)


def test_trace_context_restores_packet_alignment_and_removes_buffer_on_failure():
    worker = worker_with_traces()
    state = worker.runtime_state
    packet, alignment = worker._last_packet, state.trace_alignment_step
    with pytest.raises(RuntimeError, match="injected"):
        with controls._temporary_trace(worker, "test-control", {"resid_pre:11": torch.zeros(1, 1, 4)}):
            assert "test-control" in state.trace_caches
            worker._last_packet = {"temporary": True}
            state.trace_alignment_step = 900
            raise RuntimeError("injected")
    assert "test-control" not in state.trace_caches
    assert worker._last_packet is packet and state.trace_alignment_step == alignment


def test_budget_preflight_is_free():
    worker = worker_with_traces()
    worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run - 8
    before = diagnostic_budget.used(worker)
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("no forward")):
        with pytest.raises(ValueError, match="insufficient_budget"):
            controls.measure(worker, [1, 3], stages=STAGES)
    assert diagnostic_budget.used(worker) == before


def test_failed_control_retains_partial_receipt_and_removes_trace():
    worker = worker_with_traces()
    before, packet = state_identity(worker), worker._last_packet
    alignment, traces = worker.runtime_state.trace_alignment_step, dict(worker.runtime_state.trace_caches)
    simulate = worker._simulate_decode
    calls, retained = 0, []

    def fail_control(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 29:
            raise ValueError("injected-control-failure")
        return simulate(*a, **kw)

    with patch.object(worker, "_simulate_decode", side_effect=fail_control):
        with pytest.raises(ValueError, match="direction_control_failed"):
            controls.measure(worker, [1, 3], stages=STAGES, record=retained.append)
    assert len(retained) == 8 and retained[-1]["result"]["status"] == "error"
    assert retained[-1]["result"]["physical_replay_count"] == 1
    assert retained[-1]["result"]["temporary_trace_removed"]
    assert worker.runtime_state.trace_caches.keys() == traces.keys()
    assert worker._last_packet is packet and worker.runtime_state.trace_alignment_step == alignment
    assert state_identity(worker) == before and not worker._collect_active_edits()


@pytest.mark.parametrize("flags", [[], ["--decompose-components", "--next-natural-boundary"]])
def test_cli_refuses_unscoped_direction_controls_before_loading(flags):
    args = ["--source-jsonl", "unused", "--source-manifest", "unused", "--reference-receipt", "unused",
            "--worker-model-path", "unused", "--position", "8", "--source-token-ids", "11,1418",
            "--output-dir", "unused", "--permuted-direction-controls", *flags]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
