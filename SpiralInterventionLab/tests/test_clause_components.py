from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.runtime import clause_actuation, clause_components, clause_specificity, clause_trajectory, diagnostic_budget
from SpiralInterventionLab.runtime.compiler import compile_expr
from SpiralInterventionLab.runtime.response_probe import state_identity
from SpiralInterventionLab.tests.test_clause_actuation import make_worker


STAGES = ["resid_pre:11", "resid_post:11"]


def component_worker():
    worker = make_worker()
    read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    return worker


def evaluate(values):
    refs = [{"ref": {"name": name}} for name in ("left", "right", "target")]
    ctx = SimpleNamespace(adapter=SimpleNamespace(read_ref=lambda ref, ctx: values[ref["name"]]))
    return {name: compile_expr(expr)(ctx) for name, expr in clause_components.expressions(*refs).items()}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_sources_add_current_target_back_and_signed_difference_is_swap_equivariant(dtype):
    values = {"left": torch.tensor([6., 2., 4.], dtype=dtype),
              "right": torch.tensor([2., 6., 8.], dtype=dtype),
              "target": torch.tensor([1., 2., 3.], dtype=dtype)}
    sources = evaluate(values)
    c, d = sources["common"] - values["target"], sources["difference_positive"] - values["target"]
    assert torch.equal(c + d, values["left"] - values["target"])
    assert torch.equal(c - d, values["right"] - values["target"])
    assert torch.equal(sources["difference_negative"] - values["target"], -d)
    # Supplying only half-difference would subtract the current activation twice.
    assert not torch.equal(sources["difference_positive"], (values["left"] - values["right"]) / 2)
    swapped = evaluate({**values, "left": values["right"], "right": values["left"]})
    assert torch.equal(swapped["common"], sources["common"])
    assert torch.equal(swapped["difference_positive"], sources["difference_negative"])


def test_seven_conditions_reuse_normal_caps_and_keep_preparation_separate():
    worker = component_worker()
    before, budget = state_identity(worker), diagnostic_budget.left(worker)
    original_packet = worker.build_controller_packet

    def packet_with_canary():
        worker.runtime_state.run_with_cache(worker._current_token_tensor())
        return original_packet()

    with patch.object(worker, "build_controller_packet", side_effect=packet_with_canary):
        with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
            with clause_trajectory.count_forwards(worker.runtime_state) as counts:
                records = clause_components.measure(worker, [1, 3], stages=STAGES)
    assert counts == {"attempted": 49, "completed": 49}
    assert simulate.call_count == 28
    assert all("policy_override" not in call.kwargs for call in simulate.call_args_list)
    assert [r["label"] for r in records] == clause_components.plan([1, 3])["condition_order"]
    assert diagnostic_budget.left(worker) == budget - 7
    assert state_identity(worker) == before and worker._steps == 1
    assert not worker._collect_active_edits() and not worker._frozen_diagnostic_candidates
    result = clause_components.summarize(records)
    assert result["trajectory"]["captured_replay_forward_count"] == 28
    assert result["trajectory"]["captured_preparation_forward_count"] == 21
    assert not result["production_apply_allowed"] and not result["target_readout_effect_certified"]
    entry = result["surfaces"][0]
    assert entry["raw_reconstruction_error_norms"] == [0, 0]
    # The toy's first source equals T and its logits respond to hook presence.
    # Neither dose matching nor a zero null may be inferred just from the caps.
    assert not entry["actual_dose_matches_cap_within_2pct"]
    assert not entry["all_captured_null_repeat_and_baseline_floors_zero"]
    for row in records[4:]:
        assert "component_definition" in clause_trajectory.compact_record(row)
        assert all("ownership_role" not in r for r in row["result"]["operator_rows"])
        assert row["result"]["diagnostic_call_cost"] == 1
    assert "_simulate_decode" not in vars(worker)
    assert "run_with_cache" not in vars(worker.runtime_state)


@pytest.mark.parametrize("failure", ["budget", "stale", "degenerate", "source", "stage", "active"])
def test_component_preflight_is_free_before_partial_matrix(failure):
    worker = component_worker()
    ids, stages = [1, 3], STAGES
    if failure == "budget":
        worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run - 6
    elif failure == "stale":
        worker.runtime_state.last_tokens = torch.tensor([[0]])
    elif failure == "degenerate":
        original = worker.adapter.read_ref
        worker.adapter.read_ref = lambda ref, ctx: original({**ref, "token": {"mode": "index", "value": 1}}, ctx)
    elif failure == "source":
        ids = [1, 400]
    elif failure == "active":
        worker._collect_active_edits = lambda: [{"id": "unrelated"}]
    else:
        stages = ["missing"]
    budget = diagnostic_budget.left(worker)
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("preflight only")):
        with pytest.raises(ValueError):
            clause_components.measure(worker, ids, stages=stages)
    assert diagnostic_budget.left(worker) == budget


def test_component_analyzer_refuses_forged_inputs_missing_condition_and_reordered_plan():
    records = clause_components.measure(component_worker(), [1, 3], stages=STAGES)
    invalid = deepcopy(records)
    invalid[-1]["result"]["operator_rows"][0]["source_tensor_identity"]["sha256"] = "other"
    with pytest.raises(ValueError, match="identity_mismatch"):
        clause_components.summarize(invalid)
    with pytest.raises(ValueError, match="two_sources"):
        clause_components.summarize(records[:-1])
    with pytest.raises(ValueError, match="condition_order"):
        clause_components.summarize([*records[:4], records[5], records[4], records[6]])
    # Legacy trajectory analysis cannot silently accept an extended experiment.
    with pytest.raises(ValueError):
        clause_trajectory.summarize(records)


def test_custom_expression_control_cannot_forge_current_surface_or_bypass_policy():
    worker = component_worker()
    anchor = clause_actuation.offer(worker)["candidates"][0]
    expression = clause_actuation.materialize(worker, anchor, anchor["surfaces"][0])["source"]["expr"]
    stale = deepcopy(anchor)
    stale["surfaces"][0]["step_size"] = 2.0
    budget = diagnostic_budget.left(worker)
    result = clause_specificity.measure_expression_control(worker, "test", stale, [expression])
    assert result["blocked_reason"] == "unknown_or_stale_control_binding"
    expression["ref"]["scope"] = "not_a_runtime_scope"
    result = clause_specificity.measure_expression_control(worker, "test", anchor, [expression])
    assert result["blocked_reason"] == "normal_cap_preflight_failed"
    assert result["physical_replay_count"] == 0
    assert diagnostic_budget.left(worker) == budget


def test_expression_measurement_failure_is_retained_and_cannot_repeat_for_free():
    worker = component_worker()
    anchor = clause_actuation.offer(worker)["candidates"][0]
    expression = clause_actuation.materialize(worker, anchor, anchor["surfaces"][0])["source"]["expr"]
    budget = diagnostic_budget.left(worker)
    with patch.object(worker, "_simulate_decode", return_value=None):
        result = clause_specificity.measure_expression_control(worker, "component_failure", anchor, [expression])
    assert result["status"] == "error" and result["physical_replay_count"] == 1
    assert result["state_restored"] and result["caps_unchanged"]
    assert diagnostic_budget.left(worker) == budget - 1
    again = clause_specificity.measure_expression_control(worker, "component_failure", anchor, [expression])
    assert again["blocked_reason"] == "already_attempted_at_context"
    assert again["physical_replay_count"] == again["diagnostic_call_cost"] == 0


def test_control_capture_preserves_signed_zero_from_actual_source_expression():
    worker = component_worker()
    cached = worker.runtime_state.last_cache[STAGES[0]]
    cached[0, -1, 0] = -2.0
    # Test capture arithmetic without rerunning the toy, which would replace T.
    with patch.object(worker, "request_controller_diagnostics", return_value=[{"status": "ok"}]):
        with patch.object(clause_specificity, "measure_control", return_value={"status": "ok"}):
            records = clause_trajectory.measure(worker, [1, 3], stages=STAGES)
    radial = records[3]["inputs"][0]["source"]
    assert torch.count_nonzero(radial) == 0
    assert torch.signbit(radial[0])
    assert not torch.signbit(torch.zeros_like(radial)[0])


def test_hash_guard_does_not_silently_accept_different_signed_zero_bytes():
    records = clause_components.measure(component_worker(), [1, 3], stages=STAGES)
    radial = records[3]["inputs"][0]["source"]
    radial[0] = 0.0 if torch.signbit(radial[0]) else -0.0
    with pytest.raises(ValueError, match="identity_mismatch"):
        clause_components.summarize(records)
