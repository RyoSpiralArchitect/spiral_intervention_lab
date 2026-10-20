from copy import deepcopy
from unittest.mock import patch

import pytest

from SpiralInterventionLab.examples.replay_source_clause_diagnostic import measure_source_operators
from SpiralInterventionLab.runtime import clause_specificity, diagnostic_budget
from SpiralInterventionLab.runtime.response_probe import state_identity
from SpiralInterventionLab.tests.test_clause_actuation import make_worker


def test_controls_are_normal_cap_and_budgeted_without_authority():
    worker = make_worker()
    # The toy read_ref needs the same last selector as the actual adapter.
    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    sources = measure_source_operators(worker)
    before, budget = state_identity(worker), diagnostic_budget.left(worker)
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        controls = [clause_specificity.measure_control(worker, kind) for kind in clause_specificity.CONTROL_KINDS]
    assert all(c["status"] == "ok" for c in controls), controls
    assert simulate.call_count == 8 and diagnostic_budget.left(worker) == budget - 2
    assert all("policy_override" not in call.kwargs for call in simulate.call_args_list)
    assert all(c["state_restored"] and c["caps_unchanged"] for c in controls)
    assert before == state_identity(worker)
    assert all(not c["production_apply_allowed"] for c in controls)
    assert controls[0]["operator_rows"][0]["edit"]["source"]["expr"]["ref"]["token"] == {"mode": "last"}
    assert controls[1]["operator_rows"][0]["edit"]["source"]["expr"]["by"] == 0.0
    assert not worker._collect_active_edits() and not worker._frozen_diagnostic_candidates
    comparison = clause_specificity.compare(sources, controls)
    assert comparison["status"] == "ok" and len(comparison["contrasts"]) == 2
    assert not comparison["production_apply_allowed"]
    assert all(t[0]["blend_delta_norm"] == 0 for t in controls[0]["operator_rows"][0]["runtime_telemetry"])
    assert diagnostic_budget.left(worker) == budget - 2
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("duplicate is free")):
        duplicate = clause_specificity.measure_control(worker, "self_copy")
    assert duplicate["blocked_reason"] == "already_attempted_at_context"
    assert duplicate["diagnostic_call_cost"] == 0


@pytest.mark.parametrize("condition,blocked_reason", [
    ("budget", "diagnostic_budget_exhausted"),
    ("off", "disabled"),
    ("no_receipt", "current_prefix_structure_observation_required"),
])
def test_controls_fail_preflight_without_a_slot_or_observation(condition, blocked_reason):
    worker = make_worker()
    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read(
        {**ref, "token": {"mode": "index", "value": -1}} if ref["token"]["mode"] == "last" else ref, ctx)
    if condition == "budget":
        worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run
    elif condition == "off":
        worker.clause_structure_operator_diagnostics = False
    else:
        worker._diagnostic_results = []
        worker._clause_structure_receipts = {}
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("preflight must block")) as simulate:
        result = clause_specificity.measure_control(worker, "radial_shrink")
    simulate.assert_not_called()
    assert result["status"] == "unavailable" and result["diagnostic_call_cost"] == 0
    assert result["physical_replay_count"] == 0
    assert result["blocked_reason"] == blocked_reason


def test_control_physical_failure_is_charged_once_and_drift_stops():
    worker = make_worker()
    old_read = worker.adapter.read_ref
    worker.adapter.read_ref = lambda ref, ctx: old_read({**ref, "token": {"mode": "index", "value": -1}}, ctx)
    budget = diagnostic_budget.left(worker)
    with patch.object(worker, "_simulate_decode", return_value=None) as simulate:
        result = clause_specificity.measure_control(worker, "self_copy")
    assert result["status"] == "error" and simulate.call_count == 1
    assert result["diagnostic_call_cost"] == 1 and diagnostic_budget.left(worker) == budget - 1
    assert clause_specificity.measure_control(worker, "self_copy")["diagnostic_call_cost"] == 0
    def drift(**kwargs):
        worker._append_output_token(2)
    with patch.object(worker, "_simulate_decode", side_effect=drift) as simulate:
        with pytest.raises(RuntimeError, match="changed_worker_state"):
            clause_specificity.measure_control(worker, "radial_shrink")
    assert simulate.call_count == 1


def test_contrasts_do_not_confuse_target_alignment_with_source_specificity():
    def row(deltas):
        return {"surface": {"site": "resid_pre"}, "context_fingerprint": "same",
                "runtime_telemetry": [[{"blend_delta_norm": 0.08}]] * 2,
                "structure_response_rows": [
                    {"token_id": i, "piece": str(i), "probability_deltas": [v, v],
                     "repeat_probability_floor": 1e-12} for i, v in enumerate(deltas)]}
    sources = [{"result": {"status": "ok", "candidate_binding": {"target_token_id": i},
                           "operator_rows": [r]}} for i, r in enumerate([row([0.004, -0.004]), row([0.005, 0.001])])]
    controls = [{"kind": kind, "status": "ok", "operator_rows": [r]} for kind, r in
                [("self_copy", row([0, 0])), ("radial_shrink", row([0.006, -0.006]))]]
    controls[0]["operator_rows"][0]["runtime_telemetry"] = [
        [{"blend_delta_norm": 0.0, "source_target_delta_norm": 0.0}]] * 2
    result = clause_specificity.compare(sources, controls)
    assert result["contrasts"][0]["own_minus_best_other_source"] == pytest.approx(-0.001)
    assert result["contrasts"][0]["own_minus_radial_control"] == pytest.approx(-0.002)
    bad = deepcopy(controls)
    bad[0]["operator_rows"][0]["context_fingerprint"] = "different"
    assert clause_specificity.compare(sources, bad)["blocked_reason"] == "mismatched_measurement_context"
    assert clause_specificity.compare(sources, controls[:1])["status"] == "unavailable"
    invalid_null = deepcopy(controls)
    invalid_null[0]["operator_rows"][0]["runtime_telemetry"][0][0]["blend_delta_norm"] = 0.08
    assert clause_specificity.compare(sources, invalid_null)["blocked_reason"] == "self_copy_is_not_null_control"


def test_specificity_cli_requires_operator_opt_in_before_model_load():
    from SpiralInterventionLab.examples.replay_source_clause_diagnostic import main
    with pytest.raises(SystemExit) as exc:
        main(["--source-jsonl", "unused", "--source-manifest", "unused", "--worker-model-path", "unused",
              "--positions", "8", "--output", "unused", "--source-specificity-controls"])
    assert exc.value.code == 2


def test_source_comparison_reserves_control_budget_instead_of_reporting_a_partial_matrix():
    worker = make_worker()
    worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run - 3
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("preflight only")):
        with pytest.raises(ValueError, match="insufficient budget"):
            measure_source_operators(worker, reserve_slots=2)


def test_stale_last_cache_cannot_be_a_self_copy_and_alignment_is_accounted():
    worker = make_worker()
    worker._append_output_token(2)
    # Obtain a new-prefix observation, which intentionally restores the old cache.
    from SpiralInterventionLab.runtime import clause_structure
    worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])
    result = clause_specificity.measure_control(worker, "self_copy")
    assert result["blocked_reason"] == "cache_not_aligned_to_current_prefix"
    assert result["diagnostic_call_cost"] == result["physical_replay_count"] == 0
    budget = diagnostic_budget.left(worker)
    prefix, step = worker.final_text(), worker._steps
    report = clause_specificity.align_cache(worker)
    assert report["physical_replay_count"] == report["diagnostic_call_cost"] == 1
    assert diagnostic_budget.left(worker) == budget - 1
    assert worker.final_text() == prefix and worker._steps == step
