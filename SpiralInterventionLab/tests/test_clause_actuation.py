from copy import deepcopy
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.runtime import clause_actuation, clause_structure, diagnostic_budget, operation_cards
from SpiralInterventionLab.runtime.loop import _diagnostic_request_signature, _extract_diagnostic_requests
from SpiralInterventionLab.runtime.response_probe import state_identity
from SpiralInterventionLab.tests.test_entity_candidate_review import hints, worker_with_probe


NAME = clause_actuation.NAME


def make_worker(*, observe=True):
    worker = worker_with_probe()
    worker.reset("pac")
    worker.clause_structure_diagnostics = True
    worker.clause_structure_operator_diagnostics = True
    worker._append_output_token(1)
    worker._steps = 1
    worker.runtime_state.run_with_cache(worker._current_token_tensor())
    first = worker.surface_catalog[0]
    worker.surface_catalog = [replace(first, caps=replace(first.caps, step_size=0.08))]
    worker._surface_catalog_raw[0]["caps"]["step_size"] = 0.08
    # Bind the fake vocabulary to exact positions, exercising index refs rather
    # than the old toy adapter's hard-coded last-token source.
    worker._token_position_records = lambda: [
        {"position": 0, "token_id": 0, "piece": "SOURCE:", "provenance_class": "constraint_header", "segment_kind": "prompt"},
        {"position": 1, "token_id": 1, "piece": " before", "provenance_class": "source_body", "segment_kind": "prompt"},
        {"position": 2, "token_id": 3, "piece": ",", "provenance_class": "source_body", "segment_kind": "prompt"},
    ]
    worker.adapter.read_ref = lambda ref, ctx: ctx.runtime_state.last_cache[
        f"{ref['tensor']}:{ref['layer']}"][0, ref["token"]["value"]].detach().clone()
    if observe:
        result = worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])[0]
        assert result["status"] == "ok"
    return worker


def request(worker, index=0):
    return clause_actuation.offer(worker)["candidates"][index]["request"]


def test_observed_exact_tokens_become_optional_normal_cap_cards_not_preferred_winners():
    worker = make_worker(observe=False)
    assert clause_actuation.offer(worker)["status"] == "current_prefix_structure_observation_required"
    assert not any(r["diagnostic"] == NAME for r in hints(worker).get("available_next_diagnostics", []))
    worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("offers cannot measure")):
        offered = clause_actuation.offer(worker)
        packet = {"strategy_hints": hints(worker)}
    assert [r["target_piece"] for r in offered["candidates"]] == [" before", ","]
    assert [r["source_token_index"] for r in offered["candidates"]] == [1, 2]
    cards = [r for r in operation_cards.build_menu(packet)[0]["cards"] if r["diagnostic"] == NAME]
    assert len(cards) == 2
    assert cards[0]["cost"]["physical_replays_upper_bound"] == 4
    assert cards[0]["target_piece"] == " before" and not cards[0]["production_apply_allowed"]
    command = {"decision": "noop", "meta": {"operation_id": cards[0]["operation_id"]}}
    assert _extract_diagnostic_requests(command, packet) == [request(worker)]
    assert _compact_controller_payload(packet)["strategy_hints"]["source_clause_activation_patch"]["status"] == "available"


def test_real_hooks_four_replays_normal_caps_state_restored_and_no_authority():
    worker = make_worker()
    before, caps = state_identity(worker), repr(worker.surface_catalog)
    budget = diagnostic_budget.left(worker)
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        result = worker.request_controller_diagnostics(request(worker))[0]
    assert result["status"] == "ok", result
    assert simulate.call_count == result["physical_replay_count"] == 4
    assert result["edited_replay_count"] == 2
    assert result["new_measurement_count"] == 1
    assert diagnostic_budget.left(worker) == budget - 1
    for call in simulate.call_args_list:
        assert "policy_override" not in call.kwargs
        assert call.kwargs["max_new_tokens"] == 1 and call.kwargs["score_candidate_text"] is False
    row = result["operator_rows"][0]
    assert row["edit"]["source"]["expr"]["ref"]["token"] == {"mode": "index", "value": 1}
    assert row["edit"]["op"] == {"kind": "activation_patch", "mode": "blend", "alpha": 0.04}
    assert row["edit"]["budget"]["step_size"] == 0.08 and row["edit"]["budget"]["ttl_steps"] == 1
    assert all(t[0]["hook_call_count"] == 1 for t in row["runtime_telemetry"])
    assert row["ownership_role"] == "cross_bound"  # toy hook wins c, not intended a
    assert row["realized_lift_token_id"] == 3
    assert row["semantic_safety"] == row["clause_correctness"] == "unassessed"
    assert result["state_restored"] and state_identity(worker) == before
    assert result["caps_unchanged"] and repr(worker.surface_catalog) == caps
    assert not worker._collect_active_edits() and not worker._frozen_diagnostic_candidates
    assert not result["production_apply_allowed"] and not row["target_readout_effect_certified"]
    compact = _compact_diagnostic_result(result)
    assert compact["operator_rows"][0]["paired_deltas"] == row["paired_deltas"]
    assert compact["operator_rows"][0]["ownership_role"] == "cross_bound"


def test_duplicate_stale_and_evicted_receipts_cannot_repeat_or_retarget():
    worker = make_worker()
    req = request(worker)
    first, duplicate = worker.request_controller_diagnostics([req, req])
    assert first["status"] == "ok"
    assert duplicate["blocked_reason"] == "already_attempted_at_context"
    assert duplicate["physical_replay_count"] == duplicate["diagnostic_call_cost"] == 0
    worker._diagnostic_results = []
    worker._latest_diagnostic_results = []
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("duplicates cannot measure")):
        lost = worker.request_controller_diagnostics(req)[0]
    assert lost["blocked_reason"] == "already_attempted_at_context"
    worker._append_output_token(2)
    stale = worker.request_controller_diagnostics(req)[0]
    assert stale["physical_replay_count"] == 0
    worker.reset("p")
    assert not worker._clause_patch_attempted_ids
    assert not worker._clause_structure_receipts


@pytest.mark.parametrize("history_window", [1, 2])
def test_source_matrix_prerequisite_survives_controller_history_eviction(history_window):
    from SpiralInterventionLab.examples.replay_source_clause_diagnostic import measure_source_operators
    worker = make_worker(observe=False)
    worker.diagnostic_result_window = history_window
    observed = worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])[0]
    context = observed["structure_context_id"]
    budget = diagnostic_budget.left(worker)
    receipts = measure_source_operators(worker)
    assert len(receipts) == 2
    assert all(row["result"]["status"] == "ok" for row in receipts)
    assert diagnostic_budget.left(worker) == budget - 2
    assert len(worker._diagnostic_results) == history_window
    assert all(row["diagnostic"] != clause_structure.NAME for row in worker._diagnostic_results)
    assert worker._clause_structure_receipts[context]["structure_readout_rows"] == observed["structure_readout_rows"]
    assert clause_actuation.offer(worker)["status"] == "already_attempted_at_context"
    worker._append_output_token(2)
    assert clause_actuation.offer(worker)["status"] == "current_prefix_structure_observation_required"


def test_different_bindings_and_caps_rotate_ids_and_legacy_request_is_exclusive():
    worker = make_worker()
    req = request(worker)
    assert _diagnostic_request_signature(req) != _diagnostic_request_signature(request(worker, 1))
    surface = worker.surface_catalog[0]
    worker.surface_catalog[0] = replace(surface, caps=replace(surface.caps, step_size=0.02))
    new = request(worker)
    assert new != req
    assert clause_actuation.offer(worker)["candidates"][0]["surfaces"][0]["step_size"] == 0.02
    assert worker.request_controller_diagnostics(req)[0]["physical_replay_count"] == 0
    command = {"decision": "noop", "meta": {"diagnostic_request": new,
        "operator_recipe_expansion_mode": "readout_steering_deepening", "next_action": "request_operator_diagnostic"}}
    packet = {"strategy_hints": {"diagnostic_frontier_bundle_key": "entity_insert:budget",
        "diagnostic_frontier_canonical_request": {"diagnostic": "compare_extra_operator_diagnostics"}}}
    assert _extract_diagnostic_requests(command, packet) == [new]


@pytest.mark.parametrize("condition", ["off", "active", "budget", "prefix", "mask", "source_mismatch", "cache"])
def test_preflight_unavailable_never_performs_or_charges_work(condition):
    worker = make_worker()
    req = request(worker)
    if condition == "off":
        worker.clause_structure_operator_diagnostics = False
    elif condition == "active":
        worker._collect_active_edits = lambda: [{"edit_id": "existing"}]
    elif condition == "budget":
        worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run
    elif condition == "prefix":
        worker._append_output_token(2)
    elif condition == "mask":
        worker.allowed_token_ids = (1,)
    elif condition == "source_mismatch":
        worker._segments[0].token_ids[1] = 2
    else:
        worker.runtime_state.last_cache = None
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("preflight must block")):
        result = clause_actuation.execute(worker, req, source="test")
    assert result["status"] == "unavailable"
    assert result["diagnostic_call_cost"] == result["physical_replay_count"] == 0


def test_policy_failure_is_free_and_failed_physical_attempt_is_charged_once():
    worker = make_worker()
    req = request(worker)
    with patch("SpiralInterventionLab.runtime.clause_actuation.validate_command_against_packet",
               side_effect=ValueError("normal cap denies")):
        result = worker.request_controller_diagnostics(req)[0]
    assert result["blocked_reason"] == "normal_cap_preflight_failed" and result["diagnostic_call_cost"] == 0
    assert worker.request_controller_diagnostics(req)[0]["blocked_reason"] == "already_attempted_at_context"
    worker = make_worker()
    req = request(worker)
    with patch.object(worker, "_simulate_decode", return_value=None):
        failed, repeat = worker.request_controller_diagnostics([req, req])
    assert failed["status"] == "error"
    assert failed["physical_replay_count"] == failed["diagnostic_call_cost"] == 1
    assert failed["new_measurement_count"] == 0 and failed["state_restored"]
    assert repeat["diagnostic_call_cost"] == 0


def test_state_drift_stops_before_second_forward():
    worker = make_worker()
    def mutate(**kwargs):
        worker._append_output_token(2)
        return {"first_logits": torch.zeros(7)}
    with patch.object(worker, "_simulate_decode", side_effect=mutate) as simulate:
        with pytest.raises(RuntimeError, match="changed_worker_state"):
            worker.request_controller_diagnostics(request(worker))
    assert simulate.call_count == 1


def readout(values):
    return {"first_logits": torch.tensor(values), "entropy": 1., "top1_margin": 0.1,
            "repeat_flag": 0, "repetition_score": 0.}


def test_repeated_readout_classification_is_not_permission_and_noise_is_not_lift():
    base = readout([4., 3., 2.])
    after = readout([4., 3.2, 2.])
    candidates = [{"token_id": 1, "piece": ","}, {"token_id": 2, "piece": " while"}]
    row = clause_actuation.summarize_pair([base, base], [after, after], {"target_token_id": 1}, candidates)
    assert row["effect_role"] == "bound_target_lift_candidate" and row["ownership_role"] == "target_aligned"
    assert not row["target_readout_effect_certified"] and not row["production_apply_allowed"]
    noisy = clause_actuation.summarize_pair([base, after], [after, after], {"target_token_id": 1}, candidates)
    assert noisy["effect_role"] == "no_resolved_target_response"
    assert noisy["metric_repeat_floors"]["target_piece_prob_delta"] > 0
    harmful = deepcopy(after)
    harmful.update(entropy=0.9, top1_margin=0.2, repeat_flag=1)
    row = clause_actuation.summarize_pair([base, base], [harmful, harmful], {"target_token_id": 1}, candidates)
    assert row["collapse_sharpening_signal"] and not row["production_apply_allowed"]


def test_local_harness_uses_operation_cards_without_runtime_apply_or_controller_choice():
    from SpiralInterventionLab.examples.replay_source_clause_diagnostic import measure_source_operators
    worker = make_worker()
    receipts = measure_source_operators(worker)
    assert len(receipts) == 2
    assert all(r["result"]["status"] == "ok" for r in receipts)
    assert all(r["selection_origin"] == "source_order_diagnostic_harness_not_controller_choice" for r in receipts)
    assert sum(r["result"]["physical_replay_count"] for r in receipts) == 8
    assert not worker._collect_active_edits() and worker._steps == 1


def test_cli_and_worker_builders_keep_operator_diagnostic_opt_in():
    import inspect
    from SpiralInterventionLab.examples import digit_transform_e2e as e2e
    args = ["--provider", "openai", "--controller-model", "test", "--worker-model", "gpt2"]
    assert not e2e._build_parser().parse_args(args).clause_structure_operator_diagnostics
    assert e2e._build_parser().parse_args([*args, "--clause-structure-operator-diagnostics"]).clause_structure_operator_diagnostics
    with patch.object(e2e, "load_worker_model", side_effect=AssertionError("invalid CLI cannot load")):
        with pytest.raises(SystemExit) as exc:
            e2e.main([*args, "--clause-structure-operator-diagnostics"])
    assert exc.value.code == 2
    for fn in (e2e.build_hooked_transformer_worker_runtime, e2e.run_digit_transform_experiment,
               e2e.run_digit_transform_c1_only_experiment, e2e.run_digit_transform_sweep):
        assert inspect.signature(fn).parameters["clause_structure_operator_diagnostics"].default is False


def test_two_sites_have_six_forward_upper_bound_and_fail_closed_without_hook():
    worker = make_worker()
    first = worker.surface_catalog[0]
    worker.surface_catalog.append(replace(first, surface_id="mlp", target=replace(first.target, site="mlp_out")))
    worker.adapter.read_ref = lambda ref, ctx: ctx.runtime_state.last_cache[
        f"{ref['tensor']}:{ref['layer']}"][0, ref["token"]["value"]].detach().clone()
    candidate = clause_actuation.offer(worker)["candidates"][0]
    assert [s["site"] for s in candidate["surfaces"]] == ["resid_pre", "mlp_out"]
    assert candidate["physical_replays_upper_bound"] == 6
    # This fake model only executes resid_pre hooks. A registered but uncalled
    # MLP hook must not count as a successful measurement.
    worker._surface_catalog_raw.append(worker._surface_to_dict(worker.surface_catalog[-1]))
    result = worker.request_controller_diagnostics(candidate["request"])[0]
    assert result["status"] == "error" and result["error"] == "activation_patch_hook_execution_not_confirmed", result
    assert result["physical_replay_count"] == 6 and result["edited_replay_count"] == 4
    assert result["new_measurement_count"] == len(result["operator_rows"]) == 1
    assert result["state_restored"] and not result["production_apply_allowed"]


def test_cap_drift_cannot_continue_to_next_repeat():
    worker = make_worker()
    req = request(worker)
    def mutate(**kwargs):
        first = worker.surface_catalog[0]
        worker.surface_catalog[0] = replace(first, caps=replace(first.caps, step_size=1.0))
        return {"first_logits": torch.zeros(7)}
    with patch.object(worker, "_simulate_decode", side_effect=mutate) as simulate:
        with pytest.raises(RuntimeError, match="changed_worker_state"):
            worker.request_controller_diagnostics(req)
    assert simulate.call_count == 1
