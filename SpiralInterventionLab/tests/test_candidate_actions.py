from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.runtime import candidate_actions as actions, diagnostic_reuse, prefix_control
from SpiralInterventionLab.runtime.loop import _extract_diagnostic_requests, _diagnostic_request_signature
from SpiralInterventionLab.runtime.response_probe import identity, tensor_identity
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime, _normalize_controller_diagnostic_request
from SpiralInterventionLab.runtime import measurement_positions
from SpiralInterventionLab.runtime.debrief import event_anchored_log_digest
import json


@pytest.fixture
def fixture(monkeypatch):
    worker = object.__new__(HookedTransformerWorkerRuntime)
    worker._steps = 1
    worker._segments = [SimpleNamespace(kind="output", token_ids=[1])]
    worker._last_packet = {}
    worker._diagnostic_review_cache = {}
    worker._diagnostic_results = []
    worker._pending_diagnostic_events = []
    worker._evidence_inspection_count = 0
    worker.max_diagnostic_calls_per_run = 2
    worker.diagnostic_result_window = 12
    worker.max_generated_tokens = 64
    worker.stop_token_ids = ()
    worker.stop_checker = None
    worker._collect_active_edits = lambda: []
    worker.done = lambda: False
    worker.build_controller_packet = lambda: {}
    worker._replay_policy = lambda **kw: "diagnostic"
    worker._cache_hook_name = lambda **kw: "hook"
    worker.codec = SimpleNamespace(decode=lambda ids: "".join("a" if i == 30 else "b" for i in ids))
    worker.final_text = lambda: worker.codec.decode(worker._segments[0].token_ids)
    worker._append_output_token = lambda i: worker._segments[0].token_ids.append(i)
    worker.runtime_state = SimpleNamespace(trace_caches={}, trace_sequences={}, trace_alignment_step=1)
    worker.runtime_state.put_trace_cache = lambda key, cache: worker.runtime_state.trace_caches.update({key: cache})
    raw_source = torch.ones(4, device="cpu")
    worker.adapter = SimpleNamespace(read_ref=lambda ref, ctx: raw_source)
    def materialize(candidate, **kw):
        return {"target": {"surface_id": "s1"}, "op": {"kind": "activation_patch", "mode": "blend", "alpha": 0.04},
                "source": {"dtype": "vector", "expr": {"ref": {"tensor": "source"}}},
                "budget": {"ttl_steps": 1, "step_size": candidate["step_size"]}}
    worker._activation_patch_trial_edit_from_candidate = materialize
    before = torch.linspace(5, -5, 40, device="cpu")
    calls = []
    def simulate(**kw):
        calls.append(kw)
        logits = before.clone()
        if kw.get("command"):
            logits[30] += 0.02
        return {"first_logits": logits, "continuation_token_ids": [30],
                "edit_runtime_telemetry": [{"op": "activation_patch", "hook_call_count": 1}] if kw.get("command") else []}
    worker._simulate_decode = simulate
    context = lambda w: identity("context:", w._segments[0].token_ids)
    monkeypatch.setattr(actions, "state_identity", context)
    monkeypatch.setattr("SpiralInterventionLab.runtime.prefix_probe.state_identity", context)
    monkeypatch.setattr("SpiralInterventionLab.runtime.candidate_trial.state_identity", context)
    row = {"objective_bundle_key": "objective:a", "intended_term": "a", "operator_recipe_id": "r1",
           "seed_operator_recipe_id": "seed-r1",
           "recipe_name": "r1", "activation_patch_site": "resid_pre", "activation_patch_layer": 1,
           "activation_patch_alpha": 0.04, "activation_patch_step_size": 0.16,
           "activation_patch_source_localization": "source_term_token", "source_tensor_identity": tensor_identity(raw_source),
           "target_piece_token_id": 30, "target_piece": "a", "measurement_context_id": context(worker),
           "measurement_complete": True, "state_restored": True,
           "repeat_max_abs_logit_delta": 0.0, "no_edit_max_abs_logit_delta": 0.0,
           "actual_delta_class": "rank_carrier", "observable_id": "obs:1", "target_piece_logit_delta": 0.001}
    result = {"target_piece_binding_seed_matrix_summary": {"status": "dose_matched_response_complete", "state_restored": True},
              "target_piece_binding_seed_matrix_rows": [row]}
    actions.reset(worker)
    actions.capture_measurements(worker, result)
    assert result["frozen_candidate_receipts"][0]["status"] == "frozen", result
    return worker, calls, result, raw_source


def request(worker, action):
    hints = actions.action_hints(worker)
    choice = next(a for a in hints["candidate_diagnostic_choices"]["cards"][0]["actions"] if a["action"] == action)
    return {"diagnostic": "candidate_action", "action_id": choice["action_id"]}


def test_frozen_source_is_a_clone_and_choices_survive_compaction(fixture):
    worker, calls, result, source = fixture
    frozen = next(iter(worker._frozen_diagnostic_candidates.values()))["frozen"]
    assert frozen.descriptor["seed_operator_recipe_id"] == "seed-r1"
    source[0] = 99
    assert frozen.source_tensor[0] == 1
    hints = actions.action_hints(worker)
    compact = _compact_controller_payload({"strategy_hints": hints})["strategy_hints"]["candidate_diagnostic_choices"]
    assert compact == hints["candidate_diagnostic_choices"]
    assert [a["action"] for a in compact["cards"][0]["actions"]] == list(actions.ACTIONS)
    assert all(a["physical_replay_cost"] == 0 for a in compact["cards"][0]["actions"])
    assert not compact["production_apply_allowed"] and not calls


def test_fixed_context_review_measure_hold_and_stale_request(fixture):
    worker, calls, _, _ = fixture
    review = worker.request_controller_diagnostics(request(worker, "review_existing"))[0]
    same = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
    assert review["status"] == "reviewed" and same["status"] == "measurement_reused"
    assert not calls and len(worker._diagnostic_results) == 0
    stale = request(worker, "remeasure_current_prefix")
    worker._segments[0].token_ids.append(2)
    blocked = worker.request_controller_diagnostics(stale)[0]
    assert blocked["blocked_reason"] == "stale_measurement_context"
    assert blocked["requested_action"] == "remeasure_current_prefix" and blocked["executed_action"] is None
    now = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
    assert now["status"] == "measured", now
    assert now["physical_replay_count"] == len(calls) == 4
    assert now["new_measurement_count"] == 1 and now["evidence"]["state_restored"]
    assert now["candidate_id"] == same["candidate_id"]
    assert now["measurement_context_id"] != same["measurement_context_id"]
    assert now["budget_before"]["diagnostic_calls_left"] == 2 and now["budget_after"]["diagnostic_calls_left"] == 1
    again = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
    assert again["status"] == "measurement_reused" and again["physical_replay_count"] == 0 and len(calls) == 4
    held = worker.request_controller_diagnostics(request(worker, "hold"))[0]
    assert held["status"] == "held" and len(worker._diagnostic_results) == 1
    assert all(not r["production_apply_allowed"] and not r["certified_for_apply"] for r in (review, same, now, again, held))
    assert _compact_diagnostic_result(now)["executed_action"] == "remeasure_current_prefix"
    assert _compact_diagnostic_result(now)["evidence"]["state_restored"] is True
    assert _compact_diagnostic_result(now)["evidence"]["metrics"] == now["evidence"]["metrics"]


def test_budget_exhaustion_keeps_reads_but_blocks_new_measurement(fixture):
    worker, calls, _, _ = fixture
    worker._segments[0].token_ids.append(2)
    worker.max_diagnostic_calls_per_run = 0
    new = request(worker, "remeasure_current_prefix")
    card = actions.action_hints(worker)["candidate_diagnostic_choices"]["cards"][0]
    assert card["actions"][1]["blocked_reason"] == "diagnostic_budget_exhausted"
    result = worker.request_controller_diagnostics(new)[0]
    assert result["blocked_reason"] == "diagnostic_budget_exhausted" and result["executed_action"] is None
    for action in ("review_existing", "hold"):
        result = worker.request_controller_diagnostics(request(worker, action))[0]
        assert result["executed_action"] == action
        assert result["budget_before"] == result["budget_after"]
    assert not calls and not worker._diagnostic_results


def install_normal_cap(worker, cap=0.12):
    original = worker._activation_patch_trial_edit_from_candidate
    def materialize(candidate, **kw):
        edit = original(candidate, **kw)
        if kw.get("trial_contract", {}).get("trial_budget_class") == "primary":
            edit["budget"]["step_size"] = min(cap, candidate["step_size"])
            edit["meta"] = {"apply_kind": "production_trial", "production_trial_allowed": True,
                            "diagnostic_only": False, "production_trial_budget_class": "primary"}
        return edit
    worker._activation_patch_trial_edit_from_candidate = materialize


def test_normal_cap_is_a_new_explicit_four_replay_measurement_not_clipped_evidence(fixture):
    worker, calls, _, _ = fixture
    install_normal_cap(worker)
    parent = next(iter(worker._frozen_diagnostic_candidates.values()))["frozen"]
    before = deepcopy(worker._candidate_measurements)
    req = request(worker, "measure_normal_cap_current_prefix")
    card = actions.action_hints(worker)["candidate_diagnostic_choices"]["cards"][0]
    offered = card["actions"][2]
    assert offered["available"] and offered["normal_step_size"] == 0.12
    assert offered["diagnostic_cost"] == 1 and offered["physical_replay_cost"] == 4
    assert not calls and len(worker._frozen_diagnostic_candidates) == 1
    result = worker.request_controller_diagnostics(req)[0]
    assert result["status"] == "measured", result
    assert result["candidate_id"] == offered["new_candidate_id"] != parent.candidate_id
    assert result["parent_candidate_id"] == parent.candidate_id and not result["inherits_measurement"]
    assert len(calls) == result["physical_replay_count"] == 4
    assert result["budget_after"]["diagnostic_calls_left"] == 1
    new = worker._frozen_diagnostic_candidates[result["candidate_id"]]["frozen"]
    assert new.edit["budget"]["step_size"] == 0.12 and parent.edit["budget"]["step_size"] == 0.16
    assert new.source_identity == parent.source_identity and new.token_id == parent.token_id
    assert new.edit["op"] == parent.edit["op"] and new.edit["target"] == parent.edit["target"]
    for call in calls:
        if call.get("command"):
            meta = call["command"]["edits"][0]["meta"]
            assert meta["apply_kind"] == "diagnostic_probe" and meta["diagnostic_only"]
            assert not meta["production_trial_allowed"] and not meta["production_apply_allowed"]
            assert meta["production_trial_budget_class"] == "diagnostic_only"
    assert all(worker._candidate_measurements[key] == value for key, value in before.items())
    row = result["evidence"]["promotion_evidence"]
    assert row["actual_delta_class"] == "unassessed" and not row["production_trial_allowed"]
    from SpiralInterventionLab.runtime import candidate_trial
    assert candidate_trial.normal_edit(worker, row, {})["budget"]["step_size"] == 0.12
    compact = _compact_diagnostic_result(result)
    assert compact["parent_candidate_id"] == parent.candidate_id
    again = worker.request_controller_diagnostics(request(worker, "measure_normal_cap_current_prefix"))[0]
    assert again["status"] == "measurement_reused" and not again["diagnostic_budget_charged"]
    assert again["candidate_id"] == new.candidate_id and len(calls) == 4


def _investigation_packet(worker, *, rounds_left=1):
    return prefix_control.annotate({"strategy_hints": actions.action_hints(worker)},
                                   rounds_left=rounds_left, terminal=False)


def _strong_readout(worker, calls):
    before = torch.linspace(5, -5, 40, device="cpu")

    def simulate(**kw):
        calls.append(kw)
        logits = before.clone()
        if kw.get("command"):
            logits[30] += 5
        return {"first_logits": logits, "continuation_token_ids": [30],
                "edit_runtime_telemetry": [{"op": "activation_patch", "hook_call_count": 1}]
                if kw.get("command") else []}

    worker._simulate_decode = simulate


def test_normal_cap_investigation_reserves_two_calls_and_returns_only_a_trial_offer(fixture, monkeypatch):
    worker, calls, _, _ = fixture
    install_normal_cap(worker)
    worker.max_diagnostic_calls_per_run = 2
    entry = next(iter(worker._frozen_diagnostic_candidates.values()))
    entry["origin_row"].update(target_piece_binding_variant="canonical",
        target_piece_binding_requested_honored=True,
        target_piece_binding_manifest={"binding_id": "binding:a", "chosen_target_token_id": 30})
    _strong_readout(worker, calls)
    confirmations = []

    def confirm(current_worker, review_request, packet):
        row = current_worker._candidate_measurements[(result_child, actions.state_identity(current_worker))]["promotion_evidence"]
        assert review_request["evidence_id"] == row["evidence_id"]
        assert review_request["objective_bundle_key"] == row["objective_bundle_key"]
        assert row["candidate_id"] == result_child
        assert row["measured_edit"]["budget"]["step_size"] == 0.12
        edit = candidate_trial.normal_edit(current_worker, row, packet)
        current_worker._response_trial_grants = {"trial:toy": {
            "context_id": actions.state_identity(current_worker), "edit": edit, "row": row}}
        confirmations.append(review_request)
        return {"status": "ok", "physical_replay_count": 4, "new_measurement_count": 1,
                "production_trial_allowed": True, "trial_authorization_id": "trial:toy",
                "production_trial_candidate": {"trial_edit": edit}}

    from SpiralInterventionLab.runtime import candidate_trial
    monkeypatch.setattr("SpiralInterventionLab.runtime.response_promotion.review_response_evidence", confirm)
    packet = _investigation_packet(worker)
    choice = next(a for a in packet["strategy_hints"]["candidate_diagnostic_choices"]["cards"][0]["actions"]
                  if a["action"] == "investigate_normal_cap_current_prefix")
    assert choice["available"] and choice["diagnostic_cost"] == 2
    assert choice["physical_replay_cost"] == 8 and choice["requires_same_prefix_rounds"] == 1
    result_child = choice["new_candidate_id"]
    result = worker.request_controller_diagnostics(
        {"diagnostic": "candidate_action", "action_id": choice["action_id"]}, packet=packet)[0]
    assert result["transaction_status"] == "offer_available" and result["diagnostic_call_cost"] == 2
    assert result["physical_replay_count"] == 8 and len(calls) == 4 and len(confirmations) == 1
    assert result["budget_before"]["diagnostic_calls_left"] == 2
    assert result["budget_after"]["diagnostic_calls_left"] == 0
    assert result["candidate_id"] == result_child and not result["inherits_measurement"]
    assert result["same_prefix_followup_available"] and not result["production_apply_allowed"]
    assert candidate_trial.followup_available(worker, [result])
    compact = _compact_diagnostic_result(result)
    assert compact["transaction_status"] == "offer_available"
    assert compact["trial_authorization_id"] == "trial:toy"
    assert not compact["production_apply_allowed"]


@pytest.mark.parametrize("cached_child", [False, True])
def test_investigation_denied_by_confirmation_never_grants_trial(fixture, monkeypatch, cached_child):
    worker, calls, _, _ = fixture
    install_normal_cap(worker)
    _strong_readout(worker, calls)
    entry = next(iter(worker._frozen_diagnostic_candidates.values()))
    entry["origin_row"].update(target_piece_binding_variant="canonical",
        target_piece_binding_requested_honored=True,
        target_piece_binding_manifest={"binding_id": "binding:a", "chosen_target_token_id": 30})
    if cached_child:
        measured = worker.request_controller_diagnostics(request(worker, "measure_normal_cap_current_prefix"))[0]
        assert measured["status"] == "measured"
        assert measured["candidate_trial_handoff"]["status"] == "confirmation_available"
        calls.clear()

    def deny(current_worker, review_request, packet):
        assert review_request["evidence_id"]
        assert review_request["objective_bundle_key"] == "objective:a"
        return {"status": "blocked", "blocked_reasons": ["physical_confirmation_failed"],
                "physical_replay_count": 4, "new_measurement_count": 1,
                "production_trial_allowed": False}

    monkeypatch.setattr("SpiralInterventionLab.runtime.response_promotion.review_response_evidence", deny)
    packet = _investigation_packet(worker)
    choice = next(a for a in packet["strategy_hints"]["candidate_diagnostic_choices"]["cards"][0]["actions"]
                  if a["action"] == "investigate_normal_cap_current_prefix")
    assert choice["diagnostic_cost"] == (1 if cached_child else 2)
    result = worker.request_controller_diagnostics(
        {"diagnostic": "candidate_action", "action_id": choice["action_id"]}, packet=packet)[0]
    assert result["transaction_status"] == "confirmed_no_offer"
    assert result["confirmation_blocked_reasons"] == ["physical_confirmation_failed"]
    assert result["diagnostic_call_cost"] == choice["diagnostic_cost"]
    assert result["physical_replay_count"] == (4 if cached_child else 8)
    assert len(calls) == (0 if cached_child else 4)
    assert not result["production_trial_allowed"] and not result["production_apply_allowed"]
    assert not result["same_prefix_followup_available"]
    assert not worker._response_trial_grants


def test_investigation_does_not_claim_offer_without_exact_grant(fixture, monkeypatch):
    worker, calls, _, _ = fixture
    install_normal_cap(worker)
    _strong_readout(worker, calls)
    entry = next(iter(worker._frozen_diagnostic_candidates.values()))
    entry["origin_row"].update(target_piece_binding_variant="canonical",
        target_piece_binding_requested_honored=True,
        target_piece_binding_manifest={"binding_id": "binding:a", "chosen_target_token_id": 30})
    monkeypatch.setattr("SpiralInterventionLab.runtime.response_promotion.review_response_evidence",
        lambda *_args: {"status": "ok", "production_trial_allowed": True,
                       "production_trial_candidate": {"trial_edit": {}},
                       "physical_replay_count": 4})
    packet = _investigation_packet(worker)
    choice = next(a for a in packet["strategy_hints"]["candidate_diagnostic_choices"]["cards"][0]["actions"]
                  if a["action"] == "investigate_normal_cap_current_prefix")
    result = worker.request_controller_diagnostics(
        {"diagnostic": "candidate_action", "action_id": choice["action_id"]}, packet=packet)[0]
    assert result["transaction_status"] == "confirmed_no_offer"
    assert result["confirmation_blocked_reasons"] == ["exact_current_context_trial_grant_missing"]
    assert result["diagnostic_call_cost"] == 2 and not result["production_trial_allowed"]
    assert not result["same_prefix_followup_available"] and not worker._response_trial_grants


@pytest.mark.parametrize("budget,rounds,blocked", [
    (1, 1, "diagnostic_budget_insufficient_for_investigation"),
    (2, 0, "no_same_prefix_decision_round"),
])
def test_investigation_preflight_fails_before_replays(fixture, budget, rounds, blocked):
    worker, calls, _, _ = fixture
    install_normal_cap(worker)
    worker.max_diagnostic_calls_per_run = budget
    packet = _investigation_packet(worker, rounds_left=rounds)
    choice = next(a for a in packet["strategy_hints"]["candidate_diagnostic_choices"]["cards"][0]["actions"]
                  if a["action"] == "investigate_normal_cap_current_prefix")
    assert not choice["available"]
    result = worker.request_controller_diagnostics(
        {"diagnostic": "candidate_action", "action_id": choice["action_id"]}, packet=packet)[0]
    assert result["blocked_reason"] == blocked and result["diagnostic_call_cost"] == 0
    assert result["budget_before"] == result["budget_after"] and not calls


def test_weak_investigation_charges_only_measurement_and_does_not_confirm(fixture, monkeypatch):
    worker, calls, _, _ = fixture
    install_normal_cap(worker)
    def unexpected_review(*args, **kwargs):
        pytest.fail("Weak readout must not enter physical confirmation")
    monkeypatch.setattr("SpiralInterventionLab.runtime.response_promotion.review_response_evidence", unexpected_review)
    packet = _investigation_packet(worker)
    choice = next(a for a in packet["strategy_hints"]["candidate_diagnostic_choices"]["cards"][0]["actions"]
                  if a["action"] == "investigate_normal_cap_current_prefix")
    result = worker.request_controller_diagnostics(
        {"diagnostic": "candidate_action", "action_id": choice["action_id"]}, packet=packet)[0]
    assert result["transaction_status"] == "measurement_only"
    assert result["diagnostic_call_cost"] == 1 and result["physical_replay_count"] == len(calls) == 4
    assert result["budget_after"]["diagnostic_calls_left"] == 1
    assert not result["same_prefix_followup_available"] and not result["production_trial_allowed"]


def test_evidence_confirmation_with_no_decision_round_does_not_spend_budget(fixture, monkeypatch):
    worker, calls, _, _ = fixture
    packet = prefix_control.annotate({"strategy_hints": {}}, rounds_left=0, terminal=False)

    def unexpected_execution(*args, **kwargs):
        pytest.fail("A confirmation without a final controller decision must not run")

    monkeypatch.setattr(worker, "_execute_controller_diagnostic_request", unexpected_execution)
    result = worker.request_controller_diagnostics({
        "diagnostic": "activation_patch_production_trial_gate_review",
        "evidence_id": "obs:1", "objective_bundle_key": "objective:a",
    }, packet=packet)[0]
    assert result["status"] == "blocked"
    assert result["blocked_reasons"] == ["no_same_prefix_decision_round"]
    assert not result["diagnostic_budget_charged"] and not result["production_trial_allowed"]
    assert result["budget_before"] == result["budget_after"]
    assert not calls and not worker._diagnostic_results


@pytest.mark.parametrize("change,reason", [
    ("source", "normal_cap_source_tensor_changed"), ("cap", "normal_cap_offer_changed"),
    ("budget", "diagnostic_budget_exhausted"), ("capacity", "episode_candidate_capacity_reached"),
])
def test_normal_cap_variant_fails_closed_without_inherited_measurement(fixture, change, reason):
    worker, calls, _, source = fixture
    install_normal_cap(worker)
    req = request(worker, "measure_normal_cap_current_prefix")
    if change == "source":
        source[0] = 2
    elif change == "cap":
        install_normal_cap(worker, 0.08)
    elif change == "budget":
        worker.max_diagnostic_calls_per_run = 0
    else:
        registry = worker._frozen_diagnostic_candidates
        for i in range(actions.MAX_CANDIDATES - 1):
            registry[f"capacity:{i}"] = next(iter(registry.values()))
    result = worker.request_controller_diagnostics(req)[0]
    assert result["blocked_reason"] == reason, result
    assert not calls and not result["diagnostic_budget_charged"]
    assert len(worker._candidate_measurements) == 1


def test_truncated_history_does_not_restore_diagnostic_budget(fixture):
    from SpiralInterventionLab.runtime.diagnostic_budget import left
    worker, calls, _, _ = fixture
    worker.max_diagnostic_calls_per_run = 3
    worker.diagnostic_result_window = 1
    for token in (2, 3, 4):
        worker._segments[0].token_ids.append(token)
        result = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
        assert result["diagnostic_budget_charged"]
    assert left(worker) == 0 and len(worker._diagnostic_results) == 1
    worker._segments[0].token_ids.append(5)
    result = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
    assert result["blocked_reason"] == "diagnostic_budget_exhausted" and len(calls) == 12


def test_fresh_matched_measurement_can_offer_confirmation_without_remeasurement(fixture):
    worker, calls, result, _ = fixture
    row = result["target_piece_binding_seed_matrix_rows"][0]
    row.update(target_piece_prob_delta=0.001, target_piece_logit_delta=0.2, bound_token_top20_hit_delta=0,
        activation_patch_hook_call_count=1, target_piece_binding_variant="canonical",
        target_piece_binding_requested_honored=True,
        target_piece_binding_manifest={"chosen_target_token_id": 30, "binding_id": "binding:30"})
    actions.reset(worker)
    actions.capture_measurements(worker, result, budget_left=1)
    assert result["same_prefix_followup_available"], result["candidate_trial_handoffs"]
    from SpiralInterventionLab.runtime import candidate_trial
    assert candidate_trial.followup_available(worker, [result])
    stored = next(iter(worker._candidate_measurements.values()))["promotion_evidence"]
    assert stored["actual_delta_class"] == "unassessed" and stored["actuator_class"] == "unknown"
    assert stored["evidence_id"] == result["candidate_trial_handoffs"][0]["evidence_id"]
    assert not stored["production_trial_allowed"] and not calls


def test_masked_target_is_not_reported_as_nan_or_zero_lift(fixture):
    worker, calls, _, _ = fixture
    simulate = worker._simulate_decode
    def masked(**kwargs):
        result = simulate(**kwargs)
        result["first_logits"][30] = float("-inf")
        return result
    worker._simulate_decode = masked
    worker._segments[0].token_ids.append(2)
    result = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
    assert result["status"] == "incomplete" and not result["production_apply_allowed"]
    assert "bound_target_masked_in_decode_readout" in result["blocked_reason"]
    evidence = _compact_diagnostic_result(result)["evidence"]
    assert evidence["readout_failure_details"]["before_target_masked"]
    assert evidence["no_edit_control"]["mask_identical"] and evidence["state_restored"]


@pytest.mark.parametrize("change", ["tensor", "operator", "source_ref"])
def test_mutated_candidate_fails_closed(fixture, change):
    worker, calls, _, _ = fixture
    req = request(worker, "remeasure_current_prefix")
    frozen = next(iter(worker._frozen_diagnostic_candidates.values()))["frozen"]
    if change == "tensor":
        frozen.source_tensor[0] = 2
    elif change == "operator":
        frozen.edit["op"]["alpha"] = 0.15
    else:
        frozen.edit["source"]["expr"]["ref"]["scope"] = "runtime"
    result = worker.request_controller_diagnostics(req)[0]
    assert result["blocked_reason"] == "frozen_candidate_mutated" and not calls


def test_exclusive_offer_survives_parser_without_prose_or_memory_substitution(fixture):
    worker, _, _, _ = fixture
    req = request(worker, "hold")
    command = {"decision": "noop", "meta": {"diagnostic_request": req,
        "next_action": "request_activation_patch_candidate_review",
        "controller_memory": {"diagnostic_request": "operator_diagnostic_replay"}}}
    extracted = _extract_diagnostic_requests(command, {"strategy_hints": {
        "diagnostic_frontier_canonical_request": {"diagnostic": "operator_diagnostic_replay"}}})
    assert extracted == [req]
    assert _normalize_controller_diagnostic_request(req) == req
    different = request(worker, "review_existing")
    assert _diagnostic_request_signature(req) != _diagnostic_request_signature(different)
    assert worker.request_controller_diagnostics(extracted)[0]["executed_action"] == "hold"


def test_fresh_request_on_context_change_and_same_request_after_reset(fixture):
    worker, calls, _, _ = fixture
    first = request(worker, "remeasure_current_prefix")
    worker._segments[0].token_ids.append(2)
    second = request(worker, "remeasure_current_prefix")
    assert first != second
    actions.reset(worker)
    assert actions.action_hints(worker) == {}
    assert worker.request_controller_diagnostics(second)[0]["blocked_reason"] == "unknown_or_stale_action_id"
    assert not calls


def test_incomplete_measurement_spends_budget_but_is_not_cached(fixture):
    worker, calls, _, _ = fixture
    worker._segments[0].token_ids.append(2)
    req = request(worker, "remeasure_current_prefix")
    with patch.object(actions, "measure_prefix", return_value={"status": "incomplete", "state_restored": False,
            "physical_replay_count": 2, "error": "state_restoration_failed"}):
        result = worker.request_controller_diagnostics(req)[0]
    assert result["physical_replay_count"] == 2 and result["new_measurement_count"] == 0
    assert result["diagnostic_budget_charged"] and len(worker._candidate_measurements) == 1
    blocked = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
    assert blocked["blocked_reason"] == "prior_state_restoration_failed" and not calls


@pytest.mark.parametrize("mode", ["active", "terminal"])
def test_active_or_terminal_runtime_never_runs_diagnostic(fixture, mode):
    worker, calls, _, _ = fixture
    worker._segments[0].token_ids.append(2)
    if mode == "active":
        worker._collect_active_edits = lambda: [{"id": "active"}]
    else:
        worker.done = lambda: True
    result = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
    assert result["blocked_reason"] == ("active_edits_not_supported" if mode == "active" else "terminal_prefix")
    assert not result["diagnostic_budget_charged"] and not calls


@pytest.mark.parametrize("change,reason", [("context", "measurement_not_at_current_anchor"),
    ("source", "recorded_source_tensor_mismatch"), ("binding", "recorded_tokenizer_binding_mismatch"),
    ("safety", "unsafe_measurement_not_offered")])
def test_capture_rejects_invalid_provenance(fixture, change, reason):
    worker, _, result, _ = fixture
    actions.reset(worker)
    result = deepcopy(result)
    row = result["target_piece_binding_seed_matrix_rows"][0]
    if change == "context":
        row["measurement_context_id"] = "other"
    elif change == "source":
        row["source_tensor_identity"]["sha256"] = "bad"
    elif change == "binding":
        row["target_piece"] = "different"
    else:
        row["actual_delta_class"] = "harmful"
    actions.capture_measurements(worker, result)
    assert result["frozen_candidate_receipts"][0]["blocked_reason"] == reason
    assert not worker._frozen_diagnostic_candidates


def test_review_aliases_do_not_reopen_but_effective_parameters_do():
    evidence = [{"evidence_rows": [{"objective_bundle_key": "a", "execution_id": "e1", "operator_recipe_id": "r1"}]}]
    base = {"diagnostic": "activation_patch_candidate_review", "objective_bundle_key": "a"}
    key = diagnostic_reuse.review_key(base, {}, evidence)
    for alias in ("activation_patch_candidate_compiler_review", "activation_patch_candidate_review_on_current_prefix",
                  "current_activation_patch_candidate_review", "alternate_activation_patch_or_bridge_evidence"):
        assert diagnostic_reuse.review_key({**base, "next_evidence_needed": alias}, {}, evidence) == key
    for delta in ({"dose_grid": [0.04]}, {"candidate_ids": ["different"]}, {"comparison_axis": "source_localization"}):
        assert diagnostic_reuse.review_key({**base, **delta}, {}, evidence) != key


@pytest.mark.parametrize("prefix,trailing", [("", "empty"), (" I", "open_text"),
    (" before", "open_text"), ("word ", "whitespace"), ("word;", "punctuation")])
def test_position_is_surface_fact_not_word_completion_or_policy(fixture, prefix, trailing):
    worker, calls, _, _ = fixture
    worker.final_text = lambda: prefix
    position = measurement_positions.describe(worker)
    assert position["trailing_surface"] == trailing
    assert position["word_completion"] == "unknown_without_lookahead"
    assert "preferred" not in position and "next_action" not in position
    hints = actions.action_hints(worker)["candidate_diagnostic_choices"]
    assert hints["current_position"] == position and hints["position_selection_owner"] == "controller"
    assert all(a["available"] for a in hints["cards"][0]["actions"]
               if a["action"] not in {"measure_normal_cap_current_prefix",
                                      "investigate_normal_cap_current_prefix"})
    assert hints["cards"][0]["actions"][2]["blocked_reason"] == "already_normal_cap_candidate"
    assert not calls


def test_history_is_candidate_local_bounded_and_survives_compaction(fixture):
    worker, calls, _, _ = fixture
    worker.max_diagnostic_calls_per_run = 4
    for token in (2, 3, 4):
        worker._segments[0].token_ids.append(token)
        worker._steps += 1
        result = worker.request_controller_diagnostics(request(worker, "remeasure_current_prefix"))[0]
        assert result["evidence"]["measurement_position"] == result["requested_position"]
        assert _compact_diagnostic_result(result)["evidence"]["measurement_position"] == result["requested_position"]
    worker._candidate_measurements[("unrelated-dose-or-piece", "c:other")] = {"metrics": {"target_piece_logit_delta": 999}}
    worker._segments[0].token_ids.append(5)
    hints = actions.action_hints(worker)
    compact = _compact_controller_payload({"strategy_hints": hints})["strategy_hints"]["candidate_diagnostic_choices"]
    assert compact == hints["candidate_diagnostic_choices"]
    history = compact["cards"][0]["measurement_history"]
    assert history["measured_context_count"] == 4 and history["omitted_context_count"] == 2
    assert [row["new_output_tokens"] for row in history["recent"]] == [2, 1]
    assert all(row["prefix_relation"] == "extended" for row in history["recent"])
    assert all(row["metrics"]["target_piece_logit_delta"] < 1 for row in history["recent"])
    assert len(calls) == 12
    worker.request_controller_diagnostics(request(worker, "hold"))
    assert actions.action_hints(worker)["candidate_diagnostic_choices"]["cards"][0]["measurement_history"] == history
    worker._segments[0].token_ids[0] = 99
    divergent = actions.action_hints(worker)["candidate_diagnostic_choices"]["cards"][0]["measurement_history"]
    assert all(r["new_output_tokens"] is None and r["prefix_relation"] == "diverged" for r in divergent["recent"])


def test_debrief_distinguishes_new_measurements_reviews_and_hold(fixture, tmp_path):
    worker, _, _, _ = fixture
    events = []
    for token, action in ((2, "remeasure_current_prefix"), (3, "hold"),
                          (3, "review_existing"), (4, "remeasure_current_prefix")):
        worker._segments[0].token_ids.append(token)
        worker._steps += 1
        result = worker.request_controller_diagnostics(request(worker, action))[0]
        events.append({**result, "event": "controller_diagnostic_result"})
    (tmp_path / "c1.jsonl").write_text("\n".join(json.dumps(row) for row in events))
    digest = event_anchored_log_digest(tmp_path)
    summary = digest["candidate_action_summary"]
    assert summary["executed_actions"] == {"remeasure_current_prefix": 2, "hold": 1, "review_existing": 1}
    assert summary["new_measurement_count"] == 2 and summary["physical_replay_count"] == 8
    assert summary["diagnostic_budget_charged"] == 2
    anchors = {kind: view for view in digest["event_anchors"] for kind in view["anchor_kinds"]}
    first = anchors["first_current_prefix_measurement"]["candidate_action"]
    last = anchors["last_current_prefix_measurement"]["candidate_action"]
    assert first["measurement_context_id"] != last["measurement_context_id"]
    assert first["evidence"]["state_restored"] is True
    assert first["evidence"]["no_edit_control"]["max_abs_logit_delta"] == 0
    assert first["production_apply_allowed"] is False
    assert "first_candidate_action_held" in anchors and "first_candidate_action_reviewed" in anchors
