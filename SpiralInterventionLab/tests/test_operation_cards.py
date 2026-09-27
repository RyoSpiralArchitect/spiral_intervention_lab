from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from SpiralInterventionLab.bridge.controller_clients import ProviderControllerClient, _compact_controller_payload
from SpiralInterventionLab.controllers.base import ControllerProviderResponse
from SpiralInterventionLab.runtime import candidate_actions, operation_cards, prefix_control
from SpiralInterventionLab.runtime.loop import (
    InMemoryStructuredLogger, _build_controller_selection_report, _extract_diagnostic_requests,
    _extract_observer_check_request, _extract_tool_requests, _log_controller_trace,
)
from SpiralInterventionLab.tests.test_candidate_actions import fixture as candidate_fixture


def packet():
    return prefix_control.annotate({"step": 1, "budget": {"edits_left_this_run": 2},
        "worker_view": {"generated_tail": "prefix"},
        "task_feedback": {"missing_required_terms": ["x"]}, "strategy_hints": {
            "diagnostic_frontier_request": "operator_diagnostic_replay",
            "diagnostic_frontier_canonical_request": {
                "diagnostic": "operator_diagnostic_replay", "objective_bundle_key": "frontier"},
            "available_next_diagnostics": [
                {"priority": 10, "request": {"diagnostic": "operator_diagnostic_replay",
                    "materialize_entity_insertion_candidates": True}},
                {"priority": 20, "request": {"diagnostic": "activation_patch_candidate_review",
                    "objective_bundle_key": "objective:x",
                    "operator_recipe_expansion_mode": "activation_patch_candidate_review"}},
            ]}}, rounds_left=2, terminal=False)


def choose(card, **meta):
    return {"version": "0.1", "decision": "noop", "meta": {"operation_id": card["operation_id"], **meta}}


def test_card_projection_preserves_evidence_requests_order_and_unknown_cost():
    raw = packet()
    before = deepcopy(raw)
    compact = _compact_controller_payload(raw)
    projected = operation_cards.project_payload(compact, raw)
    assert raw == before
    assert projected["task_feedback"] == compact["task_feedback"]
    assert projected["budget"] == compact["budget"]
    assert projected["source_packet_sha256"] == compact["source_packet_sha256"]
    assert "diagnostic_frontier_canonical_request" not in projected["strategy_hints"]
    assert "available_next_diagnostics" not in projected["strategy_hints"]
    cards = projected["operation_menu"]["cards"]
    assert [row["diagnostic"] for row in cards] == [
        "operator_diagnostic_replay", "activation_patch_candidate_review"]
    for card, offered in zip(cards, raw["strategy_hints"]["available_next_diagnostics"]):
        assert operation_cards.resolve(raw, choose(card)) == offered["request"]
        assert card["cost"]["physical_replays_upper_bound"] is None
        assert card["cost"]["diagnostic_slots_upper_bound"] == 1
        assert card["production_apply_allowed"] is False
        assert "priority" not in card


def test_exact_operation_is_exclusive_and_does_not_inherit_frontier():
    raw = packet()
    card = operation_cards.build_menu(raw)[0]["cards"][1]
    command = choose(card, diagnostic_request={"diagnostic": "operator_diagnostic_replay"},
        next_action="request_operator_diagnostic", objective_bundle_key="frontier",
        tool_requests=[{"tool": "tokenize_terms"}], observer_check_request=True,
        controller_memory={"diagnostic_request": "other"})
    requests = _extract_diagnostic_requests(command, raw)
    assert requests == [raw["strategy_hints"]["available_next_diagnostics"][1]["request"]]
    assert _extract_tool_requests(command) == []
    assert _extract_observer_check_request(command) is None
    report = _build_controller_selection_report(raw, command)
    assert report["controller_operation_resolved_request"] == requests[0]
    assert report["controller_operation_blocked_reason"] is None


@pytest.mark.parametrize("change", ["prefix", "budget", "request", "unknown"])
def test_stale_operation_never_falls_back_to_canonical(change):
    raw = packet()
    card = operation_cards.build_menu(raw)[0]["cards"][0]
    command = choose(card, next_action="request_operator_diagnostic")
    if change == "prefix":
        raw["worker_view"]["generated_tail"] += " next"
    elif change == "budget":
        raw["budget"] = {"edits_left_this_run": 0}
    elif change == "request":
        raw["strategy_hints"]["available_next_diagnostics"][0]["request"]["objective_bundle_key"] = "other"
    else:
        command["meta"]["operation_id"] = "operation:unknown"
    assert _extract_diagnostic_requests(command, raw) == []
    assert operation_cards.selection_report(raw, command)["controller_operation_blocked_reason"] == (
        "unknown_or_stale_operation_id")


def test_card_without_same_prefix_round_does_not_offer_investigation():
    raw = packet()
    raw["strategy_hints"]["candidate_diagnostic_choices"] = {"cards": [{
        "candidate_id": "candidate:x", "objective_bundle_key": "objective:x", "actions": [{
            "action": "investigate_normal_cap_current_prefix", "action_id": "action:x",
            "available": True, "diagnostic_cost": 2, "physical_replay_cost": 8}]}]}
    cards = operation_cards.build_menu(raw)[0]["cards"]
    assert cards[-1]["cost"]["physical_replays_upper_bound"] == 8
    assert cards[-1]["possible_result"] == "measurement_or_confirmation_or_trial_offer"
    raw = prefix_control.annotate(raw, rounds_left=0, terminal=False)
    card = operation_cards.build_menu(raw)[0]["cards"][-1]
    assert card["available"] is False
    assert _extract_diagnostic_requests(choose(card), raw) == []
    assert operation_cards.selection_report(raw, choose(card))["controller_operation_blocked_reason"] == (
        "no_same_prefix_decision_round")


def test_operation_id_executes_existing_four_replay_action_and_charges_once(candidate_fixture):
    worker, calls, _, _ = candidate_fixture
    worker._segments[0].token_ids.append(2)
    raw = prefix_control.annotate({"step": 2, "strategy_hints": candidate_actions.action_hints(worker)},
                                 rounds_left=2, terminal=False)
    card = next(c for c in operation_cards.build_menu(raw)[0]["cards"]
                if c["action"] == "remeasure_current_prefix")
    command = choose(card, generation_action="hold_prefix")
    selected = _extract_diagnostic_requests(command, raw)
    result = worker.request_controller_diagnostics(selected, packet=raw)[0]
    assert result["executed_action"] == "remeasure_current_prefix"
    assert result["candidate_id"] == card["candidate_id"]
    assert result["physical_replay_count"] == len(calls) == 4
    assert result["budget_before"]["diagnostic_calls_left"] - result["budget_after"]["diagnostic_calls_left"] == 1
    assert result["evidence"]["state_restored"] is True
    assert result["production_apply_allowed"] is False


def test_provider_ids_resolve_against_full_packet_and_are_logged():
    raw = packet()
    seen = []
    def complete(request):
        seen.append(request)
        card = request.payload["operation_menu"]["cards"][1]
        return ControllerProviderResponse(json.dumps(choose(card)), "fake", "fake")
    provider = SimpleNamespace(provider_name="fake", model_name="fake", complete=complete)
    client = ProviderControllerClient(provider, packet_view="compact", action_view="cards")
    command = client.invoke(raw)
    assert _extract_diagnostic_requests(command, raw) == [
        raw["strategy_hints"]["available_next_diagnostics"][1]["request"]]
    assert "operation_menu" in seen[0].system_prompt
    logger = InMemoryStructuredLogger()
    _log_controller_trace(logger, step=0, trace=client.latest_trace())
    assert next(e for e in logger.events if e["event"] == "controller_operation_menu")["cards"] == (
        seen[0].payload["operation_menu"]["cards"])


def test_diagnostic_operation_is_not_an_apply_grant():
    raw = packet()
    command = choose(operation_cards.build_menu(raw)[0]["cards"][0])
    command["decision"] = "apply"
    with pytest.raises(ValueError, match="requires_diagnostic_noop"):
        operation_cards.resolve(raw, command)


def test_card_contract_allows_decline_and_inspection_but_rejects_freeform_diagnostics():
    raw = packet()
    operation_cards.validate_card_command(raw, {"decision": "noop", "meta": {}})
    operation_cards.validate_card_command(raw, {"decision": "noop", "meta": {
        "diagnostic_request": {"diagnostic": "inspect_evidence", "evidence_id": "obs:x"}}})
    with pytest.raises(ValueError, match="current_operation_id"):
        operation_cards.validate_card_command(raw, {"decision": "noop", "meta": {
            "diagnostic_request": {"diagnostic": "operator_diagnostic_replay"}}})
    with pytest.raises(ValueError, match="current_operation_id"):
        operation_cards.validate_card_command(raw, {"decision": "noop", "meta": {
            "next_action": "request_operator_diagnostic"}})


def test_legacy_provider_keeps_original_prompt_and_payload():
    seen = []
    def complete(request):
        seen.append(request)
        return ControllerProviderResponse('{"version":"0.1","decision":"noop"}', "fake", "fake")
    provider = SimpleNamespace(provider_name="fake", model_name="fake", complete=complete)
    client = ProviderControllerClient(provider, packet_view="compact", system_prompt="legacy prompt")
    raw = packet()
    client.invoke(raw)
    assert seen[0].system_prompt == "legacy prompt"
    assert seen[0].payload == _compact_controller_payload(raw)
