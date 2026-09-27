"""A common description of already-offered diagnostics, with exact ID resolution.

This module projects a packet. It does not discover, rank, execute, or certify
candidates. Requests stay in the runtime's packet, outside controller prose.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from typing import Any


DIAGNOSTICS = {
    "target_entity_insertion_probe": (
        "Inspect missing-term source spans and token reachability", "source_and_binding_evidence"),
    "entity_insertion_operator_candidate_review": (
        "Resolve entity evidence into candidate blueprints", "unmeasured_candidate_blueprints"),
    "operator_diagnostic_replay": (
        "Measure the offered operator candidates", "operator_effect_evidence"),
    "activation_patch_candidate_review": (
        "Materialize and replay activation-patch blueprints", "activation_patch_measurements_and_possible_frozen_cards"),
    "matched_response_probe": (
        "Compare the offered seed under matched controls", "matched_measurements_and_possible_frozen_cards"),
    "activation_patch_runtime_support_probe": (
        "Check runtime support for the offered activation patch", "runtime_support_evidence"),
    "readout_gap_confirmation_or_variant_sweep": (
        "Measure a bounded readout-gap confirmation or variant set", "gap_and_target_response_evidence"),
    "carrier_to_actuator_conversion_sweep": (
        "Measure target response from carrier conversion recipes", "conversion_effect_evidence"),
    "objective_rotation_pipeline": (
        "Inspect and replay the offered alternate objectives", "alternate_objective_evidence"),
    "inspect_evidence": ("Read recorded evidence", "historical_evidence"),
}
FROZEN_ACTIONS = {
    "review_existing": ("Read this candidate's recorded measurement", "historical_evidence"),
    "remeasure_current_prefix": ("Measure this frozen candidate here", "current_prefix_measurement"),
    "measure_normal_cap_current_prefix": (
        "Measure a new normal-cap child here", "current_prefix_child_measurement"),
    "investigate_normal_cap_current_prefix": (
        "Measure a normal-cap child and conditionally confirm it here", "measurement_or_confirmation_or_trial_offer"),
    "hold": ("Decline this candidate measurement", "measurement_declined"),
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    default=str).encode("utf-8")).hexdigest()


def _rows(value: Any) -> list[Mapping[str, Any]]:
    return [row for row in value if isinstance(row, Mapping)] if isinstance(value, (list, tuple)) else []


def build_menu(packet: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Keep existing offer order and exact requests; never add a preferred action."""
    hints = packet.get("strategy_hints") or {}
    generation = hints.get("generation_control") or {}
    rounds = generation.get("same_prefix_rounds_left")
    can_hold = generation.get("hold_available") is True
    context = _hash(packet)
    cards: list[dict[str, Any]] = []
    requests: dict[str, dict[str, Any]] = {}

    def add(request: Mapping[str, Any], *, purpose: str, output: str,
            prerequisites: list[str], available: bool = True, blocked: Any = None,
            slots: int = 1, replays: int | None = None,
            cost_basis: str = "dispatch_dependent", candidate: Mapping[str, Any] | None = None,
            action: str | None = None, needs_round: bool = False) -> None:
        if not request.get("diagnostic"):
            return
        operation_id = "operation:" + _hash([context, request])[:24]
        if operation_id in requests:
            return
        if needs_round and (not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0):
            available, blocked = False, "no_same_prefix_decision_round"
        explicit_clock = generation.get("clock_mode") == "explicit"
        if explicit_clock and (not isinstance(rounds, int) or rounds <= 0):
            available, blocked = False, "inspection_round_budget_exhausted"
        requests[operation_id] = deepcopy(dict(request))
        cards.append({
            "operation_id": operation_id, "diagnostic": request["diagnostic"],
            "action": action, "mode": request.get("operator_recipe_expansion_mode"),
            "objective_bundle_key": request.get("objective_bundle_key") or request.get("bundle_key")
                or (candidate or {}).get("objective_bundle_key"),
            "candidate_id": (candidate or {}).get("candidate_id"),
            "target_piece": (candidate or {}).get("target_piece"),
            "purpose": purpose, "prerequisites": prerequisites,
            "available": available, "blocked_reason": blocked,
            "cost": {"diagnostic_slots_upper_bound": slots,
                     "inspection_calls_upper_bound": int(request["diagnostic"] == "inspect_evidence"),
                     "physical_replays_upper_bound": replays, "basis": cost_basis,
                     "charge": "actual_work_under_existing_budget"},
            "possible_result": output, "result_guaranteed": False,
            "timing": {"generation_during_diagnostic": "unchanged",
                       "same_prefix_rounds_left": rounds,
                       "result_before_next_token": "inspect_prefix_returns_before_commit" if explicit_clock and available else "conditional_on_usable_result_and_explicit_hold"
                           if can_hold and action != "hold" else "not_guaranteed",
                       "requires_followup_round": needs_round},
            "production_apply_allowed": False,
        })

    for item in _rows(hints.get("available_next_diagnostics")):
        request = item.get("request")
        if not isinstance(request, Mapping):
            continue
        name = str(request.get("diagnostic") or "")
        purpose, output = DIAGNOSTICS.get(name, (
            "Execute the offered diagnostic and report its evidence", "diagnostic_evidence"))
        add(request, purpose=purpose, output=output,
            prerequisites=["offered_by_runtime", "current_request_preflight_at_dispatch"],
            available=item.get("available") is not False and not item.get("blocked_reason"),
            blocked=item.get("blocked_reason"), slots=0 if name == "inspect_evidence" else 1,
            replays=0 if name == "inspect_evidence" else None)

    choices = hints.get("candidate_diagnostic_choices") or {}
    for candidate in _rows(choices.get("cards")):
        for action in _rows(candidate.get("actions")):
            name = str(action.get("action") or "")
            if name not in FROZEN_ACTIONS or not action.get("action_id"):
                continue
            purpose, output = FROZEN_ACTIONS[name]
            add({"diagnostic": "candidate_action", "action_id": action["action_id"]},
                purpose=purpose, output=output, candidate=candidate, action=name,
                prerequisites=["exact_frozen_candidate", "current_action_id",
                               "existing_action_preflight"],
                available=action.get("available") is True, blocked=action.get("blocked_reason"),
                slots=action.get("diagnostic_cost", 0), replays=action.get("physical_replay_cost"),
                cost_basis="existing_action_preflight_upper_bound",
                needs_round=name == "investigate_normal_cap_current_prefix")

    for handoff in _rows(hints.get("candidate_trial_handoffs")):
        request = handoff.get("canonical_request")
        if not isinstance(request, Mapping):
            continue
        add(request, purpose="Physically confirm this exact current-prefix evidence",
            output="confirmation_or_trial_offer", candidate=handoff,
            prerequisites=["current_readout_evidence", "exact_normal_cap_edit",
                           "existing_ownership_and_safety_checks"],
            available=handoff.get("status") == "confirmation_available",
            blocked=handoff.get("blocked_reasons") or None, replays=4,
            cost_basis="existing_confirmation_upper_bound", needs_round=True)

    return ({"schema_version": 1, "scope": "existing_diagnostic_operations",
             "ordering": "existing_offer_order_not_a_new_ranking", "cards": cards,
             "selection_contract": "one_current_operation_id_or_no_diagnostic",
             "production_apply_allowed": False}, requests)


def resolve(packet: Mapping[str, Any], command: Any) -> dict[str, Any]:
    """Resolve against this packet only, without frontier fallback or field merging."""
    if is_dataclass(command):
        command = asdict(command)
    meta = command.get("meta") or {}
    operation_id = meta.get("operation_id")
    if command.get("decision") != "noop" or command.get("edits") or command.get("rollback_ids"):
        raise ValueError("operation_selection_requires_diagnostic_noop")
    menu, requests = build_menu(packet)
    card = next((card for card in menu["cards"] if card["operation_id"] == operation_id), None)
    if card is None:
        raise ValueError("unknown_or_stale_operation_id")
    if not card["available"]:
        raise ValueError(str(card["blocked_reason"] or "operation_unavailable"))
    return deepcopy(requests[operation_id])


def selection_report(packet: Mapping[str, Any], command: Any) -> dict[str, Any]:
    if is_dataclass(command):
        command = asdict(command)
    meta = command.get("meta") or {}
    if "operation_id" not in meta:
        return {}
    try:
        request, reason = resolve(packet, command), None
    except ValueError as exc:
        request, reason = None, str(exc)
    return {"controller_operation_id": meta.get("operation_id"),
            "controller_operation_resolved_request": request,
            "controller_operation_blocked_reason": reason}


def validate_card_command(packet: Mapping[str, Any], command: Mapping[str, Any]) -> None:
    meta = command.get("meta") or {}
    if "operation_id" in meta:
        resolve(packet, command)
        return
    request = meta.get("diagnostic_request")
    if request and not (isinstance(request, Mapping) and request.get("diagnostic") == "inspect_evidence"):
        raise ValueError("select_a_current_operation_id_for_diagnostics")
    next_action = str(meta.get("next_action") or "")
    if next_action.startswith("request_") and next_action != "request_observer_check":
        raise ValueError("select_a_current_operation_id_instead_of_a_diagnostic_alias")


def project_payload(payload: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    """Replace duplicated navigation with cards while retaining measurement facts."""
    result = deepcopy(dict(payload))
    hints = result.get("strategy_hints") or {}
    for key in list(hints):
        if (key in {"available_next_diagnostics", "diagnostic_frontier_request",
                    "diagnostic_frontier_reason_text", "diagnostic_frontier_next_evidence",
                    "diagnostic_frontier_operator_recipe_expansion_mode"}
                or key.endswith("_canonical_request") or key.endswith("_recommended")):
            hints.pop(key)
    if isinstance(hints.get("candidate_handoff"), dict):
        hints["candidate_handoff"].pop("measurement_offers", None)
    choices = hints.get("candidate_diagnostic_choices")
    if isinstance(choices, dict):
        for card in _rows(choices.get("cards")):
            card.pop("actions", None)
    for handoff in _rows(hints.get("candidate_trial_handoffs")):
        handoff.pop("canonical_request", None)
    result["strategy_hints"] = hints
    result["operation_menu"] = build_menu(packet)[0]
    result["controller_action_view"] = "cards"
    return result
