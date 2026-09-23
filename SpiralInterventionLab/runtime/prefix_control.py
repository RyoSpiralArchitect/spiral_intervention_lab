"""Bounded controller-selected inspection time; no new edit or replay budget."""
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


def annotate(packet: Mapping[str, Any], *, rounds_left: int, terminal: bool) -> dict[str, Any]:
    hints = dict(packet.get("strategy_hints") or {})
    choices = hints.get("candidate_diagnostic_choices")
    if isinstance(choices, Mapping):
        cards = []
        for card in choices.get("cards", ()):
            card = dict(card)
            actions = []
            for action in card.get("actions", ()):
                action = dict(action)
                if action.get("action") == "investigate_normal_cap_current_prefix" and rounds_left <= 0:
                    action.update(available=False, blocked_reason="no_same_prefix_decision_round",
                                  diagnostic_cost=0, physical_replay_cost=0)
                actions.append(action)
            card["actions"] = actions
            cards.append(card)
        hints["candidate_diagnostic_choices"] = {**choices, "cards": cards}
    return {**packet, "strategy_hints": {**hints,
        "generation_control": {
            "default_noop_action": "advance_after_existing_followups",
            "hold_action": "hold_prefix", "hold_available": rounds_left > 0 and not terminal,
            "same_prefix_rounds_left": max(0, rounds_left),
            "budget_pool": "shared_candidate_handoff_rounds",
            "requires_fresh_result": True, "terminal_prefix": terminal,
            "production_apply_allowed": False,
        }}}


def evaluate(command: Any, requests: Sequence[Any], results: Sequence[Any], *,
             rounds_left: int, terminal: bool) -> dict[str, Any] | None:
    if is_dataclass(command):
        command = asdict(command)
    if (command.get("meta") or {}).get("generation_action") != "hold_prefix":
        return None
    reason = None
    if command.get("decision") != "noop":
        reason = "hold_requires_noop"
    elif terminal:
        reason = "terminal_prefix"
    elif rounds_left <= 0:
        reason = "same_prefix_round_budget_exhausted"
    elif not requests or not results:
        reason = "no_fresh_requested_result"
    elif any(isinstance(r, Mapping) and (r.get("state_restored") is False
             or (isinstance(r.get("evidence"), Mapping) and r["evidence"].get("state_restored") is False))
             for r in results):
        reason = "diagnostic_state_restoration_failed"
    elif not any(isinstance(r, Mapping)
                 and r.get("status") not in {"blocked", "unavailable", "incomplete", "error", "held",
                                             "no_cached_evidence", "no_rows"}
                 and not r.get("blocked_reason") and not r.get("error")
                 and r.get("state_restored") is not False
                 and (not isinstance(r.get("evidence"), Mapping) or r["evidence"].get("state_restored") is not False)
                 for r in results):
        reason = "diagnostic_result_not_usable"
    return {"requested_action": "hold_prefix", "accepted": reason is None,
            "blocked_reason": reason, "generated_tokens_during_hold": 0,
            "same_prefix_rounds_left": max(0, rounds_left - int(reason is None)),
            "budget_pool": "shared_candidate_handoff_rounds", "production_apply_allowed": False}
