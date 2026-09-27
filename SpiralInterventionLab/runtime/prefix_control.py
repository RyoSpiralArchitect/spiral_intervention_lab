"""Bounded controller-selected inspection time; no new edit or replay budget."""
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


def annotate(packet: Mapping[str, Any], *, rounds_left: int, terminal: bool,
             clock_mode: str = "legacy", token_count: int | None = None,
             controller_round: int = 0, last_clock_report: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
    result = {**packet, "strategy_hints": {**hints,
        "generation_control": {
            "default_noop_action": "advance_after_existing_followups",
            "hold_action": "hold_prefix", "hold_available": rounds_left > 0 and not terminal,
            "same_prefix_rounds_left": max(0, rounds_left),
            "budget_pool": "shared_candidate_handoff_rounds",
            "requires_fresh_result": True, "terminal_prefix": terminal,
            "production_apply_allowed": False,
        }}}
    if clock_mode == "explicit":
        result["strategy_hints"]["generation_control"].update(
            clock_mode="explicit", phase="decision_before_token_commit",
            generated_token_count=token_count, controller_round=controller_round,
            default_noop_action="requires_explicit_generation_action",
            inspect_action="inspect_prefix", commit_action="commit_token",
            hold_action="inspect_prefix", requires_fresh_result=False,
            inspection_consumes_round_even_without_new_evidence=True,
            on_round_exhaustion="commit_required_or_logged_fallback",
            last_clock_report=dict(last_clock_report) if last_clock_report else None)
    return result


def clock_decision(command: Any, *, rounds_left: int) -> dict[str, Any]:
    """Noop is an edit decision, not permission to consume the next token."""
    if is_dataclass(command):
        command = asdict(command)
    meta = command.get("meta") or {}
    requested = meta.get("generation_action")
    action = {"hold_prefix": "inspect_prefix", "advance": "commit_token"}.get(requested, requested)
    memory = meta.get("controller_memory") or {}
    has_request = any(meta.get(key) for key in (
        "operation_id", "diagnostic_request", "tool_requests", "observer_check_request"))
    has_request = has_request or str(meta.get("next_action") or "").startswith("request_")
    if isinstance(memory, Mapping):
        has_request = has_request or bool(memory.get("diagnostic_request"))
    reason = None
    if action not in {"inspect_prefix", "commit_token"}:
        reason = "explicit_generation_action_required"
    elif action == "inspect_prefix":
        if command.get("decision") != "noop" or command.get("edits") or command.get("rollback_ids"):
            reason = "inspection_cannot_apply_or_rollback"
        elif rounds_left <= 0:
            reason = "same_prefix_round_budget_exhausted"
    elif has_request:
        reason = "commit_cannot_request_diagnostics"
    # Invalid commands get a bounded correction opportunity, not a silent advance.
    inspect = (action == "inspect_prefix" or reason is not None) and rounds_left > 0
    return {"clock_mode": "explicit", "requested_action": requested,
            "effective_phase": "inspect" if inspect else "commit",
            "command_valid": reason is None, "blocked_reason": reason,
            "diagnostics_allowed": inspect and reason is None,
            "continue_research": inspect,
            "same_prefix_rounds_left": max(0, rounds_left - int(inspect)),
            "commit_source": None if inspect else (
                "controller_explicit_commit" if reason is None else "research_budget_fallback"),
            "production_apply_allowed": False}


def validate_clock_command(packet: Mapping[str, Any], command: Any) -> None:
    control = (packet.get("strategy_hints") or {}).get("generation_control") or {}
    if control.get("clock_mode") != "explicit":
        return
    report = clock_decision(command, rounds_left=int(control.get("same_prefix_rounds_left") or 0))
    if not report["command_valid"]:
        raise ValueError(report["blocked_reason"])


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
