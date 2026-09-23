"""Episode-local choices to read evidence or remeasure a frozen candidate.

Offers carry no policy preference. Only an explicit controller request spends
diagnostic budget; neither a new prefix nor a favorable measurement grants apply.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from .prefix_probe import freeze_candidate, measure_prefix, normal_cap_variant
from . import measurement_positions
from . import diagnostic_budget
from . import candidate_trial
from .response_probe import identity, state_identity, tensor_identity


MAX_CANDIDATES = 8
MAX_VISIBLE_CANDIDATES = 4
ACTIONS = ("review_existing", "remeasure_current_prefix", "measure_normal_cap_current_prefix",
           "investigate_normal_cap_current_prefix", "hold")
PERMISSIONS = {"diagnostic_only": True, "production_apply_allowed": False,
               "certified_for_apply": False, "policy_candidate_ready": False}


def reset(worker: Any) -> None:
    worker._frozen_diagnostic_candidates = {}
    worker._candidate_action_offers = {}
    worker._candidate_measurements = {}
    worker._candidate_measurement_positions = {}
    worker._candidate_action_context_failed = False
    worker._response_trial_grants = {}


def _initialize(worker: Any) -> None:
    if not hasattr(worker, "_frozen_diagnostic_candidates"):
        reset(worker)


def _candidate_digest(frozen: Any) -> str:
    return identity("frozen_candidate:", [frozen.edit["target"], frozen.edit["op"],
        frozen.edit["budget"], tensor_identity(frozen.source_tensor), frozen.term, frozen.token_id])


def _candidate_intact(worker: Any, frozen: Any) -> bool:
    expected_source = {"dtype": "vector", "expr": {"ref": {
        "scope": "trace", "trace_id": frozen.trace_id, "tensor": frozen.descriptor["site"],
        "layer": frozen.descriptor["layer"], "token": {"mode": "last"}}}}
    return (_candidate_digest(frozen) == frozen.candidate_id
            and tensor_identity(frozen.source_tensor) == frozen.source_identity
            and frozen.edit.get("source") == expected_source
            and worker.codec.decode([frozen.token_id]) == frozen.token_piece)


def capture_measurements(worker: Any, result: dict[str, Any], *, budget_left: int | None = None) -> None:
    """Freeze only freshly measured, restored rows while still at their anchor."""
    summary = result.get("target_piece_binding_seed_matrix_summary") or {}
    if not isinstance(summary, Mapping) or summary.get("status") != "dose_matched_response_complete":
        return
    if summary.get("state_restored") is not True:
        return
    rows = result.get("target_piece_binding_seed_matrix_rows", ())
    if not rows:
        return
    _initialize(worker)
    context = state_identity(worker)
    receipts = []
    handoffs = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        receipt = {"observable_id": row.get("observable_id"), "candidate_id": None}
        try:
            if row.get("measurement_context_id") != context:
                raise ValueError("measurement_not_at_current_anchor")
            if row.get("measurement_complete") is not True or row.get("state_restored") is not True:
                raise ValueError("measurement_completion_or_restoration_unverified")
            if row.get("actual_delta_class") in {"harmful", "collapse_sharpener", "collapse_isomorphic", "replay_error"}:
                raise ValueError("unsafe_measurement_not_offered")
            for key in ("repeat_max_abs_logit_delta", "no_edit_max_abs_logit_delta"):
                if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                    raise ValueError("missing_finite_measurement_control")
            frozen = freeze_candidate(worker, row)
            if _candidate_digest(frozen) != frozen.candidate_id:
                raise ValueError("frozen_identity_mismatch")
            registry = worker._frozen_diagnostic_candidates
            if frozen.candidate_id not in registry:
                if len(registry) >= MAX_CANDIDATES:
                    raise ValueError("episode_candidate_capacity_reached")
                metrics = {key: row[key] for key in (
                    "target_piece_logit_delta", "target_piece_prob_delta", "target_top20_threshold_gap_delta",
                    "target_rank_before", "target_rank_after", "bound_token_top20_hit_delta",
                    "repeat_max_abs_logit_delta", "no_edit_max_abs_logit_delta") if key in row}
                evidence = {"status": "complete", "candidate_id": frozen.candidate_id,
                    "measurement_context_id": context, "observed_at_step": worker._steps,
                    "observable_id": row.get("observable_id"), "source_tensor_identity": frozen.source_identity,
                    "target_piece": frozen.token_piece, "target_piece_token_id": frozen.token_id,
                    "metrics": metrics, "state_restored": True, "measurement_origin": "recorded_matched_probe",
                    "measurement_position": measurement_positions.describe(worker),
                    **PERMISSIONS}
                registry[frozen.candidate_id] = {"frozen": frozen, "evidence": evidence, "origin_row": deepcopy(row)}
                evidence["promotion_evidence"] = candidate_trial.record(frozen, {
                    **evidence,
                    "no_edit_control": {"max_abs_logit_delta": row["no_edit_max_abs_logit_delta"]},
                    "repeat_control": {"max_abs_logit_delta": row["repeat_max_abs_logit_delta"]},
                    "edit_runtime_telemetry": [{"op": "activation_patch", "hook_call_count": row.get("activation_patch_hook_call_count", 0)}],
                }, row)
                handoffs.append(candidate_trial.handoff(worker, evidence["promotion_evidence"], context=context,
                    budget_left=max(0, diagnostic_budget.left(worker) - 1) if budget_left is None else budget_left))
                worker._candidate_measurements[(frozen.candidate_id, context)] = evidence
                worker._candidate_measurement_positions[(frozen.candidate_id, context)] = measurement_positions.output_tokens(worker)
            receipt.update(candidate_id=frozen.candidate_id, status="frozen",
                           measurement_context_id=context)
        except (ValueError, KeyError, TypeError) as exc:
            receipt.update(status="unavailable", blocked_reason=str(exc))
        receipts.append(receipt)
    result["frozen_candidate_receipts"] = receipts
    if handoffs:
        result["candidate_trial_handoffs"] = handoffs
        result["same_prefix_followup_available"] = any(r["same_prefix_followup_available"] for r in handoffs)


def action_hints(worker: Any) -> dict[str, Any]:
    _initialize(worker)
    registry = worker._frozen_diagnostic_candidates
    if not registry:
        return {}
    context = state_identity(worker)
    left = diagnostic_budget.left(worker)
    offers, cards = {}, []
    # Bounded insertion order, not a winner inferred from effect or target name.
    for candidate_id, entry in list(registry.items())[-MAX_VISIBLE_CANDIDATES:]:
        frozen, evidence = entry["frozen"], entry["evidence"]
        cached = (candidate_id, context) in worker._candidate_measurements
        blocked = None
        if worker._candidate_action_context_failed:
            blocked = "prior_state_restoration_failed"
        elif worker._collect_active_edits():
            blocked = "active_edits_not_supported"
        elif worker.done():
            blocked = "terminal_prefix"
        elif not cached and left <= 0:
            blocked = "diagnostic_budget_exhausted"
        actions = []
        variant, variant_blocked = None, None
        try:
            variant = normal_cap_variant(worker, frozen)
            if variant.candidate_id not in registry and len(registry) >= MAX_CANDIDATES:
                variant_blocked = "episode_candidate_capacity_reached"
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            variant_blocked = str(exc)
        for action in ACTIONS:
            investigation = action == "investigate_normal_cap_current_prefix"
            normal_cap = action in {"measure_normal_cap_current_prefix", "investigate_normal_cap_current_prefix"}
            variant_id = variant.candidate_id if normal_cap and variant else None
            action_cached = (variant_id, context) in worker._candidate_measurements if normal_cap else cached
            action_blocked = (variant_blocked or blocked) if normal_cap else blocked
            if action_cached and action_blocked == "diagnostic_budget_exhausted":
                action_blocked = None
            required_calls = (1 if action_cached else 2) if investigation else int(not action_cached)
            if (investigation or not action_cached) and left <= 0:
                action_blocked = action_blocked or "diagnostic_budget_exhausted"
            elif investigation and left < required_calls:
                action_blocked = action_blocked or "diagnostic_budget_insufficient_for_investigation"
            action_id = identity("action:", [candidate_id, context, action, variant_id])
            offers[action_id] = {"action": action, "candidate_id": candidate_id,
                                 "expected_context_id": context, "variant_candidate_id": variant_id}
            measuring = action in {"remeasure_current_prefix", "measure_normal_cap_current_prefix",
                                   "investigate_normal_cap_current_prefix"}
            actions.append({"action": action, "action_id": action_id,
                "available": not (measuring and action_blocked),
                "blocked_reason": action_blocked if measuring else None,
                "diagnostic_cost": required_calls if measuring and not action_blocked else 0,
                "physical_replay_cost": 4 * required_calls if measuring and not action_blocked else 0,
                **({"requires_same_prefix_rounds": 1,
                    "confirmation_conditional_on_readout": True,
                    "apply_requires_separate_controller_decision": True} if investigation else {}),
                **({"new_candidate_id": variant_id,
                    "normal_step_size": variant.descriptor["step_size"] if variant else None,
                    "inherits_measurement": False} if normal_cap else {})})
        cards.append({"candidate_id": candidate_id, "objective_bundle_key": frozen.objective,
            "target_piece": frozen.token_piece,
            "operator": {key: frozen.descriptor[key] for key in ("site", "layer", "alpha", "step_size")},
            "evidence_context_id": evidence["measurement_context_id"],
            "prefix_changed": context != evidence["measurement_context_id"],
            "measurement_history": measurement_positions.history(worker, candidate_id, context),
            "current_context_measured": cached, "actions": actions})
    worker._candidate_action_offers = offers
    return {"candidate_diagnostic_choices": {"diagnostic": "candidate_action",
        "current_context_id": context, "candidate_count": len(registry), "cards": cards,
        "current_position": measurement_positions.describe(worker),
        "diagnostic_calls_left": left,
        "position_selection_owner": "controller",
        "measurement_scope": "frozen_source_and_binding_current_prefix_one_token",
        "production_apply_allowed": False}}


def execute_action(worker: Any, request: Mapping[str, Any], *, budget_left: int, source: str) -> dict[str, Any]:
    _initialize(worker)
    action_id = request.get("action_id")
    offer = worker._candidate_action_offers.get(action_id) if isinstance(action_id, str) else None
    result = {"diagnostic": "candidate_action", "step": worker._steps, "source": source,
        "action_id": action_id, "requested_action": offer["action"] if offer else None,
        "executed_action": None, "status": "blocked", "blocked_reason": None,
        "candidate_id": offer["candidate_id"] if offer else None, "new_measurement_count": 0,
        "physical_replay_count": 0, "diagnostic_budget_charged": False, **PERMISSIONS}
    if offer is None:
        return {**result, "blocked_reason": "unknown_or_stale_action_id"}
    context = state_identity(worker)
    result["current_context_id"] = context
    result["requested_position"] = measurement_positions.describe(worker)
    if offer["expected_context_id"] != context:
        return {**result, "blocked_reason": "stale_measurement_context"}
    entry = worker._frozen_diagnostic_candidates[offer["candidate_id"]]
    frozen = entry["frozen"]
    if not _candidate_intact(worker, frozen):
        return {**result, "blocked_reason": "frozen_candidate_mutated"}
    action = offer["action"]
    if action == "hold":
        return {**result, "status": "held", "executed_action": action}
    if action == "review_existing":
        return {**result, "status": "reviewed", "executed_action": action,
            "evidence_scope": "historical_measurement_not_current_certification",
            "measurement_context_id": entry["evidence"]["measurement_context_id"],
            "evidence": deepcopy(entry["evidence"])}
    if worker._candidate_action_context_failed or worker._collect_active_edits() or worker.done():
        return {**result, "blocked_reason": "prior_state_restoration_failed" if worker._candidate_action_context_failed
                else "active_edits_not_supported" if worker._collect_active_edits() else "terminal_prefix"}
    if action in {"measure_normal_cap_current_prefix", "investigate_normal_cap_current_prefix"}:
        try:
            variant = normal_cap_variant(worker, frozen)
            if variant.candidate_id != offer.get("variant_candidate_id"):
                raise ValueError("normal_cap_offer_changed")
            if variant.candidate_id not in worker._frozen_diagnostic_candidates and len(worker._frozen_diagnostic_candidates) >= MAX_CANDIDATES:
                raise ValueError("episode_candidate_capacity_reached")
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            return {**result, "blocked_reason": str(exc)}
        result.update(parent_candidate_id=frozen.candidate_id, candidate_id=variant.candidate_id,
                      candidate_derivation="normal_policy_step_cap", inherits_measurement=False)
        frozen = variant
        entry = {"frozen": frozen, "origin_row": deepcopy(entry.get("origin_row") or {}),
                 "parent_candidate_id": result["parent_candidate_id"]}
    key = (frozen.candidate_id, context)
    cached = worker._candidate_measurements.get(key)
    if cached is not None:
        row = cached.get("promotion_evidence")
        handoff = candidate_trial.handoff(worker, row, context=context, budget_left=budget_left) if row else {}
        return {**result, "status": "measurement_reused", "executed_action": action,
            "candidate_trial_handoff": handoff,
            "same_prefix_followup_available": bool(handoff.get("same_prefix_followup_available")),
            "measurement_context_id": context, "cached_measurement_count": 1,
            "evidence_scope": "same_candidate_same_context_measurement", "evidence": deepcopy(cached)}
    if budget_left <= 0:
        return {**result, "blocked_reason": "diagnostic_budget_exhausted"}
    result.update(executed_action=action, diagnostic_budget_charged=True)
    try:
        measured = measure_prefix(worker, frozen, horizon=1)
    except Exception as exc:
        measured = {"status": "incomplete", "error": f"{type(exc).__name__}:{exc}",
                    "state_restored": state_identity(worker) == context,
                    "physical_replay_count": 0, "physical_replay_count_is_lower_bound": True}
    complete = measured.get("status") == "complete" and measured.get("state_restored") is True
    if not measured.get("state_restored"):
        worker._candidate_action_context_failed = True
    if complete:
        measured["measurement_position"] = result["requested_position"]
        if action in {"measure_normal_cap_current_prefix", "investigate_normal_cap_current_prefix"}:
            measured.update(parent_candidate_id=result["parent_candidate_id"],
                            candidate_derivation="normal_policy_step_cap", inherits_measurement=False)
        row = candidate_trial.record(frozen, measured, entry.get("origin_row") or {})
        measured["promotion_evidence"] = row
        result["candidate_trial_handoff"] = candidate_trial.handoff(worker, row, context=context,
                                                                   budget_left=budget_left - 1)
        result["same_prefix_followup_available"] = result["candidate_trial_handoff"]["same_prefix_followup_available"]
        if result["same_prefix_followup_available"]:
            result["next_evidence_needed"] = "activation_patch_production_trial_gate_review"
        worker._candidate_measurements[key] = deepcopy(measured)
        worker._candidate_measurement_positions[key] = measurement_positions.output_tokens(worker)
        if action in {"measure_normal_cap_current_prefix", "investigate_normal_cap_current_prefix"}:
            worker._frozen_diagnostic_candidates[frozen.candidate_id] = {**entry, "evidence": deepcopy(measured)}
    return {**result, "status": "measured" if complete else "incomplete",
        "measurement_context_id": context, "new_measurement_count": int(complete),
        "physical_replay_count": measured.get("physical_replay_count", 0),
        "blocked_reason": None if complete else measured.get("error", "measurement_incomplete"),
        "evidence_scope": "frozen_candidate_current_prefix_measurement", "evidence": measured}


def execute_investigation(worker: Any, request: Mapping[str, Any], *, budget_left: int,
                          source: str, packet: Mapping[str, Any]) -> dict[str, Any]:
    """Measure the selected normal-cap child and conditionally confirm it here."""
    _initialize(worker)
    offer = worker._candidate_action_offers.get(request.get("action_id"))
    base = {"diagnostic": "candidate_action", "action_id": request.get("action_id"),
            "requested_action": "investigate_normal_cap_current_prefix",
            "status": "blocked", "transaction_status": "blocked", "new_measurement_count": 0,
            "physical_replay_count": 0, "diagnostic_call_cost": 0,
            "diagnostic_budget_charged": False, "same_prefix_followup_available": False,
            **PERMISSIONS}
    if not offer or offer["action"] != "investigate_normal_cap_current_prefix":
        return {**base, "blocked_reason": "unknown_or_stale_action_id"}
    context = state_identity(worker)
    if offer["expected_context_id"] != context:
        return {**base, "blocked_reason": "stale_measurement_context"}
    rounds = ((packet.get("strategy_hints") or {}).get("generation_control") or {}).get("same_prefix_rounds_left")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1 or worker.done():
        return {**base, "blocked_reason": "no_same_prefix_decision_round"}
    child_id = offer.get("variant_candidate_id")
    if not child_id:
        return {**base, "blocked_reason": "normal_cap_variant_unavailable"}
    cached = (child_id, context) in worker._candidate_measurements
    required_calls = 1 if cached else 2
    if budget_left < required_calls:
        return {**base, "blocked_reason": "diagnostic_budget_insufficient_for_investigation",
                "required_diagnostic_calls": required_calls}

    measured = execute_action(worker, request, budget_left=budget_left, source=source)
    measurement_cost = int(bool(measured.get("diagnostic_budget_charged")))
    result = {**measured, "transaction_status": "measurement_only",
              "diagnostic_call_cost": measurement_cost,
              "required_diagnostic_calls": required_calls,
              "production_trial_allowed": False}
    handoff = measured.get("candidate_trial_handoff") or {}
    if measured.get("status") not in {"measured", "measurement_reused"} or handoff.get("status") != "confirmation_available":
        result["same_prefix_followup_available"] = False
        return result

    from .response_promotion import review_response_evidence
    confirmation = review_response_evidence(worker, {
        "diagnostic": "activation_patch_production_trial_gate_review",
        "evidence_id": handoff["evidence_id"],
        "objective_bundle_key": handoff["objective_bundle_key"],
    }, packet)
    cost = measurement_cost + 1
    grant_id = confirmation.get("trial_authorization_id")
    grant = getattr(worker, "_response_trial_grants", {}).get(grant_id) if isinstance(grant_id, str) else None
    candidate = confirmation.get("production_trial_candidate")
    allowed = bool(confirmation.get("production_trial_allowed") is True
                   and isinstance(candidate, Mapping)
                   and isinstance(grant, Mapping) and grant.get("context_id") == context
                   and candidate.get("trial_edit") == grant.get("edit")
                   and isinstance(grant.get("row"), Mapping)
                   and grant["row"].get("evidence_id") == handoff["evidence_id"]
                   and state_identity(worker) == context)
    blocked_reasons = (confirmation.get("blocked_reasons") or
                       confirmation.get("production_trial_blocked_reasons") or
                       ([] if allowed else ["exact_current_context_trial_grant_missing"]))
    result.update(transaction_status="offer_available" if allowed else "confirmed_no_offer",
                  confirmation_result=confirmation,
                  confirmation_status=confirmation.get("status"),
                  confirmation_blocked_reasons=blocked_reasons,
                  physical_replay_count=result.get("physical_replay_count", 0)
                      + confirmation.get("physical_replay_count", 0),
                  new_measurement_count=result.get("new_measurement_count", 0)
                      + confirmation.get("new_measurement_count", 0),
                  diagnostic_call_cost=cost, diagnostic_budget_charged=True,
                  production_trial_allowed=allowed,
                  trial_authorization_id=grant_id if allowed else None,
                  production_trial_candidate=confirmation.get("production_trial_candidate") if allowed else None,
                  same_prefix_followup_available=allowed,
                  blocked_reason=None if allowed else blocked_reasons[0])
    return result
