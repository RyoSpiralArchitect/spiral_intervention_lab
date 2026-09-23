"""Read-only seed-to-card handoff facts and episode-local diagnostic delay accounting."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from . import diagnostic_budget, measurement_positions
from .evidence_inspection import evidence_catalog
from .response_probe import identity, select_probe_seeds


SAFETY_DIAGNOSTICS = frozenset({
    "inspect_evidence",
    "activation_patch_runtime_support_probe",
    "activation_patch_promotion_gate_review",
    "activation_patch_production_shadow_replay",
    "activation_patch_production_trial_gate_review",
})
OFFER_REQUEST_FIELDS = (
    "diagnostic", "objective_bundle_key", "focus_term", "comparison_axis", "dose_grid", "candidate_ids",
)


def matches_offered_request(request: Mapping[str, Any], offer: Mapping[str, Any]) -> bool:
    expected = offer.get("request")
    return isinstance(expected, Mapping) and all(
        request.get(key) == expected.get(key) for key in OFFER_REQUEST_FIELDS)


def reset(worker: Any) -> None:
    worker._candidate_handoff_deferred = {}
    worker._candidate_handoff_attempted = set()
    worker._candidate_handoff_unavailable = set()


def report(worker: Any) -> dict[str, Any]:
    """Seed preflight is not a binding check, physical measurement, or permission."""
    catalog = evidence_catalog(getattr(worker, "_diagnostic_results", ()), full=True)
    pool = [row for row in catalog if row.get("activation_patch_site") in {"resid_pre", "resid_post", "mlp_out"}
            and str(row.get("objective_bundle_key") or row.get("bundle_key") or "")]
    objectives = sorted({str(row.get("objective_bundle_key") or row.get("bundle_key")) for row in pool})
    registry = getattr(worker, "_frozen_diagnostic_candidates", {}) or {}
    carded = {str(entry["frozen"].objective) for entry in registry.values()
              if isinstance(entry, Mapping) and entry.get("frozen") is not None}
    feedback = getattr(worker, "_last_task_feedback", {}) or {}
    present_terms = {str(term).casefold() for term in feedback.get("required_terms_present", ())}
    left = diagnostic_budget.left(worker)
    blocked = "diagnostic_budget_exhausted" if left <= 0 else (
        "terminal_prefix" if worker.done() else (
            "active_edits_not_supported" if worker._collect_active_edits() else None
        )
    )
    offers: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    prefix_id = identity("handoff-prefix:", [worker._steps, measurement_positions.output_tokens(worker)])
    attempted = getattr(worker, "_candidate_handoff_attempted", set()) or set()
    failed = getattr(worker, "_candidate_handoff_unavailable", set()) or set()
    if blocked is None:
        for objective in objectives:
            if objective in carded:
                continue
            objective_rows = [row for row in pool if str(row.get("objective_bundle_key") or row.get("bundle_key")) == objective]
            fallback_term = objective.split(":")[1] if ":" in objective else objective
            term = str(objective_rows[0].get("intended_term") or objective_rows[0].get("objective_term")
                       or fallback_term)
            if term.casefold() in present_terms:
                unavailable[objective] = "objective_term_already_present_control_only"
                continue
            seeds, reason = select_probe_seeds(catalog, objective_bundle_key=objective,
                objective_term=term, comparison_axis="source_localization")
            if reason:
                unavailable[objective] = reason
                continue
            anchor = seeds[0]
            recipe_id = str(anchor.get("seed_operator_recipe_id") or anchor.get("operator_recipe_id") or "")
            if (prefix_id, objective, recipe_id) in attempted:
                unavailable[objective] = "measurement_attempt_consumed_at_this_prefix"
                continue
            if (prefix_id, objective, recipe_id) in failed:
                unavailable[objective] = "offered_measurement_no_physical_replay_at_this_prefix"
                continue
            request = {"diagnostic": "matched_response_probe", "objective_bundle_key": objective,
                "focus_term": term, "comparison_axis": "source_localization", "dose_grid": [0.04]}
            if recipe_id:
                request["candidate_ids"] = [recipe_id]
            offers.append({"objective_bundle_key": objective, "focus_term": term,
                "seed_recipe_id": recipe_id or None, "seed_source": anchor.get("seed_source"),
                "request": request})
    if offers:
        state = "measurable"
    elif carded:
        state = "carded"
    else:
        state = "unmeasurable"
    context_id = identity("handoff-offers:", [prefix_id, [offer["objective_bundle_key"] for offer in offers]])
    deferred = getattr(worker, "_candidate_handoff_deferred", {}) or {}
    defer_count = int(deferred.get(context_id, 0)) if state == "measurable" else 0
    blocked_reason = blocked
    if blocked_reason is None and not offers and not carded:
        blocked_reason = ("no_recorded_activation_patch_seed" if not pool else
                          next(iter(unavailable.values())) if len(unavailable) == 1 else
                          "no_materializable_activation_patch_seed")
    return {"state": state, "preflight_scope": "recorded_seed_only_binding_and_replay_unverified",
        "context_id": context_id, "prefix_id": prefix_id,
        "pool_row_count": len(pool), "pool_objective_count": len(objectives),
        "measured_card_count": len(registry), "measurement_offers": offers[:4],
        "unavailable_reasons": dict(list(unavailable.items())[:4]),
        "blocked_reason": blocked_reason,
        "deferred_diagnostic_count": defer_count,
        "soft_opportunity_cost": min(3, max(0, defer_count - 1)) if state == "measurable" else 0,
        "production_apply_allowed": False}


def record_diagnostic(worker: Any, handoff: Mapping[str, Any], request: Mapping[str, Any],
                      result: dict[str, Any], *, cost: int, source: str) -> None:
    if source != "controller" or handoff.get("state") != "measurable":
        return
    diagnostic = str(request.get("diagnostic") or "")
    offered = {str(offer.get("objective_bundle_key")): offer
               for offer in handoff.get("measurement_offers", ()) if isinstance(offer, Mapping)}
    objective = str(request.get("objective_bundle_key") or "")
    if diagnostic == "matched_response_probe" and objective in offered:
        same_measurement = matches_offered_request(request, offered[objective])
        replayed = (int(result.get("physical_replay_count") or 0) > 0
                    or int(result.get("new_measurement_count") or 0) > 0)
        if same_measurement:
            attempt = (str(handoff.get("prefix_id") or ""), objective,
                       str(offered[objective].get("seed_recipe_id") or ""))
            if attempt[0]:
                attribute = "_candidate_handoff_attempted" if replayed else "_candidate_handoff_unavailable"
                history = getattr(worker, attribute, None)
                if not isinstance(history, set):
                    history = set()
                    setattr(worker, attribute, history)
                history.add(attempt)
            result["candidate_handoff_choice"] = (
                "measurement_attempted" if replayed else "measurement_unavailable_at_this_prefix")
        else:
            result["candidate_handoff_choice"] = (
                "measurement_not_executed" if not replayed else "different_measurement_executed")
        return
    if cost <= 0:
        return
    if diagnostic in SAFETY_DIAGNOSTICS or diagnostic == "candidate_action":
        result["candidate_handoff_choice"] = "safety_or_card_action_exempt"
        return
    context_id = str(handoff.get("context_id") or "")
    if not context_id:
        return
    memory = getattr(worker, "_candidate_handoff_deferred", None)
    if not isinstance(memory, dict):
        memory = {}
        worker._candidate_handoff_deferred = memory
    memory[context_id] = min(12, int(memory.get(context_id, 0)) + 1)
    if len(memory) > 16:
        memory.pop(next(iter(memory)))
    result["candidate_handoff_choice"] = "deferred_for_other_diagnostic"
    result["candidate_handoff_deferred_diagnostic_count"] = memory[context_id]
