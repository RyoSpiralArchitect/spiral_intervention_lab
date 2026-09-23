"""Read-only, bounded views of recorded evidence, separate from replay budgets."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from .response_probe import identity


ROW_FIELDS = (
    "execution_id", "observable_id", "measurement_context_id", "evidence_scope",
    "objective_bundle_key", "bundle_key", "operator_recipe_id", "recipe_name",
    "seed_source", "activation_patch_seed_source", "operator_axis", "status",
    "activation_patch_site", "activation_patch_layer", "activation_patch_alpha",
    "activation_patch_step_size", "activation_patch_source_localization",
    "target_piece", "target_piece_token_id", "target_piece_binding_id",
    "target_piece_logit_delta", "target_piece_prob_delta", "threshold20_logit_delta",
    "target_top20_threshold_gap_delta", "target_mass_delta", "target_top20_hit_delta",
    "target_rank_delta", "actual_delta_class", "actuator_class", "verdict", "repeat_delta", "entropy_delta",
    "target_piece_binding_requested_honored", "repeat_max_abs_logit_delta",
    "actual_delta_class_scope", "bound_token_response", "measurement_focus_terms",
    "comparison_axis", "source_variant", "source_variant_origin", "bound_token_top20_hit_delta",
    "no_edit_max_abs_logit_delta", "state_restored", "measurement_complete",
)


def evidence_catalog(results: Sequence[Any], *, full: bool = False) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def visit(value: Any, depth: int = 0) -> None:
        if depth > 12 or len(found) >= 128:
            return
        if isinstance(value, Mapping):
            if value.get("operator_recipe_id") and (value.get("actual_delta_class") or value.get("execution_id")):
                compact = {k: value[k] for k in ROW_FIELDS if k in value}
                key = str(value.get("observable_id") or identity("evidence:", compact))
                row = dict(value) if full else compact
                if not full:
                    invalid = [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v)]
                    for k in invalid:
                        row[k] = None
                    if invalid:
                        row["invalid_numeric_fields"] = invalid
                row["evidence_id"] = key
                row.setdefault("evidence_scope", "historical_context_not_revalidated")
                found.setdefault(key, row)
            for key, child in value.items():
                if key not in {"candidate_fingerprint", "eval_context_fingerprint", "candidate_edits", "target_piece_binding_report", "candidate_descriptor", "measured_edit"}:
                    visit(child, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value[:128]:
                visit(item, depth + 1)

    for result in reversed(results):
        visit(result)
    # Expose executed records first, without ranking by favorable outcome.
    return sorted(found.values(), key=lambda row: not bool(row.get("execution_id")))


def inspect_evidence(worker: Any, request: Mapping[str, Any], hints: Mapping[str, Any]) -> dict[str, Any]:
    base = {"diagnostic_only": True, "production_apply_allowed": False, "certified_for_apply": False,
            "new_measurement_count": 0, "physical_replay_count": 0, "cached_measurement_count": 0,
            "rows": [], "evidence_scope": "historical_view_not_current_permission"}
    limit = request.get("limit", 4)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 6:
        return {**base, "status": "invalid_request", "unavailable_reason": "limit_must_be_1_to_6"}
    rows = evidence_catalog(worker._diagnostic_results)
    for field in ("execution_id", "evidence_id", "objective_bundle_key", "operator_recipe_id"):
        if request.get(field):
            rows = [r for r in rows if r.get(field) == request[field]]
    rows = rows[:limit]
    ids = {r["evidence_id"] for r in rows}
    gate_facts = {key: hints[key] for key in (
        "diagnostic_frontier_bundle_key", "diagnostic_frontier_next_evidence", "diagnostic_budget_exhausted",
        "blocked_next_diagnostics", "production_trial_eligible", "compile_preview_blocked_reason",
    ) if key in hints}
    signature = identity("gate:", gate_facts)
    prior_facts = getattr(worker, "_inspection_gate_facts", {})
    new_gate = sum(key not in prior_facts or prior_facts[key] != value for key, value in gate_facts.items())
    context_changed = signature != worker._inspection_gate_signature
    new_views = len(ids - worker._inspection_seen_ids)
    worker._inspection_seen_ids.update(ids)
    worker._inspection_gate_signature = signature
    worker._inspection_gate_facts = gate_facts
    available = hints.get("available_next_diagnostics")
    if not isinstance(available, (list, tuple)):
        available = ()
    return {**base, "status": "ok" if rows else "no_matching_evidence", "rows": rows,
            "cached_measurement_count": len({r.get("execution_id") for r in rows if r.get("execution_id")}),
            "newly_visible_row_count": new_views, "new_gate_fact_count": new_gate,
            "gate_context_changed": context_changed, "gate_context": gate_facts,
            "gate_fact_scope": "listed_current_packet_fields_only_not_exhaustive",
            "retry_preconditions": ["new_execution_id", "explicit_context_or_dose_change", "changed_gate_context"],
            "missing_evidence": [] if rows else ["matching_execution_or_recipe_id"],
            "executable_diagnostic_ids": [str(r.get("diagnostic")) for r in available
                                          if isinstance(r, Mapping) and r.get("diagnostic")][:8]}
