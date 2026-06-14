from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence as SequenceABC
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if isinstance(value, bool):
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        return int(value)
    except Exception:
        return default


def diagnostic_objective_keys(result: Mapping[str, Any]) -> set[str]:
    """Return objective-like bundle keys mentioned by a diagnostic result."""

    result_objectives = {
        str(result.get("objective_bundle_key") or ""),
        str(result.get("bundle_key") or ""),
        str(result.get("step_actuator_bundle_key") or ""),
    }
    for nested_key in ("readout_deepening_review_summary", "operator_recipe_expansion_summary"):
        nested = result.get(nested_key)
        if isinstance(nested, Mapping):
            result_objectives.add(str(nested.get("objective_bundle_key") or ""))
            result_objectives.add(str(nested.get("bundle_key") or ""))
    for rows_key in ("evidence_rows", "operator_recipe_expansion_matrix"):
        rows = result.get(rows_key)
        if not isinstance(rows, SequenceABC) or isinstance(rows, (str, bytes, bytearray)):
            continue
        for row in rows[:8]:
            if not isinstance(row, Mapping):
                continue
            result_objectives.add(str(row.get("objective_bundle_key") or ""))
            result_objectives.add(str(row.get("bundle_key") or ""))
    result_objectives.discard("")
    return result_objectives


def readout_gap_confirmation_seen_for_objective(
    diagnostic_results: SequenceABC[Any],
    objective_key: str,
) -> bool:
    """Return whether readout gap confirmation has already covered an objective."""

    objective_key = str(objective_key or "")
    for result in diagnostic_results:
        if not isinstance(result, Mapping):
            continue
        if str(result.get("diagnostic") or "") not in {
            "compare_extra_operator_diagnostics",
            "readout_gap_confirmation_or_variant_sweep",
        }:
            continue
        if str(result.get("operator_recipe_expansion_mode") or "") != (
            "readout_gap_confirmation_or_variant_sweep"
        ):
            continue
        if not objective_key:
            return True
        if objective_key in diagnostic_objective_keys(result):
            return True
    return False


def confirmed_gap_only_objective_rows(
    diagnostic_results: SequenceABC[Any],
    *,
    term_resolver: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Return latest objectives where gap movement was confirmed without target lift."""

    rows_by_objective: dict[str, dict[str, Any]] = {}
    for result in diagnostic_results:
        if not isinstance(result, Mapping):
            continue
        diagnostic = str(result.get("diagnostic") or "")
        if diagnostic not in {
            "compare_extra_operator_diagnostics",
            "readout_gap_confirmation_or_variant_sweep",
        }:
            continue
        if str(result.get("operator_recipe_expansion_mode") or "") != (
            "readout_gap_confirmation_or_variant_sweep"
        ):
            continue
        review = result.get("readout_deepening_review_summary")
        if not isinstance(review, Mapping):
            continue
        role = str(review.get("best_candidate_role") or "")
        trial_eligible = bool(review.get("production_trial_eligible", False))
        objective_key = str(
            review.get("objective_bundle_key")
            or result.get("objective_bundle_key")
            or result.get("bundle_key")
            or ""
        )
        if not objective_key:
            continue
        target_mass_delta = _as_float(review.get("target_mass_delta"), default=0.0)
        target_top20_hit_delta = _as_int(review.get("target_top20_hit_delta"), default=0)
        direct_target_effect = target_mass_delta > 0.00002 or target_top20_hit_delta > 0
        if role != "gap_closer_candidate" or trial_eligible or direct_target_effect:
            continue
        term = term_resolver(objective_key) if term_resolver is not None else ""
        rows_by_objective[objective_key] = {
            "objective_bundle_key": objective_key,
            "term": term or None,
            "objective_status": "confirmed_gap_only_no_target_lift",
            "rotation_eligible": False,
            "ttl_reason": "until_new_operator_family_or_conversion_sweep",
            "best_candidate_role": role,
            "gap_delta": review.get("gap_delta"),
            "target_mass_delta": review.get("target_mass_delta"),
            "target_top20_hit_delta": review.get("target_top20_hit_delta"),
            "best_recipe_id": review.get("best_recipe_id"),
            "best_recipe_name": review.get("best_recipe_name"),
            "recorded_step": result.get("recorded_step"),
        }
    return sorted(
        rows_by_objective.values(),
        key=lambda row: int(row.get("recorded_step", -1) or -1),
        reverse=True,
    )


def operator_family_shift_preview_rows(
    *,
    objective_bundle_key: str,
    objective_term: str = "",
    seed_recipe_id: str = "",
    seed_recipe_family: str = "",
) -> list[dict[str, Any]]:
    """Return diagnostic-only non-KV families to try after carrier conversion stalls."""

    objective_key = str(objective_bundle_key or "")
    term = str(objective_term or "")
    seed_recipe = str(seed_recipe_id or "")
    seed_family = str(seed_recipe_family or "")
    rows = [
        {
            "operator_family": "resid_readout_direction_patch",
            "operator_axis": "non_kv_operator_search",
            "candidate_kind": "resid_add",
            "diagnostic_family": "readout_direction_patch",
            "purpose": "test whether direct residual readout steering can turn a gap carrier into target mass/top20 lift",
        },
        {
            "operator_family": "activation_patch_source_term_token",
            "operator_axis": "non_kv_operator_search",
            "candidate_kind": "activation_patch",
            "diagnostic_family": "source_term_activation_patch",
            "purpose": "patch a source-body term-local activation instead of another readout gap steer",
        },
        {
            "operator_family": "anti_attractor_suppression_patch",
            "operator_axis": "non_kv_operator_search",
            "candidate_kind": "attractor_suppression",
            "diagnostic_family": "collapse_suppression",
            "purpose": "reduce junk-attractor pressure before asking for more target lift",
        },
        {
            "operator_family": "attention_route_carrier_probe",
            "operator_axis": "non_kv_operator_search",
            "candidate_kind": "attention_route_probe",
            "diagnostic_family": "attention_route_carrier",
            "purpose": "inspect whether answer-boundary attention route is the missing carrier-to-actuator bridge",
        },
    ]
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        item = dict(row)
        item.update(
            {
                "objective_bundle_key": objective_key or None,
                "objective_term": term or None,
                "preview_rank": index + 1,
                "source": "carrier_to_actuator_conversion_failed",
                "permission": "diagnostic_only",
                "production_apply_allowed": False,
                "policy_candidate_ready": False,
                "seed_operator_recipe_id": seed_recipe or None,
                "seed_recipe_family": seed_family or None,
            }
        )
        normalized.append({key: value for key, value in item.items() if value not in (None, "", [])})
    return normalized


def operator_family_shift_canonical_request(
    *,
    objective_bundle_key: str,
    objective_term: str = "",
    seed_recipe_id: str = "",
    seed_recipe_family: str = "",
    reason: str | None = None,
) -> dict[str, Any]:
    preview_rows = operator_family_shift_preview_rows(
        objective_bundle_key=objective_bundle_key,
        objective_term=objective_term,
        seed_recipe_id=seed_recipe_id,
        seed_recipe_family=seed_recipe_family,
    )
    request = {
        "diagnostic": "compare_extra_operator_diagnostics",
        "bundle_key": objective_bundle_key or None,
        "objective_bundle_key": objective_bundle_key or None,
        "next_evidence_needed": "non_kv_operator_search",
        "operator_recipe_expansion_mode": "non_kv_operator_search",
        "operator_family_shift_requested": True,
        "operator_family_shift_source": "carrier_to_actuator_conversion_failed",
        "operator_family_shift_preview_rows": preview_rows,
        "seed_operator_recipe_id": seed_recipe_id or None,
        "seed_recipe_family": seed_recipe_family or None,
        "reason": reason
        or "carrier-to-actuator conversion stayed carrier-only; inspect bounded non-KV operator families",
        "permission": "diagnostic_only",
        "production_apply_allowed": False,
        "policy_candidate_ready": False,
    }
    return {key: value for key, value in request.items() if value not in (None, "", [])}
