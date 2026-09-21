"""Reuse closed evidence reviews, never measurements or permission decisions."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .evidence_inspection import evidence_catalog
from .response_probe import identity


REVIEW_NAMES = {"activation_patch_candidate_review", "activation_patch_runtime_support_probe"}


def prefix_identity(worker: Any) -> str:
    return identity("prefix:", {
        "segments": [(s.kind, s.token_ids) for s in worker._segments],
        "active_edits": worker._collect_active_edits(),
    })


def review_key(request: Mapping[str, Any], hints: Mapping[str, Any], results: Sequence[Any]) -> str | None:
    name = request.get("diagnostic")
    if name not in REVIEW_NAMES or request.get("evidence_id"):
        return None
    objective = (request.get("objective_bundle_key") or request.get("bundle_key")
                 or hints.get("diagnostic_frontier_bundle_key") or hints.get("gate_report_frontier_bundle_key")
                 or hints.get("selected_bundle_key"))
    if not objective:
        return None
    rows = [r for r in evidence_catalog(results, full=True)
            if (r.get("objective_bundle_key") or r.get("bundle_key")) == objective]
    if not rows:
        return None
    # Include real evidence and materialization descriptors, not a favorable score
    # or the request's prose. A new execution/binding/source invalidates reuse.
    evidence = sorted(identity("row:", r) for r in rows)
    intent = {k: v for k, v in request.items() if k not in {
        "reason", "requested_by", "source", "canonical_followup_request_applied",
        "bundle_key", "objective_bundle_key", "step_actuator_bundle_key",
    }}
    return identity("review:", [objective, intent, evidence])


def closed_review(result: Mapping[str, Any]) -> bool:
    matrix = result.get("target_piece_binding_seed_matrix_summary") or {}
    if not isinstance(matrix, Mapping):
        return False
    return bool(
        result.get("status") == "ok"
        and matrix.get("status") in {"dose_matched_response_complete", "already_replayed"}
        and not matrix.get("errors")
        and matrix.get("state_restored") is not False
        and not result.get("activation_patch_compile_preview_created")
        and result.get("activation_patch_compile_preview_blocked_reason") == "rank_carrier_not_target_actuator"
        and not result.get("production_trial_allowed")
        and not result.get("production_apply_allowed")
    )


def review_receipt(worker: Any, request: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    matrix = result.get("target_piece_binding_seed_matrix_summary") or {}
    return {
        "diagnostic": request["diagnostic"],
        "objective_bundle_key": result.get("objective_bundle_key") or result.get("bundle_key"),
        "reviewed_at_step": worker._steps,
        "reviewed_prefix_id": prefix_identity(worker),
        "measurement_context_id": matrix.get("measurement_context_id"),
        "compile_preview_blocked_reason": result.get("activation_patch_compile_preview_blocked_reason"),
        "evidence_ids": [r["evidence_id"] for r in evidence_catalog([result])][:6],
    }


def reuse_report(worker: Any, receipt: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    current = prefix_identity(worker)
    return {
        **receipt, "status": "review_reused", "source": source, "recorded_step": worker._steps,
        "current_prefix_id": current,
        "prefix_changed_since_review": current != receipt["reviewed_prefix_id"],
        "evidence_scope": "historical_review_not_current_measurement",
        "review_reused": True, "diagnostic_budget_charged": False,
        "new_measurement_count": 0, "physical_replay_count": 0,
        "next_evidence_needed": "explicit_current_prefix_measurement_or_evidence_inspection",
        "explicit_measurement_diagnostic": "matched_response_probe",
        "why_not_apply": "Reusing a closed review supplies no current-prefix certification or permission.",
        "diagnostic_only": True, "production_apply_allowed": False,
        "certified_for_apply": False, "policy_candidate_ready": False,
    }


def reuse_hints(worker: Any) -> dict[str, Any]:
    receipts = list(getattr(worker, "_diagnostic_review_cache", {}).values())[-4:]
    if not receipts:
        return {}
    current = prefix_identity(worker)
    return {"diagnostic_review_reuse": [{
        **r, "prefix_changed_since_review": current != r["reviewed_prefix_id"],
        "evidence_scope": "historical_review_not_current_measurement",
        "explicit_measurement_diagnostic": "matched_response_probe",
        "production_apply_allowed": False,
    } for r in receipts]}
