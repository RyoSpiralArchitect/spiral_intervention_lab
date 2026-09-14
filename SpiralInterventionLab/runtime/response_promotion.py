"""Evidence-ID bridge into existing controller-owned promotion/trial reviews.

This module cannot apply an edit. It rejects historical, dose-changed and
gap-only evidence, and physically confirms qualified evidence before review.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .compiler import StepContext, compile_expr
from .evidence_inspection import evidence_catalog
from .response_probe import bound_metrics, identity, logit_variation, state_identity, tensor_identity, with_ownership


REVIEW_NAMES = {
    "activation_patch_promotion_gate_review",
    "activation_patch_production_shadow_replay",
    "activation_patch_production_trial_gate_review",
}


def response_review_readiness(row: Mapping[str, Any], context_id: str) -> dict[str, Any]:
    def finite(key: str) -> bool:
        value = row.get(key)
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)

    required = ("target_piece_prob_delta", "bound_token_top20_hit_delta", "target_piece_logit_delta",
        "repeat_max_abs_logit_delta", "no_edit_max_abs_logit_delta", "alignment_margin", "self_delta",
        "repeat_flag_delta", "repetition_score_delta", "entropy_delta", "top1_margin_delta",
        "required_term_recall_delta", "required_term_span_progress_delta", "activation_patch_hook_call_count")
    numeric = all(finite(k) for k in required)
    checks = {
        "complete_restored_measurement": row.get("measurement_complete") is True and row.get("state_restored") is True,
        "same_context": row.get("measurement_context_id") == context_id,
        "canonical_binding": row.get("target_piece_binding_variant") == "canonical" and row.get("target_piece_binding_requested_honored") is True,
        "finite_controls_and_effects": numeric,
        "concrete_execution": bool(row.get("execution_id") and row.get("observable_id") and row.get("measured_edit") and row.get("candidate_descriptor")),
    }
    checks.update({
        # Match the existing promotion gate's mass floor, but bind it to this
        # observable rather than borrowing another piece's term-set lift.
        "bound_target_lift": numeric and (row["target_piece_prob_delta"] >= 0.0001 or row["bound_token_top20_hit_delta"] >= 1),
        "response_above_variation": numeric and row["target_piece_logit_delta"] > max(row["repeat_max_abs_logit_delta"], row["no_edit_max_abs_logit_delta"], 0),
        "owned_target_lift": numeric and row.get("actuator_class") == "self_actuator" and row.get("actual_delta_class") == "target_lift"
            and row["self_delta"] > 0 and row["alignment_margin"] >= 0,
        "hook_executed": numeric and row["activation_patch_hook_call_count"] > 0,
        "nonregressing": numeric and row["repeat_flag_delta"] <= 0 and row["repetition_score_delta"] <= 0
            and row["required_term_recall_delta"] >= 0 and row["required_term_span_progress_delta"] >= 0
            and not (row["entropy_delta"] < -0.02 and row["top1_margin_delta"] > 0.002),
    })
    missing = [name for name, passed in checks.items() if not passed]
    return {"review_eligible": not missing, "checks": checks, "blocked_reasons": missing,
        "next_evidence_needed": "same_context_physical_confirmation" if not missing else "matched_response_probe",
        "production_apply_allowed": False, "production_trial_allowed": False, "certified_for_apply": False}


def review_response_evidence(worker: Any, request: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    base = {"diagnostic": request["diagnostic"], "status": "blocked", "evidence_id": request.get("evidence_id"),
        "physical_replay_count": 0, "new_measurement_count": 0, "diagnostic_only": True,
        "production_apply_allowed": False, "production_trial_allowed": False, "certified_for_apply": False}
    rows = [r for r in evidence_catalog(worker._diagnostic_results, full=True) if r.get("evidence_id") == request.get("evidence_id")]
    if len(rows) != 1:
        return {**base, "blocked_reasons": ["unique_recorded_evidence_required"]}
    row = rows[0]
    context_id = state_identity(worker)
    readiness = response_review_readiness(row, context_id)
    base["response_promotion_readiness"] = readiness
    if not readiness["review_eligible"]:
        return {**base, "blocked_reasons": readiness["blocked_reasons"]}
    objective = row["objective_bundle_key"]
    if request.get("objective_bundle_key", request.get("bundle_key")) != objective:
        return {**base, "blocked_reasons": ["objective_mismatch"]}
    try:
        candidate = row["candidate_descriptor"]
        edit = worker._activation_patch_trial_edit_from_candidate(candidate, trial_contract={
            "max_alpha": 0.15, "norm_clip": 1.0, "trial_budget_class": "primary"})
        if edit is None or any(edit[k] != row["measured_edit"][k] for k in ("target", "source", "op", "budget")):
            return {**base, "blocked_reasons": ["normal_trial_edit_differs_from_diagnostic"]}
        ctx = StepContext(packet=worker.build_controller_packet(), runtime_state=worker.runtime_state,
            adapter=worker.adapter, traces={}, stats={}, active_edits={})
        tensor = compile_expr(dict(edit["source"]["expr"]))(ctx)
        execution_id = identity("exec:", [context_id, edit["target"], edit["op"], edit["budget"], tensor_identity(tensor)])
        if execution_id != row["execution_id"]:
            return {**base, "blocked_reasons": ["resolved_execution_changed"]}
        edit["meta"].update(apply_kind="diagnostic_probe", production_trial_allowed=False, diagnostic_only=True)
        edit.update(bundle_key=objective, focus_feature=row["intended_term"], phase_objective="readout_escape")
        # No cap override, no production apply. Compare two-token trajectories
        # and first-token readout using the exact normal-budget edit.
        baseline = worker._simulate_decode(max_new_tokens=2, top_k=6, score_candidate_text=True, score_observer_check=False)
        base["physical_replay_count"] += 1
        captured: dict[str, Any] = {}
        replay = worker.replay_candidate_edits_actual_delta([edit], max_new_tokens=2, top_k=6,
            score_candidate_text=True, intended_bundle_key=objective, intended_term=row["intended_term"],
            ownership_terms=(row["intended_term"],),
            target_piece_binding=row["target_piece_binding_manifest"], _baseline_snapshot=baseline,
            _measurement_capture=captured)
        base["physical_replay_count"] += 1
        if replay.get("status") != "ok" or "edited_logits" not in captured:
            raise ValueError("confirmation_replay_failed")
        replay = with_ownership(worker, replay, objective, row["intended_term"])
        metrics = bound_metrics(baseline["first_logits"], captured["edited_logits"], row["target_piece_token_id"])
        null = worker._simulate_decode(max_new_tokens=2, top_k=6, score_candidate_text=True, score_observer_check=False)
        base["physical_replay_count"] += 1
        variation = logit_variation(baseline["first_logits"], null["first_logits"])
        confirmed = {**row, **replay, **metrics, "no_edit_max_abs_logit_delta": variation["max_abs_logit_delta"],
            "target_piece_binding_requested_honored": replay.get("target_piece_token_id") == row["target_piece_token_id"]
                and replay.get("target_piece_binding_id") == row["target_piece_binding_id"],
            "state_restored": state_identity(worker) == context_id}
        confirmation = response_review_readiness(confirmed, context_id)
        base.update(new_measurement_count=1, confirmation=confirmation)
        if not confirmation["review_eligible"]:
            return {**base, "blocked_reasons": ["physical_confirmation_failed", *confirmation["blocked_reasons"]]}
        delta_keys = ("target_mass_delta", "target_top20_hit_delta", "focus_rank_delta", "self_delta", "alignment_margin")
        shadow_row = {**row, "activation_patch_actuator_class": replay.get("actuator_class"),
            "promotable_to_candidate_compiler": True, "actuator_bundle_key": objective, "objective_term": row["intended_term"],
            "actual_delta_class": replay["actual_delta_class"],
            "counterfactual_delta": {k: replay.get(k, 0.0) for k in delta_keys}}
        status = {"activation_patch_shadow_actuator": shadow_row}
        promotion = worker._activation_patch_promotion_gate_review(status)
        base["activation_patch_promotion_gate_review"] = promotion
        result = promotion or {}
        if request["diagnostic"] in {"activation_patch_production_shadow_replay", "activation_patch_production_trial_gate_review"}:
            shadow = worker._activation_patch_production_shadow_replay(status, request=request,
                packet_context=packet, promotion_gate_review=promotion)
            base["activation_patch_production_shadow_replay"] = shadow
            result = shadow or {}
            if request["diagnostic"] == "activation_patch_production_trial_gate_review":
                result = worker._activation_patch_production_trial_gate_review(status, request=request,
                    packet_context=packet, production_shadow_replay=shadow, promotion_gate_review=promotion) or {}
                base["activation_patch_production_trial_gate_review"] = result
        return {**base, **{k: v for k, v in result.items() if k in ("status", "next_evidence_needed", "why_not_apply",
            "production_trial_allowed", "production_trial_candidate", "production_trial_contract", "production_trial_blocked_reasons")},
            "execution_id": execution_id, "confirmation_scope": "same_context_two_token_normal_budget_replay"}
    except Exception as exc:
        return {**base, "blocked_reasons": ["confirmation_error"], "error": f"{type(exc).__name__}:{exc}"}
