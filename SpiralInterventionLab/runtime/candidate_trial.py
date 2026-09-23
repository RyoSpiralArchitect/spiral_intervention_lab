"""Current-prefix evidence handoff; neither a measurement nor an offer is apply permission."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any

from .response_probe import identity, state_identity, tensor_identity


def record(frozen: Any, measured: Mapping[str, Any], origin: Mapping[str, Any]) -> dict[str, Any]:
    edit = deepcopy(frozen.edit)
    edit["source"]["expr"] = deepcopy(frozen.original_source_expr)
    context = measured["measurement_context_id"]
    execution = identity("exec:", [context, edit["target"], edit["op"], edit["budget"], frozen.source_identity])
    binding = deepcopy(origin.get("target_piece_binding_manifest") or {})
    observable = identity("obs:", [execution, frozen.token_id, "current_prefix_candidate"])
    hooks = measured.get("edit_runtime_telemetry") or []
    # Do not inherit ownership or safety labels from the candidate's old anchor.
    return {
        "evidence_kind": "current_prefix_candidate_measurement", "evidence_id": observable,
        "observable_id": observable, "execution_id": execution, "measurement_context_id": context,
        "origin_observable_id": origin.get("observable_id"),
        "candidate_id": frozen.candidate_id, "candidate_descriptor": deepcopy(frozen.descriptor),
        **{"activation_patch_" + k: frozen.descriptor[k] for k in ("site", "layer", "alpha", "step_size", "source_localization")},
        "measured_edit": {k: edit[k] for k in ("target", "source", "op", "budget")},
        "operator_recipe_id": edit.get("meta", {}).get("operator_recipe_id") or frozen.descriptor["operator_recipe_id"],
        "objective_bundle_key": frozen.objective, "intended_term": frozen.term,
        "target_piece": frozen.token_piece, "target_piece_token_id": frozen.token_id,
        "target_piece_binding_id": binding.get("binding_id"), "target_piece_binding_manifest": binding,
        "target_piece_binding_variant": origin.get("target_piece_binding_variant"),
        "target_piece_binding_requested_honored": bool(
            origin.get("target_piece_binding_requested_honored") is True
            and binding.get("chosen_target_token_id") == frozen.token_id),
        "source_tensor_identity": deepcopy(frozen.source_identity),
        "measurement_complete": measured.get("status") == "complete",
        "state_restored": measured.get("state_restored") is True,
        "no_edit_max_abs_logit_delta": measured.get("no_edit_control", {}).get("max_abs_logit_delta"),
        "repeat_max_abs_logit_delta": measured.get("repeat_control", {}).get("max_abs_logit_delta"),
        "activation_patch_hook_call_count": sum(h.get("hook_call_count", 0) for h in hooks
                                                if h.get("op") == "activation_patch"),
        **dict(measured.get("metrics") or {}),
        "actual_delta_class": "unassessed", "actuator_class": "unknown",
        "ownership_status": "requires_current_prefix_confirmation",
        "safety_status": "requires_current_prefix_confirmation",
        "evidence_scope": "current_prefix_readout_only_not_trial_certification",
        "production_apply_allowed": False, "production_trial_allowed": False, "certified_for_apply": False,
    }


def normal_edit(worker: Any, row: Mapping[str, Any], packet: Mapping[str, Any]) -> dict[str, Any]:
    from .compiler import StepContext, compile_expr
    edit = worker._activation_patch_trial_edit_from_candidate(row["candidate_descriptor"], trial_contract={
        "max_alpha": 0.15, "norm_clip": 1.0, "trial_budget_class": "primary"})
    if edit is None or any(edit[k] != row["measured_edit"][k] for k in ("target", "source", "op", "budget")):
        raise ValueError("normal_trial_edit_differs_from_diagnostic")
    ctx = StepContext(packet=packet, runtime_state=worker.runtime_state, adapter=worker.adapter,
                      traces={}, stats={}, active_edits={})
    resolved = tensor_identity(compile_expr(dict(edit["source"]["expr"]))(ctx))
    execution = identity("exec:", [state_identity(worker), edit["target"], edit["op"], edit["budget"], resolved])
    if execution != row["execution_id"]:
        raise ValueError("resolved_execution_changed")
    return edit


def handoff(worker: Any, row: Mapping[str, Any], *, context: str, budget_left: int) -> dict[str, Any]:
    from .response_promotion import response_confirmation_readiness
    readiness = response_confirmation_readiness(row, context)
    reasons = list(readiness["blocked_reasons"])
    if budget_left <= 0:
        reasons.append("diagnostic_budget_exhausted")
    if not reasons:
        try:
            normal_edit(worker, row, getattr(worker, "_last_packet", None) or {})
        except Exception as exc:
            reasons.append(str(exc))
    return {
        "status": "confirmation_available" if not reasons else "blocked",
        "evidence_id": row["evidence_id"], "candidate_id": row.get("candidate_id"),
        "measurement_context_id": row["measurement_context_id"],
        "objective_bundle_key": row["objective_bundle_key"],
        "blocked_reasons": reasons, "checks": readiness["checks"],
        "canonical_request": {"diagnostic": "activation_patch_production_trial_gate_review",
            "evidence_id": row["evidence_id"], "objective_bundle_key": row["objective_bundle_key"]} if not reasons else None,
        "same_prefix_followup_available": not reasons,
        "production_apply_allowed": False, "production_trial_allowed": False,
    }


def authorize(worker: Any, result: dict[str, Any], row: Mapping[str, Any], edit: Mapping[str, Any]) -> None:
    candidate = result.get("production_trial_candidate")
    if not result.get("production_trial_allowed") or not isinstance(candidate, Mapping):
        return
    trial = candidate.get("trial_edit")
    if not isinstance(trial, Mapping) or any(trial.get(k) != edit[k] for k in ("target", "source", "op", "budget")):
        result.update(status="blocked", production_trial_allowed=False, production_trial_candidate=None,
                      blocked_reasons=["final_trial_edit_differs_from_confirmation"])
        result["activation_patch_production_trial_gate_review"] = {
            **dict(result.get("activation_patch_production_trial_gate_review") or {}),
            "status": "blocked", "production_trial_allowed": False, "production_trial_candidate": None,
            "production_trial_blocked_reasons": ["final_trial_edit_differs_from_confirmation"]}
        return
    context = state_identity(worker)
    grant_id = identity("trial:", [context, row["execution_id"], row["evidence_id"]])
    trial = deepcopy(trial)
    trial.setdefault("meta", {}).update(trial_authorization_id=grant_id, apply_kind="production_trial",
                                        diagnostic_only=False, production_trial_allowed=True)
    result["production_trial_candidate"] = {**candidate, "trial_edit": trial,
                                            "trial_authorization_id": grant_id}
    result.update(trial_authorization_id=grant_id, measurement_context_id=context,
                  same_prefix_followup_available=True)
    # A grant is ephemeral and exact. It cannot be reused after one generated token.
    worker._response_trial_grants = {grant_id: {"context_id": context, "edit": trial,
                                               "row": deepcopy(row)}}


def validate_apply(worker: Any, command: Any, packet: Mapping[str, Any]) -> dict[str, Any]:
    from .policy import PolicyViolation
    from .schema import Edit
    if is_dataclass(command):
        command = asdict(command)
    if command.get("decision") != "apply":
        return dict(command)
    edits = command.get("edits") or []
    meta = command.get("meta") or {}
    relevant = [e for e in edits if (e.get("meta", {}).get("apply_kind") or meta.get("apply_kind")) == "production_trial"]
    if not relevant:
        return dict(command)
    if callable(getattr(worker, "done", None)) and worker.done():
        raise PolicyViolation("no_next_token_for_trial")
    if len(edits) != 1:
        raise PolicyViolation("a response trial must contain exactly one approved edit")
    requested = relevant[0]
    grant_id = requested.get("meta", {}).get("trial_authorization_id") or meta.get("trial_authorization_id")
    grant = getattr(worker, "_response_trial_grants", {}).get(grant_id)
    if not grant or grant["context_id"] != state_identity(worker):
        raise PolicyViolation("missing_or_stale_trial_authorization")
    approved = grant["edit"]
    requested_edit, approved_edit = Edit.from_dict(requested), Edit.from_dict(approved)
    if any(getattr(requested_edit, k) != getattr(approved_edit, k) for k in ("target", "source", "op", "budget")):
        raise PolicyViolation("trial_edit_differs_from_authorized_measurement")
    try:
        normal_edit(worker, grant["row"], packet)
    except Exception as exc:
        raise PolicyViolation(str(exc)) from exc
    # Controller-provided accounting metadata must not change the budget pool.
    return {**command, "meta": {**meta, "apply_kind": "production_trial"}, "edits": [deepcopy(approved)]}


def consume(worker: Any, command: Any) -> None:
    if is_dataclass(command):
        command = asdict(command)
    for edit in command.get("edits", ()):
        grant_id = edit.get("meta", {}).get("trial_authorization_id")
        getattr(worker, "_response_trial_grants", {}).pop(grant_id, None)


def require_physical_confirmation(result: Mapping[str, Any]) -> dict[str, Any]:
    def close(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: False if key in {"production_trial_allowed", "production_trial_eligible"}
                    else close(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [close(child) for child in value]
        return value
    closed = close(result)
    closed.update(trial_review_checks_passed=True, production_trial_allowed=False,
        next_evidence_needed="same_context_physical_confirmation",
        why_not_apply="Review checks passed, but exact current-prefix physical confirmation is still required.",
        production_trial_blocked_reasons=[*result.get("production_trial_blocked_reasons", ()),
                                          "same_context_physical_confirmation_required"])
    return closed


def packet_hints(worker: Any) -> dict[str, Any]:
    from .diagnostic_budget import left
    if not getattr(worker, "_candidate_measurements", {}) and not getattr(worker, "_response_trial_grants", {}):
        return {}
    context = state_identity(worker)
    reports = []
    for (candidate_id, measured_context), measured in list(getattr(worker, "_candidate_measurements", {}).items())[-8:]:
        row = measured.get("promotion_evidence")
        if measured_context == context and isinstance(row, Mapping):
            reports.append(handoff(worker, row, context=context, budget_left=left(worker)))
    hints = {"candidate_trial_handoffs": reports[-4:]} if reports else {}
    for grant_id, grant in getattr(worker, "_response_trial_grants", {}).items():
        if grant["context_id"] == context:
            hints["candidate_trial_offer"] = {"trial_authorization_id": grant_id,
                "measurement_context_id": context, "trial_edit": deepcopy(grant["edit"]),
                "production_trial_allowed": True, "production_apply_allowed": False,
                "scope": "one_exact_edit_current_prefix_only"}
    return hints


def followup_available(worker: Any, results: list[dict[str, Any]]) -> bool:
    from .diagnostic_budget import left
    if callable(getattr(worker, "done", None)) and worker.done():
        return False
    if not any(r.get("same_prefix_followup_available") is True for r in results):
        return False
    context = state_identity(worker)
    for result in results:
        if result.get("trial_authorization_id"):
            grant = getattr(worker, "_response_trial_grants", {}).get(result["trial_authorization_id"])
            if grant and grant["context_id"] == context:
                return True
        reports = [result.get("candidate_trial_handoff") or {}, *result.get("candidate_trial_handoffs", ())]
        for report in reports:
            if report.get("same_prefix_followup_available") is True and report.get("measurement_context_id") == context and left(worker) > 0:
                return True
    return False
