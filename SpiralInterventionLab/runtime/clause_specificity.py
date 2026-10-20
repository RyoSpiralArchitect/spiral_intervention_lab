"""Offline-only controls for source-clause patches, with no new operation cards."""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from . import clause_actuation, clause_structure, diagnostic_budget
from .compiler import StepContext, compile_expr
from .policy import validate_command_against_packet
from .response_probe import identity, state_identity, tensor_identity


CONTROL_KINDS = ("self_copy", "radial_shrink")


def align_cache(worker: Any) -> dict[str, Any]:
    """An explicitly charged no-edit forward, before the comparison snapshot."""
    if worker._collect_active_edits() or diagnostic_budget.left(worker) <= 0:
        raise ValueError("cache alignment requires an unedited prefix and one diagnostic slot")
    tokens = worker._current_token_tensor().detach().clone()
    step, caps, before = worker._steps, repr(worker.surface_catalog), state_identity(worker)
    worker._diagnostic_calls_used = diagnostic_budget.used(worker) + 1
    worker.runtime_state.run_with_cache(tokens, return_type="logits")
    if (step != worker._steps or caps != repr(worker.surface_catalog)
            or not torch.equal(tokens, worker._current_token_tensor())):
        raise RuntimeError("cache_alignment_changed_generation_state")
    cached = getattr(worker.runtime_state, "last_tokens", None)
    if not isinstance(cached, torch.Tensor) or not torch.equal(tokens, cached):
        raise ValueError("cache_alignment_did_not_bind_current_tokens")
    return {"status": "ok", "kind": "current_prefix_cache_alignment",
            "physical_replay_count": 1, "diagnostic_call_cost": 1,
            "context_before": before, "context_after": state_identity(worker),
            "tokens_identity": tensor_identity(tokens), "generated_tokens": len(worker._output_token_ids()),
            "generation_unchanged": True, "caps_unchanged": True,
            "cache_refreshed_not_restored": True, "production_apply_allowed": False}


def measure_control(worker: Any, kind: str) -> dict[str, Any]:
    """Charge one slot for a bounded control; never enter the controller ledger."""
    if kind not in CONTROL_KINDS:
        raise ValueError("unknown source-specificity control")
    offered = clause_actuation.offer(worker)
    context = offered["structure_context_id"]
    control_id = identity("clause-control:", [context, kind, offered["candidates"][:1]])
    report = {
        "kind": kind, "control_id": control_id, "structure_context_id": context,
        "selection_origin": "offline_control_harness_not_controller_choice",
        "status": "unavailable", "physical_replay_count": 0, "edited_replay_count": 0,
        "diagnostic_call_cost": 0, "operator_rows": [], "diagnostic_only": True,
        "production_apply_allowed": False,
    }
    if offered["status"] not in {"available", "already_attempted_at_context"} or not offered["candidates"]:
        return {**report, "blocked_reason": offered["status"]}
    cached = getattr(worker.runtime_state, "last_tokens", None)
    if not isinstance(cached, torch.Tensor) or not torch.equal(cached, worker._current_token_tensor()):
        return {**report, "blocked_reason": "cache_not_aligned_to_current_prefix"}
    # Availability changes as source requests are consumed; it is not control identity.
    anchor = {k: v for k, v in offered["candidates"][0].items() if k not in {"available", "request"}}
    control_id = identity("clause-control:", [kind, anchor])
    report["control_id"] = control_id
    attempted = getattr(worker, "_clause_patch_attempted_ids", set())
    if control_id in attempted:
        return {**report, "blocked_reason": "already_attempted_at_context"}
    before, step, caps = state_identity(worker), worker._steps, repr(worker.surface_catalog)
    prepared = []
    try:
        packet = worker.build_controller_packet()
        ctx = StepContext(packet=packet, runtime_state=worker.runtime_state,
                          adapter=worker.adapter, traces={}, stats={}, active_edits={})
        for surface in anchor["surfaces"]:
            edit = clause_actuation.materialize(worker, anchor, surface)
            edit["id"] = identity("clause-control-edit:", [control_id, surface])
            ref = deepcopy(edit["source"]["expr"])
            ref["ref"]["token"] = {"mode": "last"}
            edit["source"]["expr"] = ref if kind == "self_copy" else {"fn": "scale", "by": 0.0, "arg": ref}
            edit["meta"]["activation_patch_source_localization"] = kind
            command = {"version": "0.1", "decision": "apply", "edits": [edit]}
            validate_command_against_packet(command, packet)
            value = compile_expr(edit["source"]["expr"])(ctx)
            if not torch.isfinite(value).all():
                raise ValueError("nonfinite_control_source")
            prepared.append((surface, edit, command, tensor_identity(value)))
    except Exception as exc:
        report.update(blocked_reason="normal_cap_preflight_failed", error=str(exc))
    else:
        worker._clause_patch_attempted_ids = attempted | {control_id}
        report["diagnostic_budget_before"] = diagnostic_budget.left(worker)
        worker._diagnostic_calls_used = diagnostic_budget.used(worker) + 1
        report["diagnostic_call_cost"] = 1
        report["physical_replays_upper_bound"] = 2 + 2 * len(prepared)

        def simulate(command=None):
            report["physical_replay_count"] += 1
            report["edited_replay_count"] += int(command is not None)
            kwargs = {"command": command} if command is not None else {}
            value = worker._simulate_decode(max_new_tokens=1, top_k=6,
                score_candidate_text=False, score_observer_check=False, **kwargs)
            if before != state_identity(worker) or step != worker._steps or caps != repr(worker.surface_catalog):
                raise RuntimeError("source_specificity_control_changed_worker_state")
            if not isinstance(value, dict) or not isinstance(value.get("first_logits"), torch.Tensor):
                raise ValueError("control_replay_failed")
            return value

        try:
            baselines = [simulate(), simulate()]
            for surface, edit, command, source_id in prepared:
                edited = [simulate(command), simulate(command)]
                telemetry = [[r for r in value["edit_runtime_telemetry"] if r.get("edit_id") == edit["id"]]
                             for value in edited]
                if any(len(t) != 1 or t[0].get("hook_call_count") != 1 for t in telemetry):
                    raise ValueError("control_hook_execution_not_confirmed")
                summary = clause_actuation.summarize_pair(
                    baselines, edited, anchor, clause_structure.source_candidates(worker))
                # Controls have no intended semantic/readout objective.
                summary = {k: summary[k] for k in (
                    "structure_response_rows", "baseline_repeat_variation", "edited_repeat_variation",
                    "safety_deltas", "collapse_sharpening_signal", "semantic_safety", "clause_correctness")}
                report["operator_rows"].append({
                    "control_kind": kind, "context_fingerprint": before,
                    "candidate_fingerprint": identity("clause-control-candidate:", [edit, source_id, before]),
                    "surface": surface, "edit": edit, "source_tensor_identity": source_id,
                    "runtime_telemetry": telemetry, "production_apply_allowed": False, **summary})
                if kind == "self_copy" and any(
                    t[0].get("source_target_delta_norm", float("inf")) > 1e-6
                    or t[0].get("blend_delta_norm", float("inf")) > 1e-6 for t in telemetry
                ):
                    raise ValueError("self_copy_is_not_null_control")
            report["status"] = "ok"
        except Exception as exc:
            report.update(status="error", error=str(exc))
        report["diagnostic_budget_after"] = diagnostic_budget.left(worker)
    report["state_restored"] = before == state_identity(worker) and step == worker._steps
    report["caps_unchanged"] = caps == repr(worker.surface_catalog)
    if not report["state_restored"] or not report["caps_unchanged"]:
        raise RuntimeError("source_specificity_control_changed_worker_state")
    return report


def compare(source_receipts: list[dict], controls: list[dict]) -> dict[str, Any]:
    """Continuous, within-context contrasts, not a significance or apply test."""
    result = {"status": "unavailable", "contrasts": [], "production_apply_allowed": False,
              "scope": "source_specificity_within_observed_token_set_not_semantic_rescue"}
    if (not source_receipts or {r["kind"] for r in controls} != set(CONTROL_KINDS)
            or len(controls) != len(CONTROL_KINDS)
            or any(r["result"]["status"] != "ok" for r in source_receipts)
            or any(r["status"] != "ok" for r in controls)):
        return {**result, "blocked_reason": "incomplete_source_or_control_measurements"}
    sources = [(r["result"]["candidate_binding"]["target_token_id"], row)
               for r in source_receipts for row in r["result"]["operator_rows"]]
    control_rows = [(r["kind"], row) for r in controls for row in r["operator_rows"]]
    if not sources or len({r["context_fingerprint"] for _, r in sources + control_rows}) != 1:
        return {**result, "blocked_reason": "mismatched_measurement_context"}
    null_rows = [row for kind, row in control_rows if kind == "self_copy"]
    if any(not row.get("runtime_telemetry") or any(
        len(repeat) != 1 or repeat[0].get("source_target_delta_norm", float("inf")) > 1e-6
        or repeat[0].get("blend_delta_norm", float("inf")) > 1e-6 for repeat in row["runtime_telemetry"]
    ) for row in null_rows):
        return {**result, "blocked_reason": "self_copy_is_not_null_control"}
    for target, row in sources:
        surface = row["surface"]
        peers = [(tid, r) for tid, r in sources if r["surface"] == surface and tid != target]
        matched_controls = {kind: r for kind, r in control_rows if r["surface"] == surface}
        if not peers or set(matched_controls) != set(CONTROL_KINDS):
            return {**result, "contrasts": [], "blocked_reason": "missing_matched_site_controls"}

        def response(value):
            return next(r for r in value["structure_response_rows"] if r["token_id"] == target)

        own = response(row)
        other = [response(r) for _, r in peers]
        radial, null = response(matched_controls["radial_shrink"]), response(matched_controls["self_copy"])
        norms = lambda r: [t["blend_delta_norm"] for repeat in r["runtime_telemetry"] for t in repeat]
        source_norms, radial_norms = norms(row), norms(matched_controls["radial_shrink"])
        norm_matched = all(abs(a - b) <= max(1e-6, 0.02 * max(a, b))
                           for a, b in zip(source_norms, radial_norms, strict=True))
        result["contrasts"].append({
            "target_token_id": target, "target_piece": own["piece"], "surface": surface,
            "own_source_probability_deltas": own["probability_deltas"],
            "best_other_source_probability_delta": max(max(r["probability_deltas"]) for r in other),
            "own_minus_best_other_source": min(own["probability_deltas"])
            - max(max(r["probability_deltas"]) for r in other),
            "radial_control_probability_deltas": radial["probability_deltas"],
            "self_copy_probability_deltas": null["probability_deltas"],
            "own_minus_radial_control": min(own["probability_deltas"]) - max(radial["probability_deltas"]),
            "repeat_probability_floor_max": max(r["repeat_probability_floor"]
                                                for r in [own, *other, radial, null]),
            "radial_step_norm_matched_within_2pct": norm_matched,
            "own_source_blend_norms": source_norms, "radial_control_blend_norms": radial_norms,
        })
    result["status"] = "ok"
    return result
