"""Source-bound clause-token activation patches under unchanged runtime caps.

This instrument measures one-token effects, not clause correctness. It never
mints frozen apply candidates, trial handoffs, or production permission.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
from typing import Any

import torch

from . import clause_structure, diagnostic_budget
from .compiler import StepContext, compile_expr
from .policy import validate_command_against_packet
from .response_probe import bound_metrics, identity, logit_variation, state_identity, tensor_identity


NAME = "source_clause_activation_patch_probe"
ALPHA = 0.04
STEP_SIZE = 0.08
MASS_REPORT_THRESHOLD = 1e-4
SITES = ("resid_pre", "mlp_out")


def _surfaces(worker: Any) -> list[dict[str, Any]]:
    """Last available layer per site, never selected by historical rescue outcomes."""
    selected = {}
    cache = getattr(worker.runtime_state, "last_cache", None) or {}
    for surface in worker.surface_catalog:
        target, caps = surface.target, surface.caps
        if (target.kind != "activation" or target.site not in SITES
                or target.token.mode != "last" or "activation_patch" not in surface.allow_ops):
            continue
        values = (caps.max_alpha, caps.step_size, caps.norm_clip)
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(x) or x <= 0 for x in values) or caps.max_ttl_steps < 1:
            continue
        hook_name = worker.adapter.bind_surface(surface).hook_name
        tensor = cache.get(hook_name)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            continue
        row = {"surface_id": surface.surface_id, "site": target.site, "layer": target.layer,
               "cache_hook_name": hook_name,
               "caps": asdict(caps), "alpha": min(ALPHA, caps.max_alpha),
               "step_size": min(STEP_SIZE, caps.step_size), "norm_clip": min(1.0, caps.norm_clip)}
        prior = selected.get(target.site)
        if prior is None or (row["layer"], row["surface_id"]) > (prior["layer"], prior["surface_id"]):
            selected[target.site] = row
    return [selected[site] for site in SITES if site in selected]


def offer(worker: Any) -> dict[str, Any]:
    enabled = getattr(worker, "clause_structure_operator_diagnostics", False)
    structure = clause_structure.offer(worker) if enabled else {}
    context = structure.get("structure_context_id")
    receipt = next((r for r in reversed(getattr(worker, "_diagnostic_results", []))
                    if r.get("diagnostic") == clause_structure.NAME
                    and r.get("structure_context_id") == context and context
                    and r.get("status") == "ok" and r.get("state_restored") is True), None)
    surfaces = _surfaces(worker) if receipt else []
    status = "available"
    if not enabled:
        status = "disabled"
    elif structure.get("status") not in {"available", "already_attempted_at_context"}:
        status = structure.get("status", "structure_observation_unavailable")
    elif receipt is None:
        status = "current_prefix_structure_observation_required"
    elif not surfaces:
        status = "no_normal_cap_activation_surfaces_with_cache"
    elif diagnostic_budget.left(worker) <= 0:
        status = "diagnostic_budget_exhausted"
    candidates = []
    if status == "available":
        tokens = worker._current_token_tensor().flatten().detach().cpu().tolist()
        observed = {r["token_id"]: r for r in receipt.get("structure_readout_rows", [])}
        for target in structure["source_candidates"]:
            row = observed.get(target["token_id"])
            if not row or row.get("status") != "measured":
                continue
            # Keep occurrence order, not the occurrence with the best downstream effect.
            position = target["source_positions"][0]
            if (position < 0 or position >= len(tokens) or tokens[position] != target["token_id"]
                    or row.get("source_positions") != target["source_positions"]):
                continue
            if any(position >= worker.runtime_state.last_cache[s["cache_hook_name"]].shape[1] for s in surfaces):
                continue
            descriptor = {
                "structure_context_id": context, "target_token_id": target["token_id"],
                "target_piece": target["piece"], "source_token_index": position,
                "source_provenance": "source_body", "source_localization": "exact_source_token",
                "source_occurrence_rule": "first_source_occurrence", "surfaces": surfaces,
                "observation_id": identity("clause-observation:", receipt["structure_readout_rows"]),
            }
            candidate_id = identity("clause-patch:", descriptor)
            candidates.append({**descriptor, "structure_operator_id": candidate_id,
                "available": candidate_id not in getattr(worker, "_clause_patch_attempted_ids", set()),
                "physical_replays_upper_bound": 2 + 2 * len(surfaces),
                "request": {"diagnostic": NAME, "structure_operator_id": candidate_id}})
        if not candidates:
            status = "no_exact_source_binding"
        elif not any(row["available"] for row in candidates):
            status = "already_attempted_at_context"
    return {
        "status": status, "structure_context_id": context, "candidates": candidates,
        "surface_rule": "last_available_layer_per_resid_pre_and_mlp_out",
        "ordering": "source_occurrence_not_expected_utility",
        "evidence_scope": "exact_token_one_step_operator_effect_not_clause_rescue",
        "diagnostic_only": True, "production_apply_allowed": False,
    }


def materialize(worker: Any, candidate: Mapping[str, Any], surface: Mapping[str, Any]) -> dict[str, Any]:
    edit_id = identity("clause-edit:", [candidate["structure_operator_id"], surface])
    return {
        "id": edit_id, "target": {"surface_id": surface["surface_id"]},
        "source": {"dtype": "vector", "expr": {"ref": {
            "scope": "runtime", "worker": worker.worker_id, "tensor": surface["site"],
            "layer": surface["layer"], "token": {"mode": "index", "value": candidate["source_token_index"]},
        }}},
        "op": {"kind": "activation_patch", "mode": "blend", "alpha": surface["alpha"]},
        "budget": {"ttl_steps": 1, "step_size": surface["step_size"],
                   "norm_clip": surface["norm_clip"], "revertible": True},
        "meta": {"diagnostic_only": True, "production_apply_allowed": False,
                 "activation_patch_source_localization": "exact_source_token"},
    }


def summarize_pair(baselines: list[Mapping[str, Any]], edited: list[Mapping[str, Any]],
                   target: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep effect, ownership and safety distinct; tolerances are reporting only."""
    logits = [r["first_logits"] for r in [*baselines, *edited]]
    baseline_noise = logit_variation(logits[0], logits[1])
    edited_noise = logit_variation(logits[2], logits[3])
    deltas = [bound_metrics(baselines[i]["first_logits"], edited[i]["first_logits"],
                            target["target_token_id"]) for i in range(2)]
    controls = [bound_metrics(logits[i], logits[i + 1], target["target_token_id"]) for i in (0, 2)]
    floors = {key: max(abs(control[key]) for control in controls) for key in (
        "target_piece_prob_delta", "target_top20_threshold_gap_delta", "target_rank_delta",
        "bound_token_top20_hit_delta")}
    # Require agreement in both repeats, above observed repeat variation.
    mass_lift = min(d["target_piece_prob_delta"] for d in deltas) > max(
        MASS_REPORT_THRESHOLD, floors["target_piece_prob_delta"])
    hit_lift = min(d["bound_token_top20_hit_delta"] for d in deltas) > floors["bound_token_top20_hit_delta"]
    gap_lift = max(d["target_top20_threshold_gap_delta"] for d in deltas) < -max(
        1e-8, floors["target_top20_threshold_gap_delta"])
    rank_lift = min(d["target_rank_delta"] for d in deltas) > floors["target_rank_delta"]
    regression = max(d["target_piece_prob_delta"] for d in deltas) < -max(
        1e-12, floors["target_piece_prob_delta"])
    effect = ("bound_target_lift_candidate" if mass_lift or hit_lift else
              "bound_gap_movement" if gap_lift else "bound_rank_movement" if rank_lift else
              "bound_readout_regression" if regression else "no_resolved_target_response")
    structure_rows = []
    for candidate in candidates:
        tid = candidate["token_id"]
        rows_before = [clause_structure.readout_rows(b["first_logits"], [candidate])[0][0] for b in baselines]
        rows_after = [clause_structure.readout_rows(a["first_logits"], [candidate])[0][0] for a in edited]
        # Use the same CPU float64 probability arithmetic as bound_metrics.
        probs = [float(value.detach().cpu().flatten().double().softmax(0)[tid]) for value in logits]
        probability_deltas = [probs[i + 2] - probs[i] for i in range(2)]
        floor = max(abs(probs[1] - probs[0]), abs(probs[3] - probs[2]), 1e-12)
        structure_rows.append({**candidate, "before": rows_before, "after": rows_after,
            "probability_deltas": probability_deltas, "repeat_probability_floor": floor,
            "repeat_supported_positive": min(probability_deltas) > floor})
    positive = [r for r in structure_rows if r["repeat_supported_positive"]]
    winners = [max(positive, key=lambda r: r["probability_deltas"][i])["token_id"] for i in range(2)] if positive else []
    realized = winners[0] if len(set(winners)) == 1 else None
    intended = next(r for r in structure_rows if r["token_id"] == target["target_token_id"])
    others = [r for r in structure_rows if r["token_id"] != target["target_token_id"]]
    self_delta = min(intended["probability_deltas"])
    cross_delta = max((max(r["probability_deltas"]) for r in others), default=0.0)
    margin_floor = intended["repeat_probability_floor"] + max(
        (r["repeat_probability_floor"] for r in others), default=0.0)
    ownership = ("target_aligned" if realized == target["target_token_id"]
                 and self_delta - cross_delta > margin_floor else
                 "cross_bound" if realized is not None and realized != target["target_token_id"] else
                 "unresolved" if positive else "no_resolved_lift")
    safety = [{key + "_delta": float(edited[i][key]) - float(baselines[i][key])
               for key in ("entropy", "top1_margin", "repeat_flag", "repetition_score")} for i in range(2)]
    collapse_signal = any(
        (r["repeat_flag_delta"] > 0 or r["repetition_score_delta"] > 0)
        and r["entropy_delta"] < -0.02 and r["top1_margin_delta"] > 0.002 for r in safety)
    return {
        "paired_deltas": deltas, "baseline_repeat_variation": baseline_noise,
        "edited_repeat_variation": edited_noise, "metric_repeat_floors": floors,
        "effect_role": effect, "ownership_role": ownership, "structure_response_rows": structure_rows,
        "realized_lift_token_id": realized, "self_delta": self_delta, "cross_delta": cross_delta,
        "alignment_margin": self_delta - cross_delta, "alignment_repeat_floor": margin_floor,
        "ownership_scope": "observed_source_structure_token_set_not_semantic_roles",
        "safety_deltas": safety, "collapse_sharpening_signal": collapse_signal,
        "semantic_safety": "unassessed", "clause_correctness": "unassessed",
        "target_readout_effect_certified": False, "production_apply_allowed": False,
    }


def execute(worker: Any, request: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    offered = offer(worker)
    candidate = next((r for r in offered["candidates"]
                      if r["structure_operator_id"] == request.get("structure_operator_id")), None)
    result: dict[str, Any] = {
        "diagnostic": NAME, "source": source, "recorded_step": worker._steps,
        "structure_context_id": offered["structure_context_id"],
        "structure_operator_id": request.get("structure_operator_id"),
        "evidence_scope": offered["evidence_scope"], "diagnostic_only": True,
        "production_apply_allowed": False, "certified_for_apply": False, "policy_candidate_ready": False,
        "diagnostic_call_cost": 0, "new_measurement_count": 0, "physical_replay_count": 0,
        "edited_replay_count": 0, "operator_rows": [],
    }
    if candidate is None or not candidate["available"] or offered["status"] != "available":
        return {**result, "status": "unavailable", "blocked_reason": (
            "already_attempted_at_context" if candidate and not candidate["available"] else
            offered["status"] if offered["status"] != "available" else "unknown_or_stale_structure_operator_id")}
    before, step, caps = state_identity(worker), worker._steps, repr(worker.surface_catalog)
    result["candidate_binding"] = {k: v for k, v in candidate.items() if k not in {"request", "available"}}
    result["physical_replays_upper_bound"] = candidate["physical_replays_upper_bound"]
    preparations = []
    try:
        packet = worker.build_controller_packet()
        ctx = StepContext(packet=packet, runtime_state=worker.runtime_state,
                          adapter=worker.adapter, traces={}, stats={}, active_edits={})
        for surface in candidate["surfaces"]:
            edit = materialize(worker, candidate, surface)
            command = {"version": "0.1", "decision": "apply", "edits": [edit]}
            validate_command_against_packet(command, packet)
            tensor = compile_expr(edit["source"]["expr"])(ctx)
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite_source_activation")
            preparations.append((surface, edit, command, tensor_identity(tensor)))
    except Exception as exc:
        if not hasattr(worker, "_clause_patch_attempted_ids"):
            worker._clause_patch_attempted_ids = set()
        # Even a free, impossible preflight must not become a phantom offer.
        worker._clause_patch_attempted_ids.add(candidate["structure_operator_id"])
        result.update(status="unavailable", blocked_reason="normal_cap_preflight_failed", error=str(exc))
    else:
        if not hasattr(worker, "_clause_patch_attempted_ids"):
            worker._clause_patch_attempted_ids = set()
        worker._clause_patch_attempted_ids.add(candidate["structure_operator_id"])
        result["diagnostic_call_cost"] = 1

        def simulate(command: dict[str, Any] | None = None) -> dict[str, Any]:
            result["physical_replay_count"] += 1
            result["edited_replay_count"] += int(command is not None)
            kwargs = {"command": command} if command is not None else {}
            value = worker._simulate_decode(max_new_tokens=1, top_k=6,
                score_candidate_text=False, score_observer_check=False, **kwargs)
            if before != state_identity(worker) or worker._steps != step or caps != repr(worker.surface_catalog):
                raise RuntimeError("source_clause_activation_patch_probe_changed_worker_state")
            if not isinstance(value, Mapping) or not isinstance(value.get("first_logits"), torch.Tensor):
                raise ValueError(getattr(worker, "_last_simulate_decode_error", None) or "operator_replay_failed")
            return dict(value)

        try:
            baselines = [simulate(), simulate()]
            result["baseline_repeat_variation"] = logit_variation(
                baselines[0]["first_logits"], baselines[1]["first_logits"])
            for surface, edit, command, source_id in preparations:
                edited = [simulate(command), simulate(command)]
                telemetry = [[r for r in row["edit_runtime_telemetry"] if r.get("edit_id") == edit["id"]]
                             for row in edited]
                if any(len(rows) != 1 or rows[0].get("hook_call_count") != 1 for rows in telemetry):
                    raise ValueError("activation_patch_hook_execution_not_confirmed")
                summary = summarize_pair(baselines, edited, candidate, clause_structure.source_candidates(worker))
                result["operator_rows"].append({
                    "candidate_fingerprint": identity("clause-operator:", [candidate, edit, source_id, before]),
                    "context_fingerprint": before, "edit": edit, "surface": surface,
                    "source_tensor_identity": source_id, "runtime_telemetry": telemetry,
                    "activation_patch_op_kind": "activation_patch", "activation_patch_mode": "blend",
                    "activation_patch_source_localization": "exact_source_token",
                    **summary,
                })
                result["new_measurement_count"] += 1
            result["status"] = "ok"
        except Exception as exc:
            result.update(status="error", error=str(exc))
    result["state_restored"] = before == state_identity(worker) and worker._steps == step
    result["caps_unchanged"] = caps == repr(worker.surface_catalog)
    if not result["state_restored"] or not result["caps_unchanged"]:
        raise RuntimeError("source_clause_activation_patch_probe_changed_worker_state")
    return result
