"""Frozen-source, single-token edits across prespecified no-apply prefixes.

Offline diagnostic helpers only. A continuation does not keep the edit active:
the first-token simulation rolls back its hook before the unedited suffix.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import gc
import re
from typing import Any

import torch

from .compiler import StepContext, compile_expr
from .response_probe import bound_metrics, identity, logit_variation, state_identity, tensor_identity


BOUNDARY = re.compile(r"(?:\b(?:and|but|while|before|after|then)|[,;:])\s*$", re.IGNORECASE)


def prefix_plan(observations: Sequence[Mapping[str, Any]], anchor_step: int) -> list[dict[str, Any]]:
    """Anchor plus before/after the first later connective; never rank by effect."""
    by_step = {int(r["step"]): r for r in observations}
    if anchor_step not in by_step or len(by_step) != len(observations):
        raise ValueError("missing_anchor_or_duplicate_observation_step")
    selected = {anchor_step: "recorded_candidate_anchor"}
    for step in sorted(by_step):
        if step <= anchor_step:
            continue
        text = str(by_step[step].get("generated_tail") or "")
        if BOUNDARY.search(text):
            previous = by_step.get(step - 1)
            if previous is None or not text.startswith(str(previous["generated_tail"])):
                raise ValueError("noncontiguous_boundary_observations")
            if step - 1 > anchor_step:
                selected[step - 1] = "before_first_later_connective"
            selected[step] = "after_first_later_connective"
            break
    if len(selected) < 2:
        raise ValueError("no_later_connective_do_not_invent_a_prefix")
    return [{"worker_step": step, "prefix": by_step[step]["generated_tail"], "selection_reason": reason}
            for step, reason in sorted(selected.items())]


@dataclass(frozen=True)
class FrozenCandidate:
    candidate_id: str
    edit: dict[str, Any]
    descriptor: dict[str, Any]
    source_tensor: torch.Tensor
    source_identity: dict[str, Any]
    original_source_expr: dict[str, Any]
    trace_id: str
    hook_name: str
    objective: str
    term: str
    token_id: int
    token_piece: str


def freeze_candidate(worker: Any, row: Mapping[str, Any]) -> FrozenCandidate:
    """Materialize one recorded recipe at its anchor, then bind its raw source."""
    if worker._collect_active_edits():
        raise ValueError("active_edits_not_supported")
    candidate = {
        "objective_bundle_key": row["objective_bundle_key"],
        "actuator_bundle_key": row["objective_bundle_key"],
        "objective_term": row["intended_term"],
        "site": row["activation_patch_site"], "layer": row["activation_patch_layer"],
        "alpha": row["activation_patch_alpha"], "step_size": row["activation_patch_step_size"],
        "source_localization": row["activation_patch_source_localization"], "patch_mode": "blend",
        "recipe_name": row["recipe_name"], "operator_recipe_id": row["operator_recipe_id"],
        "contrast_mode": row.get("activation_patch_contrast_mode"),
        "contrast_scale": row.get("activation_patch_contrast_scale"),
        "stealer_term": row.get("activation_patch_stealer_term"),
        "stealer_bundle_key": row.get("activation_patch_stealer_bundle_key"),
    }
    if candidate["source_localization"] != "source_term_token" or candidate.get("contrast_mode") not in (None, "none", ""):
        raise ValueError("v1_requires_recorded_source_term_token_without_contrast")
    if candidate["step_size"] not in (0.04, 0.16) or isinstance(candidate["step_size"], bool):
        raise ValueError("unsupported_recorded_dose")
    edit = worker._activation_patch_trial_edit_from_candidate(candidate, trial_contract={
        "max_alpha": max(0.08, float(candidate["alpha"])), "norm_clip": 1.0,
        "trial_budget_class": "diagnostic_only", "allow_step_size_cap_release": True,
        "max_step_size": candidate["step_size"], "production_trial_followup_allowed": False,
    })
    if not edit or edit["op"] != {"kind": "activation_patch", "mode": "blend", "alpha": candidate["alpha"]}:
        raise ValueError("recorded_operator_materialization_changed")
    if edit["budget"]["step_size"] != candidate["step_size"] or edit["budget"]["ttl_steps"] != 1:
        raise ValueError("recorded_budget_materialization_changed")
    packet = worker.build_controller_packet()
    ctx = StepContext(packet=packet, runtime_state=worker.runtime_state, adapter=worker.adapter,
                      traces={}, stats={}, active_edits={})
    source = compile_expr(edit["source"]["expr"])(ctx).detach().clone()
    source_id = tensor_identity(source)
    if not torch.isfinite(source).all() or source_id != row["source_tensor_identity"]:
        raise ValueError("recorded_source_tensor_mismatch")
    token_id = row["target_piece_token_id"]
    if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
        raise ValueError("invalid_recorded_binding")
    piece = worker.codec.decode([token_id])
    if piece != row["target_piece"]:
        raise ValueError("recorded_tokenizer_binding_mismatch")
    candidate_id = identity("frozen_candidate:", [edit["target"], edit["op"], edit["budget"], source_id,
                                                 candidate["objective_term"], token_id])
    trace_id = candidate_id + ":source"
    original = deepcopy(edit["source"]["expr"])
    edit = deepcopy(edit)
    edit["source"] = {"dtype": "vector", "expr": {"ref": {
        "scope": "trace", "trace_id": trace_id, "tensor": candidate["site"], "layer": candidate["layer"],
        "token": {"mode": "last"},
    }}}
    edit.update(id="prefix_probe", bundle_key=candidate["objective_bundle_key"],
                focus_feature=candidate["objective_term"], phase_objective="readout_escape")
    return FrozenCandidate(candidate_id, edit, candidate, source, source_id, original, trace_id,
                           worker._cache_hook_name(layer=candidate["layer"], site=candidate["site"]),
                           candidate["objective_bundle_key"], candidate["objective_term"], token_id, piece)


def _release_probe_buffers(worker: Any) -> None:
    gc.collect()
    model = getattr(worker.runtime_state, "model", None)
    if str(getattr(getattr(model, "cfg", None), "device", "cpu")) == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()


def _one_token_then_unedited(worker: Any, *, horizon: int, edit: Mapping[str, Any] | None) -> dict[str, Any]:
    command = {"version": "0.1", "decision": "apply", "edits": [deepcopy(edit)]} if edit else None
    first = worker._simulate_decode(max_new_tokens=1, top_k=6, command=command,
        policy_override=worker._replay_policy(max_edits_per_step_override=1) if edit else None,
        score_candidate_text=False, score_observer_check=False)
    if first is None:
        raise ValueError(f"first_token_failed:{worker._last_simulate_decode_error}")
    ids = list(first["continuation_token_ids"])
    if len(ids) != 1 or worker._collect_active_edits():
        raise ValueError("first_token_or_hook_rollback_contract_failed")
    segments = deepcopy(worker._segments)
    try:
        worker._append_output_token(ids[0])
        _release_probe_buffers(worker)
        # Independent one-token calls avoid retaining two long-sequence cache
        # generations at once. No dose, decode rule, or horizon is changed.
        for _ in range(horizon - 1):
            if ids[-1] in worker.stop_token_ids or (worker.stop_checker and worker.stop_checker(worker.final_text())):
                break
            rest = worker._simulate_decode(max_new_tokens=1, top_k=6,
                score_candidate_text=False, score_observer_check=False)
            if rest is None or len(rest["continuation_token_ids"]) != 1:
                raise ValueError(f"unedited_suffix_failed:{worker._last_simulate_decode_error}")
            ids.extend(rest["continuation_token_ids"])
            worker._append_output_token(ids[-1])
            del rest
            _release_probe_buffers(worker)
    finally:
        worker._segments = segments
    return {**first, "continuation_token_ids": ids, "continuation": worker.codec.decode(ids)}


def measure_prefix(worker: Any, frozen: FrozenCandidate, *, horizon: int = 8,
                   completion: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 16:
        raise ValueError("horizon_must_be_1_to_16")
    if worker._collect_active_edits() or worker.done():
        raise ValueError("active_or_terminal_prefix_not_supported")
    if tensor_identity(frozen.source_tensor) != frozen.source_identity:
        raise ValueError("frozen_source_mutated")
    context = state_identity(worker)
    prefix = worker.final_text()
    horizon = min(horizon, worker.max_generated_tokens - worker._steps)
    state = worker.runtime_state
    if frozen.trace_id in state.trace_caches or frozen.trace_id in state.trace_sequences:
        raise ValueError("frozen_trace_id_collision")
    saved_packet, saved_alignment = worker._last_packet, state.trace_alignment_step
    state.put_trace_cache(frozen.trace_id, {frozen.hook_name: frozen.source_tensor.reshape(1, 1, -1)})
    worker._last_packet = None
    report = {"status": "incomplete", "candidate_id": frozen.candidate_id,
              "measurement_context_id": context, "prefix": prefix, "worker_step": worker._steps,
              "source_tensor_identity": frozen.source_identity, "target_piece": frozen.token_piece,
              "target_piece_token_id": frozen.token_id, "objective_term": frozen.term,
              "horizon": horizon, "physical_replay_count": 0, "model_forward_token_count": 0,
              "completed_replay_count": 0,
              "edit_active_for_tokens": 1, "diagnostic_only": True,
              "production_apply_allowed": False, "certified_for_apply": False}
    try:
        trials = []
        for edit in (None, None, frozen.edit, frozen.edit):
            report["physical_replay_count"] += 1
            trial = _one_token_then_unedited(worker, horizon=horizon, edit=edit)
            trials.append(trial)
            report["completed_replay_count"] += 1
            report["model_forward_token_count"] += len(trial["continuation_token_ids"])
        baseline, null, edited, repeat = trials
        no_edit_control = logit_variation(baseline["first_logits"], null["first_logits"])
        repeat_control = logit_variation(edited["first_logits"], repeat["first_logits"])
        if baseline["continuation_token_ids"] != null["continuation_token_ids"] or edited["continuation_token_ids"] != repeat["continuation_token_ids"]:
            raise ValueError("continuation_repeat_drift")
        metrics = bound_metrics(baseline["first_logits"], edited["first_logits"], frozen.token_id)
        for label, trial in (("before", baseline), ("after", edited)):
            logits = trial["first_logits"].detach().cpu().flatten().double()
            metrics[f"target_piece_logit_{label}"] = float(logits[frozen.token_id])
            metrics[f"target_piece_prob_{label}"] = float(logits.softmax(0)[frozen.token_id])
            metrics[f"target_top20_threshold_gap_{label}"] = float(logits.topk(min(20, logits.numel())).values[-1] - logits[frozen.token_id])
        telemetry = edited.get("edit_runtime_telemetry") or []
        hooks = [r for r in telemetry if r.get("op") == "activation_patch"]
        if len(hooks) != 1 or hooks[0].get("hook_call_count") != 1:
            raise ValueError("activation_hook_not_called_exactly_once")
        def outcome(trial):
            text = prefix + trial["continuation"]
            return {"text": text, "continuation": trial["continuation"],
                    "token_ids": trial["continuation_token_ids"],
                    "whole_term_present": bool(re.search(r"(?<!\w)" + re.escape(frozen.term) + r"(?!\w)", text, re.IGNORECASE)),
                    "stopped": bool(trial["continuation_token_ids"][-1] in worker.stop_token_ids
                                    or (worker.stop_checker and worker.stop_checker(text))),
                    "semantic_faithfulness_certified": False}
        report.update(status="complete", baseline=outcome(baseline), edited=outcome(edited),
                      no_edit_control=no_edit_control, repeat_control=repeat_control,
                      continuation_identical=baseline["continuation_token_ids"] == edited["continuation_token_ids"],
                      metrics=metrics, edit_runtime_telemetry=telemetry)
        if completion is not None:
            from .completion_probe import measure_completion
            extra = measure_completion(worker, frozen, baseline["first_logits"], edited["first_logits"],
                                       completion, horizon=horizon)
            report.update(primary_natural_probe_status="complete", completion_probe=extra,
                total_physical_replay_count=report["physical_replay_count"] + extra["physical_replay_count"],
                total_model_forward_token_count=report["model_forward_token_count"] + extra["model_forward_token_count"])
            if extra["status"] != "complete":
                raise ValueError(f"completion_probe_incomplete:{extra.get('error')}")
    except Exception as exc:
        report.update(status="incomplete", error=f"{type(exc).__name__}:{exc}",
                      model_forward_token_count_is_lower_bound=True)
    finally:
        del state.trace_caches[frozen.trace_id]
        state.trace_alignment_step = saved_alignment
        worker._last_packet = saved_packet
    report["state_restored"] = state_identity(worker) == context
    if not report["state_restored"]:
        report.update(status="incomplete", error="state_restoration_failed")
    return report
