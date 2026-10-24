"""Offline common/difference interventions at a fixed source-clause boundary.

Average and half-difference refer to source-minus-target displacements, not a
discovered semantic subspace. Each direction has the same normal-cap blend;
separate clipping makes this a direction test, not an additive attribution.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import torch

from . import clause_actuation, clause_specificity, clause_trajectory, diagnostic_budget
from .compiler import StepContext, compile_expr
from .response_probe import state_identity, tensor_identity


COMPONENTS = ("common", "difference_positive", "difference_negative")


def expressions(left: dict, right: dict, current: dict) -> dict[str, dict]:
    """Blend subtracts T itself: difference sources must explicitly add T back."""
    half_difference = {"fn": "scale", "by": 0.5,
                       "arg": {"fn": "sub", "args": [left, right]}}
    return deepcopy({
        "common": {"fn": "mean", "args": [left, right]},
        "difference_positive": {"fn": "add", "args": [current, half_difference]},
        "difference_negative": {"fn": "sub", "args": [current, half_difference]},
    })


def plan(token_ids: Sequence[int]) -> dict[str, Any]:
    if (len(token_ids) != 2 or len(set(token_ids)) != 2
            or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in token_ids)):
        raise ValueError("two_distinct_exact_source_token_ids_required")
    return {
        "source_token_ids": list(token_ids),
        "condition_order": [f"source:{t}" for t in token_ids] + ["self_copy", "radial_shrink", *COMPONENTS],
        "displacement_definitions": {"left": "S_left - T", "right": "S_right - T",
                                     "common": "(left + right) / 2",
                                     "difference_positive": "(left - right) / 2",
                                     "difference_negative": "(right - left) / 2"},
        "dose_contract": "same_alpha_and_step_cap_report_actual_rounded_norms_no_rescaling",
        "decomposition_contract": "raw_displacements_not_orthogonal_or_semantic_subspaces",
        "clipped_effects_are_not_additive": True,
        "diagnostic_slots_for_conditions": 7,
        "controller_calls": 0, "production_apply_allowed": False,
    }


def measure(worker: Any, token_ids: Sequence[int], *, stages: Sequence[str] | None = None,
            record: Any = None) -> list[dict]:
    plan(token_ids)
    if diagnostic_budget.left(worker) < 7:
        raise ValueError("insufficient_budget_for_seven_component_conditions")
    offered = clause_actuation.offer(worker)
    candidates = [next((c for c in offered["candidates"] if c["target_token_id"] == t
                        and c["available"]), None) for t in token_ids]
    if offered["status"] != "available" or any(c is None for c in candidates):
        raise ValueError("requested_source_pair_not_currently_offered")
    tokens = worker._current_token_tensor().detach().clone()
    cached_tokens = getattr(worker.runtime_state, "last_tokens", None)
    if not isinstance(cached_tokens, torch.Tensor) or not torch.equal(tokens, cached_tokens):
        raise ValueError("components_require_aligned_cache")
    stages = list(stages) if stages is not None else clause_trajectory.stage_names(worker.runtime_state.model.cfg.n_layers)
    if not stages or any(s not in worker.runtime_state.last_cache for s in stages):
        raise ValueError("component_stage_not_available")
    if candidates[0]["surfaces"] != candidates[1]["surfaces"]:
        raise ValueError("component_sources_have_different_surfaces")
    prepared = {kind: {"expressions": [], "inputs": []} for kind in COMPONENTS}
    ctx = StepContext(packet={}, runtime_state=worker.runtime_state, adapter=worker.adapter,
                      traces={}, stats={}, active_edits={})
    for surface in candidates[0]["surfaces"]:
        refs = [clause_actuation.materialize(worker, c, surface)["source"]["expr"] for c in candidates]
        current = deepcopy(refs[0])
        current["ref"]["token"] = {"mode": "last"}
        target = compile_expr(current)(ctx).detach().cpu().clone()
        for kind, expr in expressions(*refs, current).items():
            value = compile_expr(expr)(ctx).detach().cpu().clone()
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError("nonfinite_or_misaligned_component")
            if float((value.double() - target.double()).norm()) == 0:
                raise ValueError(f"degenerate_component:{kind}")
            prepared[kind]["expressions"].append(expr)
            prepared[kind]["inputs"].append({"source": value, "target": target.clone()})
    before, step = state_identity(worker), worker._steps
    records = clause_trajectory.measure(worker, token_ids, stages=stages, record=record)
    for kind in COMPONENTS:
        with clause_trajectory.capture_probe(worker, tokens, stages) as frames:
            result = clause_specificity.measure_expression_control(
                worker, kind, candidates[0], prepared[kind]["expressions"])
        item = {"label": kind, "kind": "component", "result": result,
                "inputs": prepared[kind]["inputs"], "frames": frames,
                "component_definition": {"source_token_ids": list(token_ids),
                                         "source_expressions": prepared[kind]["expressions"]}}
        records.append(item)
        if record is not None:
            record(item)
        if result["status"] != "ok":
            raise ValueError(f"component_measurement_failed:{kind}:{result.get('error', result.get('blocked_reason'))}")
        if (before != state_identity(worker) or step != worker._steps or worker._collect_active_edits()):
            raise RuntimeError("component_measurement_changed_worker_state")
    return records


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trajectory = clause_trajectory.summarize(records, component_labels=COMPONENTS)
    sources = [r for r in records if r["kind"] == "source"]
    token_ids = [r["result"]["candidate_binding"]["target_token_id"] for r in sources]
    if [r["label"] for r in records] != plan(token_ids)["condition_order"]:
        raise ValueError("component_condition_order_differs_from_plan")
    result = {"status": "ok", "plan": plan(token_ids), "surfaces": [], "trajectory": trajectory,
              "semantic_safety": "unassessed", "clause_correctness": "unassessed",
              "target_readout_effect_certified": False, "production_apply_allowed": False}
    for index, surface_summary in enumerate(trajectory["surfaces"]):
        surface = surface_summary["surface"]
        hook = surface["cache_hook_name"]
        left, right = [r["inputs"][index] for r in sources]
        target = left["target"].double()
        da, db = left["source"].double() - target, right["source"].double() - target
        common, difference = (da + db) / 2, (da - db) / 2
        ideal = {"common": common, "difference_positive": difference, "difference_negative": -difference}
        entry = {"surface": surface, "ideal_component_geometry": clause_trajectory.geometry(common, difference),
                 "raw_reconstruction_error_norms": [float((da - common - difference).norm()),
                                                    float((db - common + difference).norm())],
                 "conditions": []}
        written = {}
        for record in records:
            row = record["result"]["operator_rows"][index]
            inputs = record["inputs"][index]
            if (not torch.equal(inputs["target"].double(), target)
                    or tensor_identity(inputs["source"]) != row["source_tensor_identity"]):
                raise ValueError("component_input_or_source_identity_mismatch")
            values = clause_trajectory._deltas(record, index, hook)
            written[record["label"]] = values
            logits = clause_trajectory._deltas(record, index, clause_trajectory.RAW_LOGITS)
            responses = row["structure_response_rows"]
            first, second = [next(r for r in responses if r["token_id"] == t) for t in token_ids]
            source_delta = inputs["source"].double() - target
            raw = (inputs["source"] - inputs["target"]) * surface["alpha"]
            raw_norm = float(raw.norm())
            frames = clause_trajectory.replay_frames(record)
            entry["conditions"].append({
                "label": record["label"], "kind": record["kind"],
                "source_displacement_norm": float(source_delta.norm()),
                "source_rounding_error_vs_ideal_component_norm": (
                    float((source_delta - ideal[record["label"]]).norm()) if record["kind"] == "component" else None),
                "pre_clip_alpha_displacement_norm": raw_norm,
                "step_clip_scale": min(1.0, surface["step_size"] / (raw_norm + 1e-8)),
                "actual_written_delta_norms": [float(v.norm()) for v in values],
                "step_size_clip_saturated": [t[0]["step_size_clip_saturated"] for t in row["runtime_telemetry"]],
                "first_minus_second_logit_contrast_deltas": [float(v[token_ids[0]] - v[token_ids[1]]) for v in logits],
                "first_minus_second_probability_deltas": [first["probability_deltas"][r] - second["probability_deltas"][r]
                                                          for r in range(2)],
                "raw_greedy_token_ids": [int(f["vectors"][clause_trajectory.RAW_LOGITS].argmax())
                                         for f in frames[2 + 2 * index:4 + 2 * index]],
                "readout_rows": [{"token_id": r["token_id"], "piece": r["piece"],
                                  "probability_deltas": r["probability_deltas"],
                                  "repeat_probability_floor": r["repeat_probability_floor"],
                                  "baseline_ranks": [v["rank"] for v in r["before"]],
                                  "edited_ranks": [v["rank"] for v in r["after"]],
                                  "raw_logit_deltas": [float(v[r["token_id"]]) for v in logits]}
                                 for r in responses],
                "safety_deltas": row["safety_deltas"],
                "collapse_sharpening_signal": row["collapse_sharpening_signal"],
                "production_apply_allowed": False,
            })
        nonnull_norms = [float(v.norm()) for label, values in written.items() if label != "self_copy" for v in values]
        entry["actual_dose_matches_cap_within_2pct"] = all(
            abs(n - surface["step_size"]) <= max(1e-6, 0.02 * surface["step_size"]) for n in nonnull_norms)
        entry["signed_difference_sum_written_norms"] = [float((a + b).norm()) for a, b in zip(
            written["difference_positive"], written["difference_negative"], strict=True)]
        entry["all_captured_null_repeat_and_baseline_floors_zero"] = all(
            row[key] == 0 for row in surface_summary["stages"] for key in (
                "baseline_drift_norm_max", "preparation_vs_baseline_norm_max",
                "delta_repeat_difference_norm_max", "self_copy_delta_norm_max"))
        result["surfaces"].append(entry)
    return result
