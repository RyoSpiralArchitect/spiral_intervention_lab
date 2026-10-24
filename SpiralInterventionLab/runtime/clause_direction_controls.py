"""Offline fixed-permutation controls, not a new operator or semantic null.

Permutation preserves a displacement's coordinates and norm, but not its angle
to the current activation. One predeclared direction pair is not a null ensemble.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
from typing import Any, Sequence

import torch

from . import clause_actuation, clause_components, clause_specificity, clause_trajectory, diagnostic_budget
from .compiler import StepContext, compile_expr
from .response_probe import identity, state_identity, tensor_identity


CONTROLS = ("permuted_common_positive", "permuted_common_negative")
LABELS = (*clause_components.COMPONENTS, *CONTROLS)
PERMUTATION_SPEC = "sha256-index-sort:source-direction-control-v1"


def permutation(width: int) -> torch.Tensor:
    if isinstance(width, bool) or not isinstance(width, int) or width < 2:
        raise ValueError("permutation_requires_at_least_two_coordinates")
    order = sorted(range(width), key=lambda i: (
        hashlib.sha256(f"{PERMUTATION_SPEC}:{width}:{i}".encode()).digest(), i))
    return torch.tensor(order, dtype=torch.long, device="cpu")


def permuted_sources(common: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if (common.shape != target.shape or common.ndim != 1 or common.numel() < 2
            or common.dtype != target.dtype or common.device.type != "cpu" or target.device.type != "cpu"
            or not torch.isfinite(common).all() or not torch.isfinite(target).all()):
        raise ValueError("invalid_permutation_inputs")
    delta = common - target
    shuffled = delta[permutation(delta.numel())]
    if float(delta.double().norm()) == 0 or torch.equal(delta, shuffled):
        raise ValueError("degenerate_permuted_common_control")
    return target + shuffled, target - shuffled


def plan(token_ids: Sequence[int]) -> dict:
    base = clause_components.plan(token_ids)
    return {**base, "condition_order": [*base["condition_order"], *CONTROLS],
            "diagnostic_slots_for_conditions": 9, "permutation_spec": PERMUTATION_SPEC,
            "displacement_definitions": {**base["displacement_definitions"],
                "permuted_common_positive": "permute(rounded_common_source - T)",
                "permuted_common_negative": "-permute(rounded_common_source - T)"},
            "control_scope": "fixed_coordinate_permutation_not_semantic_null_or_random_ensemble",
            "preserves_angle_to_current_activation": False, "adaptive_resampling_allowed": False}


@contextmanager
def _temporary_trace(worker: Any, trace_id: str, cache: dict):
    state = worker.runtime_state
    if trace_id in state.trace_caches or trace_id in state.trace_sequences or trace_id in worker.trace_metadata:
        raise ValueError("direction_control_trace_collision")
    saved_packet, alignment = worker._last_packet, state.trace_alignment_step
    state.put_trace_cache(trace_id, cache)
    worker._last_packet = None
    try:
        yield
    finally:
        del state.trace_caches[trace_id]
        state.trace_alignment_step = alignment
        worker._last_packet = saved_packet


def measure(worker: Any, token_ids: Sequence[int], *, stages: Sequence[str] | None = None,
            record: Any = None) -> list[dict]:
    packet, alignment = worker._last_packet, worker.runtime_state.trace_alignment_step
    try:
        return _measure(worker, token_ids, stages=stages, record=record)
    finally:
        worker._last_packet = packet
        worker.runtime_state.trace_alignment_step = alignment


def _measure(worker: Any, token_ids: Sequence[int], *, stages: Sequence[str] | None = None,
             record: Any = None) -> list[dict]:
    plan(token_ids)
    if diagnostic_budget.left(worker) < 9:
        raise ValueError("insufficient_budget_for_nine_direction_conditions")
    offered = clause_actuation.offer(worker)
    candidates = [next((c for c in offered["candidates"] if c["target_token_id"] == tid and c["available"]), None)
                  for tid in token_ids]
    if offered["status"] != "available" or any(c is None for c in candidates):
        raise ValueError("requested_source_pair_not_currently_offered")
    if candidates[0]["surfaces"] != candidates[1]["surfaces"]:
        raise ValueError("direction_controls_require_identical_surfaces")
    tokens = worker._current_token_tensor().detach().clone()
    cached = worker.runtime_state.last_tokens
    if not isinstance(cached, torch.Tensor) or not torch.equal(cached, tokens):
        raise ValueError("direction_controls_require_aligned_cache")
    stages = list(stages) if stages is not None else clause_trajectory.stage_names(worker.runtime_state.model.cfg.n_layers)
    if not stages or any(s not in worker.runtime_state.last_cache for s in stages):
        raise ValueError("direction_control_stage_not_available")
    before, step, caps = state_identity(worker), worker._steps, repr(worker.surface_catalog)
    ctx = StepContext(packet={}, runtime_state=worker.runtime_state, adapter=worker.adapter,
                      traces={}, stats={}, active_edits={})
    prepared = {label: [] for label in CONTROLS}
    for surface in candidates[0]["surfaces"]:
        refs = [clause_actuation.materialize(worker, c, surface)["source"]["expr"] for c in candidates]
        current = deepcopy(refs[0])
        current["ref"]["token"] = {"mode": "last"}
        target = compile_expr(current)(ctx).detach().cpu().clone()
        common = compile_expr({"fn": "mean", "args": refs})(ctx).detach().cpu().clone()
        for label, source in zip(CONTROLS, permuted_sources(common, target), strict=True):
            prepared[label].append({"source": source, "target": target.clone()})
    records = clause_components.measure(worker, token_ids, stages=stages, record=record)
    for label in CONTROLS:
        inputs = prepared[label]
        trace_id = identity("offline-direction:", [before, label, [tensor_identity(v["source"]) for v in inputs]])
        cache = {s["cache_hook_name"]: v["source"].reshape(1, 1, -1)
                 for s, v in zip(candidates[0]["surfaces"], inputs, strict=True)}
        expressions = []
        for surface in candidates[0]["surfaces"]:
            expr = clause_actuation.materialize(worker, candidates[0], surface)["source"]["expr"]
            expr["ref"].update(scope="trace", trace_id=trace_id, token={"mode": "last"})
            expressions.append(expr)
        with _temporary_trace(worker, trace_id, cache):
            with clause_trajectory.capture_probe(worker, tokens, stages) as frames:
                result = clause_specificity.measure_expression_control(worker, label, candidates[0], expressions)
        result["temporary_trace_removed"] = trace_id not in worker.runtime_state.trace_caches
        item = {"label": label, "kind": "component", "result": result, "inputs": inputs, "frames": frames,
                "component_definition": {"permutation_spec": PERMUTATION_SPEC,
                    "source_expressions": expressions, "source_token_ids": list(token_ids),
                    "control_scope": plan(token_ids)["control_scope"]}}
        records.append(item)
        if record is not None:
            record(item)
        if result["status"] != "ok":
            raise ValueError(f"direction_control_failed:{label}:{result.get('error', result.get('blocked_reason'))}")
        if before != state_identity(worker) or worker._steps != step or repr(worker.surface_catalog) != caps:
            raise RuntimeError("direction_control_changed_worker_state")
    return records


def summarize(records: Sequence[dict]) -> dict:
    trajectory = clause_trajectory.summarize(records, component_labels=LABELS)
    token_ids = [r["result"]["candidate_binding"]["target_token_id"] for r in records[:2]]
    if [r["label"] for r in records] != plan(token_ids)["condition_order"]:
        raise ValueError("direction_control_order_differs_from_plan")
    base = clause_components.summarize(records[:7])
    result = {"status": "ok", "plan": plan(token_ids), "components": base, "trajectory": trajectory,
              "surfaces": [], "production_apply_allowed": False, "target_readout_effect_certified": False}
    for index, surface in enumerate(trajectory["surfaces"]):
        common = records[4]["inputs"][index]
        expected = permuted_sources(common["source"], common["target"])
        entry = {"surface": surface["surface"], "controls": []}
        for record, source in zip(records[7:], expected, strict=True):
            inputs, row = record["inputs"][index], record["result"]["operator_rows"][index]
            if (not torch.equal(inputs["target"], common["target"])
                    or tensor_identity(inputs["source"]) != tensor_identity(source)
                    or row["source_tensor_identity"] != tensor_identity(source)
                    or not record["result"].get("temporary_trace_removed")
                    or record["component_definition"]["permutation_spec"] != PERMUTATION_SPEC):
                raise ValueError("direction_control_source_identity_or_restoration_mismatch")
            hook = surface["surface"]["cache_hook_name"]
            delta = inputs["source"].double() - inputs["target"].double()
            common_delta = common["source"].double() - common["target"].double()
            common_written = clause_trajectory._deltas(records[4], index, hook)
            written = clause_trajectory._deltas(record, index, hook)
            logits = clause_trajectory._deltas(record, index, clause_trajectory.RAW_LOGITS)
            entry["controls"].append({"label": record["label"],
                "displacement_vs_common": clause_trajectory.geometry(delta, common_delta),
                "displacement_vs_current_activation": clause_trajectory.geometry(delta, common["target"]),
                "actual_written_delta_norms": [float(v.norm()) for v in written],
                "dose_matches_common_within_2pct": all(
                    abs(float(a.norm() - b.norm())) <= max(1e-6, 0.02 * float(b.norm()))
                    for a, b in zip(written, common_written, strict=True)),
                "first_minus_second_logit_contrast_deltas": [float(v[token_ids[0]] - v[token_ids[1]]) for v in logits],
                "safety_deltas": row["safety_deltas"],
                "collapse_sharpening_signal": row["collapse_sharpening_signal"],
                "production_apply_allowed": False})
        result["surfaces"].append(entry)
    return result
