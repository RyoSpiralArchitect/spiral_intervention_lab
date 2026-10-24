"""Offline, read-only trajectory capture for normal-cap source-clause probes.

Compare actual post-hook displacement, not the cosine of large unedited states.
No controller cards, additional forwards, or apply certificates originate here.
"""
from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
from typing import Any, Iterator, Mapping, Sequence

import torch

from . import clause_actuation, clause_specificity, diagnostic_budget
from .response_probe import state_identity, tensor_identity


RAW_LOGITS = "raw_logits"


def stage_names(n_layers: int) -> list[str]:
    if isinstance(n_layers, bool) or not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError("a known positive layer count is required")
    return [f"blocks.{layer}.hook_{site}" for layer in range(n_layers)
            for site in ("resid_pre", "resid_mid", "mlp_out", "resid_post")] + ["ln_final.hook_normalized"]


@contextmanager
def _override(obj: Any, name: str, value: Any) -> Iterator[None]:
    had_override, prior = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had_override:
            setattr(obj, name, prior)
        else:
            delattr(obj, name)


@contextmanager
def count_forwards(runtime: Any) -> Iterator[dict[str, int]]:
    original = runtime.run_with_cache
    counts = {"attempted": 0, "completed": 0}

    def run(*args, **kwargs):
        counts["attempted"] += 1
        value = original(*args, **kwargs)
        counts["completed"] += 1
        return value

    with _override(runtime, "run_with_cache", run):
        yield counts


@contextmanager
def capture_forwards(runtime: Any, tokens: torch.Tensor, stages: Sequence[str],
                     *, context: Any = None) -> Iterator[list[dict]]:
    """Copy last-position vectors after existing hooks; restore the method exactly."""
    if not stages or len(set(stages)) != len(stages) or RAW_LOGITS in stages:
        raise ValueError("unique, nonempty activation stages are required")
    original = runtime.run_with_cache
    expected_tokens = tokens.detach().clone()
    records: list[dict] = []

    def run(model_input, **kwargs):
        origin = dict(context()) if context is not None else {"kind": "unattributed"}
        if not isinstance(model_input, torch.Tensor) or not torch.equal(model_input, expected_tokens):
            raise ValueError("trajectory_prefix_changed")
        output, cache = original(model_input, **kwargs)
        vectors = {}
        for name in stages:
            value = cache[name] if name in cache else None
            if (not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1
                    or value.shape[1] != expected_tokens.shape[1]):
                raise ValueError(f"trajectory_stage_missing_or_misaligned:{name}")
            vectors[name] = value[0, -1].detach().cpu().clone()
        if (not isinstance(output, torch.Tensor) or output.ndim != 3
                or output.shape[:2] != expected_tokens.shape):
            raise ValueError("trajectory_logits_missing_or_misaligned")
        vectors[RAW_LOGITS] = output[0, -1].detach().cpu().clone()
        if any(not torch.isfinite(value).all() for value in vectors.values()):
            raise ValueError("nonfinite_trajectory_vector")
        records.append({"tokens_identity": tensor_identity(model_input), "vectors": vectors,
                        "origin": origin, "output_device": str(output.device), "output_dtype": str(output.dtype)})
        return output, cache

    with _override(runtime, "run_with_cache", run):
        yield records


@contextmanager
def capture_probe(worker: Any, tokens: torch.Tensor, stages: Sequence[str]) -> Iterator[list[dict]]:
    """Separate packet canaries from the explicit one-token replay, without skipping either."""
    original_simulate, original_packet = worker._simulate_decode, worker.build_controller_packet
    origin = {"kind": "preparation"}
    replay_index = 0

    def simulate(**kwargs):
        nonlocal origin, replay_index
        previous = origin
        origin = {"kind": "replay", "replay_index": replay_index,
                  "edit_ids": [e["id"] for e in (kwargs.get("command") or {}).get("edits", [])]}
        replay_index += 1
        try:
            return original_simulate(**kwargs)
        finally:
            origin = previous

    def packet(*args, **kwargs):
        nonlocal origin
        previous = origin
        origin = {**origin, "kind": "packet_preparation"}
        try:
            return original_packet(*args, **kwargs)
        finally:
            origin = previous

    with _override(worker, "_simulate_decode", simulate), _override(worker, "build_controller_packet", packet):
        with capture_forwards(worker.runtime_state, tokens, stages, context=lambda: origin) as frames:
            yield frames


def replay_frames(record: Mapping[str, Any]) -> list[dict]:
    frames = [f for f in record["frames"] if f.get("origin", {}).get("kind") == "replay"]
    if [f["origin"]["replay_index"] for f in frames] != list(range(len(frames))):
        raise ValueError("ambiguous_replay_capture_sequence")
    return frames


def geometry(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    a, b = a.detach().cpu().flatten().double(), b.detach().cpu().flatten().double()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("incompatible_or_nonfinite_geometry_vectors")
    na, nb = float(a.norm()), float(b.norm())
    distance = float((a - b).norm())
    return {"left_norm": na, "right_norm": nb, "distance": distance,
            "relative_separation": distance / max(na, nb) if max(na, nb) else 0.0,
            "cosine": float(torch.clamp(torch.dot(a, b) / (na * nb), -1, 1)) if na and nb else None}


def _deltas(record: Mapping[str, Any], index: int, stage: str) -> list[torch.Tensor]:
    frames = replay_frames(record)
    return [frames[2 + index * 2 + r]["vectors"][stage].double()
            - frames[r]["vectors"][stage].double() for r in range(2)]


def summarize(records: Sequence[Mapping[str, Any]], *, component_labels: Sequence[str] = ()) -> dict[str, Any]:
    """Continuous geometry and repeat/null floors, not a mediation or rescue test."""
    expected_kinds = {"source", "self_copy", "radial_shrink"} | ({"component"} if component_labels else set())
    if (len(records) != 4 + len(component_labels) or len({r["label"] for r in records}) != len(records)
            or len(set(component_labels)) != len(component_labels)
            or sum(r["kind"] == "source" for r in records) != 2
            or {r["label"] for r in records if r["kind"] == "component"} != set(component_labels)
            or {r["kind"] for r in records} != expected_kinds):
        raise ValueError("two_sources_and_two_distinct_controls_required")
    if any(r["result"].get("status") != "ok" or not r["result"].get("state_restored")
           or not r["result"].get("caps_unchanged") for r in records):
        raise ValueError("incomplete_or_unrestored_trajectory")
    reference_rows = records[0]["result"]["operator_rows"]
    surfaces = [r["surface"] for r in reference_rows]
    if not surfaces or any([r["surface"] for r in record["result"]["operator_rows"]] != surfaces
                           for record in records):
        raise ValueError("trajectory_surface_mismatch")
    if len({r["context_fingerprint"] for record in records for r in record["result"]["operator_rows"]}) != 1:
        raise ValueError("trajectory_context_mismatch")
    frames = [frame for record in records for frame in record["frames"]]
    if (any(len(replay_frames(r)) != 2 + 2 * len(surfaces) for r in records)
            or len({str(f["tokens_identity"]) for f in frames}) != 1):
        raise ValueError("trajectory_forward_count_or_tokens_mismatch")
    for record in records:
        expected_ids = [[], []] + [[row["edit"]["id"]] for row in record["result"]["operator_rows"] for _ in range(2)]
        if [f["origin"].get("edit_ids") for f in replay_frames(record)] != expected_ids:
            raise ValueError("trajectory_edit_identity_mismatch")
    stages = list(frames[0]["vectors"])
    if any(list(frame["vectors"]) != stages for frame in frames):
        raise ValueError("trajectory_stages_mismatch")
    null = next(r for r in records if r["kind"] == "self_copy")
    source_records = [r for r in records if r["kind"] == "source"]
    result = {"status": "ok", "surfaces": [], "production_apply_allowed": False,
              "captured_forward_count": len(frames),
              "captured_replay_forward_count": sum(len(replay_frames(r)) for r in records),
              "captured_preparation_forward_count": sum(len(r["frames"]) - len(replay_frames(r)) for r in records),
              "scope": "last_position_displacement_geometry_not_causal_mediation_or_clause_rescue",
              "probability_contract": "existing_receipt_post_constraint_cpu_float64",
              "raw_logit_contract": "pre_token_constraints_and_decoder_control"}
    for index, surface in enumerate(surfaces):
        hook = surface["cache_hook_name"]
        if hook not in stages:
            raise ValueError("intervention_stage_not_captured")
        capture_checks = []
        for record in records:
            telemetry = record["result"]["operator_rows"][index]["runtime_telemetry"]
            actual = _deltas(record, index, hook)
            source, target = record["inputs"][index]["source"], record["inputs"][index]["target"]
            # Match the adapter's arithmetic and casting, not an idealized fp64 blend.
            raw = (source - target) * surface["alpha"]
            norm = float(raw.norm())
            clipped = raw * (surface["step_size"] / (norm + 1e-8)) if norm > surface["step_size"] else raw
            expected = (target + clipped).double() - target.double()
            residuals = [float((value - expected).norm()) for value in actual]
            tolerance = max(1e-6, 0.005 * surface["step_size"])
            if (any(len(t) != 1 or t[0].get("hook_call_count") != 1 for t in telemetry)
                    or max(residuals) > tolerance):
                raise ValueError("cache_is_not_verified_post_patch_activation")
            capture_checks.append({"label": record["label"], "expected_vs_captured_residual_norms": residuals,
                                   "verification_tolerance": tolerance})
        inputs = [r["inputs"][index] for r in source_records]
        target = inputs[0]["target"].double()
        if not torch.equal(inputs[0]["target"], inputs[1]["target"]):
            raise ValueError("source_input_targets_differ")
        entry = {"surface": surface, "capture_checks": capture_checks,
                 "source_geometry": geometry(inputs[0]["source"], inputs[1]["source"]),
                 "source_minus_target_geometry": geometry(inputs[0]["source"].double() - target,
                                                           inputs[1]["source"].double() - target),
                 "stages": []}
        for stage in stages:
            changes = {r["label"]: _deltas(r, index, stage) for r in records}
            all_baselines = [f["vectors"][stage].double() for r in records for f in replay_frames(r)[:2]]
            preparations = [f["vectors"][stage].double() for r in records for f in r["frames"]
                            if f.get("origin", {}).get("kind") != "replay"]
            baseline_drift = max(float((v - all_baselines[0]).norm()) for v in all_baselines)
            repeat_noise = max(float((values[1] - values[0]).norm()) for values in changes.values())
            row = {"stage": stage, "baseline_drift_norm_max": baseline_drift,
                   "preparation_vs_baseline_norm_max": max((float((v - all_baselines[0]).norm())
                                                            for v in preparations), default=0.0),
                   "delta_repeat_difference_norm_max": repeat_noise,
                   "self_copy_delta_norm_max": max(float(v.norm()) for v in changes[null["label"]]),
                   "delta_norms": {k: [float(v.norm()) for v in values] for k, values in changes.items()},
                   "pairwise": []}
            for a, b in combinations(records, 2):
                pair = {"left": a["label"], "right": b["label"],
                        "repeats": [geometry(changes[a["label"]][r], changes[b["label"]][r]) for r in range(2)]}
                if stage == RAW_LOGITS:
                    pair["centered_repeats"] = [geometry(
                        changes[a["label"]][r] - changes[a["label"]][r].mean(),
                        changes[b["label"]][r] - changes[b["label"]][r].mean()) for r in range(2)]
                row["pairwise"].append(pair)
            entry["stages"].append(row)
        result["surfaces"].append(entry)
    return result


def measure(worker: Any, token_ids: Sequence[int], *, stages: Sequence[str] | None = None,
            record: Any = None) -> list[dict[str, Any]]:
    """Predeclared two-token contrast plus two controls, reusing existing probes."""
    if (len(token_ids) != 2 or len(set(token_ids)) != 2
            or any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in token_ids)):
        raise ValueError("two_distinct_exact_source_token_ids_required")
    offered = clause_actuation.offer(worker)
    candidates = [next((c for c in offered["candidates"] if c["target_token_id"] == tid
                        and c["available"]), None) for tid in token_ids]
    if offered["status"] != "available" or any(c is None for c in candidates):
        raise ValueError("requested_source_pair_not_currently_offered")
    if diagnostic_budget.left(worker) < 4:
        raise ValueError("insufficient_budget_for_source_pair_and_controls")
    tokens = worker._current_token_tensor().detach().clone()
    cached_tokens = getattr(worker.runtime_state, "last_tokens", None)
    if not isinstance(cached_tokens, torch.Tensor) or not torch.equal(tokens, cached_tokens):
        raise ValueError("trajectory_requires_aligned_cache")
    stages = list(stages) if stages is not None else stage_names(worker.runtime_state.model.cfg.n_layers)
    if any(name not in worker.runtime_state.last_cache for name in stages):
        raise ValueError("trajectory_required_stage_not_available")
    records = []
    before, step = state_identity(worker), worker._steps
    for candidate, kind in [(candidates[0], "source"), (candidates[1], "source"),
                            (candidates[0], "self_copy"), (candidates[0], "radial_shrink")]:
        inputs = []
        for surface in candidate["surfaces"]:
            cache = worker.runtime_state.last_cache[surface["cache_hook_name"]]
            target = cache[0, -1].detach().cpu().clone()
            source = (cache[0, candidate["source_token_index"]].detach().cpu().clone() if kind == "source"
                      else target.clone() if kind == "self_copy" else target * 0.0)
            inputs.append({"source": source, "target": target})
        label = f"source:{candidate['target_token_id']}" if kind == "source" else kind
        with capture_probe(worker, tokens, stages) as frames:
            result = (worker.request_controller_diagnostics(candidate["request"], source="offline_trajectory_harness")[0]
                      if kind == "source" else clause_specificity.measure_control(worker, kind))
        item = {"label": label, "kind": kind, "result": result, "inputs": inputs, "frames": frames}
        records.append(item)
        if record is not None:
            record(item)
        if result.get("status") != "ok":
            raise ValueError(f"trajectory_measurement_failed:{label}:{result.get('error', result.get('blocked_reason'))}")
        if before != state_identity(worker) or step != worker._steps or worker._collect_active_edits():
            raise RuntimeError("trajectory_measurement_changed_worker_state")
    return records


def compact_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"label": record["label"], "kind": record["kind"], "result": record["result"],
            **({"component_definition": record["component_definition"]} if "component_definition" in record else {}),
            "inputs": [{k: tensor_identity(v) for k, v in row.items()} for row in record["inputs"]],
            "frames": [{"tokens_identity": f["tokens_identity"], "origin": f.get("origin"),
                        "output_device": f.get("output_device"), "output_dtype": f.get("output_dtype"),
                        "vectors": {name: tensor_identity(v) for name, v in f["vectors"].items()}}
                       for f in record["frames"]]}
