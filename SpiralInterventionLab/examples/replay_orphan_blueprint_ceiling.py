"""Diagnostic-only ceiling replay for entity blueprints a live run never measured.

Rebuilds the recorded no-edit prefix and asks three separate questions:

1. orphan replay: the recorded blueprints, replayed through the runtime's own
   materialization paths with an unconstrained diagnostic budget;
2. insertion ceiling: forcing the objective's first piece at each prefix bounds
   what an insertion could buy at task level, independent of any operator;
3. actuator reach: the existing bounded operator families, swept at each prefix,
   bound whether any of them makes that first piece the greedy choice.

Every edit is a quarantined replay with state restored. No controller call,
rollout edit, production trial or apply permission is involved.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ..runtime.worker import _DEFAULT_BAD_ATTRACTOR_TERMS, _TokenSegment
from .digit_transform_e2e import (
    build_hooked_transformer_worker_runtime,
    create_readout_analyzer,
    create_task_env,
    load_worker_model,
)

READOUT_SPECS = (
    ("entity_target_readout_l005_a040", "entity_target_readout", "target_readout_minus_attractor", 0.05, 0.04),
    ("entity_target_readout_l005_a080", "entity_target_readout", "target_readout_minus_attractor", 0.05, 0.08),
    ("target_readout_pure_a080", "target_readout", "target_readout_pure", 0.0, 0.08),
)
PATCH_LOCALIZATIONS = ("source_term_token", "source_centered_pm1", "source_span_mean")
PATCH_ALPHAS = (0.04, 0.08)
TRIAL_CONTRACT = {
    "max_alpha": 0.08, "norm_clip": 1.0, "trial_budget_class": "diagnostic_only",
    "production_trial_followup_allowed": False, "promotion_reason": "orphan_blueprint_ceiling_replay",
}
PARTS = ("orphan", "ceiling", "sweep")
ORPHAN_ROW_KEYS = (
    "intended_term", "recipe_name", "operator_recipe_id", "diagnostic_family", "status", "actual_delta_class",
    "actuator_class", "target_piece", "target_rank_after", "target_piece_logit_delta", "target_mass_delta",
    "target_top20_hit_delta", "target_top20_threshold_gap_delta", "focus_rank_delta", "realized_lift_bundle_key",
    "replay_error", "simulate_error", "blocked_by", "production_apply_allowed",
)


def _release_memory() -> None:
    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_source(path: Path) -> dict[str, Any]:
    """Read one recorded episode: prompt, per-length prefixes, final output and diagnostics."""
    prompt: str | None = None
    prefixes: dict[int, str] = {}
    results: list[tuple[int, dict[str, Any]]] = []
    final_output: str | None = None
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            event = json.loads(line)
            name = event.get("event")
            if name == "episode_start":
                if prompt is not None:
                    raise ValueError("source JSONL must contain one episode")
                prompt = str(event["prompt"])
            elif name == "controller_observation" and isinstance(event.get("step"), int):
                prefixes.setdefault(int(event["step"]), str(event.get("generated_tail") or ""))
            elif name == "controller_diagnostic_result":
                results.append((line_no, dict(event)))
            elif name == "episode_end":
                final_output = str(event.get("output") or "")
    if prompt is None or final_output is None:
        raise ValueError("source episode is incomplete")
    reviews = [(no, row) for no, row in results
               if row.get("diagnostic") == "entity_insertion_operator_candidate_review"
               and row.get("candidate_blueprints")]
    if not reviews:
        raise ValueError("source episode has no blueprint review")
    review_line, review = reviews[-1]
    probes = [(no, row) for no, row in results
              if row.get("diagnostic") == "target_entity_insertion_probe" and no < review_line]
    if not probes:
        raise ValueError("blueprint review has no preceding entity probe")
    replayed_terms = {
        str(row.get("intended_term") or "").casefold()
        for no, result in results if no > review_line
        for row in result.get("evidence_rows", ()) if isinstance(row, Mapping)
    }
    return {
        "prompt": prompt, "prefixes": prefixes, "final_output": final_output,
        "review_line": review_line, "review": review, "probe_line": probes[-1][0], "probe": probes[-1][1],
        "orphan_terms": [str(bp.get("objective_term")) for bp in review["candidate_blueprints"]
                         if str(bp.get("objective_term") or "").casefold() not in replayed_terms],
    }


def verify_checkpoint(model_path: Path, recorded: Mapping[str, str]) -> dict[str, str]:
    """Refuse a checkpoint whose files differ from the source run's recorded hashes."""
    if not recorded:
        raise ValueError("source manifest records no checkpoint_sha256 map")
    verified: dict[str, str] = {}
    for name, expected in sorted(recorded.items()):
        path = model_path / name
        if not path.is_file():
            raise ValueError(f"checkpoint file missing: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 24), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f"checkpoint file differs from source manifest: {name}")
        verified[name] = expected
    return verified


def first_piece_won(row: Mapping[str, Any], first_piece_id: int) -> bool:
    """Gate on the replay's own target rank; the top-k diff table is truncated by probability delta."""
    rank = row.get("target_rank_after")
    if isinstance(rank, int) and not isinstance(rank, bool):
        return rank == 1
    return (row.get("edited_top1") or {}).get("token_id") == first_piece_id


def _edited_top1(replay: Mapping[str, Any]) -> dict[str, Any] | None:
    rows = [row for row in replay.get("topk_token_diff", ()) if isinstance(row, Mapping)]
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row.get("edited_logit", float("-inf"))))
    return {"piece": best.get("piece"), "token_id": best.get("token_id"), "edited_prob": best.get("edited_prob")}


def _replay_summary(replay: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("status", "error", "actual_delta_class", "target_piece", "target_rank_before", "target_rank_after",
            "target_piece_logit_delta", "target_piece_prob_delta", "target_mass_delta", "target_top20_hit_delta",
            "target_top20_threshold_gap_delta", "focus_rank_delta", "repeat_flag_delta",
            "activation_patch_hook_call_count", "activation_patch_step_size_clip_saturated")
    summary = {key: replay.get(key) for key in keys if key in replay}
    summary["edited_top1"] = _edited_top1(replay)
    return summary


def _scored(task: Any, text: str) -> dict[str, Any]:
    feedback = task.task_feedback(text)
    scored = {"text": text, "score": task.score(text), "word_count": len(text.split()),
              "missing_required_terms": feedback.get("missing_required_terms"),
              "constraint_violations": feedback.get("constraint_violations")}
    role_report = getattr(task, "role_binding_report", None)
    if callable(role_report):
        roles = role_report(text)
        scored.update(role_binding_ok=roles["role_binding_ok"], role_violations=roles["role_violations"])
    return scored


def _branch(worker: Any, task: Any, *, forced_token_ids: Sequence[int], horizon: int) -> dict[str, Any]:
    """Greedy continuation after optionally forcing tokens; restores the prefix afterwards."""
    saved = [_TokenSegment(kind=segment.kind, token_ids=list(segment.token_ids)) for segment in worker._segments]
    try:
        for token_id in forced_token_ids:
            worker._append_output_token(int(token_id))
        simulated = worker._simulate_decode(max_new_tokens=max(1, horizon), top_k=5,
                                            score_candidate_text=False, score_observer_check=False)
        if simulated is None:
            return {"status": "error", "error": worker._last_simulate_decode_error}
        branch_text = worker.codec.decode(worker._output_token_ids() + list(simulated["continuation_token_ids"]))
    finally:
        worker._segments = saved
        _release_memory()
    return {"status": "ok", **_scored(task, branch_text)}


def _next_token_view(worker: Any, token_id: int, *, top_k: int = 5) -> dict[str, Any]:
    """Unedited next-token distribution at the current prefix (one forward, state restored)."""
    import torch

    simulated = worker._simulate_decode(max_new_tokens=1, top_k=top_k,
                                        score_candidate_text=False, score_observer_check=False)
    _release_memory()
    if simulated is None:
        return {"status": "error", "error": worker._last_simulate_decode_error}
    logits = simulated["first_logits"].float()
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k=top_k)
    return {
        "status": "ok",
        "first_piece_rank": int((logits > logits[token_id]).sum().item()) + 1,
        "first_piece_prob": round(float(probs[token_id].item()), 6),
        "top": [{"piece": worker.codec.decode([int(index)]), "prob": round(float(prob), 6)}
                for prob, index in zip(top.values.tolist(), top.indices.tolist())],
    }


def _patch_edit(worker: Any, *, term: str, bundle_key: str, surface: Any,
                localization: str, alpha: float) -> dict[str, Any] | None:
    layer = int(getattr(surface.target, "layer", 0) or 0)
    site = str(getattr(surface.target, "site", "") or "resid_pre")
    recipe_id = f"readout_escape|activation_patch|{site}|L{layer}|source_body|{localization}|blend|a{alpha:.4f}"
    edit = worker._activation_patch_trial_edit_from_candidate({
        "objective_bundle_key": bundle_key, "actuator_bundle_key": bundle_key, "bundle_key": bundle_key,
        "objective_term": term, "layer": layer, "site": site, "alpha": alpha,
        "source_localization": localization, "patch_mode": "blend",
        "recipe_name": f"activation_patch_{site}_l{layer}_{localization}_a{int(round(alpha * 1000)):03d}",
        "operator_recipe_id": recipe_id, "promotion_reason": TRIAL_CONTRACT["promotion_reason"],
    }, trial_contract=TRIAL_CONTRACT)
    if edit is None:
        return None
    edit.update(bundle_key=bundle_key, focus_feature=term, phase_objective="readout_escape",
                bundle_family="activation_patch", provenance_class="source_body",
                recipe_localization=localization, recipe_pooling="blend", recipe_alpha=alpha,
                operator_recipe_id=recipe_id)
    return edit


def operator_candidates(worker: Any, *, term: str, bundle_key: str) -> list[tuple[str, dict[str, Any]]]:
    candidates: list[tuple[str, dict[str, Any]]] = []
    for name, kind, contrast, negative_scale, alpha in READOUT_SPECS:
        candidate = worker._readout_steering_candidate(
            bundle_key=bundle_key, members=(), intended_term=term, recipe_name=name, steering_kind=kind,
            target_terms=(term,), negative_terms=_DEFAULT_BAD_ATTRACTOR_TERMS if negative_scale else (),
            negative_scale=negative_scale, target_scale=1.0, alpha=alpha, contrast_mode=contrast)
        if candidate is not None:
            candidates.append((f"readout_steering|{name}", candidate))
    surfaces = [surface for surface in worker.surface_catalog
                if getattr(surface.target, "kind", None) == "activation"
                and "activation_patch" in set(getattr(surface, "allow_ops", ()) or ())
                and getattr(getattr(surface.target, "token", None), "mode", None) == "last"]
    for surface in surfaces:
        for localization in PATCH_LOCALIZATIONS:
            for alpha in PATCH_ALPHAS:
                edit = _patch_edit(worker, term=term, bundle_key=bundle_key, surface=surface,
                                   localization=localization, alpha=alpha)
                if edit is not None:
                    candidates.append((str(edit["operator_recipe_id"]), edit))
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective-term", default="Selim")
    parser.add_argument("--positions", default="", help="comma-separated generated-token counts; default: review onward")
    parser.add_argument("--parts", default=",".join(PARTS), help=f"comma-separated subset of {PARTS}")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    args = parser.parse_args(argv)
    parts = {item.strip() for item in args.parts.split(",") if item.strip()}
    if not parts or not parts <= set(PARTS):
        parser.error(f"--parts must be a non-empty subset of {PARTS}")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    manifest_path = args.source_manifest.expanduser().resolve(strict=True)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    source = load_source(source_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    profile = manifest["profile"]
    verified_checkpoint = verify_checkpoint(model_path, manifest.get("checkpoint_sha256") or {})

    import torch

    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    task = create_task_env(str(profile["task"]))
    prompt = task.reset(int(profile["seed"]))
    if prompt != source["prompt"]:
        raise ValueError("task fixture differs from recorded episode")
    model = load_worker_model(str(profile["worker_model"]), model_path=model_path, device=args.device,
                              dtype=str(profile.get("dtype") or "float16"), hf_offline=True,
                              mps_mode="conservative")
    worker = build_hooked_transformer_worker_runtime(
        model, task, seed=int(profile["seed"]), activation_surface_profile="activation_patch_expanded",
        max_diagnostic_calls_per_run=10_000, diagnostic_result_window=64, candidate_handoff_mode="soft",
        readout_sidecar_analyzer=create_readout_analyzer("sae_scaffold"), readout_analyzer_rerank_mode="apply")
    worker.reset(prompt)

    review_step = int(source["review"]["recorded_step"])
    final_length = max(source["prefixes"]) + 1
    positions = sorted({int(item) for item in args.positions.split(",") if item.strip()} or
                       set(range(review_step, final_length)))
    term = args.objective_term
    bundle_key = f"entity_insert:{term.casefold()}:ceiling"
    blueprint = next((bp for bp in source["review"]["candidate_blueprints"]
                      if str(bp.get("objective_term") or "").casefold() == term.casefold()), None)
    if blueprint is None or not isinstance(blueprint.get("first_token_id"), int):
        raise ValueError(f"no recorded blueprint with a first-piece token for {term!r}")
    first_piece_ids = [int(blueprint["first_token_id"])]
    first_piece = worker.codec.decode(first_piece_ids)
    if first_piece != blueprint.get("first_piece"):
        raise ValueError(f"tokenizer disagrees with recorded first piece: {first_piece!r}")
    max_tokens = int(task.worker_runtime_kwargs().get("max_generated_tokens") or 64)
    report: dict[str, Any] = {
        "kind": "orphan_blueprint_ceiling_replay",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "worker_model": profile["worker_model"], "worker_model_path": str(model_path),
        "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "checkpoint_sha256_verified": verified_checkpoint,
        "device": args.device, "dtype": profile.get("dtype"), "seed": profile["seed"], "task": profile["task"],
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "recorded_final_output": source["final_output"],
        "recorded_final": _scored(task, source["final_output"]),
        "recorded_review_line": source["review_line"], "recorded_probe_line": source["probe_line"],
        "recorded_review_step": review_step, "orphan_terms": source["orphan_terms"],
        "objective_term": term, "first_piece": first_piece, "first_piece_token_id": first_piece_ids[0],
        "positions": positions, "scope": "diagnostic_only_ceiling_not_controller_choice_or_rollout_outcome",
        "production_apply_allowed": False, "positions_report": [],
    }

    def advance_to(length: int) -> None:
        while len(worker._output_token_ids()) < length:
            worker.step()
        recorded = source["prefixes"].get(length)
        if recorded is not None and worker.final_text() != recorded:
            raise ValueError(f"prefix drifted at {length}: {worker.final_text()!r} != {recorded!r}")

    started = time.monotonic()
    report["parts"] = sorted(parts)
    if "orphan" in parts:
        advance_to(review_step)
        worker._diagnostic_results = [dict(source["probe"]), dict(source["review"])]
        worker._diagnostic_calls_used = 0
        packet = worker.build_controller_packet()
        orphan = {"prefix": worker.final_text(), "results": []}
        orphan_keys = {term.casefold() for term in source["orphan_terms"]}
        requests = [{"diagnostic": "operator_diagnostic_replay",
                     "next_evidence_needed": "operator_diagnostic_replay_for_entity_candidates",
                     "materialize_entity_insertion_candidates": True}]
        for recorded in source["review"]["candidate_blueprints"]:
            objective = str(recorded.get("candidate_key"))
            requests.append({"diagnostic": "activation_patch_candidate_review", "bundle_key": objective,
                             "objective_bundle_key": objective,
                             "operator_recipe_expansion_mode": "activation_patch_candidate_review"})
        for request in requests:
            results = worker.request_controller_diagnostics(request, source="forced_diagnostic_shadow", packet=packet)
            for result in results:
                rows = [row for row in result.get("evidence_rows", ()) if isinstance(row, Mapping)
                        and str(row.get("intended_term") or "").casefold() in orphan_keys]
                orphan["results"].append({
                    "diagnostic": result.get("diagnostic"), "status": result.get("status"),
                    "objective_bundle_key": request.get("objective_bundle_key"),
                    "rows": [{key: row.get(key) for key in ORPHAN_ROW_KEYS} for row in rows],
                })
            _release_memory()
        report["orphan_replay"] = orphan
        report["orphan_replay_elapsed_seconds"] = round(time.monotonic() - started, 1)
        worker._diagnostic_results = []
        getattr(worker, "_frozen_diagnostic_candidates", {}).clear()
        _release_memory()

    for position in positions if parts & {"ceiling", "sweep"} else ():
        advance_to(position)
        prefix = worker.final_text()
        horizon = max_tokens - position
        entry: dict[str, Any] = {"generated_tokens": position, "prefix": prefix,
                                 "next_token": _next_token_view(worker, first_piece_ids[0])}
        if "ceiling" in parts:
            entry["natural"] = _branch(worker, task, forced_token_ids=(), horizon=horizon)
            entry["forced_first_piece"] = _branch(worker, task, forced_token_ids=first_piece_ids,
                                                  horizon=horizon - 1)
        if "sweep" in parts:
            sweep = []
            for recipe_id, candidate in operator_candidates(worker, term=term, bundle_key=bundle_key):
                replay = worker.replay_candidate_edits_actual_delta(
                    [candidate], max_new_tokens=1, top_k=8, max_edits_per_step_override=1,
                    score_candidate_text=False, label=f"ceiling:{recipe_id}", ownership_terms=(term,),
                    intended_bundle_key=bundle_key, intended_term=term)
                row = {"operator_recipe_id": recipe_id, **_replay_summary(replay)}
                if first_piece_won(row, first_piece_ids[0]):
                    full = worker.replay_candidate_edits_actual_delta(
                        [candidate], max_new_tokens=horizon, top_k=5, max_edits_per_step_override=1,
                        score_candidate_text=False, label=f"ceiling_full:{recipe_id}", ownership_terms=(term,),
                        intended_bundle_key=bundle_key, intended_term=term)
                    row["full_continuation"] = _scored(task, prefix + str(full.get("continuation_candidate") or ""))
                sweep.append(row)
                _release_memory()
            ranks = [row["target_rank_after"] for row in sweep if isinstance(row.get("target_rank_after"), int)]
            entry["operator_sweep"] = {
                "candidate_count": len(sweep),
                "first_piece_wins": sum(1 for row in sweep if "full_continuation" in row),
                "best_target_rank_after": min(ranks) if ranks else None,
                "rows": sweep,
            }
        report["positions_report"].append(entry)
        print(json.dumps({"generated_tokens": position, "prefix": prefix,
                          "first_piece_rank": entry["next_token"].get("first_piece_rank"),
                          "forced": (entry.get("forced_first_piece") or {}).get("text"),
                          "forced_score": (entry.get("forced_first_piece") or {}).get("score"),
                          "best_rank": (entry.get("operator_sweep") or {}).get("best_target_rank_after"),
                          "wins": (entry.get("operator_sweep") or {}).get("first_piece_wins"),
                          "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
