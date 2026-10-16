"""Diagnostic-only probe of structure tokens (connectives) instead of entity names.

At recorded no-edit prefixes this asks whether nudging a clause connective, rather
than writing the missing name, lets the worker produce the missing clause itself:

1. forced connectives: force each candidate token once, decode greedily, score with
   the unchanged evaluator and the offline role-binding check;
2. readout dose: push one target token through the existing readout-steering
   operator (direction retargeted to that token id, shape unchanged) at increasing
   dose, with the target rank computed exactly from the full edited logits;
3. scale: the norm of the residual stream the readout operator edits, so a dose
   can be read as a fraction of it.

Every edit is a one-token replay in a quarantined worker with released limits and
state restored. No controller call, rollout edit, production trial or apply
permission is involved, and shared defaults are not modified.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from .digit_transform_e2e import (
    build_default_activation_surface_catalog,
    build_hooked_transformer_worker_runtime,
    create_task_env,
    load_worker_model,
)
from .replay_operator_dose_response import (
    PATCH_DOSES,
    READOUT_RECIPES,
    RELEASED,
    _patch_edit,
    _readout_edit,
    released_replay_policy,
)
from .replay_orphan_blueprint_ceiling import (
    PATCH_LOCALIZATIONS,
    _branch,
    _release_memory,
    _scored,
    load_source,
    verify_checkpoint,
)

STRUCTURE_DOSES = (0.08, 0.16, 0.32, 0.64, 1.28, 1.92, 2.56, 5.12)


def single_token_id(codec: Any, piece: str, *, special_ids: set[int]) -> int | None:
    ids = [int(i) for i in codec.encode(piece).reshape(-1).tolist() if int(i) not in special_ids]
    return ids[0] if len(ids) == 1 else None


def retarget_readout(edit: dict[str, Any], token_id: int) -> int:
    """Point every readout_direction node at one token id; returns the node count."""
    def visit(node: Any) -> int:
        if isinstance(node, dict):
            if node.get("fn") == "readout_direction":
                node["target_token_ids"] = [int(token_id)]
                node["negative_token_ids"] = []
                return 1
            return sum(visit(value) for value in node.values())
        if isinstance(node, list):
            return sum(visit(value) for value in node)
        return 0
    return visit(edit.get("source"))


def token_rank(logits: Any, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def patch_edit_size(simulated: dict[str, Any]) -> dict[str, Any]:
    """Blend movement relative to the edited site's own norm, from the hook telemetry."""
    rows = [row for row in simulated.get("edit_runtime_telemetry") or () if isinstance(row, dict)
            and str(row.get("op") or "") == "activation_patch"]
    if not rows:
        return {}
    before = rows[0].get("target_norm_before")
    delta = rows[0].get("blend_delta_norm")
    size = {"site_norm_before": before, "blend_delta_norm": delta}
    if isinstance(before, (int, float)) and isinstance(delta, (int, float)) and before > 0:
        size["blend_delta_over_site_norm"] = round(float(delta) / float(before), 5)
    return size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", default="8,9")
    parser.add_argument("--connectives", default=",| and| while| whereas|;")
    parser.add_argument("--dose-target", default=",")
    parser.add_argument("--dose-positions", default="8")
    parser.add_argument("--patch-source-terms", default="",
                        help="comma-separated source-body terms whose activations are blended in (e.g. while)")
    parser.add_argument("--skip-forced-and-readout", action="store_true",
                        help="run only the activation-patch curves")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    args = parser.parse_args(argv)
    positions = sorted({int(item) for item in args.positions.split(",") if item.strip()})
    dose_positions = sorted({int(item) for item in args.dose_positions.split(",") if item.strip()})
    connectives = [piece for piece in args.connectives.split("|") if piece]
    patch_terms = [item.strip() for item in args.patch_source_terms.split(",") if item.strip()]
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
    catalog = build_default_activation_surface_catalog(
        model, profile="activation_patch_expanded", max_alpha=RELEASED, step_size=RELEASED)
    worker = build_hooked_transformer_worker_runtime(
        model, task, seed=int(profile["seed"]), surface_catalog=catalog,
        activation_surface_profile="activation_patch_expanded", max_total_alpha=RELEASED)
    worker.reset(prompt)
    special_ids = {int(i) for i in (getattr(model.tokenizer, "all_special_ids", None) or [])}
    dose_token = single_token_id(worker.codec, args.dose_target, special_ids=special_ids)
    if dose_token is None:
        raise ValueError(f"dose target {args.dose_target!r} is not a single token")
    max_tokens = int(task.worker_runtime_kwargs().get("max_generated_tokens") or 64)
    readout_layer = int(model.cfg.n_layers) - 1
    report: dict[str, Any] = {
        "kind": "structure_token_probe",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "worker_model": profile["worker_model"], "checkpoint_sha256_verified": verified_checkpoint,
        "device": args.device, "dtype": profile.get("dtype"), "seed": profile["seed"], "task": profile["task"],
        "connectives": connectives, "dose_target": args.dose_target, "dose_target_token_id": dose_token,
        "dose_recipe": READOUT_RECIPES[1][0], "doses": list(STRUCTURE_DOSES),
        "patch_source_terms": patch_terms, "patch_doses": list(PATCH_DOSES),
        "patch_localizations": list(PATCH_LOCALIZATIONS),
        "readout_residual_hook": f"blocks.{readout_layer}.hook_resid_pre",
        "release": {"surface_max_alpha": RELEASED, "surface_step_size": RELEASED, "worker_max_total_alpha": RELEASED,
                    "edit_ttl_tokens": 1, "scope": "quarantined_worker_only_shared_defaults_unchanged"},
        "scope": "diagnostic_only_structure_probe_not_controller_choice_or_rollout_outcome",
        "production_apply_allowed": False, "positions_report": [],
    }

    def advance_to(length: int) -> None:
        while len(worker._output_token_ids()) < length:
            worker.step()
        recorded = source["prefixes"].get(length)
        if recorded is not None and worker.final_text() != recorded:
            raise ValueError(f"prefix drifted at {length}: {worker.final_text()!r} != {recorded!r}")

    def simulate(edit: dict[str, Any] | None, tokens: int) -> dict[str, Any] | None:
        command = None if edit is None else {"version": "0.1", "decision": "apply", "edits": [edit]}
        return worker._simulate_decode(max_new_tokens=tokens, top_k=5, command=command,
                                       policy_override=released_replay_policy(max_edits_per_step_override=1),
                                       score_candidate_text=False, score_observer_check=False)

    def patch_curves(position: int, prefix: str, horizon: int, base_logits: Any, residual_norm: float) -> list[dict[str, Any]]:
        surfaces = [surface for surface in worker.surface_catalog
                    if getattr(surface.target, "kind", None) == "activation"
                    and "activation_patch" in set(getattr(surface, "allow_ops", ()) or ())
                    and getattr(getattr(surface.target, "token", None), "mode", None) == "last"]
        result = []
        for term in patch_terms:
            for surface in surfaces:
                for localization in PATCH_LOCALIZATIONS:
                    points = []
                    for dose in PATCH_DOSES:
                        edit = _patch_edit(worker, term=term, bundle_key=f"structure:patch:{term}", surface=surface,
                                           localization=localization, dose=dose)
                        if edit is None:
                            points.append({"dose": dose, "status": "materialization_failed"})
                            continue
                        edit.setdefault("id", f"structure_patch_{dose:g}")
                        edited = simulate(edit, 1)
                        if edited is None:
                            points.append({"dose": dose, "status": "error", "error": worker._last_simulate_decode_error})
                            continue
                        logits = edited["first_logits"].float()
                        point = {"dose": dose, "status": "ok", "target_rank": token_rank(logits, dose_token),
                                 "target_logit_delta": round(float(logits[dose_token] - base_logits[dose_token]), 5),
                                 "gap_to_top1": round(float(logits.max() - logits[dose_token]), 5),
                                 "top1": worker.codec.decode([int(logits.argmax())]),
                                 **patch_edit_size(edited)}
                        if point["target_rank"] == 1:
                            full = simulate(edit, horizon)
                            if full is not None:
                                point["full_continuation"] = _scored(task, worker.codec.decode(
                                    worker._output_token_ids() + list(full["continuation_token_ids"])))
                        points.append(point)
                        _release_memory()
                    wins = [point["dose"] for point in points if point.get("target_rank") == 1]
                    result.append({"source_term": term, "site": str(getattr(surface.target, "site", "")),
                                   "layer": int(getattr(surface.target, "layer", 0) or 0),
                                   "localization": localization, "points": points,
                                   "minimal_winning_dose": min(wins) if wins else None})
        return result

    started = time.monotonic()
    for position in sorted(set(positions) | set(dose_positions)):
        advance_to(position)
        prefix = worker.final_text()
        horizon = max_tokens - position
        baseline = simulate(None, 1)
        if baseline is None:
            raise RuntimeError(worker._last_simulate_decode_error)
        base_logits = baseline["first_logits"].float()
        _, cache = model.run_with_cache(worker._current_token_tensor().to(model.cfg.device),
                                        names_filter=lambda name: name == report["readout_residual_hook"])
        residual_norm = float(cache[report["readout_residual_hook"]][0, -1].float().norm())
        del cache
        entry: dict[str, Any] = {"generated_tokens": position, "prefix": prefix,
                                 "readout_residual_norm": round(residual_norm, 4),
                                 "baseline_top1": worker.codec.decode([int(base_logits.argmax())])}
        if position in positions and not args.skip_forced_and_readout:
            forced = []
            for piece in connectives:
                token_id = single_token_id(worker.codec, piece, special_ids=special_ids)
                if token_id is None:
                    forced.append({"piece": piece, "status": "not_single_token"})
                    continue
                probs = torch.softmax(base_logits, dim=-1)
                forced.append({"piece": piece, "token_id": token_id, "rank": token_rank(base_logits, token_id),
                               "prob": round(float(probs[token_id]), 6),
                               **_branch(worker, task, forced_token_ids=(token_id,), horizon=horizon - 1)})
            entry["forced_connectives"] = forced
        if position in dose_positions and patch_terms:
            entry["patch_dose_curves"] = patch_curves(position, prefix, horizon, base_logits, residual_norm)
        if position in dose_positions and not args.skip_forced_and_readout:
            curve = []
            for dose in STRUCTURE_DOSES:
                edit = copy.deepcopy(_readout_edit(worker, term=args.dose_target, bundle_key="structure:dose",
                                                   recipe=READOUT_RECIPES[1], dose=dose))
                if edit is None or not retarget_readout(edit, dose_token):
                    curve.append({"dose": dose, "status": "materialization_failed"})
                    continue
                edit.setdefault("id", f"structure_dose_{dose:g}")
                edited = simulate(edit, 1)
                if edited is None:
                    curve.append({"dose": dose, "status": "error", "error": worker._last_simulate_decode_error})
                    continue
                logits = edited["first_logits"].float()
                point = {"dose": dose, "status": "ok", "dose_over_residual_norm": round(dose / residual_norm, 5),
                         "target_rank": token_rank(logits, dose_token),
                         "target_logit_delta": round(float(logits[dose_token] - base_logits[dose_token]), 5),
                         "gap_to_top1": round(float(logits.max() - logits[dose_token]), 5),
                         "top1": worker.codec.decode([int(logits.argmax())])}
                if point["target_rank"] == 1:
                    full = simulate(edit, horizon)
                    if full is not None:
                        point["full_continuation"] = _scored(task, worker.codec.decode(
                            worker._output_token_ids() + list(full["continuation_token_ids"])))
                curve.append(point)
                _release_memory()
            wins = [point["dose"] for point in curve if point.get("target_rank") == 1]
            entry["readout_dose_curve"] = {"target": args.dose_target, "baseline_rank": token_rank(base_logits, dose_token),
                                           "baseline_gap": round(float(base_logits.max() - base_logits[dose_token]), 5),
                                           "minimal_winning_dose": min(wins) if wins else None, "points": curve}
        report["positions_report"].append(entry)
        print(json.dumps({"generated_tokens": position, "prefix": prefix,
                          "residual_norm": entry["readout_residual_norm"],
                          "min_dose": (entry.get("readout_dose_curve") or {}).get("minimal_winning_dose"),
                          "patch_wins": sum(1 for curve in entry.get("patch_dose_curves", ())
                                            if curve["minimal_winning_dose"] is not None),
                          "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
