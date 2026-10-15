"""Diagnostic-only dose response for the operator families the ceiling replay found too weak.

At recorded no-edit prefixes, each readout-steering recipe and each activation
patch (every patchable surface x source localization) is replayed at increasing
dose. The question is the smallest dose that makes the objective's first piece
the greedy token, and whether the sentence finished from that dose is intact.

Dose limits are released only inside this quarantined worker: its surface caps,
alpha budget and replay policy are raised, while operator shape (direction
construction and norm clip) is unchanged. Every edit is a one-token replay with
state restored. No controller call, rollout edit, production trial or apply
permission is involved, and shared defaults are not modified.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ..runtime.policy import HarnessPolicy
from ..runtime.worker import _DEFAULT_BAD_ATTRACTOR_TERMS
from .digit_transform_e2e import (
    build_default_activation_surface_catalog,
    build_hooked_transformer_worker_runtime,
    create_readout_analyzer,
    create_task_env,
    load_worker_model,
)
from .replay_orphan_blueprint_ceiling import (
    PATCH_LOCALIZATIONS,
    _next_token_view,
    _release_memory,
    _replay_summary,
    _scored,
    first_piece_won,
    load_source,
    verify_checkpoint,
)

READOUT_RECIPES = (
    ("entity_target_readout_l005", "entity_target_readout", "target_readout_minus_attractor", 0.05),
    ("target_readout_pure", "target_readout", "target_readout_pure", 0.0),
)
READOUT_DOSES = (0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96)
PATCH_DOSES = (0.08, 0.16, 0.32, 0.64, 1.0)
RELEASED = 1e9
# Finite on purpose: the worker's packet edit-cost budget rejects a 1e9 step size.
# 1000 is far above late-layer residual norms, so it does not bind the blend.
PATCH_STEP_SIZE = 1000.0
FULL_DECODES_PER_CURVE = 3


def released_replay_policy(*, max_edits_per_step_override: int | None = None) -> HarnessPolicy:
    base = HarnessPolicy.default_v0()
    budget = replace(base.global_budget, max_edits_per_step=max(1, int(max_edits_per_step_override or 1)),
                     max_total_alpha=RELEASED, max_total_edit_cost=RELEASED)
    return replace(base, global_budget=budget)


def minimal_winning_dose(points: Sequence[Mapping[str, Any]]) -> float | None:
    wins = [float(point["dose"]) for point in points if point.get("won")]
    return min(wins) if wins else None


def _readout_edit(worker: Any, *, term: str, bundle_key: str, recipe: tuple, dose: float) -> dict[str, Any] | None:
    name, kind, contrast, negative_scale = recipe
    return worker._readout_steering_candidate(
        bundle_key=bundle_key, members=(), intended_term=term, recipe_name=f"{name}_dose{dose:g}",
        steering_kind=kind, target_terms=(term,),
        negative_terms=_DEFAULT_BAD_ATTRACTOR_TERMS if negative_scale else (),
        negative_scale=negative_scale, target_scale=1.0, alpha=dose, contrast_mode=contrast)


def _patch_edit(worker: Any, *, term: str, bundle_key: str, surface: Any,
                localization: str, dose: float) -> dict[str, Any] | None:
    layer = int(getattr(surface.target, "layer", 0) or 0)
    site = str(getattr(surface.target, "site", "") or "resid_pre")
    recipe_id = f"readout_escape|activation_patch|{site}|L{layer}|source_body|{localization}|blend"
    edit = worker._activation_patch_trial_edit_from_candidate({
        "objective_bundle_key": bundle_key, "actuator_bundle_key": bundle_key, "bundle_key": bundle_key,
        "objective_term": term, "layer": layer, "site": site, "alpha": dose, "step_size": PATCH_STEP_SIZE,
        "source_localization": localization, "patch_mode": "blend",
        "recipe_name": f"activation_patch_{site}_l{layer}_{localization}_dose{dose:g}",
        "operator_recipe_id": f"{recipe_id}|a{dose:.4f}", "promotion_reason": "operator_dose_response",
    }, trial_contract={"max_alpha": RELEASED, "norm_clip": 1.0, "trial_budget_class": "diagnostic_only",
                       "production_trial_followup_allowed": False, "promotion_reason": "operator_dose_response"})
    if edit is None:
        return None
    edit.update(bundle_key=bundle_key, focus_feature=term, phase_objective="readout_escape",
                bundle_family="activation_patch", provenance_class="source_body",
                recipe_localization=localization, recipe_pooling="blend", recipe_alpha=dose)
    return edit


def curves(worker: Any, *, term: str, bundle_key: str, families: set[str]) -> list[dict[str, Any]]:
    """Each curve is one operator recipe; its points are built lazily per dose."""
    specs: list[dict[str, Any]] = []
    if "readout" in families:
        for recipe in READOUT_RECIPES:
            specs.append({"family": "readout_steering", "recipe": recipe[0], "doses": READOUT_DOSES,
                          "build": lambda dose, recipe=recipe: _readout_edit(
                              worker, term=term, bundle_key=bundle_key, recipe=recipe, dose=dose)})
    if "patch" in families:
        surfaces = [surface for surface in worker.surface_catalog
                    if getattr(surface.target, "kind", None) == "activation"
                    and "activation_patch" in set(getattr(surface, "allow_ops", ()) or ())
                    and getattr(getattr(surface.target, "token", None), "mode", None) == "last"]
        for surface in surfaces:
            for localization in PATCH_LOCALIZATIONS:
                site = str(getattr(surface.target, "site", ""))
                layer = int(getattr(surface.target, "layer", 0) or 0)
                specs.append({"family": "activation_patch", "recipe": f"{site}|L{layer}|{localization}",
                              "doses": PATCH_DOSES,
                              "build": lambda dose, surface=surface, localization=localization: _patch_edit(
                                  worker, term=term, bundle_key=bundle_key, surface=surface,
                                  localization=localization, dose=dose)})
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective-term", default="Selim")
    parser.add_argument("--positions", default="9,10,13")
    parser.add_argument("--families", default="readout,patch")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    args = parser.parse_args(argv)
    families = {item.strip() for item in args.families.split(",") if item.strip()}
    if not families or not families <= {"readout", "patch"}:
        parser.error("--families must be a non-empty subset of readout,patch")
    positions = sorted({int(item) for item in args.positions.split(",") if item.strip()})
    if not positions:
        parser.error("--positions must name at least one prefix length")
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
    released_catalog = build_default_activation_surface_catalog(
        model, profile="activation_patch_expanded", max_alpha=RELEASED, step_size=RELEASED)
    worker = build_hooked_transformer_worker_runtime(
        model, task, seed=int(profile["seed"]), surface_catalog=released_catalog,
        activation_surface_profile="activation_patch_expanded", max_total_alpha=RELEASED,
        max_diagnostic_calls_per_run=10_000, diagnostic_result_window=64, candidate_handoff_mode="soft",
        readout_sidecar_analyzer=create_readout_analyzer("sae_scaffold"), readout_analyzer_rerank_mode="apply")
    worker._replay_policy = released_replay_policy
    worker.reset(prompt)

    term = args.objective_term
    blueprint = next((bp for bp in source["review"]["candidate_blueprints"]
                      if str(bp.get("objective_term") or "").casefold() == term.casefold()), None)
    if blueprint is None or not isinstance(blueprint.get("first_token_id"), int):
        raise ValueError(f"no recorded blueprint with a first-piece token for {term!r}")
    first_piece_id = int(blueprint["first_token_id"])
    bundle_key = f"entity_insert:{term.casefold()}:dose_response"
    max_tokens = int(task.worker_runtime_kwargs().get("max_generated_tokens") or 64)
    report: dict[str, Any] = {
        "kind": "operator_dose_response",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "worker_model": profile["worker_model"], "worker_model_path": str(model_path),
        "checkpoint_sha256_verified": verified_checkpoint,
        "device": args.device, "dtype": profile.get("dtype"), "seed": profile["seed"], "task": profile["task"],
        "objective_term": term, "first_piece": blueprint.get("first_piece"), "first_piece_token_id": first_piece_id,
        "positions": positions, "families": sorted(families),
        "readout_doses": list(READOUT_DOSES), "patch_doses": list(PATCH_DOSES),
        "release": {"surface_max_alpha": RELEASED, "surface_step_size": RELEASED, "worker_max_total_alpha": RELEASED,
                    "patch_requested_step_size": PATCH_STEP_SIZE,
                    "replay_policy_max_total_alpha": RELEASED, "replay_policy_max_total_edit_cost": RELEASED,
                    "norm_clip": "unchanged (catalog default; patch trial contract 1.0)",
                    "edit_ttl_tokens": 1, "scope": "quarantined_worker_only_shared_defaults_unchanged"},
        "win_criterion": "target_rank_after == 1 for the objective's first piece",
        "scope": "diagnostic_only_dose_response_not_controller_choice_or_rollout_outcome",
        "production_apply_allowed": False, "positions_report": [],
    }

    def advance_to(length: int) -> None:
        while len(worker._output_token_ids()) < length:
            worker.step()
        recorded = source["prefixes"].get(length)
        if recorded is not None and worker.final_text() != recorded:
            raise ValueError(f"prefix drifted at {length}: {worker.final_text()!r} != {recorded!r}")

    started = time.monotonic()
    for position in positions:
        advance_to(position)
        prefix = worker.final_text()
        horizon = max_tokens - position
        entry: dict[str, Any] = {"generated_tokens": position, "prefix": prefix,
                                 "next_token": _next_token_view(worker, first_piece_id), "curves": []}
        for spec in curves(worker, term=term, bundle_key=bundle_key, families=families):
            points: list[dict[str, Any]] = []
            full_decodes = 0
            for dose in spec["doses"]:
                edit = spec["build"](dose)
                if edit is None:
                    points.append({"dose": dose, "status": "materialization_failed", "won": False})
                    continue
                replay = worker.replay_candidate_edits_actual_delta(
                    [edit], max_new_tokens=1, top_k=8, max_edits_per_step_override=1, score_candidate_text=False,
                    label=f"dose:{spec['recipe']}:{dose:g}", ownership_terms=(term,),
                    intended_bundle_key=bundle_key, intended_term=term)
                point = {"dose": dose, "applied_alpha": (edit.get("op") or {}).get("alpha"),
                         "applied_step_size": (edit.get("budget") or {}).get("step_size"), **_replay_summary(replay)}
                if replay.get("simulate_error"):
                    point["simulate_error"] = str(replay["simulate_error"])[:200]
                point["won"] = first_piece_won(point, first_piece_id)
                if point["won"] and full_decodes < FULL_DECODES_PER_CURVE:
                    full = worker.replay_candidate_edits_actual_delta(
                        [edit], max_new_tokens=horizon, top_k=5, max_edits_per_step_override=1,
                        score_candidate_text=False, label=f"dose_full:{spec['recipe']}:{dose:g}",
                        ownership_terms=(term,), intended_bundle_key=bundle_key, intended_term=term)
                    point["full_continuation"] = _scored(task, prefix + str(full.get("continuation_candidate") or ""))
                    full_decodes += 1
                points.append(point)
                _release_memory()
            entry["curves"].append({"family": spec["family"], "recipe": spec["recipe"], "points": points,
                                    "minimal_winning_dose": minimal_winning_dose(points)})
        wins = [curve for curve in entry["curves"] if curve["minimal_winning_dose"] is not None]
        entry["winning_curve_count"] = len(wins)
        report["positions_report"].append(entry)
        print(json.dumps({"generated_tokens": position, "prefix": prefix,
                          "first_piece_rank": entry["next_token"].get("first_piece_rank"),
                          "curves": len(entry["curves"]), "winning_curves": len(wins),
                          "min_doses": sorted({(c["family"], c["minimal_winning_dose"]) for c in wins})[:6],
                          "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
