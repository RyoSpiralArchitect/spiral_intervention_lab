"""Replay one frozen activation patch across recorded connective boundaries.

Local-only, no provider calls or policy promotion. Paths are CLI supplied; the
source manifest supplies model/task settings, not instructions or API keys.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import torch

from .digit_transform_e2e import (
    _build_parser, build_hooked_transformer_worker_runtime, create_task_env, load_worker_model,
)
from .rewrite_ladder_baseline import sha256_file
from ..runtime.prefix_probe import freeze_candidate, measure_prefix, prefix_plan
from ..runtime.response_probe import state_identity
from ..runtime.completion_probe import completion_binding


def plan_from_events(events: list[dict], manifest: dict, *, controller_step: int, dose: float) -> dict:
    """Select by recorded step/dose/canonical binding, never by measured gain."""
    starts = [e for e in events if e.get("event") == "episode_start"]
    if len(starts) != 1 or starts[0].get("prompt") != manifest["fixture"]["prompt"]:
        raise ValueError("recorded_prompt_manifest_mismatch")
    commands = [e for e in events if e.get("event") == "controller_command"]
    observations = [e for e in events if e.get("event") == "controller_observation"]
    if not commands or any(e.get("command", {}).get("decision") != "noop" for e in commands):
        raise ValueError("requires_recorded_no_apply_trajectory")
    if any(e.get("active_edit_ids") for e in observations):
        raise ValueError("recorded_active_edits")
    measured = [e for e in events if e.get("event") == "controller_diagnostic_result"
                and e.get("step") == controller_step and e.get("target_piece_binding_seed_matrix_executed")]
    if len(measured) != 1:
        raise ValueError("requires_one_recorded_matrix_at_anchor")
    matrix = measured[0]["target_piece_binding_seed_matrix_summary"]
    if matrix.get("status") != "dose_matched_response_complete" or not matrix.get("state_restored") or matrix.get("errors"):
        raise ValueError("recorded_matrix_incomplete")
    selected = [r for r in measured[0]["target_piece_binding_seed_matrix_rows"]
                if r.get("activation_patch_step_size") == dose and r.get("target_piece_binding_variant") == "canonical"]
    if len(selected) != 1 or selected[0].get("measurement_context_id") != matrix["measurement_context_id"]:
        raise ValueError("ambiguous_or_unbound_recorded_candidate")
    points = prefix_plan(observations, controller_step + 1)
    return {"controller_anchor_step": controller_step, "candidate": selected[0], "points": points,
            "recorded_context_id": matrix["measurement_context_id"],
            "selection_rule": "recorded canonical binding at requested dose; anchor plus first later connective before/after",
            "planned_physical_replays": 4 * len(points), "production_apply_allowed": False}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _release_buffers(device: str) -> dict:
    gc.collect()
    if device == "mps":
        torch.mps.synchronize()
        before = torch.mps.driver_allocated_memory()
        torch.mps.empty_cache()
        return {"driver_bytes_before_release": before, "driver_bytes_after_release": torch.mps.driver_allocated_memory()}
    return {}


def completion_reference(directory: Path, plan: dict, *, worker_step: int,
                         source_sha256: str) -> dict:
    """Bind an explicitly chosen measured prefix, not a newly winning prefix."""
    manifest = json.loads((directory / "manifest.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    candidate = json.loads((directory / "frozen_candidate.json").read_text())
    if (status.get("status") != "complete" or status.get("rows_sha256") != sha256_file(directory / "rows.jsonl")
            or manifest.get("source_jsonl_sha256") != source_sha256):
        raise ValueError("completion_reference_not_hash_verified")
    rows = [json.loads(line) for line in (directory / "rows.jsonl").read_text().splitlines() if line.strip()]
    selected = [r for r in rows if r.get("worker_step") == worker_step]
    points = [p for p in plan["points"] if p["worker_step"] == worker_step]
    if len(selected) != 1 or len(points) != 1:
        raise ValueError("completion_prefix_not_in_frozen_plan")
    row = selected[0]
    if (row.get("status") != "complete" or not row.get("state_restored") or row.get("prefix") != points[0]["prefix"]
            or row.get("candidate_id") != candidate.get("candidate_id")
            or row.get("target_piece_token_id") != plan["candidate"].get("target_piece_token_id")
            or row.get("source_tensor_identity") != plan["candidate"].get("source_tensor_identity")
            or candidate.get("edit", {}).get("budget", {}).get("step_size") != plan["candidate"].get("activation_patch_step_size")
            or candidate.get("edit", {}).get("op", {}).get("alpha") != plan["candidate"].get("activation_patch_alpha")):
        raise ValueError("completion_reference_candidate_or_context_mismatch")
    return {"worker_step": worker_step, "prefix": row["prefix"], "candidate_id": row["candidate_id"],
            "measurement_context_id": row["measurement_context_id"],
            "competitor_token_id": row["baseline"]["token_ids"][0],
            "reference_artifact_sha256": {name: sha256_file(directory / name)
                for name in ("manifest.json", "status.json", "rows.jsonl", "frozen_candidate.json")},
            "reference_directory": str(directory.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--controller-step", type=int, default=4)
    parser.add_argument("--recorded-dose", type=float, choices=(0.04, 0.16), default=0.16)
    parser.add_argument("--horizon", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--completion-worker-step", type=int,
                        help="Remeasure only this previously measured prefix with competition/completion diagnostics")
    parser.add_argument("--reference-prefix-dir", type=Path,
                        help="Completed frozen-prefix replay to pin the candidate, context and competitor")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source_manifest = json.loads(args.source_manifest.read_text())
    events = [json.loads(line) for line in args.source_jsonl.read_text().splitlines() if line.strip()]
    plan = plan_from_events(events, source_manifest, controller_step=args.controller_step, dose=args.recorded_dose)
    plan["candidate_anchor_point"] = dict(plan["points"][0])
    if (args.completion_worker_step is None) != (args.reference_prefix_dir is None):
        raise ValueError("completion_mode_requires_worker_step_and_reference_directory")
    reference = None
    if args.completion_worker_step is not None:
        reference = completion_reference(args.reference_prefix_dir, plan, worker_step=args.completion_worker_step,
                                         source_sha256=sha256_file(args.source_jsonl))
        plan["points"] = [p for p in plan["points"] if p["worker_step"] == args.completion_worker_step]
        plan.pop("planned_physical_replays")
        plan["planned_primary_natural_replays"] = 4
        plan["max_total_physical_replays"] = 18
        plan["selection_rule"] = "explicitly fixed previously measured prefix, candidate, source, dose and competitor"
        plan["completion_reference"] = reference
        plan["conditional_branch_contract"] = {
            "source": "recorded canonical required-term tokenization only", "max_target_pieces": 8,
            "natural_replays": 4, "conditional_greedy_replays": 2, "repeats_per_extra_name_factor": 2,
            "no_patch_in_conditioned_branch": True, "eligible_for_task_comparison": False}
    settings = _build_parser().parse_args(source_manifest["argv"])
    if settings.worker_first_n_layers is not None or settings.worker_decoder_control_mode != "off":
        raise ValueError("requires_full_depth_without_decoder_control")
    if settings.worker_tokenizer_path:
        raise ValueError("separate_tokenizer_requires_a_separate_frozen_manifest")
    env = create_task_env(settings.task)
    prompt = env.reset(settings.seed)
    if (prompt != source_manifest["fixture"]["prompt"] or settings.seed != source_manifest["fixture"]["seed"]
            or settings.task != source_manifest["fixture"]["task"]):
        raise ValueError("task_or_seed_changed_since_recording")
    checkpoint = args.worker_model_path.expanduser().resolve(strict=True)
    expected_hashes = source_manifest["checkpoint_sha256"]
    if not expected_hashes or not any(name.endswith(".safetensors") for name in expected_hashes):
        raise ValueError("missing_recorded_checkpoint_hashes")
    checkpoint_hashes = {}
    for name, digest in expected_hashes.items():
        path = (checkpoint / name).resolve(strict=True)
        if not path.is_relative_to(checkpoint) or sha256_file(path) != digest:
            raise ValueError(f"checkpoint_mismatch:{name}")
        checkpoint_hashes[name] = digest
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "frozen_prefix_competition_completion_v1" if reference else "frozen_prefix_response_v1",
        "source_jsonl": str(args.source_jsonl.resolve()), "source_jsonl_sha256": sha256_file(args.source_jsonl),
        "source_manifest": str(args.source_manifest.resolve()), "source_manifest_sha256": sha256_file(args.source_manifest),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_hashes,
        "model": settings.worker_model, "model_layers": source_manifest["model_layers"],
        "device": settings.worker_device, "dtype": settings.worker_dtype, "mps_mode": settings.worker_mps_mode,
        "fixture": source_manifest["fixture"], "plan": plan, "horizon": args.horizon,
        "sampling": "greedy", "edit_active_for_tokens": 1, "mps_memory_fraction": 0.7,
        "continuation_execution": "single_token_simulations_with_unoccupied_buffer_release",
        "provider_calls": 0, "production_policy_changed": False, "production_apply_allowed": False,
        "controller_scope": "local replay of previously nominated candidate; not a new Luna trajectory",
        "analyzer_scope": "not invoked; recorded candidate/source/token binding are fixed",
        "source_sha256": {str(p.relative_to(root)): sha256_file(p)
                          for p in (root / "SpiralInterventionLab").rglob("*") if p.suffix in (".py", ".txt")},
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    _write_json(args.output_dir / "status.json", {"status": "running"})
    started, reports, rebuilt = time.monotonic(), [], []
    try:
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        if settings.worker_device == "mps":
            torch.mps.set_per_process_memory_fraction(0.7)
        model = load_worker_model(settings.worker_model, model_path=checkpoint,
            device=settings.worker_device, dtype=settings.worker_dtype, hf_offline=True,
            mps_mode=settings.worker_mps_mode)
        if model.cfg.n_layers != source_manifest["model_layers"]:
            raise ValueError("model_depth_changed")
        worker = build_hooked_transformer_worker_runtime(model, env, seed=settings.seed,
            activation_surface_profile=settings.activation_surface_profile,
            max_diagnostic_calls_per_run=settings.max_diagnostic_calls_per_run,
            diagnostic_result_window=settings.diagnostic_result_window)
        worker.reset(prompt)
        if worker.max_generated_tokens != source_manifest["fixture"]["max_new_tokens"]:
            raise ValueError("generation_ceiling_changed")
        points = {point["worker_step"]: point for point in plan["points"]}
        observations = {e["step"]: e for e in events if e.get("event") == "controller_observation"}
        frozen = None
        for step in range(1, max(points) + 1):
            if worker.done():
                raise ValueError("reconstruction_stopped_early")
            worker.step()
            worker.tick_ttl()
            worker.cleanup_expired()
            if worker.final_text() != observations[step]["generated_tail"]:
                raise ValueError(f"recorded_prefix_diverged_at_step:{step}")
            rebuilt.append({"worker_step": step, "prefix": worker.final_text(), "matched": True})
            memory = _release_buffers(settings.worker_device)
            if step == plan["candidate_anchor_point"]["worker_step"]:
                context = state_identity(worker)
                if context != plan["recorded_context_id"]:
                    raise ValueError(f"recorded_anchor_state_mismatch:{context}")
                frozen = freeze_candidate(worker, plan["candidate"])
                torch.save(frozen.source_tensor.detach().cpu(), args.output_dir / "frozen_source.pt")
                _write_json(args.output_dir / "frozen_candidate.json", {
                    "candidate_id": frozen.candidate_id, "descriptor": frozen.descriptor,
                    "edit": frozen.edit, "original_source_expr": frozen.original_source_expr,
                    "source_tensor_identity": frozen.source_identity,
                    "source_file_sha256": sha256_file(args.output_dir / "frozen_source.pt"),
                    "target_piece": frozen.token_piece, "target_piece_token_id": frozen.token_id,
                    "recorded_anchor_context_matched": True, "production_apply_allowed": False,
                })
            if step not in points:
                continue
            binding = None
            if reference:
                if state_identity(worker) != reference["measurement_context_id"] or frozen.candidate_id != reference["candidate_id"]:
                    raise ValueError("completion_reference_state_or_candidate_changed")
                surface = plan["candidate"]["target_piece_binding_manifest"]["canonical_surface_variant"]
                binding = completion_binding(worker.codec, term=frozen.term, surface=surface,
                    first_token_id=frozen.token_id, competitor_token_id=reference["competitor_token_id"])
                _write_json(args.output_dir / "completion_plan.json", {
                    "binding": binding, "reference": reference, "horizon": args.horizon,
                    "planned_total_replays": 4 + 2 + 2 * (len(binding["token_ids"]) - 2),
                    "production_apply_allowed": False, "candidate_id": frozen.candidate_id})
            report = measure_prefix(worker, frozen, horizon=args.horizon, completion=binding)
            report.update(selection_reason=points[step]["selection_reason"], memory_before=memory)
            for label in ("baseline", "edited"):
                outcome = report.get(label)
                if outcome:
                    outcome["legacy_score"] = env.score(outcome["text"]) if outcome["stopped"] else None
                    outcome["legacy_task_done"] = env.done(outcome["text"]) if outcome["stopped"] else None
                    outcome["task_feedback"] = env.task_feedback(outcome["text"]) if outcome["stopped"] else None
            report["memory_after"] = _release_buffers(settings.worker_device)
            reports.append(report)
            with (args.output_dir / "rows.jsonl").open("a") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")
            print(json.dumps({k: report.get(k) for k in ("worker_step", "prefix", "status", "metrics", "continuation_identical", "error")}), flush=True)
            if report["status"] != "complete":
                raise ValueError("prefix_measurement_incomplete")
        _write_json(args.output_dir / "summary.json", {
            "status": "complete", "prefix_count": len(reports), "candidate_count": 1,
            "physical_replay_count": sum(r.get("total_physical_replay_count", r["physical_replay_count"]) for r in reports),
            "model_forward_token_count": sum(r.get("total_model_forward_token_count", r["model_forward_token_count"]) for r in reports),
            "all_state_restored": all(r["state_restored"] for r in reports),
            "all_continuations_identical": all(r["continuation_identical"] for r in reports),
            "whole_term_present_in_edited": sum(r["edited"]["whole_term_present"] for r in reports),
            "points": [{k: r[k] for k in ("worker_step", "prefix", "metrics", "baseline", "edited", "completion_probe") if k in r}
                       for r in reports],
            "provider_calls": 0, "production_apply_allowed": False, "certified_for_apply": False,
        })
        _write_json(args.output_dir / "status.json", {"status": "complete", "seconds": time.monotonic() - started,
            "rows_sha256": sha256_file(args.output_dir / "rows.jsonl"), "reconstructed_prefixes": rebuilt})
    except Exception as exc:
        _write_json(args.output_dir / "status.json", {"status": "incomplete", "error": f"{type(exc).__name__}:{exc}",
            "seconds": time.monotonic() - started, "completed_prefixes": len(reports), "reconstructed_prefixes": rebuilt})
        raise


if __name__ == "__main__":
    main()
