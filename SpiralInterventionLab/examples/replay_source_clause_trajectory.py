"""Trace two predeclared source patches at one naturally reconstructed prefix.

Normal runtime caps, two repeats, self-copy and radial controls. No provider calls
or production apply. A backend/dtype change is an explicit numerical replication.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Any

from ..runtime import clause_actuation, clause_choice, clause_components, clause_specificity, clause_trajectory, diagnostic_budget
from ..runtime.response_probe import tensor_identity
from .digit_transform_e2e import build_hooked_transformer_worker_runtime, create_task_env, load_worker_model
from .replay_orphan_blueprint_ceiling import verify_checkpoint
from .replay_source_clause_diagnostic import measure_current_prefix, source_prefixes


def verify_reference_prefix(worker: Any, reference: dict) -> None:
    if reference.get("status") != "complete" or len(reference.get("measurements", [])) != 1:
        raise ValueError("a complete single-prefix reference is required")
    measurement = reference["measurements"][0]
    if (measurement["prefix"] != worker.final_text()
            or reference["cache_alignment"]["tokens_identity"] != tensor_identity(worker._current_token_tensor())):
        raise ValueError("reference_prefix_or_tokens_differ")


def _verify_source_bindings(worker: Any, reference: dict, token_ids: list[int]) -> dict:
    measurement = reference["measurements"][0]
    keys = ("target_token_id", "target_piece", "source_token_index", "source_provenance",
            "source_localization", "source_occurrence_rule", "surfaces")
    offered = clause_actuation.offer(worker)
    bindings = []
    for tid in token_ids:
        previous = next((r["result"]["candidate_binding"] for r in measurement["operator_measurements"]
                         if r["result"].get("status") == "ok"
                         and r["result"]["candidate_binding"]["target_token_id"] == tid), None)
        current = next((c for c in offered["candidates"] if c["target_token_id"] == tid), None)
        if previous is None or current is None or any(previous[k] != current[k] for k in keys):
            raise ValueError(f"reference_source_binding_or_normal_caps_differ:{tid}")
        bindings.append({k: current[k] for k in keys})
    return {"status": "matched", "bindings": bindings, "reference_device": reference["device"],
            "reference_dtype": reference["dtype"], "tensor_values_may_differ_across_numerics": True}


def verify_reference(worker: Any, reference: dict, token_ids: list[int]) -> dict:
    """Default path stays strict: same token prefix, source positions, sites and caps."""
    verify_reference_prefix(worker, reference)
    return _verify_source_bindings(worker, reference, token_ids)


def next_boundary_plan(reference: dict, prefixes: dict, position: int,
                       source_sha256: str, manifest_sha256: str) -> dict:
    if reference.get("status") != "complete" or len(reference.get("measurements", [])) != 1:
        raise ValueError("a complete single-prefix reference is required")
    measurement = reference["measurements"][0]
    anchor = measurement.get("generated_tokens")
    if (not isinstance(anchor, int) or isinstance(anchor, bool) or position != anchor + 1
            or prefixes.get(anchor) != measurement["prefix"] or position not in prefixes):
        raise ValueError("next_boundary_requires_exactly_one_recorded_natural_token")
    if (reference.get("source_jsonl_sha256") != source_sha256
            or reference.get("source_manifest_sha256") != manifest_sha256):
        raise ValueError("next_boundary_reference_source_hashes_differ")
    return {"mode": "adjacent_natural_boundary_same_source_bindings_not_same_prefix",
            "anchor_position": anchor, "measurement_position": position,
            "anchor_prefix": measurement["prefix"], "expected_prefix": prefixes[position],
            "fallback_position_search_allowed": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--reference-receipt", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--position", type=int, required=True)
    parser.add_argument("--next-natural-boundary", action="store_true",
                        help="Verify the reference prefix, then measure exactly one natural token later; no fallback search")
    parser.add_argument("--source-token-ids", required=True, help="Two exact source token IDs, declared before measurement")
    parser.add_argument("--decompose-components", action="store_true",
                        help="Also measure common and signed half-difference directions; seven conditions, nine slots")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.7)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        token_ids = [int(s) for s in args.source_token_ids.split(",")]
    except ValueError:
        parser.error("--source-token-ids requires two distinct nonnegative integers")
    if len(token_ids) != 2 or len(set(token_ids)) != 2 or min(token_ids) < 0:
        parser.error("--source-token-ids requires two distinct nonnegative integers")
    if args.position < 1 or not 1 <= args.torch_threads <= 16 or not 0 < args.mps_memory_fraction <= 1:
        parser.error("invalid position, thread count or memory fraction")
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    manifest_path = args.source_manifest.expanduser().resolve(strict=True)
    reference_path = args.reference_receipt.expanduser().resolve(strict=True)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    prompt, prefixes = source_prefixes(source_path)
    if args.position not in prefixes:
        raise ValueError("position not present in the recorded trajectory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    boundary = next_boundary_plan(
        reference, prefixes, args.position, hashlib.sha256(source_path.read_bytes()).hexdigest(),
        hashlib.sha256(manifest_path.read_bytes()).hexdigest()) if args.next_natural_boundary else None
    profile = manifest["profile"]
    task = create_task_env(profile["task"])
    if task.reset(int(profile["seed"])) != prompt:
        raise ValueError("task fixture differs from the recorded prompt")
    output.mkdir(parents=True)
    report = {"kind": "source_clause_components" if args.decompose_components else "source_clause_trajectory",
              "schema_version": 1, "status": "preflight",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (source_path, manifest_path, reference_path)},
              "profile": profile, "device": args.device, "dtype": args.dtype,
              "torch_threads": args.torch_threads, "source_token_ids": token_ids,
              "position": args.position, "expected_prefix": prefixes[args.position],
              "reconstruction": "natural_greedy_no_edit_refuse_token_prefix_mismatch",
              "selection_origin": "predeclared_offline_pair_not_controller_choice",
              "controller_calls": 0, "production_apply_allowed": False, "rollout_apply_count": 0,
              "caps": "unchanged_runtime_defaults", "diagnostic_budget_limit": 12,
              "records": [], "versions": {name: importlib.metadata.version(name)
                                            for name in ("torch", "transformers", "transformer-lens")}}
    if args.decompose_components:
        report["component_plan"] = clause_components.plan(token_ids)
    if boundary is not None:
        report["boundary_plan"] = boundary
    root = Path(__file__).resolve().parents[2]
    report["base_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    report["code_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted((root / "SpiralInterventionLab").rglob("*.py"))}

    def save():
        (output / "trajectory.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    save()
    worker = None
    counts = None
    monitors = ExitStack()
    try:
        report["checkpoint_sha256_verified"] = verify_checkpoint(model_path, manifest.get("checkpoint_sha256") or {})
        report["status"] = "loading_model"
        save()
        import torch

        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(args.torch_threads)
        if args.device == "mps":
            torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
        model = load_worker_model(profile["worker_model"], model_path=model_path, device=args.device,
                                  dtype=args.dtype, hf_offline=True, mps_mode="conservative")
        report["model_execution"] = {
            "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "n_layers": model.cfg.n_layers,
        }
        if (any(p.device.type != args.device for p in model.parameters())
                or any(str(p.dtype) != "torch." + args.dtype for p in model.parameters())):
            raise ValueError("model_device_or_dtype_differs_from_requested_numerics")
        worker = build_hooked_transformer_worker_runtime(
            model, task, seed=int(profile["seed"]), clause_structure_diagnostics=True,
            clause_structure_operator_diagnostics=True, max_diagnostic_calls_per_run=12)
        counts = monitors.enter_context(clause_trajectory.count_forwards(worker.runtime_state))
        worker.reset(prompt)
        report["status"] = "reconstructing_prefix"
        save()
        anchor_tokens = None
        while len(worker._output_token_ids()) < args.position:
            if worker.done():
                raise ValueError("worker terminated before requested prefix")
            worker.step()
            report["reconstruction_tokens"] = len(worker._output_token_ids())
            if boundary is not None and len(worker._output_token_ids()) == boundary["anchor_position"]:
                verify_reference_prefix(worker, reference)
                anchor_tokens = worker._current_token_tensor().detach().clone()
                report["verified_anchor_tokens_identity"] = tensor_identity(anchor_tokens)
            save()
        report["observed_prefix"] = worker.final_text()
        if worker.final_text() != prefixes[args.position]:
            raise ValueError("prefix differs; refusing non-matched measurement")
        if boundary is not None:
            current_tokens = worker._current_token_tensor()
            if anchor_tokens is None or not torch.equal(current_tokens[:, :-1], anchor_tokens):
                raise ValueError("next_boundary_is_not_exact_one_token_extension_of_verified_anchor")
            report["verified_boundary_tokens_identity"] = tensor_identity(current_tokens)
        report["forward_counts_before_diagnostics"] = dict(counts)
        report["cache_alignment"] = clause_specificity.align_cache(worker)
        report["structure_observation"] = measure_current_prefix(worker)
        if report["structure_observation"]["result"]["status"] != "ok":
            raise ValueError("source-clause observation failed")
        report["reference_match"] = (
            {**_verify_source_bindings(worker, reference, token_ids), "prefix_contract": boundary["mode"]}
            if boundary is not None else verify_reference(worker, reference, token_ids))
        report["status"] = "measuring"
        save()

        def record(item):
            filename = f"vectors_{len(report['records']):02d}.pt"
            torch.save(item, output / filename)
            report["records"].append({**clause_trajectory.compact_record(item), "vectors_artifact": filename,
                                      "vectors_sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest()})
            save()
            print(json.dumps({"condition": item["label"], "status": item["result"]["status"],
                              "captured_forwards": len(item["frames"])}), flush=True)

        instrument = clause_components if args.decompose_components else clause_trajectory
        records = instrument.measure(worker, token_ids, record=record)
        report["declared_diagnostic_replay_count"] = 2 + sum(r["result"]["physical_replay_count"] for r in records)
        report["comparison"] = instrument.summarize(records)
        report["choice_comparison"] = clause_choice.summarize(
            records, component_labels=clause_components.COMPONENTS if args.decompose_components else (),
            decode_token=lambda tid: worker.codec.decode([tid]))
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        monitors.close()
        if counts is not None:
            report["forward_counts_total"] = dict(counts)
            if "forward_counts_before_diagnostics" in report:
                report["diagnostic_interval_forward_counts"] = {
                    k: v - report["forward_counts_before_diagnostics"][k] for k, v in counts.items()}
        if worker is not None:
            report["diagnostic_budget_used"] = diagnostic_budget.used(worker)
            report["diagnostic_budget_left"] = diagnostic_budget.left(worker)
        report["recorded_condition_replay_count"] = sum(
            r["result"].get("physical_replay_count", 0) for r in report["records"])
        report["recorded_condition_count"] = len(report["records"])
        report["failed_condition_labels"] = [r["label"] for r in report["records"]
                                               if r["result"].get("status") != "ok"]
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
