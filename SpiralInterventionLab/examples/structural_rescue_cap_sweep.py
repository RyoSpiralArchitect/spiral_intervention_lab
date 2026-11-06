"""Calibration-only cap sweep for the norm-relative structural write contract.

Measures the reach curve of the contract's readout push: each eligible source
token's last-layer readout recipe at every frozen cap, on the three v1
calibration cases only. One B0 branch per case supplies the fixed baseline
winner. No provider, no policies, no confirmation or rescue cases. Freeze with
--prepare, then execute with --run and the printed manifest hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from ..runtime import clause_specificity, clause_structure, structural_contract
from ..runtime.response_probe import tensor_identity
from ..tasks.structural_rescue import CASES, StructuralRescueEnv, benchmark_manifest
from .digit_transform_e2e import (
    build_default_activation_surface_catalog, build_hooked_transformer_worker_runtime, load_worker_model,
)
from .replay_orphan_blueprint_ceiling import _release_memory, verify_checkpoint
from .structural_rescue_benchmark import Meter, capture_sources, readout
from .structural_rescue_benchmark_v2 import CAST_TOLERANCE, branch, required_terms
from .structural_rescue_screen import DISK_FLOOR_GIB


CAPS = (0.04, 0.06, 0.08, 0.10, 0.12)
CALIBRATION_CASES = tuple(c.case_id for c in CASES if c.split == "calibration")


def manifest() -> dict[str, Any]:
    v1 = benchmark_manifest()
    return {"version": "structural_rescue_cap_sweep_v1",
            "cases": [c for c in v1["cases"] if c["case_id"] in CALIBRATION_CASES],
            "confirmation_status": "calibration_only_reach_measurement_not_a_rescue_split",
            "relation_to_v2": "same_cases_boundary_and_readout_recipe_rule; cap_varied; "
                              "no_patch_recipes_policies_or_provider",
            "contract": {"id": structural_contract.CONTRACT,
                         "readout_layer_rule": "n_layers-1",
                         "readout_alpha_rule": "cap*||h_boundary||",
                         "source_rule": "every_eligible_source_token_in_source_order",
                         "ttl_steps": 1, "edits_per_branch": 1, "site": "resid_pre", "token": "last",
                         "guard": "per_position_hook_output_minus_input_vs_input_norm",
                         "cast_tolerance": CAST_TOLERANCE,
                         "structural_lexemes": sorted(clause_structure.SOURCE_LEXEMES),
                         "protected_rule": "structural_lexemes_inside_required_terms"},
            "caps": list(CAPS),
            "max_words": v1["max_words"], "max_generated_tokens": v1["max_generated_tokens"],
            "boundary_rule": v1["boundary_rule"],
            "baseline_branch_per_case": "b0_once; two_repeats_before_every_branch",
            "diagnostic_forward_cap_per_branch": 24, "continuation_forward_cap_per_branch": 64,
            "disk_floor_gib": DISK_FLOOR_GIB,
            "protocol_deviations": ["disk_floor_6_gib_not_v1_10_gib_user_directed_2026_11_03_swap_headroom"],
            "apply_scope": "restored_contract_diagnostic_branch_only", "production_apply_allowed": False}


def manifest_bytes() -> bytes:
    return (json.dumps(manifest(), indent=2, sort_keys=True) + "\n").encode()


def sweep_recipes(worker: Any, task: StructuralRescueEnv, layer: int, cap: float) -> tuple[list[dict], Any]:
    """One readout recipe per eligible source token at this cap, with its policy."""
    sources = structural_contract.eligible_sources(worker, required_terms(task))
    if len(sources) > structural_contract.MAX_SOURCES:
        raise ValueError("sweep_covers_every_eligible_source_but_catalog_truncates_at_three")
    policy = structural_contract.policy_for(worker, required_terms(task), cap=cap)
    last = structural_contract.surface(worker, layer)
    return structural_contract.readout_recipes(worker, sources, last, cap=cap), policy


def summarize(cases: list[dict]) -> dict[str, Any]:
    """Reach table: per case, per source piece, per cap."""
    rows = []
    for case in cases:
        b0 = case["branches"].get("b0", {})
        for branch in case["branches"].get("sweep", []):
            if branch["status"] != "ok":
                rows.append({**{k: branch[k] for k in ("cap", "source_piece", "recipe_id")},
                             "case_id": case["case_id"], "status": branch["status"]})
                continue
            target = next((r for r in branch["readout"]["rows"]
                           if r["token_id"] == branch["source_token_id"]), {})
            rows.append({"case_id": case["case_id"], "cap": branch["cap"],
                         "source_piece": branch["source_piece"], "recipe_id": branch["recipe_id"],
                         "status": "ok",
                         "first_token_changed": (branch["first_token_id"] != b0["first_token_id"]
                                                 if b0.get("status") == "ok" else None),
                         "joint_success": branch["joint_success"],
                         "baseline_gap": target.get("baseline_gap_to_fixed_winner"),
                         "gap_closure": target.get("gap_closure_fraction"),
                         "relative_write_norm": (branch["runtime_telemetry"][0] or {}).get("relative_write_norm")
                         if branch["runtime_telemetry"] else None})
    return {"reach_rows": rows}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--worker-model-path", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.7)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    frozen = manifest_bytes()
    digest = hashlib.sha256(frozen).hexdigest()
    if args.prepare:
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_bytes(frozen)
        print(json.dumps({"manifest_sha256": digest, "output_dir": str(output)}))
        return 0
    if args.manifest_sha256 != digest or (output / "manifest.json").read_bytes() != frozen:
        raise ValueError("predeclared_manifest_hash_required_and_must_match")
    if args.worker_model_path is None or args.checkpoint_manifest is None:
        parser.error("--run requires local model and checkpoint manifest")
    if not 60 <= args.wall_seconds <= 7200:
        raise ValueError("wall_seconds must be between 60 and 7200")
    if not 0 < args.mps_memory_fraction <= 0.85:
        raise ValueError("mps_memory_fraction must be >0 and <=0.85")
    result_path = output / "results.json"
    for path in (result_path, output / "execution_plan.json"):
        if path.exists():
            raise FileExistsError(path)
    code_hashes = capture_sources(output)
    execution = {"manifest_sha256": digest, "device": args.device, "dtype": args.dtype,
                 "caps": list(CAPS), "case_ids": list(CALIBRATION_CASES),
                 "wall_seconds": args.wall_seconds, "mps_memory_fraction": args.mps_memory_fraction,
                 "code_sha256": code_hashes}
    (output / "execution_plan.json").write_text(json.dumps(execution, indent=2))
    started = time.monotonic()
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest, "checkpoint_sha256_verified": None,
              "device": args.device, "dtype": args.dtype, "surface_catalog": [], "cases": [],
              "scope": "calibration_reach_curve_not_a_rescue_or_population_efficacy_result",
              "production_apply_allowed": False}

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["summary"] = summarize(report["cases"])
        temp = result_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temp.replace(result_path)

    persist()
    worker = None
    try:
        checkpoint = json.loads(args.checkpoint_manifest.read_text())["checkpoint_sha256"]
        report["checkpoint_sha256_verified"] = verify_checkpoint(args.worker_model_path, checkpoint)
        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        if args.device == "mps":
            torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
        model = load_worker_model(args.worker_model, model_path=args.worker_model_path, device=args.device,
                                  dtype=args.dtype, hf_offline=True, mps_mode="conservative")
        last_layer = model.cfg.n_layers - 1
        catalog = [s for s in build_default_activation_surface_catalog(model, layers=[last_layer])
                   if s["target"].get("site") == "resid_pre" and s["target"]["token"]["mode"] == "last"]
        if [s["target"]["layer"] for s in catalog] != [last_layer]:
            raise ValueError("contract_resid_surface_unavailable")
        report["surface_catalog"] = catalog
        for case in CASES:
            if case.case_id not in CALIBRATION_CASES:
                continue
            task = StructuralRescueEnv(case)
            prompt = task.reset(0)
            worker = build_hooked_transformer_worker_runtime(model, task, seed=0, surface_catalog=catalog,
                clause_structure_diagnostics=True, clause_structure_operator_diagnostics=True,
                max_diagnostic_calls_per_run=12)
            worker.reset(prompt)
            meter = Meter(worker, deadline=started + args.wall_seconds, disk_path=output,
                          disk_floor_gib=DISK_FLOOR_GIB)
            row = {"case_id": case.case_id, "split": case.split, "branches": {"sweep": []}}
            row["setup_forwards"] = meter.counts
            report["cases"].append(row)
            with meter.install():
                recipient = case.frames[0][3]
                while not worker.done() and len(worker._output_token_ids()) < 32:
                    worker.step()
                    if re.search(r"\bto\s+" + re.escape(recipient) + r"$", worker.final_text()):
                        break
                boundary = bool(re.search(r"\bto\s+" + re.escape(recipient) + r"$", worker.final_text())) and not worker.done()
                row.update(prefix=worker.final_text(), boundary_found=boundary,
                           setup_forwards=meter.counts, tokens_identity=tensor_identity(worker._current_token_tensor()))
                if not boundary:
                    # The sweep does not decode a fallback: a missing boundary is reported, not rescued.
                    if worker.done():
                        row["b0_terminal"] = task.evaluate(worker.final_text())
                    persist()
                    continue
                meter.boundary_length = worker._current_token_tensor().shape[-1]
                clause_specificity.align_cache(worker)
                observation = worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])[0]
                if observation["status"] != "ok":
                    raise RuntimeError("structure_observation_failed")
                token_ids = [r["token_id"] for r in observation["structure_readout_rows"]]
                row.update(comparison_eligible=True, shared_setup_forwards=dict(meter.counts),
                           eligible_sources=[{k: s[k] for k in ("token_id", "piece", "source_positions")}
                                             for s in structural_contract.eligible_sources(worker, required_terms(task))])
                meter.reset("diagnostic")
                b0 = {}
                row["branches"]["b0"] = b0
                b0_raw = record(worker, task, None, meter, b0, None)
                for cap in CAPS:
                    recipes, policy = sweep_recipes(worker, task, last_layer, cap)
                    for recipe in recipes:
                        meter.check_limits()
                        meter.reset("diagnostic")
                        outcome = {"cap": cap, "recipe_id": recipe["recipe_id"],
                                   "source_piece": recipe["sources"][0]["piece"],
                                   "source_token_id": recipe["sources"][0]["token_id"]}
                        row["branches"]["sweep"].append(outcome)
                        raw = record(worker, task, recipe, meter, outcome, policy, cap=cap)
                        outcome["readout"] = readout(b0_raw, raw, token_ids)
                        outcome["first_token_changed_vs_b0"] = outcome["first_token_id"] != b0["first_token_id"]
                        persist()
                        print(json.dumps({"case": case.case_id, "cap": cap,
                            "source": outcome["source_piece"], "joint_success": outcome["joint_success"],
                            "first_token_changed": outcome["first_token_changed_vs_b0"],
                            "output": outcome["output"]}), flush=True)
            worker.runtime_state.clear_edits()
            # Each case's worker keeps a full activation cache on the device.
            worker.runtime_state.last_cache = None
            _release_memory()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        persist()
        raise
    finally:
        if worker is not None:
            worker.runtime_state.clear_edits()
    persist()
    return 0


def record(worker, task, recipe, meter, destination: dict, policy,
           *, cap: float = structural_contract.DEFAULT_RELATIVE_WRITE_CAP):
    """Retain attempted work even if simulation or a contract check fails."""
    destination["status"] = "running"
    try:
        outcome, raw = branch(worker, task, recipe, meter, policy, cap=cap)
        destination.update(outcome)
        return raw
    except Exception as exc:
        destination.update(status="error", error_type=type(exc).__name__, error=str(exc)[:300],
                           forwards=dict(meter.counts), production_apply_allowed=False)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
