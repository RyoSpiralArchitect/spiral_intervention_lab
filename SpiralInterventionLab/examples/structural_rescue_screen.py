"""Pre-registered B0-only screen for baseline-failing structural rescue cases.

--prepare freezes the generator, seed, count, selection rule and follow-up
before any model call. --run decodes each candidate once at the v2 boundary,
without edits, exactly as the v2 B0 branch does, and records which candidates
meet the rule. The selected cases then feed the v2 contract pilot unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

from ..tasks.structural_rescue import RescueCase, StructuralRescueEnv
from ..tasks.structural_rescue_generator import (
    CONNECTIVES, DEADLINES, FAMILIES, GENERATOR_VERSION, NAMES, OBJECTS, VERBS, candidate_rows, generate,
)
from .digit_transform_e2e import (
    build_default_activation_surface_catalog, build_hooked_transformer_worker_runtime, load_worker_model,
)
from .replay_orphan_blueprint_ceiling import _release_memory, verify_checkpoint
from .structural_rescue_benchmark import ForwardLimit, Meter, capture_sources, prepare_unembed_storage


SEED = 20261103
COUNT = 72
SELECTION_LIMIT = 12
SELECTION_RULE = "b0_joint_failure_with_natural_boundary_first_12_in_generation_order"
# v1 stops below 10 GiB free. Here macOS swap takes free disk to about 8 GiB
# while the model is loaded, so the user set 6 GiB for the contract pilot
# and this screen (2026-11-03).
DISK_FLOOR_GIB = 6


def manifest() -> dict[str, Any]:
    return {"version": "structural_rescue_screen_v1", "generator": GENERATOR_VERSION,
            "seed": SEED, "count": COUNT, "candidates": candidate_rows(generate(SEED, COUNT)),
            "pools": {"names": NAMES, "objects": OBJECTS, "verbs": VERBS, "deadlines": DEADLINES,
                      "connectives": CONNECTIVES, "families": FAMILIES},
            "screen": "greedy_b0_from_the_v2_boundary_no_edits_no_controller",
            "boundary_rule": "first_natural_prefix_ending_in_to_first_source_recipient",
            "selection_rule": SELECTION_RULE, "selection_limit": SELECTION_LIMIT,
            "selected_split": "rescue",
            "follow_up": "structural_rescue_benchmark_v2_unchanged_contract_all_policies_on_selected_cases",
            "analysis_plan": {"primary": "joint_success_count_per_policy_on_the_rescue_split",
                              "comparison": "controller_vs_fixed_readout_vs_fixed_vs_random_paired_by_case",
                              "harm_check": "losses_vs_b0_on_the_v1_confirmation_split",
                              "b0_reproduction": "a_selected_case_where_b0_succeeds_is_reported_not_dropped",
                              "inference": "descriptive_counts_only_no_significance_claims_at_n_le_12"},
            "max_words": 18, "max_generated_tokens": 64, "disk_floor_gib": DISK_FLOOR_GIB,
            "production_apply_allowed": False}


def manifest_bytes() -> bytes:
    return (json.dumps(manifest(), indent=2, sort_keys=True) + "\n").encode()


def selected_cases(screen_dir: Path) -> tuple[tuple[RescueCase, ...], dict[str, Any]]:
    """Rebuild the selected cases and bind them to the frozen screen receipts."""
    frozen = manifest_bytes()
    if (screen_dir / "manifest.json").read_bytes() != frozen:
        raise ValueError("screen_manifest_differs_from_generator")
    results_bytes = (screen_dir / "results.json").read_bytes()
    results = json.loads(results_bytes)
    if results.get("status") != "complete" or results.get("manifest_sha256") != hashlib.sha256(frozen).hexdigest():
        raise ValueError("screen_incomplete_or_unbound")
    by_id = {case.case_id: case for case in generate(SEED, COUNT)}
    cases = tuple(RescueCase(case_id, "rescue", by_id[case_id].source, by_id[case_id].frames,
                             by_id[case_id].deadline) for case_id in results["selected_case_ids"])
    return cases, {"screen_manifest_sha256": hashlib.sha256(frozen).hexdigest(),
                   "screen_results_sha256": hashlib.sha256(results_bytes).hexdigest(),
                   "selected_case_ids": list(results["selected_case_ids"])}


def load_selected_cases(screen_dir: Path) -> tuple[tuple[RescueCase, ...], dict[str, Any]]:
    """Dispatch on the frozen screen manifest version; v1 receipts stay readable."""
    version = json.loads((screen_dir / "manifest.json").read_text()).get("version")
    if version == "structural_rescue_screen_v4":
        from .structural_rescue_screen_v4 import selected_cases as pick
    elif version == "structural_rescue_screen_v3":
        from .structural_rescue_screen_v3 import selected_cases as pick
    elif version == "structural_rescue_screen_v2":
        from .structural_rescue_screen_v2 import selected_cases as pick
    elif version == "structural_rescue_screen_v1":
        pick = selected_cases
    else:
        raise ValueError(f"unknown_screen_manifest_version: {version}")
    return pick(screen_dir)


def screen_case(model: Any, catalog: list[dict], case: RescueCase, *,
                meter_options: dict | None = None, attempt: dict | None = None) -> dict[str, Any]:
    task = StructuralRescueEnv(case)
    worker = build_hooked_transformer_worker_runtime(model, task, seed=0, surface_catalog=catalog,
        clause_structure_diagnostics=True, clause_structure_operator_diagnostics=True,
        max_diagnostic_calls_per_run=12)
    worker.reset(task.reset(0))
    meter = Meter(worker, **meter_options) if meter_options is not None else None
    try:
        if meter is None:
            return _decode_case(worker, task, case)
        with meter.install():
            row = _decode_case(worker, task, case, meter=meter)
        row["forwards"] = deepcopy(meter.counts)
        return row
    except Exception as exc:
        if attempt is not None:
            attempt.update(status="error", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        if attempt is not None and meter is not None:
            attempt["forwards"] = deepcopy(meter.counts)
        worker.runtime_state.clear_edits()
        worker.runtime_state.last_cache = None


def _decode_case(worker: Any, task: StructuralRescueEnv, case: RescueCase, *,
                 meter: Meter | None = None) -> dict[str, Any]:
    pattern = r"\bto\s+" + re.escape(case.frames[0][3]) + r"$"
    while not worker.done() and len(worker._output_token_ids()) < 32:
        worker.step()
        if re.search(pattern, worker.final_text()):
            break
    boundary = bool(re.search(pattern, worker.final_text())) and not worker.done()
    prefix = worker.final_text()
    text = prefix
    if not worker.done():
        if meter is not None:
            meter.phase = "branch"
            meter.boundary_length = worker._current_token_tensor().shape[-1]
            # This screen discards the worker after its unedited continuation.
            # Do not clone a full prefix cache solely to restore that disposable state.
            worker.runtime_state.last_cache = None
            worker.runtime_state.last_logits = None
            _release_memory()
        simulated = worker._simulate_decode(max_new_tokens=64 - len(worker._output_token_ids()), top_k=6,
                                            score_candidate_text=False, score_observer_check=False)
        if simulated is None:
            raise RuntimeError(worker._last_simulate_decode_error)
        text = worker.codec.decode(worker._output_token_ids() + simulated["continuation_token_ids"])
    return {"case_id": case.case_id, "boundary_found": boundary, "prefix": prefix, **task.evaluate(text)}


def select(rows: list[dict]) -> list[str]:
    return [r["case_id"] for r in rows
            if r.get("boundary_found") is True and r.get("joint_success") is False][:SELECTION_LIMIT]


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
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    args = parser.parse_args(argv)
    return execute(args, seed=SEED, count=COUNT, verbs=VERBS, frozen=manifest_bytes())


def execute(args: Any, *, seed: int, count: int, verbs: tuple[str, ...], frozen: bytes,
            families: tuple[str, ...] = FAMILIES) -> int:
    protocol = json.loads(frozen)
    if any(getattr(args, key) != value for key, value in protocol.get("run", {}).items()):
        raise ValueError("screen_execution_profile_differs_from_frozen_protocol")
    disk_floor = protocol.get("disk_floor_gib", DISK_FLOOR_GIB)
    output = args.output_dir.expanduser().resolve()
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
    if not 0 < args.mps_memory_fraction <= 0.85:
        raise ValueError("mps_memory_fraction must be >0 and <=0.85")
    result_path = output / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest, "code_sha256": capture_sources(output), "rows": []}
    report.update(device=args.device, dtype=args.dtype, production_apply_allowed=False)
    started = time.monotonic()

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["selected_case_ids"] = select(report["rows"])
        temp = result_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temp.replace(result_path)

    persist()
    try:
        source = json.loads(args.checkpoint_manifest.read_text())
        checkpoint = source.get("checkpoint_sha256") or source.get("checkpoint_sha256_verified")
        if "checkpoint_sha256" in protocol and checkpoint != protocol["checkpoint_sha256"]:
            raise ValueError("screen_checkpoint_differs_from_frozen_protocol")
        if shutil.disk_usage(output).free < disk_floor * 2**30:
            raise ForwardLimit(f"disk_below_predeclared_{disk_floor:g}_gib_floor")
        report["checkpoint_sha256_verified"] = verify_checkpoint(args.worker_model_path, checkpoint)
        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        if args.device == "mps":
            torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
        model = load_worker_model(args.worker_model, model_path=args.worker_model_path, device=args.device,
                                  dtype=args.dtype, hf_offline=True, mps_mode="conservative")
        if args.device == "mps":
            _release_memory()
            prepare_unembed_storage(model)
            torch.mps.synchronize()
            _release_memory()
        layers = sorted({model.cfg.n_layers // 2, model.cfg.n_layers - 1})
        catalog = [s for s in build_default_activation_surface_catalog(model, layers=layers)
                   if s["target"].get("site") == "resid_pre" and s["target"]["token"]["mode"] == "last"]
        for case in generate(seed, count, verbs, families):
            if shutil.disk_usage(output).free < disk_floor * 2**30:
                raise ForwardLimit(f"disk_below_predeclared_{disk_floor:g}_gib_floor")
            options = ({"deadline": started + protocol["wall_seconds"], "disk_path": output,
                        "disk_floor_gib": disk_floor} if "forward_caps" in protocol else None)
            attempt = {"case_id": case.case_id, "status": "running"} if options is not None else None
            if attempt is not None:
                report["rows"].append(attempt)
            row = screen_case(model, catalog, case, meter_options=options, attempt=attempt)
            _release_memory()
            if attempt is None:
                report["rows"].append(row)
            else:
                attempt.update(row, status="ok")
            persist()
            print(json.dumps({k: row[k] for k in ("case_id", "boundary_found", "joint_success",
                                                  "role_status", "output")}), flush=True)
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        persist()
        raise
    persist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
