"""Second pre-registered B0-only screen, judged by evaluator v2.

Same generator pools and boundary rule as the first screen and a fresh seed.
--prepare freezes the candidates, both selection quotas, the stopping rule and
the whole follow-up treatment (passed in as a JSON file) before any model call.
--run decodes candidates in generation order, records each boundary's first
token and the eligible sources' gaps, and stops once both quotas are met, which
cannot change which candidates "the first N in generation order" selects.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any

from ..runtime import structural_contract
from ..tasks.structural_rescue import EVALUATOR_V2, RescueCase, StructuralRescueEnv
from ..tasks.structural_rescue_generator import (
    CONNECTIVES, DEADLINES, FAMILIES, GENERATOR_VERSION, NAMES, OBJECTS, VERBS, candidate_rows, generate,
)
from . import run_hygiene
from .digit_transform_e2e import (
    build_default_activation_surface_catalog, build_hooked_transformer_worker_runtime, load_worker_model,
)
from .replay_orphan_blueprint_ceiling import _release_memory, verify_checkpoint
from .structural_rescue_benchmark import ForwardLimit, capture_sources
from .structural_rescue_benchmark_v2 import required_terms
from .structural_rescue_screen import COUNT as FIRST_COUNT, DISK_FLOOR_GIB, SEED as FIRST_SEED


VERSION = "structural_rescue_screen_v2"
SEED = 20261104
COUNT = 144
RESCUE_LIMIT = 12
HARM_LIMIT = 24
SELECTION_RULE = ("rescue: b0_joint_failure_evaluator_v2_with_natural_boundary_first_12_in_generation_order; "
                  "harm: b0_joint_success_evaluator_v2_with_natural_boundary_first_24_in_generation_order")
STOPPING_RULE = "stop_decoding_once_both_quotas_are_met"


def candidates() -> tuple[tuple[RescueCase, ...], list[str]]:
    """Generation order, minus any source the first screen already decoded.

    IDs are renamed s2_NNN so they cannot be mistaken for first-screen gen_NNN cases.
    """
    seen = {case.source for case in generate(FIRST_SEED, FIRST_COUNT)}
    drawn = [replace(c, case_id="s2_" + c.case_id.removeprefix("gen_")) for c in generate(SEED, COUNT)]
    return tuple(c for c in drawn if c.source not in seen), [c.case_id for c in drawn if c.source in seen]


def manifest(treatment: dict[str, Any]) -> dict[str, Any]:
    kept, excluded = candidates()
    return {"version": VERSION, "generator": GENERATOR_VERSION, "seed": SEED, "count": COUNT,
            "candidates": candidate_rows(kept), "excluded_repeats_of_first_screen": excluded,
            "pools": {"names": NAMES, "objects": OBJECTS, "verbs": VERBS, "deadlines": DEADLINES,
                      "connectives": CONNECTIVES, "families": FAMILIES},
            "screen": "greedy_b0_from_the_v2_boundary_no_edits_no_controller",
            "evaluator": EVALUATOR_V2, "evaluator_sha256": run_hygiene.evaluator_sha256(),
            "boundary_rule": "first_natural_prefix_ending_in_to_first_source_recipient",
            "selection_rule": SELECTION_RULE, "rescue_limit": RESCUE_LIMIT, "harm_limit": HARM_LIMIT,
            "stopping_rule": STOPPING_RULE, "recorded_per_candidate": "b0_first_token_and_eligible_source_gaps",
            "treatment": treatment, "max_words": 18, "max_generated_tokens": 64, "disk_floor_gib": DISK_FLOOR_GIB,
            "launch_headroom_gib": run_hygiene.LOAD_HEADROOM_GIB,
            "protocol_deviations": ["disk_floor_6_gib_not_v1_10_gib_user_directed_2026_11_03_swap_headroom",
                                    "launch_headroom_5_gib_user_directed_2026_11_03_concurrent_build"],
            "production_apply_allowed": False}


def manifest_bytes(treatment: dict[str, Any]) -> bytes:
    return (json.dumps(manifest(treatment), indent=2, sort_keys=True) + "\n").encode()


def select(rows: list[dict]) -> dict[str, list[str]]:
    boundary = [r for r in rows if r["boundary_found"]]
    return {"rescue": [r["case_id"] for r in boundary if not r["joint_success"]][:RESCUE_LIMIT],
            "harm": [r["case_id"] for r in boundary if r["joint_success"]][:HARM_LIMIT]}


def selected_cases(screen_dir: Path, treatment: dict[str, Any]) -> tuple[tuple, tuple, dict[str, Any]]:
    frozen = manifest_bytes(treatment)
    if (screen_dir / "manifest.json").read_bytes() != frozen:
        raise ValueError("screen_manifest_differs_from_generator")
    results_bytes = (screen_dir / "results.json").read_bytes()
    results = json.loads(results_bytes)
    if results.get("status") != "complete" or results.get("manifest_sha256") != hashlib.sha256(frozen).hexdigest():
        raise ValueError("screen_incomplete_or_unbound")
    chosen = select(results["rows"])
    if results["selected"] != chosen:
        raise ValueError("screen_selection_does_not_follow_the_rule")
    by_id = {case.case_id: case for case in candidates()[0]}
    rescue = tuple(replace(by_id[i], split="rescue") for i in chosen["rescue"])
    harm = tuple(replace(by_id[i], split="harm") for i in chosen["harm"])
    return rescue, harm, {"screen2_manifest_sha256": hashlib.sha256(frozen).hexdigest(),
                          "screen2_results_sha256": hashlib.sha256(results_bytes).hexdigest(), **chosen}


def screen_case(model: Any, catalog: list[dict], case: RescueCase) -> dict[str, Any]:
    task = StructuralRescueEnv(case)
    worker = build_hooked_transformer_worker_runtime(model, task, seed=0, surface_catalog=catalog,
        clause_structure_diagnostics=True, clause_structure_operator_diagnostics=True,
        max_diagnostic_calls_per_run=12)
    worker.reset(task.reset(0))
    pattern = r"\bto\s+" + re.escape(case.frames[0][3]) + r"$"
    while not worker.done() and len(worker._output_token_ids()) < 32:
        worker.step()
        if re.search(pattern, worker.final_text()):
            break
    boundary = bool(re.search(pattern, worker.final_text())) and not worker.done()
    prefix, text, extra = worker.final_text(), worker.final_text(), {}
    if not worker.done():
        sources = structural_contract.eligible_sources(worker, required_terms(task)) if boundary else []
        simulated = worker._simulate_decode(max_new_tokens=64 - len(worker._output_token_ids()), top_k=6,
                                            score_candidate_text=False, score_observer_check=False)
        if simulated is None:
            raise RuntimeError(worker._last_simulate_decode_error)
        text = worker.codec.decode(worker._output_token_ids() + simulated["continuation_token_ids"])
        logits = simulated["first_logits"]
        winner = int(logits.argmax())
        extra = {"b0_first_token_id": simulated["continuation_token_ids"][0], "b0_winner_id": winner,
                 "eligible_source_gaps": [{"token_id": int(s["token_id"]), "piece": s["piece"],
                                           "gap_to_top1": float(logits[winner] - logits[int(s["token_id"])])}
                                          for s in sources]}
    worker.runtime_state.clear_edits()
    # Each worker keeps a full activation cache on the device; drop it before the next one.
    worker.runtime_state.last_cache = None
    return {"case_id": case.case_id, "boundary_found": boundary, "prefix": prefix, **extra,
            **task.evaluate_v2(text)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--treatment", type=Path, required=True, help="the frozen rescue-2 treatment JSON")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--worker-model-path", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    parser.add_argument("--allow-dirty", action="store_true", help="tests only")
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    treatment = json.loads(args.treatment.read_text())
    frozen = manifest_bytes(treatment)
    digest = hashlib.sha256(frozen).hexdigest()
    if args.prepare:
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_bytes(frozen)
        print(json.dumps({"manifest_sha256": digest, "output_dir": str(output), "candidates": len(candidates()[0])}))
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
    run_hygiene.preflight(output, floor_gib=DISK_FLOOR_GIB, allow_dirty=args.allow_dirty)
    execution = {"manifest_sha256": digest, "device": args.device, "dtype": args.dtype,
                 "mps_memory_fraction": args.mps_memory_fraction, "code_sha256": capture_sources(output),
                 **run_hygiene.provenance()}
    (output / "execution_plan.json").write_text(json.dumps(execution, indent=2))
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest, "rows": []}
    started = time.monotonic()

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["selected"] = select(report["rows"])
        temp = result_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temp.replace(result_path)

    persist()
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
        layers = sorted({model.cfg.n_layers // 2, model.cfg.n_layers - 1})
        catalog = [s for s in build_default_activation_surface_catalog(model, layers=layers)
                   if s["target"].get("site") == "resid_pre" and s["target"]["token"]["mode"] == "last"]
        for case in candidates()[0]:
            chosen = select(report["rows"])
            if len(chosen["rescue"]) >= RESCUE_LIMIT and len(chosen["harm"]) >= HARM_LIMIT:
                report["stopped_early_after"] = report["rows"][-1]["case_id"]
                break
            if shutil.disk_usage(output).free < DISK_FLOOR_GIB * 2**30:
                raise ForwardLimit(f"disk_below_predeclared_{DISK_FLOOR_GIB:g}_gib_floor")
            row = screen_case(model, catalog, case)
            _release_memory()
            row["telemetry"] = run_hygiene.telemetry(output)
            report["rows"].append(row)
            persist()
            print(json.dumps({k: row.get(k) for k in ("case_id", "boundary_found", "joint_success",
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
