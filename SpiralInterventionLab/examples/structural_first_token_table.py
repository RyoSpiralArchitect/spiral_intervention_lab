"""First-token table of every structural contract push at a natural boundary.

A TTL-1 edit is rolled back after the first decoded token and every later step
recomputes the whole sequence, so a restored branch's output is fixed by its
first token. At each case's boundary this runner measures every catalog push at
every listed cap with one edited first-token forward, then decodes one restored
edited branch (the PR #39 path) per distinct first token for its outcome, and
repeats that decode on a pre-declared subset to check the claim. Policies are
then lookups in the table. Freeze with --prepare, run parts with --run.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from ..runtime import clause_specificity, clause_structure, structural_contract
from ..runtime.compiler import compile_expr
from ..runtime.response_probe import state_identity, tensor_identity
from ..tasks.structural_rescue import CASES, EVALUATOR_V2, RescueCase, StructuralRescueEnv
from ..tasks.structural_rescue_generator import candidate_rows, generate
from . import run_hygiene
from .digit_transform_e2e import (
    build_default_activation_surface_catalog, build_hooked_transformer_worker_runtime, load_worker_model,
)
from .replay_orphan_blueprint_ceiling import _release_memory, verify_checkpoint
from .structural_rescue_benchmark import ForwardLimit, Meter, capture_sources, readout
from .structural_rescue_benchmark_v2 import CAST_TOLERANCE, branch, compact_packet, required_terms
from .structural_rescue_screen import COUNT as SCREEN1_COUNT, DISK_FLOOR_GIB, SEED as SCREEN1_SEED


VERSION = "structural_first_token_table_v1"
SWEEP_CAPS = (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12)
# The outcome of a first token is decoded at the cap closest to this one, ties to the smaller.
OUTCOME_CAP = 0.08
SPOT_CHECK_EVERY = 6
MAX_WALL_SECONDS = 4 * 3600
SCREEN1_RECEIPTS = ("precheck_r4", "pilot", "rescue", "screen_r2")


def screen1_cases(screen_dir: Path) -> tuple[dict, dict]:
    from .structural_rescue_screen import manifest_bytes as screen_manifest_bytes
    frozen = screen_manifest_bytes()
    if (screen_dir / "manifest.json").read_bytes() != frozen:
        raise ValueError("screen_manifest_differs_from_generator")
    raw = (screen_dir / "results.json").read_bytes()
    results = json.loads(raw)
    if results.get("status") != "complete" or results.get("manifest_sha256") != hashlib.sha256(frozen).hexdigest():
        raise ValueError("screen_incomplete_or_unbound")
    return results, {"screen1_manifest_sha256": hashlib.sha256(frozen).hexdigest(),
                     "screen1_results_sha256": hashlib.sha256(raw).hexdigest()}


def case_set(name: str, *, screen1_dir: Path, screen2_dir: Path | None = None,
             screen2_plan: dict | None = None) -> tuple[tuple[RescueCase, ...], dict]:
    """known16: the 16 PR #39 cases. pool62: calibration + screen-1 boundary cases.
    rescue2: screen-2 rescue and harm quotas plus the six v1 confirmation cases."""
    results, binding = screen1_cases(screen1_dir)
    by_id = {case.case_id: case for case in generate(SCREEN1_SEED, SCREEN1_COUNT)}
    selected = results["selected_case_ids"]
    if name == "known16":
        rescue = tuple(RescueCase(i, "rescue", by_id[i].source, by_id[i].frames, by_id[i].deadline) for i in selected)
        return CASES + rescue, {"name": name, **binding}
    if name == "pool62":
        screened = tuple(RescueCase(r["case_id"], "screen1_rescue" if r["case_id"] in selected else "screen1_other",
                                    by_id[r["case_id"]].source, by_id[r["case_id"]].frames, by_id[r["case_id"]].deadline)
                         for r in results["rows"] if r["boundary_found"])
        return tuple(c for c in CASES if c.split == "calibration") + screened, {
            "name": name, "excluded": "v1_confirmation_cases_kept_for_harm_replay", **binding}
    if name == "rescue2":
        from .structural_rescue_screen_v2 import selected_cases
        rescue, harm, screen2 = selected_cases(screen2_dir, screen2_plan)
        confirmation = tuple(c for c in CASES if c.split == "confirmation")
        return rescue + harm + confirmation, {"name": name, "screen2": screen2, **binding}
    raise ValueError("unknown case set")


def manifest(cases: tuple, binding: dict, *, caps: list[float], patch_caps: list[float], patch_scope: str,
             packet_caps: list[float], references: dict) -> dict[str, Any]:
    return {"version": VERSION, "cases": candidate_rows(cases), "case_set": binding,
            "caps": caps, "readout_rule": "readout_push_for_each_of_the_first_three_eligible_source_tokens",
            "patch_caps": patch_caps, "patch_scope": patch_scope, "packet_caps": packet_caps,
            "measurement": "one_edited_first_token_forward_per_push_raw_full_vocabulary_logits_prebuilt_packet",
            "outcome_rule": "one_restored_pr39_path_edited_branch_per_distinct_first_token_cap_closest_to_0.08_ties_smaller",
            "spot_check": f"pool_index_mod_{SPOT_CHECK_EVERY}_also_decode_a_second_push_with_the_same_first_token",
            "evaluator": EVALUATOR_V2, "evaluator_sha256": run_hygiene.evaluator_sha256(),
            "secondary_evaluator": "structural_rescue_v1_evaluate", "contract": structural_contract.CONTRACT,
            "cast_tolerance": CAST_TOLERANCE, "realized_write_check": "readouts_two_sided_abs_rel_minus_cap",
            "boundary_rule": "first_natural_prefix_ending_in_to_first_source_recipient",
            "max_generated_tokens": 64, "baseline_repeats": 2, "references": references,
            "disk_floor_gib": DISK_FLOOR_GIB, "launch_headroom_gib": run_hygiene.LOAD_HEADROOM_GIB,
            "protocol_deviations": ["disk_floor_6_gib_not_v1_10_gib_user_directed_2026_11_03_swap_headroom"],
            "apply_scope": "restored_contract_diagnostic_branch_only", "production_apply_allowed": False}


def manifest_bytes(*args, **kwargs) -> bytes:
    return (json.dumps(manifest(*args, **kwargs), indent=2, sort_keys=True) + "\n").encode()


def outcome_cap(caps: list[float]) -> float:
    return min(caps, key=lambda cap: (round(abs(cap - OUTCOME_CAP), 9), cap))


def push(worker: Any, recipe: dict, meter: Meter, policy: Any, cap: float, packet: dict) -> tuple[dict, Any]:
    """One edited forward at the boundary, rolled back after the first token."""
    command = structural_contract.command(recipe)
    if tensor_identity(compile_expr(command["edits"][0]["source"]["expr"])(structural_contract._context(worker))) \
            != recipe["input_identity"]:
        raise ValueError("recipe_source_changed_before_push")
    meter.phase, meter.capture_edited, meter.raw_first = "branch", True, None
    simulated = worker._simulate_decode(max_new_tokens=1, top_k=6, command=command, policy_override=policy,
                                        score_candidate_text=False, score_observer_check=False, packet=packet)
    if simulated is None or meter.raw_first is None:
        raise RuntimeError(f"push_failed: {worker._last_simulate_decode_error}")
    if worker._collect_active_edits():
        raise RuntimeError("diagnostic_push_failed_to_restore")
    telemetry = simulated["edit_runtime_telemetry"]
    if len(telemetry) != 1 or telemetry[0].get("guard_call_count") != 1:
        raise RuntimeError("one_token_contract_hook_not_confirmed")
    realized = telemetry[0].get("relative_write_norm", float("inf"))
    if not realized <= cap + CAST_TOLERANCE or (recipe["kind"] == "readout" and abs(realized - cap) > CAST_TOLERANCE):
        raise RuntimeError("contract_write_outside_tolerance")
    return {"recipe_id": recipe["recipe_id"], "kind": recipe["kind"], "component": recipe["component"],
            "sources": recipe["sources"], "cap": cap, "first_token_id": simulated["continuation_token_ids"][0],
            "realized_relative_write": realized, "runtime_telemetry": telemetry,
            "forwards": deepcopy(meter.counts)}, meter.raw_first


def push_record(row: dict, raw: Any, b0_raw: Any, token_ids: list[int]) -> dict:
    row["readout"] = readout(b0_raw, raw, token_ids)
    first = row["first_token_id"]
    row["edited_top5"] = [int(i) for i in raw.topk(min(5, raw.numel())).indices.tolist()]
    row["emitted_b0_rank"] = int((b0_raw > b0_raw[first]).sum()) + 1
    row["emitted_b0_gap"] = float(b0_raw.max() - b0_raw[first])
    row["edited_winner_matches_first_token"] = row["readout"]["edited_winner"] == first
    return row


def table_case(worker: Any, task: StructuralRescueEnv, case: RescueCase, n_layers: int, meter: Meter, row: dict,
               persist: Any, *, caps: list[float], patch_caps: list[float], patch_scope: str,
               packet_caps: list[float], spot_check: bool, output: Path) -> None:
    pattern = r"\bto\s+" + re.escape(case.frames[0][3]) + r"$"
    while not worker.done() and len(worker._output_token_ids()) < 32:
        worker.step()
        if re.search(pattern, worker.final_text()):
            break
    boundary = bool(re.search(pattern, worker.final_text())) and not worker.done()
    row.update(prefix=worker.final_text(), boundary_found=boundary, setup_forwards=deepcopy(meter.counts))
    if not boundary:
        return
    meter.boundary_length = worker._current_token_tensor().shape[-1]
    clause_specificity.align_cache(worker)
    observation = worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])[0]
    if observation["status"] != "ok":
        raise RuntimeError("structure_observation_failed")
    token_ids = [r["token_id"] for r in observation["structure_readout_rows"]]
    sources = structural_contract.eligible_sources(worker, required_terms(task))
    row.update(structure_readout=observation["structure_readout_rows"], shared_setup_forwards=deepcopy(meter.counts),
               eligible_sources=[{k: s[k] for k in ("token_id", "piece", "source_positions")} for s in sources])
    start = state_identity(worker)
    meter.reset("diagnostic")
    b0, b0_raw = branch(worker, task, None, meter, None, evaluate=task.evaluate_v2)
    row["b0"] = b0
    row["b0_winner_id"] = int(b0_raw.argmax())
    persist()
    middle = structural_contract.surface(worker, n_layers // 2)
    last = structural_contract.surface(worker, n_layers - 1)
    # Built once: every push restores the state it was built from.
    packet = worker.build_controller_packet()
    recipes, policies = {}, {}
    for cap in sorted(set(caps) | set(patch_caps)):
        policies[cap] = structural_contract.policy_for(worker, required_terms(task), cap=cap)
        readouts = structural_contract.readout_recipes(worker, sources, last, cap=cap)
        patches = structural_contract.patch_recipes(worker, sources, middle, cap=cap)
        if cap in packet_caps:
            # The v2 catalog order (patches, then readouts) and packet, for controller sampling.
            row["packets"][f"{cap:.2f}"] = compact_packet(worker, patches + readouts, case.case_id, observation, cap=cap)
        batch = readouts if cap in caps else []
        if cap in patch_caps:
            batch = batch + [p for p in patches if patch_scope == "all" or spot_check
                             or p["component"] == "common_orthogonal"]
        for recipe in batch:
            recipes[recipe["recipe_id"]] = recipe
            meter.reset("diagnostic")
            measured, raw = push(worker, recipe, meter, policies[cap], cap, packet)
            row["pushes"].append(push_record(measured, raw, b0_raw, token_ids))
        persist()
    groups: dict[int, list[dict]] = {}
    for measured in row["pushes"]:
        if measured["first_token_id"] != b0["first_token_id"]:
            groups.setdefault(measured["first_token_id"], []).append(measured)
    for token, members in sorted(groups.items()):
        ordered = sorted(members, key=lambda m: (round(abs(m["cap"] - OUTCOME_CAP), 9), m["cap"], m["kind"] != "readout"))
        chosen = ordered[:2] if spot_check and len(ordered) > 1 else ordered[:1]
        for index, representative in enumerate(chosen):
            meter.reset("diagnostic")
            cap = representative["cap"]
            outcome, _raw = branch(worker, task, recipes[representative["recipe_id"]], meter, policies[cap], cap=cap,
                                   evaluate=task.evaluate_v2)
            outcome.update(push_first_token_id=token, representative_recipe_id=representative["recipe_id"], cap=cap,
                           role="spot_check" if index else "outcome")
            row["outcomes"].append(outcome)
            persist()
    # The decomposition fails if a branch starts differently from its push, or one
    # first token decodes to two outputs. Recorded, never raised: it is the measurement.
    decoded: dict[int, set] = {}
    for outcome in row["outcomes"]:
        decoded.setdefault(outcome["push_first_token_id"], set()).add(outcome["output"])
    row["decomposition_mismatches"] = sorted(
        {o["push_first_token_id"] for o in row["outcomes"] if o["first_token_id"] != o["push_first_token_id"]}
        | {t for t, outputs in decoded.items() if len(outputs) > 1})
    row["state_restored"] = start == state_identity(worker)
    if not row["state_restored"]:
        raise RuntimeError("table_case_failed_to_restore")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="the prepared run directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--case-set", choices=("known16", "pool62", "rescue2"), required=True)
    parser.add_argument("--screen1-dir", type=Path, required=True)
    parser.add_argument("--screen2-dir", type=Path)
    parser.add_argument("--screen2-plan", type=Path, help="the frozen rescue-2 analysis plan (rescue2 only)")
    parser.add_argument("--caps", required=True, help="comma-separated readout caps")
    parser.add_argument("--patch-caps", default="")
    parser.add_argument("--patch-scope", choices=("all", "orthogonal_plus_spot_checks"), default="all")
    parser.add_argument("--packet-caps", default="")
    parser.add_argument("--references", type=Path, nargs="*", default=[],
                        help="results.json files the analyzer must reproduce")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--case-range", default=None, help="start:stop, one part per process")
    parser.add_argument("--worker-model-path", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--wall-seconds", type=int, default=MAX_WALL_SECONDS)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    parser.add_argument("--allow-dirty", action="store_true", help="tests only")
    args = parser.parse_args(argv)
    floats = lambda text: [float(v) for v in text.split(",") if v]
    caps, patch_caps, packet_caps = floats(args.caps), floats(args.patch_caps), floats(args.packet_caps)
    if not caps or any(not 0 < c <= 1 for c in caps + patch_caps + packet_caps):
        raise ValueError("caps must be in (0, 1]")
    plan = json.loads(args.screen2_plan.read_text()) if args.screen2_plan else None
    cases, binding = case_set(args.case_set, screen1_dir=args.screen1_dir.expanduser().resolve(),
                              screen2_dir=args.screen2_dir and args.screen2_dir.expanduser().resolve(),
                              screen2_plan=plan)
    # Repo-relative, so the manifest does not depend on the working directory.
    references = {p.expanduser().resolve().relative_to(run_hygiene.REPO).as_posix():
                  hashlib.sha256(p.read_bytes()).hexdigest() for p in args.references}
    frozen = manifest_bytes(cases, binding, caps=caps, patch_caps=patch_caps, patch_scope=args.patch_scope,
                            packet_caps=packet_caps, references=references)
    digest = hashlib.sha256(frozen).hexdigest()
    root = args.output_dir.expanduser().resolve()
    if args.prepare:
        root.mkdir(parents=True, exist_ok=False)
        (root / "manifest.json").write_bytes(frozen)
        print(json.dumps({"manifest_sha256": digest, "output_dir": str(root), "cases": len(cases)}))
        return 0
    if args.manifest_sha256 != digest or (root / "manifest.json").read_bytes() != frozen:
        raise ValueError("predeclared_manifest_hash_required_and_must_match")
    if args.worker_model_path is None or args.checkpoint_manifest is None:
        parser.error("--run requires local model and checkpoint manifest")
    if not 60 <= args.wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError(f"wall_seconds must be between 60 and {MAX_WALL_SECONDS}")
    if not 0 < args.mps_memory_fraction <= 0.85:
        raise ValueError("mps_memory_fraction must be >0 and <=0.85")
    start, stop = (int(v) for v in (args.case_range or f"0:{len(cases)}").split(":"))
    if not 0 <= start < stop <= len(cases):
        raise ValueError("case_range outside the case set")
    attempt = 1
    while (root / f"part_{start:03d}_{stop:03d}_r{attempt}").exists():
        attempt += 1
    output = root / f"part_{start:03d}_{stop:03d}_r{attempt}"
    run_hygiene.preflight(root, floor_gib=DISK_FLOOR_GIB, allow_dirty=args.allow_dirty)
    output.mkdir()
    result_path = output / "results.json"
    execution = {"manifest_sha256": digest, "case_range": [start, stop], "attempt": attempt, "device": args.device,
                 "dtype": args.dtype, "wall_seconds": args.wall_seconds,
                 "mps_memory_fraction": args.mps_memory_fraction, "code_sha256": capture_sources(output),
                 **run_hygiene.provenance()}
    (output / "execution_plan.json").write_text(json.dumps(execution, indent=2))
    started = time.monotonic()
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest, "case_range": [start, stop], "attempt": attempt,
              "checkpoint_sha256_verified": None, "device": args.device, "dtype": args.dtype, "cases": [],
              "scope": "restored_contract_branch_first_token_table_not_production", "production_apply_allowed": False}

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
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
        n_layers = model.cfg.n_layers
        layers = sorted({n_layers // 2, n_layers - 1})
        catalog = [s for s in build_default_activation_surface_catalog(model, layers=layers)
                   if s["target"].get("site") == "resid_pre" and s["target"]["token"]["mode"] == "last"]
        if sorted(s["target"]["layer"] for s in catalog) != layers:
            raise ValueError("contract_resid_surfaces_unavailable")
        report["after_model_load"] = run_hygiene.telemetry(output)
        for index in range(start, stop):
            case = cases[index]
            task = StructuralRescueEnv(case)
            worker = build_hooked_transformer_worker_runtime(model, task, seed=0, surface_catalog=catalog,
                clause_structure_diagnostics=True, clause_structure_operator_diagnostics=True,
                max_diagnostic_calls_per_run=12)
            worker.reset(task.reset(0))
            meter = Meter(worker, deadline=started + args.wall_seconds, disk_path=output,
                          disk_floor_gib=DISK_FLOOR_GIB)
            row = {"case_id": case.case_id, "split": case.split, "pool_index": index, "status": "running",
                   "pushes": [], "outcomes": [], "packets": {}}
            report["cases"].append(row)
            meter.check_limits()
            try:
                with meter.install():
                    table_case(worker, task, case, n_layers, meter, row, persist, caps=caps, patch_caps=patch_caps,
                               patch_scope=args.patch_scope, packet_caps=packet_caps,
                               spot_check=index % SPOT_CHECK_EVERY == 0, output=output)
                row["status"] = "ok"
            except ForwardLimit:
                row["status"] = "error"
                raise
            except Exception as exc:
                # One case's failure is a measurement; the part goes on.
                row.update(status="error", error_type=type(exc).__name__, error=str(exc)[:500])
            row["telemetry"] = run_hygiene.telemetry(output)
            persist()
            print(json.dumps({"case": case.case_id, "status": row["status"], "boundary": row.get("boundary_found"),
                              "b0_joint": row.get("b0", {}).get("joint_success"), "pushes": len(row["pushes"]),
                              "outcomes": len(row["outcomes"]), **row["telemetry"]}), flush=True)
            worker.runtime_state.clear_edits()
            # Each case's worker keeps a full activation cache on the device.
            worker.runtime_state.last_cache = None
            _release_memory()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc)[:500])
        persist()
        raise
    finally:
        if worker is not None:
            worker.runtime_state.clear_edits()
    persist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
