"""Normal-cap, one-boundary policy comparison in restored diagnostic branches.

Freeze with --prepare, then execute with --run and the printed manifest hash.
This pilot does not expose new production edits or force any output token.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import shutil
import tarfile
import time
from typing import Any

from ..controllers.base import ControllerProviderRequest
from ..controllers.factory import create_controller_provider
from ..runtime import clause_actuation, clause_radial, clause_specificity, clause_structure
from ..runtime.compiler import StepContext, compile_expr
from ..runtime.policy import validate_command_against_packet
from ..runtime.response_probe import identity, state_identity, tensor_identity
from ..tasks.structural_rescue import CASES, StructuralRescueEnv, manifest_bytes
from .digit_transform_e2e import (
    build_default_activation_surface_catalog, build_hooked_transformer_worker_runtime, load_worker_model,
)
from .replay_orphan_blueprint_ceiling import verify_checkpoint


POLICIES = ("b0", "fixed", "random", "controller")
SYSTEM_PROMPT = """Select one bounded internal intervention recipe at this current worker prefix.
Return strict JSON only: {"decision":"noop"} or {"decision":"probe","recipe_id":"<offered ID>"}.
You receive generated text, next-token readout, and a catalog of source-bound geometric recipes.
Choose a small recipe only if it plausibly helps the worker preserve independent clauses.
You may choose noop. Do not output an answer, a token to force, new expressions, or any other fields.
This selects one restored diagnostic continuation under existing caps, not production permission.
Role annotations, reference continuations and candidate outcomes are unavailable.
"""


class ForwardLimit(RuntimeError):
    pass


def capture_sources(output: Path) -> dict[str, str]:
    """Save the source bytes and versions before loading a model or provider."""
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "SpiralInterventionLab").rglob("*.py")) + [root / "pyproject.toml"]
    hashes = {}
    with tarfile.open(output / "source_snapshot.tgz", "x:gz") as archive:
        for path in files:
            relative, data = path.relative_to(root).as_posix(), path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            member = tarfile.TarInfo(relative)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    versions = {}
    for package in ("torch", "transformers", "transformer-lens", "openai"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    (output / "environment.json").write_text(json.dumps(versions, indent=2) + "\n")
    return hashes


def unavailable_policies(baseline: dict, policies: list[str]) -> dict:
    """Reuse the observed fallback without charging fictitious model calls."""
    return {p: {**deepcopy(baseline), "forwards": {}, "reused_baseline": True,
                "selection": {"status": "no_eligible_boundary", "provider_calls": 0}}
            for p in policies}


class Meter:
    """Count and cap actual calls, including packet preparation and failed calls."""
    def __init__(self, worker: Any, *, deadline: float, disk_path: Path):
        self.worker, self.deadline, self.disk_path = worker, deadline, disk_path
        self.phase = "setup"
        self.boundary_length: int | None = None
        self.counts: dict[str, dict[str, int]] = {}
        self.raw_first: Any = None
        self.capture_edited = False

    def reset(self, phase: str) -> None:
        self.phase, self.counts, self.raw_first = phase, {}, None

    def check_limits(self) -> None:
        if time.monotonic() > self.deadline:
            raise ForwardLimit("wall_clock_budget_exhausted")
        if shutil.disk_usage(self.disk_path).free < 10 * 2**30:
            raise ForwardLimit("disk_below_predeclared_10_gib_floor")

    @contextmanager
    def install(self):
        runtime = self.worker.runtime_state
        original = runtime.run_with_cache
        had_override, prior = "run_with_cache" in vars(runtime), vars(runtime).get("run_with_cache")

        def run(tokens, *args, **kwargs):
            self.check_limits()
            phase = self.phase
            if phase == "branch":
                phase = "diagnostic" if tokens.shape[-1] == self.boundary_length else "continuation"
            cap = {"setup": 40, "diagnostic": 24, "continuation": 64}[phase]
            count = self.counts.setdefault(phase, {"attempted": 0, "completed": 0})
            if count["attempted"] >= cap:
                raise ForwardLimit(phase + "_forward_budget_exhausted")
            count["attempted"] += 1
            active = bool(self.worker._collect_active_edits())
            value = original(tokens, *args, **kwargs)
            count["completed"] += 1
            if (self.phase == "branch" and tokens.shape[-1] == self.boundary_length
                    and active == self.capture_edited and self.raw_first is None):
                self.raw_first = value[0][0, -1].detach().cpu().double()
            return value

        runtime.run_with_cache = run
        try:
            yield self
        finally:
            if had_override:
                runtime.run_with_cache = prior
            else:
                delattr(runtime, "run_with_cache")


def choose(policy: str, packet: dict, *, provider=None, seed: int = 17,
           fixed_recipe_id: str | None = None) -> tuple[str | None, dict]:
    recipes = packet["recipes"]
    if policy == "b0" or not recipes:
        return None, {"status": "noop", "provider_calls": 0}
    if policy == "fixed":
        if fixed_recipe_id not in {r["recipe_id"] for r in recipes}:
            return None, {"status": "fixed_recipe_unavailable", "provider_calls": 0}
        return fixed_recipe_id, {"status": "selected", "provider_calls": 0}
    if policy == "random":
        digest = hashlib.sha256((packet["case_id"] + ":" + str(seed)).encode()).digest()
        return recipes[int.from_bytes(digest, "big") % len(recipes)]["recipe_id"], {
            "status": "selected", "provider_calls": 0, "seed": seed}
    if policy != "controller" or provider is None:
        raise ValueError("controller policy requires an explicit provider")
    started = time.monotonic()
    request = ControllerProviderRequest(system_prompt=SYSTEM_PROMPT, payload=packet,
                                        max_output_tokens=1600, expect_json=True)
    trace = {"provider_calls": 1, "provider": provider.provider_name, "model": provider.model_name,
             "packet_sha256": identity("packet:", packet),
             "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()}
    try:
        response = provider.complete(request)
        trace.update(usage=dict(response.usage), response=response.text,
                     response_metadata=dict(response.metadata),
                     latency_seconds=round(time.monotonic() - started, 3))
        command = json.loads(response.text)
        if command == {"decision": "noop"}:
            return None, {**trace, "status": "noop"}
        if (set(command) != {"decision", "recipe_id"} or command["decision"] != "probe"
                or command["recipe_id"] not in {r["recipe_id"] for r in recipes}):
            raise ValueError("controller_choice_outside_offered_schema")
        return command["recipe_id"], {**trace, "status": "selected"}
    except Exception as exc:
        # Provider errors may contain credential fragments. Keep safe metadata only.
        return None, {**trace, "status": "error", "error_type": type(exc).__name__,
                      "latency_seconds": round(time.monotonic() - started, 3),
                      "http_status": getattr(exc, "status_code", None)}


def compact_packet(worker: Any, recipes: list[dict], case_id: str, observation: dict) -> dict:
    return {"version": "structural_rescue_choice_v1", "case_id": case_id,
            "generated_tail": worker.final_text(), "structure_readout": observation["structure_readout_rows"],
            "recipes": [{"recipe_id": r["component_recipe_id"], "component": r["component"],
                         "source_pair_id": r["source_pair_id"], "source_pair": r["source_pair"],
                         "surfaces": r["surfaces"], "dsl_formula": r["dsl_formula"],
                         "geometry": r["geometry"]} for r in recipes],
            "budget": {"max_edits": 1, "diagnostic_forwards": 24, "continuation_forwards": 64},
            "production_apply_allowed": False}


def command_for(worker: Any, recipe: dict) -> dict:
    if len(recipe["surfaces"]) != 1:
        raise ValueError("pilot_requires_exactly_one_predeclared_surface")
    anchor = next(c for c in clause_actuation.offer(worker)["candidates"]
                  if c["structure_operator_id"] == recipe["source_pair"][0]["structure_operator_id"])
    edit = clause_actuation.materialize(worker, anchor, recipe["surfaces"][0])
    edit["id"] = recipe["component_recipe_id"]
    edit["source"]["expr"] = deepcopy(recipe["source_expressions"][0])
    return {"version": "0.1", "decision": "apply", "edits": [edit]}


def readout(baseline, edited, token_ids: list[int]) -> dict:
    if baseline is None or edited is None:
        raise ValueError("raw_first_token_capture_missing")
    winner = int(baseline.argmax())
    p, q = baseline.softmax(0), edited.softmax(0)
    rows = []
    for tid in token_ids:
        gap = float(baseline[winner] - baseline[tid])
        after = float(edited[winner] - edited[tid])
        rows.append({"token_id": tid, "baseline_gap_to_fixed_winner": gap,
                     "edited_gap_to_fixed_winner": after,
                     "gap_closure_fraction": (gap - after) / gap if gap > 0 else None,
                     "baseline_probability": float(p[tid]), "edited_probability": float(q[tid]),
                     "probability_delta": float(q[tid] - p[tid]),
                     "rank_before": int((baseline > baseline[tid]).sum()) + 1,
                     "rank_after": int((edited > edited[tid]).sum()) + 1})
    return {"stage": "raw_pre_constraint_full_vocabulary_cpu_float64",
            "fixed_baseline_winner": winner, "edited_winner": int(edited.argmax()),
            "raw_greedy_branch_changed": int(edited.argmax()) != winner, "rows": rows}


def branch(worker, task, recipe, meter) -> tuple[dict, Any]:
    before, caps = state_identity(worker), repr(worker.surface_catalog)
    baseline_raw = []
    meter.phase, meter.capture_edited = "branch", False
    for _ in range(2):
        meter.raw_first = None
        baseline = worker._simulate_decode(max_new_tokens=1, top_k=6,
            score_candidate_text=False, score_observer_check=False)
        if baseline is None or meter.raw_first is None:
            raise RuntimeError("repeated_baseline_failed")
        baseline_raw.append(meter.raw_first)
    command = command_for(worker, recipe) if recipe is not None else None
    input_identity = None
    if command:
        packet = worker.build_controller_packet()
        validate_command_against_packet(command, packet)
        ctx = StepContext(packet=packet, runtime_state=worker.runtime_state, adapter=worker.adapter,
                          traces={}, stats={}, active_edits={})
        source = compile_expr(command["edits"][0]["source"]["expr"])(ctx)
        input_identity = tensor_identity(source)
        if input_identity != recipe["input_identities"][0]["source"]:
            raise ValueError("recipe_source_changed_before_branch")
    meter.phase, meter.capture_edited, meter.raw_first = "branch", command is not None, None
    simulated = worker._simulate_decode(max_new_tokens=64 - len(worker._output_token_ids()), top_k=6,
        command=command, score_candidate_text=False, score_observer_check=False)
    if simulated is None:
        raise RuntimeError(worker._last_simulate_decode_error)
    text = worker.codec.decode(worker._output_token_ids() + simulated["continuation_token_ids"])
    restored = before == state_identity(worker) and caps == repr(worker.surface_catalog)
    if not restored or worker._collect_active_edits():
        raise RuntimeError("diagnostic_branch_failed_to_restore")
    telemetry = simulated["edit_runtime_telemetry"]
    if command and (len(telemetry) != 1 or telemetry[0].get("hook_call_count") != 1):
        raise RuntimeError("one_token_edit_hook_not_confirmed")
    for event in telemetry:
        norm = event.get("target_norm_before")
        event["relative_write_norm"] = event.get("blend_delta_norm", 0) / norm if norm else None
    return {"status": "ok", **task.evaluate(text), "state_restored": restored,
            "baseline_repeat_max_abs_logit_delta": float((baseline_raw[0] - baseline_raw[1]).abs().max()),
            "baseline_logits_sha256": [tensor_identity(v)["sha256"] for v in baseline_raw],
            "normal_caps_unchanged": True, "command": command,
            "source_tensor_identity": input_identity, "runtime_telemetry": telemetry,
            "first_token_id": simulated["continuation_token_ids"][0],
            "forwards": deepcopy(meter.counts), "production_apply_allowed": False}, meter.raw_first


def record_branch(worker, task, recipe, meter, destination: dict):
    """Retain attempted work even if simulation or a budget check fails."""
    destination["status"] = "running"
    try:
        outcome, raw = branch(worker, task, recipe, meter)
        destination.update(outcome)
        return raw
    except Exception as exc:
        destination.update(status="error", error_type=type(exc).__name__,
                           forwards=deepcopy(meter.counts), production_apply_allowed=False)
        raise


def aggregate(rows: list[dict]) -> dict:
    results = {}
    for split in ("calibration", "confirmation"):
        selected = [r for r in rows if r["split"] == split]
        results[split] = {}
        for policy in POLICIES:
            paired = [(r["policies"].get("b0"), r["policies"].get(policy)) for r in selected]
            valid = [(a, b) for a, b in paired if a and b and a.get("status") == b.get("status") == "ok"
                     and b.get("selection", {}).get("status") != "error"]
            results[split][policy] = {"completed_pairs": len(valid),
                "eligible_pairs": sum(r.get("comparison_eligible") is True
                    and r["policies"].get("b0", {}).get("status") == "ok"
                    and r["policies"].get(policy, {}).get("status") == "ok"
                    and r["policies"].get(policy, {}).get("selection", {}).get("status") != "error"
                    for r in selected),
                "unavailable_boundary_cases": sum(r.get("comparison_eligible") is False for r in selected),
                "joint_successes": sum(b["joint_success"] for _, b in valid),
                "wins_vs_b0": sum(b["joint_success"] and not a["joint_success"] for a, b in valid),
                "losses_vs_b0": sum(a["joint_success"] and not b["joint_success"] for a, b in valid),
                "output_changes": sum(a["output"] != b["output"] for a, b in valid),
                "applied_diagnostic_edits": sum(b.get("command") is not None for _, b in valid),
                "unknown_roles": sum(b["role_status"] == "unknown" for _, b in valid),
                "selection_errors": sum(bool(b and b.get("selection", {}).get("status") == "error") for _, b in paired)}
    return results


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
    parser.add_argument("--controller-provider", choices=("openai",), default="openai")
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--case-ids", default=",".join(c.case_id for c in CASES))
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
    if (args.manifest_sha256 != digest or (output / "manifest.json").read_bytes() != frozen):
        raise ValueError("predeclared_manifest_hash_required_and_must_match")
    if args.worker_model_path is None or args.checkpoint_manifest is None:
        parser.error("--run requires local model and checkpoint manifest")
    policies, case_ids = args.policies.split(","), args.case_ids.split(",")
    if (set(policies) - set(POLICIES) or policies[0] != "b0"
            or len(set(policies)) != len(policies) or len(set(case_ids)) != len(case_ids)
            or set(case_ids) - {c.case_id for c in CASES}):
        raise ValueError("unknown or duplicate policy/case, or missing b0")
    if not 60 <= args.wall_seconds <= 7200:
        raise ValueError("wall_seconds must be between 60 and 7200")
    if not 0 < args.mps_memory_fraction <= 0.85:
        raise ValueError("mps_memory_fraction must be >0 and <=0.85")
    result_path = output / "results.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    if (output / "execution_plan.json").exists():
        raise FileExistsError(output / "execution_plan.json")
    code_hashes = capture_sources(output)
    execution = {"manifest_sha256": digest, "device": args.device, "dtype": args.dtype,
                 "policies": policies, "case_ids": case_ids, "wall_seconds": args.wall_seconds,
                 "mps_memory_fraction": args.mps_memory_fraction,
                 "controller_model": args.controller_model if "controller" in policies else None,
                 "code_sha256": code_hashes}
    (output / "execution_plan.json").write_text(json.dumps(execution, indent=2))
    started = time.monotonic()
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest, "checkpoint_sha256_verified": None,
              "device": args.device, "dtype": args.dtype, "surface_catalog": [],
              "policies_requested": policies, "case_ids_requested": case_ids, "cases": [],
              "scope": "restored_normal_cap_branch_pilot_not_production_or_population_efficacy",
              "production_apply_allowed": False}

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["aggregate"] = aggregate(report["cases"])
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
        catalog = build_default_activation_surface_catalog(model)
        catalog = [s for s in catalog if s["target"].get("site") == "resid_pre"
                   and s["target"]["layer"] == model.cfg.n_layers // 2
                   and s["target"]["token"]["mode"] == "last"]
        if len(catalog) != 1:
            raise ValueError("standard_middle_resid_surface_unavailable")
        report["surface_catalog"] = catalog
        provider = None
        if "controller" in policies:
            from openai import OpenAI
            provider = create_controller_provider(args.controller_provider, model=args.controller_model,
                                                   client=OpenAI(timeout=90, max_retries=0))
        for case in CASES:
            if case.case_id not in case_ids:
                continue
            task = StructuralRescueEnv(case)
            prompt = task.reset(0)
            worker = build_hooked_transformer_worker_runtime(model, task, seed=0, surface_catalog=catalog,
                clause_structure_diagnostics=True, clause_structure_operator_diagnostics=True,
                max_diagnostic_calls_per_run=12)
            worker.reset(prompt)
            meter = Meter(worker, deadline=started + args.wall_seconds, disk_path=output)
            row = {"case_id": case.case_id, "split": case.split, "policies": {}, "calibration_matrix": []}
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
                    meter.reset("branch")
                    meter.boundary_length = worker._current_token_tensor().shape[-1]
                    if worker.done():
                        baseline = {"status": "ok", **task.evaluate(worker.final_text()), "command": None,
                                    "forwards": {}, "production_apply_allowed": False}
                    else:
                        baseline = {}
                        row["fallback_result"] = baseline
                        record_branch(worker, task, None, meter, baseline)
                    row["shared_fallback_forwards"] = deepcopy(baseline["forwards"])
                    row["policies"] = unavailable_policies(baseline, policies)
                    row["comparison_eligible"] = False
                    persist()
                    continue
                meter.boundary_length = worker._current_token_tensor().shape[-1]
                clause_specificity.align_cache(worker)
                observation = worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])[0]
                if observation["status"] != "ok":
                    raise RuntimeError("structure_observation_failed")
                offered = clause_radial.offer(worker)["recipes"]
                fixed = next((r for r in offered if r["component"] == "common_orthogonal"), None)
                fixed_id = fixed["component_recipe_id"] if fixed and fixed["available"] else None
                recipes = [r for r in offered
                           if r["available"] and r["component"] in {"common", "common_parallel", "common_orthogonal"}]
                packet = compact_packet(worker, recipes, case.case_id, observation)
                (output / (case.case_id + "_observation.json")).write_text(json.dumps(packet, indent=2))
                row.update(comparison_eligible=True, shared_setup_forwards=deepcopy(meter.counts),
                           eligible_recipe_count=len(recipes), recipe_ids=[r["component_recipe_id"] for r in recipes])
                baseline_raw = None
                for policy in policies:
                    meter.check_limits()
                    selected_id, trace = choose(policy, packet, provider=provider, fixed_recipe_id=fixed_id)
                    recipe = next((r for r in recipes if r["component_recipe_id"] == selected_id), None)
                    meter.reset("diagnostic")
                    outcome = {"selection": trace, "selected_recipe_id": selected_id}
                    row["policies"][policy] = outcome
                    raw = record_branch(worker, task, recipe, meter, outcome)
                    if policy == "b0":
                        baseline_raw = raw
                    outcome.update(selection=trace, selected_recipe_id=selected_id,
                        readout=readout(baseline_raw, raw, [r["token_id"] for r in observation["structure_readout_rows"]]))
                    outcome["natural_branch_changed"] = outcome["first_token_id"] != row["policies"].get("b0", outcome)["first_token_id"]
                    persist()
                    print(json.dumps({"case": case.case_id, "policy": policy, "joint_success": outcome["joint_success"],
                        "score": outcome["legacy_score"], "selection": trace["status"],
                        "forwards": outcome["forwards"], "output": outcome["output"]}), flush=True)
                if case.split == "calibration" and recipes:
                    first_pair = recipes[0]["source_pair_id"]
                    for recipe in (r for r in recipes if r["source_pair_id"] == first_pair):
                        meter.reset("diagnostic")
                        outcome = {"component": recipe["component"], "recipe_id": recipe["component_recipe_id"]}
                        row["calibration_matrix"].append(outcome)
                        raw = record_branch(worker, task, recipe, meter, outcome)
                        outcome["readout"] = readout(baseline_raw, raw, [r["token_id"] for r in observation["structure_readout_rows"]])
                        persist()
            worker.runtime_state.clear_edits()
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


if __name__ == "__main__":
    raise SystemExit(main())
