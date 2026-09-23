"""Repeatable model-local B0/C1 checks, with no cross-model score pooling.

Run each profile in a separate process. Checkpoints are CLI inputs; prompts,
seeds, model depths and policy limits are not retuned to obtain an intervention.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time


PROFILES = {
    "gpt2_control": {"worker_model": "gpt2", "model_type": "gpt2", "layers": 12,
                     "task": "constrained_rewrite", "seed": 7, "dtype": "float32"},
    "llama_l1": {"worker_model": "meta-llama/Llama-3.2-3B-Instruct", "model_type": "llama", "layers": 28,
                 "task": "constrained_rewrite_l1", "seed": 0, "dtype": "float16"},
}


def build_argv(profile: str, checkpoint: Path, root: Path, *, device: str, controller: str,
               candidate_handoff_mode: str = "off") -> list[str]:
    spec = PROFILES[profile]
    return ["--provider", "openai", "--controller-model", controller,
        "--worker-model", spec["worker_model"], "--worker-model-path", str(checkpoint),
        "--worker-hf-offline", "--worker-device", device, "--worker-dtype", spec["dtype"],
        "--worker-mps-mode", "conservative", "--task", spec["task"], "--seed", str(spec["seed"]), "--no-b1",
        "--controller-prompt-profile", "compact", "--controller-packet-view", "compact",
        "--readout-analyzer", "sae_scaffold", "--readout-analyzer-rerank-mode", "apply",
        "--activation-surface-profile", "activation_patch_expanded",
        "--max-diagnostic-calls-per-run", "12", "--diagnostic-result-window", "12",
        "--candidate-handoff-rounds", "2",
        "--candidate-handoff-mode", candidate_handoff_mode,
        "--post-run-debrief", "controller", "--post-run-debrief-max-output-tokens", "1200",
        "--log-dir", str(root / "live")]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_action_identity(row: dict, offered: dict) -> None:
    if row.get("parent_candidate_id") is not None:
        assert offered["action"] in {"measure_normal_cap_current_prefix", "investigate_normal_cap_current_prefix"}
        assert row["parent_candidate_id"] == offered["candidate_id"]
        assert row["candidate_id"] == offered["new_candidate_id"] != offered["candidate_id"]
        assert row["inherits_measurement"] is False
    else:
        assert row["candidate_id"] == offered["candidate_id"]
        if offered["action"] == "measure_normal_cap_current_prefix":
            assert row["executed_action"] is None


def audit(root: Path, manifest: dict) -> dict:
    traces = {label: [json.loads(line) for line in (root / "live" / f"{label}.jsonl").read_text().splitlines()]
              for label in ("b0", "c1")}
    outcomes = {}
    for label, rows in traces.items():
        ends = [row for row in rows if row.get("event") == "episode_end"]
        assert len(ends) == 1
        start = next(row for row in rows if row.get("event") == "episode_start")
        assert start["prompt"] == manifest["fixture"]["prompt"] and start["seed"] == manifest["profile"]["seed"]
        outcomes[label] = ends[0]
    rows = traces["c1"]
    actions, offers, requests, timeline = [], {}, Counter(), []
    charges, diagnostics, usage = Counter(), Counter(), Counter()
    exposure_steps = []
    handoff_measurable_steps = []
    handoff_deferred_results = 0
    handoff_missing_reasons = 0
    first_card = None
    diagnostic_used = 0
    for row in rows:
        if row.get("event") == "controller_selection" and row.get("candidate_handoff_state") == "measurable":
            handoff_measurable_steps.append(row["step"])
            handoff_missing_reasons += int(row.get("candidate_handoff_defer_reason") == "not_stated_by_controller")
        if row.get("event") == "controller_selection" and row.get("candidate_diagnostic_choices"):
            choices = row["candidate_diagnostic_choices"]
            exposure_steps.append(row["step"])
            for card in choices["cards"]:
                for action in card["actions"]:
                    offers[action["action_id"]] = {**action, "candidate_id": card["candidate_id"],
                        "context": choices["current_context_id"], "position": choices.get("current_position"),
                        "history": card.get("measurement_history")}
        if row.get("event") == "controller_diagnostic_request" and row.get("diagnostic") == "candidate_action":
            requests[row["action_id"]] += 1
        if row.get("event") == "controller_diagnostic_result":
            diagnostics[row["diagnostic"]] += 1
            for key in ("diagnostic_calls_left", "inspection_calls_left"):
                charges[key] += row["budget_before"][key] - row["budget_after"][key]
            diagnostic_used += row["budget_before"]["diagnostic_calls_left"] - row["budget_after"]["diagnostic_calls_left"]
            handoff_deferred_results += int(row.get("candidate_handoff_choice") == "deferred_for_other_diagnostic")
            if first_card is None and any(receipt.get("status") == "frozen"
                                         for receipt in row.get("frozen_candidate_receipts", ())):
                first_card = {"step": row["step"], "diagnostics_used": diagnostic_used}
            if row.get("diagnostic") == "candidate_action":
                offered = offers[row["action_id"]]
                assert requests[row["action_id"]] > 0
                requests[row["action_id"]] -= 1
                validate_action_identity(row, offered)
                assert row["requested_action"] == offered["action"]
                assert row["executed_action"] in (None, row["requested_action"])
                assert not row["production_apply_allowed"] and not row["certified_for_apply"]
                if row["status"] == "measured":
                    assert offered["available"] and row["measurement_context_id"] == offered["context"]
                    assert row["evidence"]["state_restored"] is True
                    if offered["action"] == "investigate_normal_cap_current_prefix":
                        assert row["transaction_status"] in {"measurement_only", "confirmed_no_offer", "offer_available"}
                        if row["transaction_status"] == "measurement_only":
                            assert row["physical_replay_count"] == 4
                        else:
                            assert 4 <= row["physical_replay_count"] <= 8
                        assert row["diagnostic_call_cost"] == (1 if row["transaction_status"] == "measurement_only" else 2)
                        assert not row["production_trial_allowed"] or row["transaction_status"] == "offer_available"
                    else:
                        assert row["physical_replay_count"] == 4
                actions.append(row)
                timeline.append({"step": row["step"], "candidate_id": row["candidate_id"],
                    "parent_candidate_id": row.get("parent_candidate_id"),
                    "action": row["executed_action"], "status": row["status"],
                    "transaction_status": row.get("transaction_status"),
                    "diagnostic_call_cost": row.get("diagnostic_call_cost"),
                    "production_trial_allowed": row.get("production_trial_allowed", False),
                    "trial_authorization_id": row.get("trial_authorization_id"),
                    "position": row.get("requested_position"), "offered_history": offered["history"],
                    "evidence": row.get("evidence"), "physical_replay_count": row["physical_replay_count"]})
        if row.get("event") == "controller_provider_attempt":
            u = row.get("usage") or {}
            for key in ("input_tokens", "output_tokens"):
                usage[key] += int(u.get(key) or 0)
            usage["cached_input_tokens"] += int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    assert charges["diagnostic_calls_left"] <= 12
    return {"profile": manifest["profile_name"], "outcomes": outcomes,
        "output_identical": outcomes["b0"]["output"] == outcomes["c1"]["output"],
        "score_delta": outcomes["c1"]["score"] - outcomes["b0"]["score"],
        "compiled_rollout_edit_count": sum(row.get("event") == "compiled_edit" for row in rows),
        "provider_parse_results": dict(Counter(str(row.get("parse_ok")) for row in rows if row.get("event") == "controller_provider_attempt")),
        "command_decisions": dict(Counter(row["command"]["decision"] for row in rows if row.get("event") == "controller_command")),
        "diagnostic_counts": dict(diagnostics), "charges": dict(charges),
        "controller_usage_excluding_debrief": dict(usage), "position_exposure_steps": exposure_steps,
        "candidate_handoff_measurable_steps": handoff_measurable_steps,
        "candidate_handoff_deferred_results": handoff_deferred_results,
        "candidate_handoff_missing_controller_reasons": handoff_missing_reasons,
        "first_measured_card": first_card,
        "candidate_action_counts": dict(Counter(row["requested_action"] for row in actions)),
        "candidate_action_statuses": dict(Counter(row["status"] for row in actions)),
        "action_measurement_count": sum(row["new_measurement_count"] for row in actions),
        "action_physical_replay_count": sum(row["physical_replay_count"] for row in actions),
        "action_transaction_statuses": dict(Counter(row.get("transaction_status") for row in actions
                                              if row.get("transaction_status"))),
        "visible_trial_offer_count": sum(row.get("event") == "controller_selection" and
            isinstance(row.get("candidate_trial_offer"), dict) for row in rows),
        "action_contract_evidence": "validated_recorded_actions" if actions else "not_exercised",
        "action_timeline": timeline, "production_policy_changed": manifest.get("production_policy_changed", False),
        "physical_policy_caps_changed": False, "candidate_handoff_rounds": 2,
        "prefix_hold_requests": sum(row.get("event") == "controller_prefix_hold" for row in rows),
        "prefix_holds_accepted": sum(row.get("event") == "controller_prefix_hold" and row.get("accepted") is True for row in rows),
        "scope": "one model-local paired trajectory, not cross-model efficacy or a causal prompt ablation"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--candidate-handoff-mode", choices=("off", "soft"), default="off")
    parser.add_argument("--baseline-reference", type=Path, help="Optional previous experiment_summary.json; stop if B0 changed")
    args = parser.parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is not present; no credentials are written or logged")
    checkpoint = args.worker_model_path.expanduser().resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    profile = PROFILES[args.profile]
    if config.get("model_type") != profile["model_type"] or config.get("num_hidden_layers", config.get("n_layer")) != profile["layers"]:
        parser.error("Checkpoint architecture/depth does not match the selected profile")
    reference = json.loads(args.baseline_reference.read_text()) if args.baseline_reference else None
    if reference is not None and not reference.get("b0"):
        parser.error("The reference must contain a B0 result")
    root = args.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    call = build_argv(args.profile, checkpoint, root, device=args.device, controller=args.controller_model,
                      candidate_handoff_mode=args.candidate_handoff_mode)
    import torch
    from SpiralInterventionLab.examples.digit_transform_e2e import create_task_env, main as run_pair
    from SpiralInterventionLab.runtime import baselines
    from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime

    env = create_task_env(profile["task"])
    prompt = env.reset(profile["seed"])
    source = sorted(p for p in (repo / "SpiralInterventionLab").rglob("*")
                    if p.is_file() and p.suffix in {".py", ".txt", ".json"} and "__pycache__" not in p.parts)
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "profile_name": args.profile,
        "profile": profile, "fixture": {"prompt": prompt, "max_generated_tokens": env.worker_runtime_kwargs()["max_generated_tokens"]},
        "argv": call, "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": {p.name: sha(p) for p in sorted(checkpoint.iterdir())
                              if p.is_file() and p.suffix in {".json", ".txt", ".model", ".safetensors", ".bin"}},
        "code_hashes": {str(p.relative_to(repo)): sha(p) for p in source},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "transformer-lens", "openai")},
        "baseline_reference_sha256": sha(args.baseline_reference) if reference else None,
        "mps_memory_fraction": 0.7 if args.device == "mps" else None,
        "production_policy_changed": True, "physical_policy_caps_changed": False,
        "controller_handoff_changed": True, "candidate_handoff_rounds": 2,
        "candidate_handoff_mode": args.candidate_handoff_mode,
        "policy_change_scope": "exact_context_not_frontier_preference_no_looser_caps",
        "normal_cap_variants": "new_identity_explicit_measurement_no_inherited_evidence",
        "prefix_hold_budget_pool": "shared_candidate_handoff_rounds",
        "full_depth": True, "fresh_b0_before_c1": True,
        "old_run_or_debrief_in_controller_context": False, "forced_diagnostic": False,
        "scope": "independent model-local regression profile; do not pool scores across profiles"}
    write(root / "manifest.json", manifest)
    with tarfile.open(root / "source_snapshot.tgz", "w:gz") as archive:
        for p in source:
            archive.add(p, arcname=str(p.relative_to(repo)))
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.7)
    original_b0, original_cleanup = baselines.run_b0, HookedTransformerWorkerRuntime.cleanup_expired

    def checked_b0(*positional, **keyword):
        result = original_b0(*positional, **keyword)
        observed = {k: getattr(result, k) for k in ("prompt", "output", "score", "steps")}
        matched = None if reference is None else all(observed[k] == reference["b0"][k] for k in observed)
        write(root / "baseline_identity_check.json", {"observed": observed, "reference_match": matched})
        if matched is False:
            raise ValueError("Fresh B0 differs from the supplied reference; stopping before C1")
        return result

    def cleanup(worker):
        original_cleanup(worker)
        gc.collect()
        if args.device == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    baselines.run_b0 = checked_b0
    HookedTransformerWorkerRuntime.cleanup_expired = cleanup
    started = time.monotonic()
    try:
        print(json.dumps({"event": "pair_manifest_sealed", "profile": args.profile}), flush=True)
        code = run_pair(call)
        if code != 0:
            raise RuntimeError(f"Pair exited with code {code}")
        if any(sha(repo / name) != value for name, value in manifest["code_hashes"].items()):
            raise RuntimeError("Source changed during the pair; artifacts retained but comparison not sealed")
        report = audit(root, manifest)
        report["artifact_hashes"] = {str(p.relative_to(root)): sha(p) for p in root.rglob("*")
                                    if p.is_file() and p.suffix != ".log"}
        write(root / "audit.json", report)
        write(root / "run_status.json", {"status": "complete", "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"event": "pair_complete", "profile": args.profile,
            "score_delta": report["score_delta"], "action_counts": report["candidate_action_counts"]}), flush=True)
        return 0
    except Exception as exc:
        message = str(exc).replace(os.environ.get("OPENAI_API_KEY", "no-key"), "[redacted]")
        message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", message)
        write(root / "run_status.json", {"status": "failed", "error_type": type(exc).__name__,
            "error": message, "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"event": "pair_failed", "error_type": type(exc).__name__, "error": message}), flush=True)
        return 1
    finally:
        baselines.run_b0 = original_b0
        HookedTransformerWorkerRuntime.cleanup_expired = original_cleanup


if __name__ == "__main__":
    raise SystemExit(main())
