"""Local-only remeasurement of a recorded no-apply prefix. No provider calls."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from .digit_transform_e2e import build_hooked_transformer_worker_runtime, create_task_env, load_worker_model
from ..runtime.response_probe import matched_response_probe
from ..runtime.evidence_inspection import evidence_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--worker-model-path", required=True)
    parser.add_argument("--worker-model", default="gpt2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--controller-step", type=int, default=4)
    parser.add_argument("--comparison-axis", choices=("seed_provenance", "source_localization"), default="seed_provenance")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.source_jsonl.read_bytes()
    if args.source_jsonl.suffix == ".gz":
        raw = gzip.decompress(raw)
    events = [json.loads(line) for line in raw.splitlines()]
    # Observation steps count emitted worker tokens (1-based); command and
    # diagnostic events use controller rounds (0-based).
    observation = next(e for e in events if e.get("event") == "controller_observation" and e.get("step") == args.controller_step + 1)
    measured = next(e for e in events if e.get("event") == "controller_diagnostic_result" and e.get("step") == args.controller_step
                    and e.get("target_piece_binding_seed_matrix_executed"))
    if observation.get("active_edit_ids") or any(e.get("command", {}).get("decision") not in (None, "noop") for e in events
            if e.get("event") == "controller_command" and e.get("step", 0) <= args.controller_step):
        raise ValueError("Only an unmodified no-apply trajectory can be rebuilt by this harness")
    env = create_task_env("constrained_rewrite")
    env.reset(args.seed)
    if observation.get("task_id") != env.task_id:
        raise ValueError("This harness reconstructs constrained_rewrite episodes only")
    prompt = events[0]["prompt"]
    model = load_worker_model(args.worker_model, model_path=args.worker_model_path, hf_offline=True)
    worker = build_hooked_transformer_worker_runtime(model, env, seed=args.seed,
        activation_surface_profile="activation_patch_expanded", max_diagnostic_calls_per_run=12, diagnostic_result_window=12)
    worker.reset(prompt)
    for _ in range(args.controller_step + 1):
        worker.step()
    if worker.final_text() != observation["generated_tail"]:
        raise ValueError("Rebuilt prefix differs from the recorded prefix")
    # Replay recorded seed descriptors rather than relabeling compact pool rows
    # whose original seed provenance may have been omitted.
    seeds = measured["target_piece_binding_seed_matrix_rows"]
    objective = measured["target_piece_binding_seed_matrix_summary"]["objective_bundle_key"]
    report = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term=objective.split(":")[1],
        comparison_axis=args.comparison_axis)
    from ..runtime.response_promotion import response_review_readiness
    report["promotion_readiness_by_observable"] = {
        r["observable_id"]: response_review_readiness(r, report["measurement_context_id"]) for r in report["rows"]}
    original = measured["target_piece_binding_seed_matrix_summary"]
    report["source_log"] = {"path": str(args.source_jsonl), "sha256": hashlib.sha256(raw).hexdigest(),
        "controller_step": args.controller_step, "worker_step": worker._steps, "prefix": worker.final_text(), "seed": args.seed,
        "local_only": True, "provider_calls": 0,
        "measurement_context_matches_live": report.get("measurement_context_id") == original.get("measurement_context_id"),
        "execution_ids_match_live": {r.get("execution_id") for r in report["rows"]} ==
                                    {r.get("execution_id") for r in measured.get("target_piece_binding_seed_matrix_rows", ())}}
    worker._diagnostic_results = [measured]
    worker.max_diagnostic_calls_per_run = 1
    packet = worker.build_controller_packet()
    catalog = evidence_catalog([measured])
    report["inspection_after_replay_budget_exhaustion"] = worker.request_controller_diagnostics(
        {"diagnostic": "inspect_evidence", "execution_id": next(r["execution_id"] for r in catalog if r.get("execution_id")), "limit": 4},
        packet=packet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report.get(k) for k in ("status", "row_count", "new_measurement_count", "physical_replay_count",
        "no_edit_control", "state_restored", "source_log", "errors")}))


if __name__ == "__main__":
    main()
