"""Diagnostic-only survey: do clause connectives rescue failed ladder rewrites?

For every rewrite-ladder case, decode the no-edit greedy output, score it with the
unchanged evaluator plus the offline role-binding check, and for each failure walk
the natural trajectory. At each prefix, every listed connective ranked within the
top-k next tokens is forced once and the sentence is finished greedily. The rank
of each missing term's first piece is recorded from the same logits.

A rescue is a forced continuation with score 1.0, no constraint violations and
correct roles. Forcing is an operator-free ceiling; no controller call, rollout
edit or apply permission is involved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from ..tasks.rewrite_ladder import REWRITE_LADDER_SEEDS, REWRITE_LADDER_TASKS
from .digit_transform_e2e import build_hooked_transformer_worker_runtime, create_task_env, load_worker_model
from .replay_orphan_blueprint_ceiling import _branch, _release_memory, _scored, verify_checkpoint
from .replay_structure_token_probe import single_token_id, token_rank

DEFAULT_CONNECTIVES = ",| while|;| and| whereas| but"


def first_piece_id(codec: Any, text: str, *, special_ids: set[int]) -> int | None:
    ids = [int(i) for i in codec.encode(text).reshape(-1).tolist() if int(i) not in special_ids]
    return ids[0] if ids else None


def is_rescue(scored: dict[str, Any]) -> bool:
    return (scored.get("score") == 1.0 and not scored.get("constraint_violations")
            and scored.get("role_binding_ok") is True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--checkpoint-manifest", type=Path, required=True,
                        help="manifest whose checkpoint_sha256 map the checkpoint must match")
    parser.add_argument("--reference-conditions", type=Path,
                        help="optional conditions.jsonl with recorded no-edit outputs to compare against")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(REWRITE_LADDER_TASKS))
    parser.add_argument("--connectives", default=DEFAULT_CONNECTIVES)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    args = parser.parse_args(argv)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    if not tasks or not set(tasks) <= set(REWRITE_LADDER_TASKS):
        parser.error(f"--tasks must be a subset of {REWRITE_LADDER_TASKS}")
    connectives = [piece for piece in args.connectives.split("|") if piece]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    manifest_path = args.checkpoint_manifest.expanduser().resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verified_checkpoint = verify_checkpoint(model_path, manifest.get("checkpoint_sha256") or {})
    reference: dict[tuple[str, int], str] = {}
    if args.reference_conditions is not None:
        for line in args.reference_conditions.expanduser().resolve(strict=True).read_text().splitlines():
            row = json.loads(line)
            reference[(str(row.get("task")), int(row.get("seed", -1)))] = str(row.get("output") or "")

    import torch

    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
    model = load_worker_model(args.worker_model, model_path=model_path, device=args.device, dtype=args.dtype,
                              hf_offline=True, mps_mode="conservative")
    special_ids = {int(i) for i in (getattr(model.tokenizer, "all_special_ids", None) or [])}
    report: dict[str, Any] = {
        "kind": "clause_structure_survey",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "worker_model": args.worker_model, "checkpoint_sha256_verified": verified_checkpoint,
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "device": args.device, "dtype": args.dtype, "connectives": connectives, "top_k": args.top_k,
        "rescue_criterion": "score 1.0, no constraint violations, role_binding_ok",
        "scope": "diagnostic_only_forced_connective_ceiling_not_controller_choice_or_rollout_outcome",
        "production_apply_allowed": False, "cases": [],
    }
    started = time.monotonic()
    for task_name in tasks:
        for seed in REWRITE_LADDER_SEEDS:
            task = create_task_env(task_name)
            prompt = task.reset(seed)
            case_id = task.benchmark_manifest()["case_id"]
            worker = build_hooked_transformer_worker_runtime(model, task, seed=seed)
            worker.reset(prompt)
            connective_ids = {piece: single_token_id(worker.codec, piece, special_ids=special_ids)
                              for piece in connectives}
            max_tokens = int(task.worker_runtime_kwargs().get("max_generated_tokens") or 64)
            positions: list[dict[str, Any]] = []
            while not worker.done():
                generated = len(worker._output_token_ids())
                simulated = worker._simulate_decode(max_new_tokens=1, top_k=5,
                                                    score_candidate_text=False, score_observer_check=False)
                if simulated is None:
                    raise RuntimeError(worker._last_simulate_decode_error)
                logits = simulated["first_logits"].float()
                probs = torch.softmax(logits, dim=-1)
                natural_id = int(logits.argmax())
                missing = task.task_feedback(worker.final_text()).get("missing_required_terms") or []
                entry: dict[str, Any] = {
                    "generated_tokens": generated, "prefix": worker.final_text(),
                    "natural_next": worker.codec.decode([natural_id]),
                    "missing_term_first_piece_ranks": {
                        term: token_rank(logits, token_id) for term in missing
                        if (token_id := first_piece_id(worker.codec, " " + term, special_ids=special_ids)) is not None},
                    "forced": [],
                }
                for piece, token_id in connective_ids.items():
                    if token_id is None or token_id == natural_id:
                        continue
                    rank = token_rank(logits, token_id)
                    if rank > args.top_k:
                        continue
                    branch = _branch(worker, task, forced_token_ids=(token_id,),
                                     horizon=max_tokens - generated - 1)
                    entry["forced"].append({
                        "piece": piece, "rank": rank, "prob": round(float(probs[token_id]), 6),
                        "gap_to_top1": round(float(logits[natural_id] - logits[token_id]), 5),
                        "rescue": is_rescue(branch), **branch})
                positions.append(entry)
                worker.step()
                _release_memory()
            natural = _scored(task, worker.final_text())
            rescues = [{"generated_tokens": entry["generated_tokens"], "prefix": entry["prefix"], **forced}
                       for entry in positions for forced in entry["forced"] if forced["rescue"]]
            case = {
                "task": task_name, "seed": seed, "case_id": case_id, "natural": natural,
                "natural_success": is_rescue(natural),
                "matches_reference": (reference.get((task_name, seed)) == natural["text"]
                                      if (task_name, seed) in reference else None),
                "rescue_count": len(rescues),
                "best_rescue": min(rescues, key=lambda row: row["gap_to_top1"]) if rescues else None,
                "positions": positions,
            }
            report["cases"].append(case)
            print(json.dumps({"task": task_name, "case": case_id, "natural_score": natural["score"],
                              "natural_roles": natural.get("role_binding_ok"), "rescues": len(rescues),
                              "best": (case["best_rescue"] or {}).get("text"),
                              "best_gap": (case["best_rescue"] or {}).get("gap_to_top1"),
                              "matches_reference": case["matches_reference"],
                              "elapsed_s": round(time.monotonic() - started, 1)}, ensure_ascii=False), flush=True)
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
