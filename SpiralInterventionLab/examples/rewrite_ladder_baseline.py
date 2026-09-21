"""Local, no-edit capability screen for the prespecified rewrite ladder.

This bypasses controller/observer orchestration, not model layers. Confirm a
selected failure with full-runtime B0 before making a Luna intervention claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from transformers import AutoConfig

from .digit_transform_e2e import create_task_env, load_worker_model
from ..runtime.codecs import ModelTokenizerCodec
from ..tasks.rewrite_ladder import REWRITE_LADDER_SEEDS, REWRITE_LADDER_TASKS, SpiralRewriteLadderEnv


SELECTION_RULE = (
    "Keep the legacy control successful; take the first nonempty, non-truncated "
    "lexical failure in ascending level and manifest case order for human "
    "coherence review and full-runtime B0 confirmation, not automatic promotion."
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_fixtures() -> list[dict]:
    fixtures = []
    conditions = [("constrained_rewrite", 7)] + [
        (task, seed) for task in REWRITE_LADDER_TASKS for seed in REWRITE_LADDER_SEEDS
    ]
    for task, seed in conditions:
        env = create_task_env(task)
        prompt = env.reset(seed)
        episode = env.current_episode
        ladder = env.benchmark_manifest() if isinstance(env, SpiralRewriteLadderEnv) else {
            "case_id": "legacy_budget", "difficulty_level": 0,
        }
        fixtures.append({
            "condition_id": f"{task}:{ladder['case_id']}:seed{seed}",
            "task": task, "task_id": env.task_id, "seed": seed, "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "source_text": episode.source_text,
            "required_terms": list(episode.required_terms),
            "forbidden_terms": list(episode.forbidden_terms), "max_words": episode.max_words,
            "max_new_tokens": env.worker_runtime_kwargs()["max_generated_tokens"],
            "benchmark": ladder,
        })
    return fixtures


@torch.inference_mode()
def generate_condition(model, fixture: dict, *, device: str, eos_ids: set[int]) -> dict:
    env = create_task_env(fixture["task"])
    if env.reset(fixture["seed"]) != fixture["prompt"]:
        raise ValueError("Task changed after the manifest was sealed")
    tokenizer = model.tokenizer
    ids = tokenizer.encode(fixture["prompt"], add_special_tokens=False)
    if ModelTokenizerCodec(model).encode(fixture["prompt"]).tolist() != ids:
        raise ValueError("Runtime codec and input tokenizer disagree")
    if len(ids) + fixture["max_new_tokens"] > model.cfg.n_ctx:
        raise ValueError("Fixture would exceed the model context window")
    torch.manual_seed(fixture["seed"])
    output_ids, steps = [], []
    stop_reason = "max_new_tokens"
    started = time.monotonic()
    for step in range(fixture["max_new_tokens"]):
        logits = model(torch.tensor([ids + output_ids], device=device), return_type="logits")[0, -1].float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits; do not label this a task failure")
        token = int(logits.argmax())
        logp = logits.log_softmax(-1)
        steps.append({"step": step, "token_id": token, "piece": tokenizer.decode([token]),
                      "top1_probability": float(logp[token].exp()),
                      "entropy": float(-(logp.exp() * logp).sum())})
        output_ids.append(token)
        output = tokenizer.decode(output_ids, skip_special_tokens=True)
        if token in eos_ids:
            stop_reason = "eos"
            break
        if env.stop_checker(output):
            stop_reason = "task_sentence_stop"
            break
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    words = output.strip().split()
    return {
        "condition_id": fixture["condition_id"], "task": fixture["task"],
        "seed": fixture["seed"], "difficulty_level": fixture["benchmark"]["difficulty_level"],
        "case_id": fixture["benchmark"]["case_id"], "prompt_sha256": fixture["prompt_sha256"],
        "input_token_ids": ids, "output_token_ids": output_ids, "output": output,
        "score": env.score(output), "task_done": env.done(output),
        "task_feedback": env.task_feedback(output), "word_count": len(words),
        "stop_reason": stop_reason, "truncated": stop_reason == "max_new_tokens",
        "adjacent_word_repeat_count": sum(a.lower() == b.lower() for a, b in zip(words, words[1:])),
        "steps": steps, "seconds": time.monotonic() - started,
    }


def summarize(rows: list[dict], fixtures: list[dict]) -> dict:
    if [row["condition_id"] for row in rows] != [fixture["condition_id"] for fixture in fixtures]:
        raise ValueError("Summary requires every prespecified condition, in manifest order")
    selected = None
    if rows[0]["task_done"] and not rows[0]["truncated"]:
        selected = next((row for row in rows[1:] if not row["task_done"]
                         and not row["truncated"] and row["output"].strip()), None)
    levels = []
    for level in range(4):
        members = [row for row in rows if row["difficulty_level"] == level]
        levels.append({
            "level": level, "count": len(members),
            "passed": sum(row["task_done"] for row in members),
            "truncated": sum(row["truncated"] for row in members),
            "violation_counts": dict(Counter(reason for row in members
                                              for reason in row["task_feedback"]["constraint_violations"])),
        })
    return {
        "levels": levels, "selection_rule": SELECTION_RULE,
        "selected_condition_id": selected["condition_id"] if selected else None,
        "selected_next_step": "human_coherence_review_then_full_runtime_b0" if selected else "no_failure_selected",
        "controller_calls": 0, "interventions": 0, "production_apply_allowed": False,
        "measurement_scope": "full_depth_tlens_no_edit_raw_greedy_without_observer_or_controller",
        "semantic_preservation_certified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float16", "float32", "bfloat16"), default="float16")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.worker_model_path.expanduser().resolve(strict=True)
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    fixtures = build_fixtures()
    source_root = Path(__file__).resolve().parents[2]
    manifest = {
        "benchmark_version": "rewrite_ladder_v1", "selection_rule": SELECTION_RULE,
        "checkpoint_path": str(checkpoint), "worker_model_alias": args.worker_model,
        "model_type": config.model_type, "layers": config.num_hidden_layers,
        "vocab_size": config.vocab_size, "device": args.device, "dtype": args.dtype,
        "prompt_format": "raw", "sampling": "greedy", "kv_cache": False,
        "checkpoint_sha256": {},
        "source_sha256": {str(path.relative_to(source_root)): sha256_file(path) for path in (
            Path(__file__).resolve(), source_root / "SpiralInterventionLab/tasks/rewrite_ladder.py",
            source_root / "SpiralInterventionLab/tasks/language_tasks.py",
            source_root / "SpiralInterventionLab/examples/digit_transform_e2e.py",
        )},
        "fixtures": fixtures,
    }
    checkpoint_files = sorted(set(checkpoint.glob("*.safetensors")) | set(checkpoint.glob("pytorch_model*.bin")))
    checkpoint_files += [checkpoint / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    for path in checkpoint_files:
        if path.is_file():
            manifest["checkpoint_sha256"][path.name] = sha256_file(path)
    # Exclusive directory creation prevents accidental replacement of old runs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"event": "manifest_sealed", "sha256": sha256_file(manifest_path),
                      "conditions": len(fixtures)}), flush=True)
    rows = []
    try:
        model = load_worker_model(args.worker_model, model_path=checkpoint, device=args.device,
                                  dtype=args.dtype, hf_offline=True, mps_mode="conservative").eval()
        if model.cfg.n_layers != config.num_hidden_layers:
            raise ValueError("Full-depth worker required; no truncation fallback")
        eos = config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        with (args.output_dir / "conditions.jsonl").open("x") as stream:
            for fixture in fixtures:
                row = generate_condition(model, fixture, device=args.device, eos_ids=eos_ids)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                print(json.dumps({"event": "condition_complete", "condition_id": row["condition_id"],
                                  "output": row["output"], "score": row["score"], "done": row["task_done"],
                                  "stop_reason": row["stop_reason"]}), flush=True)
        report = {"manifest_sha256": sha256_file(manifest_path),
                  "conditions_sha256": sha256_file(args.output_dir / "conditions.jsonl"),
                  **summarize(rows, fixtures)}
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report), flush=True)
    except Exception as exc:
        (args.output_dir / "error.json").write_text(json.dumps({
            "error_type": type(exc).__name__, "error": str(exc), "completed_conditions": len(rows),
        }, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
