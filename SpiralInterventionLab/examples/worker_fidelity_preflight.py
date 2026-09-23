"""No-edit worker comparison, with HF and TransformerLens in separate processes.

Raw text is the historical runtime condition. Native chat is a separate prompt
format control, never an architecture-only comparison. No controller calls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .digit_transform_e2e import create_task_env, load_worker_model
from ..runtime.codecs import ModelTokenizerCodec


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def checkpoint_file_hashes(checkpoint: Path) -> dict[str, str]:
    metadata = {
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "model.safetensors.index.json", "pytorch_model.bin.index.json",
    }
    files = sorted(
        path for path in checkpoint.iterdir()
        if path.is_file() and (path.name in metadata or path.suffix in {".safetensors", ".bin"})
    )
    hashes = {}
    for path in files:
        sha = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                sha.update(chunk)
        hashes[path.name] = sha.hexdigest()
    return hashes


def compare_logits(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    reference, actual = reference.float().cpu(), actual.float().cpu()
    if reference.shape != actual.shape or not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        return {"valid": False, "reason": "shape_or_nonfinite_logits", "passed": False}
    # A shared logit offset is distribution-invariant (e.g. centered unembed).
    delta = actual - reference
    centered = delta - delta.mean()
    ref_logp, logp = reference.log_softmax(-1), actual.log_softmax(-1)
    kl = max(0.0, float((ref_logp.exp() * (ref_logp - logp)).sum()))
    rms = float(centered.square().mean().sqrt())
    same_top1 = int(reference.argmax()) == int(actual.argmax())
    return {
        "valid": True, "top1_match": same_top1, "kl_hf_to_tlens": kl,
        "centered_logit_rmse": rms, "max_abs_logit_delta": float(delta.abs().max()),
        "top20_overlap": len(set(reference.topk(min(20, reference.numel())).indices.tolist())
                             & set(actual.topk(min(20, actual.numel())).indices.tolist())),
        "passed": same_top1 and kl <= 0.01 and rms <= 0.1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf", "tlens"), required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=18)
    parser.add_argument("--prompt-format", choices=("raw", "native_chat", "both"), default="both")
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.backend == "tlens" and args.reference_dir is None:
        parser.error("tlens requires --reference-dir from a completed HF preflight")
    output_dir = args.output_dir.expanduser().resolve()
    reference_dir = args.reference_dir.expanduser().resolve() if args.reference_dir else None
    if reference_dir == output_dir:
        parser.error("--output-dir must differ from --reference-dir")
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.worker_model_path.expanduser().resolve()
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    env = create_task_env("constrained_rewrite")
    raw_prompt = env.reset(args.seed)
    formats = ("raw", "native_chat") if args.prompt_format == "both" else (args.prompt_format,)
    prompts = {}
    for fmt in formats:
        prompts[fmt] = raw_prompt if fmt == "raw" else tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}], tokenize=False,
            add_generation_prompt=True, date_string="14 Sep 2026",
        )
    identity = {
        "model_type": config.model_type, "layers": config.num_hidden_layers,
        "attention_heads": config.num_attention_heads,
        "kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "vocab_size": config.vocab_size, "dtype": args.dtype, "device": args.device,
        "seed": args.seed, "task": env.task_id, "max_new_tokens": args.max_new_tokens,
        "checkpoint_files": checkpoint_file_hashes(checkpoint),
    }
    reference = None
    if reference_dir:
        reference = json.loads((reference_dir / "report.json").read_text())
        if reference["identity"] != identity:
            raise ValueError("Reference checkpoint/dtype/device/seed/budget mismatch")
    print(json.dumps({"event": "loading", "backend": args.backend, "identity": identity}), flush=True)
    if args.backend == "hf":
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=getattr(torch, args.dtype),
        ).to(args.device).eval()
        forward = lambda tokens: model(tokens, use_cache=False).logits
    else:
        model = load_worker_model(args.worker_model, model_path=checkpoint, device=args.device,
                                  dtype=args.dtype, hf_offline=True, mps_mode="conservative").eval()
        if model.cfg.n_layers != config.num_hidden_layers:
            raise ValueError("Full-depth comparison required; truncated worker rejected")
        forward = lambda tokens: model(tokens, return_type="logits")
    print(json.dumps({"event": "loaded", "seconds": time.monotonic() - started}), flush=True)
    results = {}
    for fmt, prompt in prompts.items():
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if args.backend == "tlens" and ModelTokenizerCodec(model).encode(prompt).tolist() != ids:
            raise ValueError("Runtime codec and HF input tokens differ")
        eos = set(config.eos_token_id if isinstance(config.eos_token_id, list) else [config.eos_token_id])
        output_ids, rows, tensors = [], [], []
        for step in range(args.max_new_tokens):
            logits = forward(torch.tensor([ids + output_ids], device=args.device))[0, -1].float().cpu()
            token = int(logits.argmax())
            tensors.append(logits)
            rows.append({"step": step, "prefix_token_ids": ids + output_ids,
                         "token_id": token, "piece": tokenizer.decode([token]),
                         "top1_probability": float(logits.softmax(-1)[token])})
            output_ids.append(token)
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            if token in eos or env.stop_checker(text):
                break
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        result = {"prompt": prompt, "prompt_hash": digest(prompt), "input_token_ids": ids,
                  "output": text, "output_token_ids": output_ids, "score": env.score(text),
                  "task_done": env.done(text), "task_feedback": env.task_feedback(text), "steps": rows}
        torch.save(torch.stack(tensors), output_dir / f"{fmt}_logits.pt")
        if reference is not None:
            ref = reference["conditions"][fmt]
            if ref["input_token_ids"] != ids or ref["prompt_hash"] != digest(prompt):
                raise ValueError("Reference prompt/token identity mismatch")
            ref_logits = torch.load(reference_dir / f"{fmt}_logits.pt", weights_only=True, map_location="cpu")
            comparisons = []
            # Teacher-forced HF prefixes prevent generated trajectory drift from
            # masquerading as adapter error at later positions.
            for row, expected in zip(ref["steps"], ref_logits, strict=True):
                actual = forward(torch.tensor([row["prefix_token_ids"]], device=args.device))[0, -1]
                comparisons.append({"step": row["step"], **compare_logits(expected, actual)})
            result["fidelity"] = {"teacher_forced_rows": comparisons,
                                  "all_passed": all(row["passed"] for row in comparisons),
                                  "greedy_token_ids_match": ref["output_token_ids"] == output_ids}
        results[fmt] = result
        report = {"backend": args.backend, "identity": identity, "conditions": results,
                  "provider_calls": 0, "production_apply_allowed": False,
                  "checkpoint_path": str(checkpoint), "seconds": time.monotonic() - started,
                  "fidelity_thresholds": {"kl_max": 0.01, "centered_logit_rmse_max": 0.1, "top1_match_required": True}}
        (output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"event": "condition_complete", "format": fmt,
                          "output": text, "score": result["score"], "task_done": result["task_done"],
                          "fidelity": result.get("fidelity")}), flush=True)


if __name__ == "__main__":
    main()
