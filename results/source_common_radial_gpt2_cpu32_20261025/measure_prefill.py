"""One fixed-text GPT-2 diagnostic; no generation, controller, or apply permission."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from SpiralInterventionLab.examples.digit_transform_e2e import (
    build_hooked_transformer_worker_runtime, create_task_env, load_worker_model,
)
from SpiralInterventionLab.examples.replay_source_clause_diagnostic import measure_current_prefix, source_prefixes
from SpiralInterventionLab.runtime import clause_actuation, clause_choice, clause_radial, clause_specificity, clause_trajectory, diagnostic_budget
from SpiralInterventionLab.runtime.response_probe import tensor_identity

PREFIX = " Nora will take the sample to Ivo"
LEXEMES = (",", " while")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.worker_model_path.expanduser().resolve(strict=True)
    source = ROOT / "results/entity_probe_refresh_llama_20261014_retry1/live/c1.jsonl"
    manifest_path = source.parent.parent / "manifest.json"
    reference_path = ROOT / "results/source_clause_specificity_llama_20261021_aligned/specificity.json"
    prompt, prefixes = source_prefixes(source)
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads(reference_path.read_text())
    if prefixes[8] != PREFIX or reference["measurements"][0]["prefix"] != PREFIX:
        raise ValueError("fixed_text_prefix_reference_mismatch")
    if (reference["source_jsonl_sha256"] != digest(source)
            or reference["source_manifest_sha256"] != digest(manifest_path)):
        raise ValueError("source_provenance_mismatch")
    task = create_task_env(manifest["profile"]["task"])
    if task.reset(0) != prompt:
        raise ValueError("task_fixture_changed")
    report = {
        "kind": "source_common_radial_controls", "schema_version": 1, "status": "preflight",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {str(p): digest(p) for p in (source, manifest_path, reference_path)},
        "profile": {"task": manifest["profile"]["task"], "seed": 0, "worker_model": "gpt2-small"},
        "device": "cpu", "dtype": "float32", "torch_threads": 4,
        "expected_prefix": PREFIX, "reconstruction": "explicit_fixed_text_diagnostic_prefill_not_natural_generation",
        "natural_generated_token_count": 0, "selection_origin": "predeclared_offline_lexeme_pair_not_controller_choice",
        "controller_calls": 0, "production_apply_allowed": False, "rollout_apply_count": 0,
        "caps": "unchanged_runtime_defaults", "diagnostic_budget_limit": 12, "records": [],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "transformer-lens")},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "code_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "SpiralInterventionLab").rglob("*.py"))},
        "harness_sha256": digest(Path(__file__)),
    }

    def save():
        (output / "trajectory.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    worker, counts = None, None
    monitors = ExitStack()
    try:
        files = ("config.json", "generation_config.json", "model.safetensors", "tokenizer_config.json", "merges.txt", "vocab.json")
        report["checkpoint_sha256_verified"] = {name: digest(checkpoint / name) for name in files}
        report["status"] = "loading_model"
        save()
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        model = load_worker_model("gpt2-small", model_path=checkpoint, device="cpu", dtype="float32", hf_offline=True, mps_mode="conservative")
        report["model_execution"] = {"parameter_devices": sorted({str(p.device) for p in model.parameters()}),
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}), "n_layers": model.cfg.n_layers}
        if model.cfg.n_layers != 12 or any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("model_architecture_or_numerics_changed")
        worker = build_hooked_transformer_worker_runtime(model, task, seed=0,
            clause_structure_diagnostics=True, clause_structure_operator_diagnostics=True, max_diagnostic_calls_per_run=12)
        counts = monitors.enter_context(clause_trajectory.count_forwards(worker.runtime_state))
        worker.reset(prompt)
        prompt_tokens = worker._current_token_tensor().detach().clone()
        prefix_ids = worker.codec.encode(PREFIX).reshape(-1).tolist()
        if worker.codec.decode(prefix_ids) != PREFIX:
            raise ValueError("prefix_roundtrip_failed")
        for tid in prefix_ids:
            worker._append_output_token(tid)
        worker._steps = len(prefix_ids)
        if (worker.final_text() != PREFIX or not torch.equal(worker._current_token_tensor()[:, :prompt_tokens.shape[1]], prompt_tokens)):
            raise ValueError("prefill_changed_prompt_or_prefix")
        token_ids = []
        for text in LEXEMES:
            encoded = worker.codec.encode(text).reshape(-1).tolist()
            if len(encoded) != 1 or worker.codec.decode(encoded) != text:
                raise ValueError(f"source_lexeme_not_exact_single_token:{text!r}")
            token_ids.append(encoded[0])
        report.update(observed_prefix=worker.final_text(), prefill_token_ids=prefix_ids,
            prefill_token_count=len(prefix_ids), position=len(prefix_ids), source_token_ids=token_ids,
            prompt_tokens_identity=tensor_identity(prompt_tokens), source_lexemes=list(LEXEMES),
            radial_control_plan=clause_radial.plan(token_ids), forward_counts_before_diagnostics=dict(counts))
        report["cache_alignment"] = clause_specificity.align_cache(worker)
        report["structure_observation"] = measure_current_prefix(worker)
        if report["structure_observation"]["result"]["status"] != "ok":
            raise ValueError("structure_observation_failed")
        offered = clause_actuation.offer(worker)
        bindings = []
        for tid, text in zip(token_ids, LEXEMES, strict=True):
            rows = [r for r in offered["candidates"] if r["target_token_id"] == tid]
            if len(rows) != 1 or rows[0]["target_piece"] != text or rows[0]["source_provenance"] != "source_body":
                raise ValueError("exact_source_binding_missing")
            if any(s["alpha"] != .04 or s["step_size"] != .08 or s["norm_clip"] != 1.0 for s in rows[0]["surfaces"]):
                raise ValueError("normal_cap_mismatch")
            bindings.append(rows[0])
        report.update(source_bindings=bindings, status="measuring")
        save()

        def record(item):
            filename = f"vectors_{len(report['records']):02d}.pt"
            torch.save(item, output / filename)
            report["records"].append({**clause_trajectory.compact_record(item),
                "vectors_artifact": filename, "vectors_sha256": digest(output / filename)})
            save()
            print(json.dumps({"condition": item["label"], "status": item["result"]["status"], "captured_forwards": len(item["frames"])}), flush=True)

        records = clause_radial.measure(worker, token_ids, record=record)
        report["declared_diagnostic_replay_count"] = 2 + sum(r["result"]["physical_replay_count"] for r in records)
        report["comparison"] = clause_radial.summarize(records)
        report["choice_comparison"] = clause_choice.summarize(records, component_labels=clause_radial.LABELS, decode_token=lambda tid: worker.codec.decode([tid]))
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        monitors.close()
        if counts is not None:
            report["forward_counts_total"] = dict(counts)
            if "forward_counts_before_diagnostics" in report:
                report["diagnostic_interval_forward_counts"] = {k: v - report["forward_counts_before_diagnostics"][k] for k, v in counts.items()}
        if worker is not None:
            report["diagnostic_budget_used"] = diagnostic_budget.used(worker)
            report["diagnostic_budget_left"] = diagnostic_budget.left(worker)
        report["recorded_condition_replay_count"] = sum(r["result"].get("physical_replay_count", 0) for r in report["records"])
        report["recorded_condition_count"] = len(report["records"])
        report["failed_condition_labels"] = [r["label"] for r in report["records"] if r["result"].get("status") != "ok"]
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
