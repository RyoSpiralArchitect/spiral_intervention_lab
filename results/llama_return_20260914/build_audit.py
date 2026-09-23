"""Build the comparison receipt from completed local reports, not debrief claims."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent


def read(name):
    return json.loads((ROOT / name).read_text())


def events(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    baselines = []
    for directory in ("hf", "tlens", "gpt2_hf_fp16", "gpt2_tlens_fp16", "gpt2_hf_fp32", "gpt2_tlens_fp32"):
        report = read(f"{directory}/report.json")
        for fmt, row in report["conditions"].items():
            fidelity = row.get("fidelity", {})
            comparisons = fidelity.get("teacher_forced_rows", [])
            baselines.append({
                "artifact": f"{directory}/report.json", "worker": report["identity"]["model_type"],
                "backend": report["backend"], "format": fmt, "dtype": report["identity"]["dtype"],
                "prompt_hash": row["prompt_hash"], "input_token_count": len(row["input_token_ids"]),
                "output": row["output"], "score": row["score"], "task_done": row["task_done"],
                "fidelity_passed": fidelity.get("all_passed"),
                "greedy_tokens_match": fidelity.get("greedy_token_ids_match"),
                "teacher_forced_count": len(comparisons),
                "max_kl": max((r["kl_hf_to_tlens"] for r in comparisons), default=None),
                "max_centered_logit_rmse": max((r["centered_logit_rmse"] for r in comparisons), default=None),
            })
    b0_events = events("luna_live_r2/b0.jsonl")
    c1_events = events("luna_live_r2/c1.jsonl")
    b0 = next(row for row in b0_events if row["event"] == "episode_end")
    c1 = next(row for row in c1_events if row["event"] == "episode_end")
    commands = [row for row in c1_events if row["event"] == "controller_command"]
    diagnostics = [row for row in c1_events if row["event"] == "controller_diagnostic_result"]
    audit = {
        "measurement_scope": "one-seed worker comparison; not an architecture-only causal claim",
        "preflight": baselines,
        "raw_prompt_identical_across_models": len({r["prompt_hash"] for r in baselines if r["format"] == "raw"}) == 1,
        "runtime_pair": {
            "controller": "gpt-5.6-luna", "worker_layers": 28, "dtype": "float16", "device": "mps",
            "b0": b0, "c1": c1, "output_identical": b0["output"] == c1["output"],
            "prompt_identical": b0_events[0]["prompt"] == c1_events[0]["prompt"],
            "command_decisions": dict(collections.Counter(row["command"]["decision"] for row in commands)),
            "event_counts": dict(collections.Counter(row["event"] for row in c1_events)),
            "diagnostic_count": len(diagnostics),
            "diagnostic_names": [row.get("diagnostic") for row in diagnostics],
            "rollout_apply_command_count": sum(row["command"]["decision"] == "apply" for row in commands),
            "matched_response_controls": [row["target_piece_binding_seed_matrix_summary"]
                                          for row in diagnostics if row.get("target_piece_binding_seed_matrix_executed")],
            "selection_sources": dict(collections.Counter(row.get("controller_selection_source") for row in commands)),
            "policy_changed": False, "paired_baseline_trace_available": True,
        },
        "incomplete_attempt": read("luna_live/attempt_status.json"),
        "confounds": ["parameter_count", "pretraining_and_instruction_tuning", "tokenizer",
                      "architecture", "Llama_unprocessed_RMSNorm_vs_GPT2_weight_processing",
                      "older_GPT2_live_runs_used_FP32_and_older_observer_code"],
        "not_claimed": ["controller_caused_task_success", "architecture_is_the_unique_cause",
                        "operator_certification_from_baseline_success", "generalization_beyond_seed_7"],
        "code_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "code_hash_scope": "working-tree implementations used by completed r2 run, not a committed release",
        "code_hashes": {},
    }
    for name in ("examples/digit_transform_e2e.py", "examples/worker_fidelity_preflight.py", "runtime/worker.py"):
        path = REPO / "SpiralInterventionLab" / name
        audit["code_hashes"]["SpiralInterventionLab/" + name] = sha(path)
    audit["artifacts"] = {
        str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts
    }
    (ROOT / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit["runtime_pair"], indent=2))


if __name__ == "__main__":
    main()
