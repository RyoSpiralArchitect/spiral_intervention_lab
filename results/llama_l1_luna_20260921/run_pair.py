"""Hash-bound invocation of the existing paired B0/C1 CLI; no policy changes."""
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import torch

import SpiralInterventionLab.runtime.baselines as baselines
from SpiralInterventionLab.examples.digit_transform_e2e import _build_parser, create_task_env, main
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SCREEN = REPO / "results/llama_rewrite_ladder_20260914"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(name, payload):
    (ROOT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def run():
    args = _build_parser().parse_args()
    if args.controller_api_key:
        raise ValueError("Use the existing environment key, never a logged CLI secret")
    if args.c1_only or not args.no_b1 or args.num_seeds != 1:
        raise ValueError("This experiment requires paired B0/C1, no B1, and one seed")
    if args.provider != "openai" or args.controller_model != "gpt-5.6-luna":
        raise ValueError("Frozen controller identity mismatch")
    if args.worker_first_n_layers is not None or args.worker_dtype != "float16":
        raise ValueError("Use full-depth FP16, as in the screen")
    if args.worker_decoder_control_mode != "off" or args.worker_loop_rescue_edits_per_run != 0:
        raise ValueError("Do not change decoder/apply policy for this comparison")
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=False)
    summary = json.loads((SCREEN / "summary.json").read_text())
    manifest = json.loads((SCREEN / "manifest.json").read_text())
    if sha(SCREEN / "manifest.json") != summary["manifest_sha256"]:
        raise ValueError("Screen manifest identity mismatch")
    if sha(SCREEN / "conditions.jsonl") != summary["conditions_sha256"]:
        raise ValueError("Screen output identity mismatch")
    fixture = next(row for row in manifest["fixtures"] if row["condition_id"] == summary["selected_condition_id"])
    expected_b0 = next(row for row in map(json.loads, (SCREEN / "conditions.jsonl").read_text().splitlines())
                       if row["condition_id"] == fixture["condition_id"])
    if args.task != fixture["task"] or args.seed != fixture["seed"]:
        raise ValueError("Use the prespecified selected fixture, not a retuned task")
    env = create_task_env(args.task)
    if env.reset(args.seed) != fixture["prompt"]:
        raise ValueError("Selected prompt changed")
    if env.worker_runtime_kwargs()["max_generated_tokens"] != fixture["max_new_tokens"]:
        raise ValueError("Selected generation budget changed")
    checkpoint = Path(args.worker_model_path).expanduser().resolve(strict=True)
    for name, expected in manifest["checkpoint_sha256"].items():
        if sha(checkpoint / name) != expected:
            raise ValueError(f"Checkpoint identity mismatch: {name}")
    for name, expected in manifest["source_sha256"].items():
        if sha(REPO / name) != expected:
            raise ValueError(f"Screen source changed: {name}")
    code_hashes = {
        str(path.relative_to(REPO)): sha(path)
        for path in sorted((REPO / "SpiralInterventionLab").rglob("*"))
        if path.is_file() and path.suffix in {".py", ".txt", ".json"} and "__pycache__" not in path.parts
    }
    config = json.loads((checkpoint / "config.json").read_text())
    if config["num_hidden_layers"] != 28:
        raise ValueError("Expected the same 28-layer Llama checkpoint")
    write("manifest.json", {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition_id": fixture["condition_id"], "fixture": fixture,
        "screen_manifest_sha256": summary["manifest_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"], "checkpoint_path": str(checkpoint),
        "model_layers": config["num_hidden_layers"], "model_type": config["model_type"],
        "argv": sys.argv[1:], "code_hashes": code_hashes,
        "wrapper_sha256": sha(Path(__file__)),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "code_scope": "current working tree, not a committed release",
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "transformer-lens", "openai")},
        "baseline_scope": "fresh paired B0 before C1; old seed7 control retained as historical reference",
        "primary_outcome": "binary presence of all required terms plus forbidden/word-budget compliance",
        "secondary_outcomes": ["rollout_edit_count", "diagnostic_measurements", "shadow_evidence", "constraint_regressions"],
        "production_policy_changed": False,
        "debrief_scope": "qualitative only; not scoring, certification, or next-run context",
    })
    print(json.dumps({"event": "manifest_sealed", "condition_id": fixture["condition_id"],
                      "model_layers": config["num_hidden_layers"]}), flush=True)
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    original_step = HookedTransformerWorkerRuntime.step
    original_b0 = baselines.run_b0

    def checked_b0(*b0_args, **b0_kwargs):
        result = original_b0(*b0_args, **b0_kwargs)
        matched = result.prompt == fixture["prompt"] and result.output == expected_b0["output"] and result.score == expected_b0["score"]
        write("baseline_identity_check.json", {
            "prompt_output_score_match": matched, "output": result.output,
            "score": result.score, "steps": result.steps,
        })
        if not matched:
            raise ValueError("Fresh B0 differs from the frozen fixture; stopping before controller calls")
        return result

    def timed_step(self):
        started = time.monotonic()
        result = original_step(self)
        print(json.dumps({"event": "worker_step_timing", "step": self._steps,
                          "seconds": time.monotonic() - started, "output": self.final_text()}), flush=True)
        return result

    HookedTransformerWorkerRuntime.step = timed_step
    baselines.run_b0 = checked_b0
    faulthandler.dump_traceback_later(180, repeat=True)
    started = time.monotonic()
    try:
        result = main()
        write("run_status.json", {"status": "complete", "cli_exit_code": result,
                                  "elapsed_seconds": time.monotonic() - started})
        return result
    except Exception as exc:
        message = str(exc).replace(os.environ.get("OPENAI_API_KEY", "no-key-present"), "[redacted]")
        message = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted]", message)
        write("run_status.json", {"status": "failed", "error_type": type(exc).__name__,
                                  "error": message, "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"event": "run_failed", "error_type": type(exc).__name__, "error": message}), flush=True)
        return 1
    finally:
        HookedTransformerWorkerRuntime.step = original_step
        baselines.run_b0 = original_b0
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    raise SystemExit(run())
