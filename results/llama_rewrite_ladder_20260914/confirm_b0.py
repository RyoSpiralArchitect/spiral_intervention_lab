"""Confirm the prespecified screen selection with the unmodified B0 runtime."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from SpiralInterventionLab.examples.digit_transform_e2e import (
    build_hooked_transformer_worker_runtime, create_task_env, load_worker_model,
)
from SpiralInterventionLab.examples.rewrite_ladder_baseline import sha256_file
from SpiralInterventionLab.runtime import JSONLStructuredLogger, run_b0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.screen_dir / "manifest.json").read_text())
    summary = json.loads((args.screen_dir / "summary.json").read_text())
    assert sha256_file(args.screen_dir / "manifest.json") == summary["manifest_sha256"]
    assert sha256_file(args.screen_dir / "conditions.jsonl") == summary["conditions_sha256"]
    selected = summary["selected_condition_id"]
    if selected is None:
        raise ValueError("The frozen screen selected no failure")
    fixture = next(row for row in manifest["fixtures"] if row["condition_id"] == selected)
    screened = next(json.loads(line) for line in (args.screen_dir / "conditions.jsonl").read_text().splitlines()
                    if json.loads(line)["condition_id"] == selected)
    checkpoint = args.worker_model_path.expanduser().resolve(strict=True)
    for name, digest in manifest["checkpoint_sha256"].items():
        assert sha256_file(checkpoint / name) == digest, name
    env = create_task_env(fixture["task"])
    assert env.reset(fixture["seed"]) == fixture["prompt"]
    assert env.worker_runtime_kwargs()["max_generated_tokens"] == fixture["max_new_tokens"]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    started = time.monotonic()
    model = load_worker_model(manifest["worker_model_alias"], model_path=checkpoint,
                              device=manifest["device"], dtype=manifest["dtype"],
                              hf_offline=True, mps_mode="conservative").eval()
    assert model.cfg.n_layers == manifest["layers"]
    worker = build_hooked_transformer_worker_runtime(
        model, env, seed=fixture["seed"], activation_surface_profile="activation_patch_expanded",
        run_id="rewrite_ladder_b0_confirmation", episode_id=selected,
    )
    result = run_b0(env, worker, seed=fixture["seed"], logger=JSONLStructuredLogger(args.output_dir / "b0.jsonl"))
    report = {
        "condition_id": selected, "result": asdict(result), "task_done": env.done(result.output),
        "task_feedback": env.task_feedback(result.output),
        "screen_output_match": result.output == screened["output"],
        "screen_score_match": result.score == screened["score"],
        "measurement_scope": "full_runtime_b0_including_observer_and_packet_construction",
        "controller_calls": 0, "rollout_edits": 0,
        "production_apply_permission_changed": False,
        "model_layers": model.cfg.n_layers, "dtype": manifest["dtype"], "device": manifest["device"],
        "screen_manifest_sha256": summary["manifest_sha256"],
        "b0_jsonl_sha256": sha256_file(args.output_dir / "b0.jsonl"),
        "confirmation_script_sha256": sha256_file(Path(__file__)), "seconds": time.monotonic() - started,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
