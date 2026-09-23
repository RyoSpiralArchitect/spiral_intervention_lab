"""Local-only allocator/lifetime probe; retains full model and trace contents."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import torch

from SpiralInterventionLab.examples.digit_transform_e2e import build_hooked_transformer_worker_runtime, create_task_env, load_worker_model


def memory(tag):
    import os
    record = {
        "event": tag, "rss_bytes": int(subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "rss="])) * 1024,
        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        "mps_recommended_bytes": torch.mps.recommended_max_memory(),
    }
    print(json.dumps(record), flush=True)


def clear_unused():
    gc.collect()
    torch.mps.synchronize()
    torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-model-path", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    # Fail in Python rather than exhausting system swap again.
    torch.mps.set_per_process_memory_fraction(0.7)
    memory("before_load")
    model = load_worker_model("meta-llama/Llama-3.2-3B-Instruct", model_path=args.worker_model_path,
                              device="mps", dtype="float16", hf_offline=True, mps_mode="conservative").eval()
    memory("after_load")
    clear_unused()
    memory("after_load_collect")
    env = create_task_env("constrained_rewrite_l1")
    prompt = env.reset(0)
    worker = build_hooked_transformer_worker_runtime(model, env, seed=0, activation_surface_profile="activation_patch_expanded")
    worker.reset(prompt)
    for step in range(3):
        start = time.monotonic()
        worker.step()
        memory(f"step_{step}_after_forward")
        packet = worker.build_controller_packet()
        memory(f"step_{step}_after_packet")
        print(json.dumps({"event": "output", "step": step, "output": worker.final_text(),
                          "seconds": time.monotonic() - start,
                          "trace_frames": worker.trace_recorder.step_count}), flush=True)
        worker.observe_recent_effects()
        worker.tick_ttl()
        worker.cleanup_expired()
        del packet
        clear_unused()
        memory(f"step_{step}_after_collect")


if __name__ == "__main__":
    main()
