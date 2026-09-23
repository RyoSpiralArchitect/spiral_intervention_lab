"""Retry the sealed pair with allocator housekeeping, not weaker model controls."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import torch

from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ORIGINAL = REPO / "results/llama_l1_luna_20260921/run_pair.py"


def memory():
    return {
        "rss_bytes": int(subprocess.check_output(["ps", "-p", str(os.getpid()), "-o", "rss="])) * 1024,
        "mps_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_bytes": torch.mps.driver_allocated_memory(),
    }


def run():
    spec = importlib.util.spec_from_file_location("sealed_pair", ORIGINAL)
    pair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pair)
    pair.ROOT = ROOT
    original_write = pair.write
    original_cleanup = HookedTransformerWorkerRuntime.cleanup_expired
    torch.mps.set_per_process_memory_fraction(0.7)

    def annotated_write(name, payload):
        if name == "manifest.json":
            payload["memory_retry"] = {
                "previous_attempt": str(ORIGINAL.parent.relative_to(REPO)),
                "previous_status": "external_kill_paging_exhaustion",
                "retry_wrapper_sha256": pair.sha(Path(__file__)),
                "mps_memory_fraction": 0.7,
                "mps_recommended_max_bytes": torch.mps.recommended_max_memory(),
                "housekeeping": "collect unreachable objects and release unoccupied MPS buffers after cleanup_expired",
                "paired_trace_contents_changed": False,
                "model_task_decoder_policy_changed": False,
                "new_independent_controller_trajectory": True,
            }
        original_write(name, payload)

    def cleanup(self):
        original_cleanup(self)
        before = memory()
        gc.collect()
        torch.mps.synchronize()
        torch.mps.empty_cache()
        with (ROOT / "memory_boundaries.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": self._steps, "before": before, "after": memory()}) + "\n")

    pair.write = annotated_write
    HookedTransformerWorkerRuntime.cleanup_expired = cleanup
    try:
        return pair.run()
    finally:
        HookedTransformerWorkerRuntime.cleanup_expired = original_cleanup


if __name__ == "__main__":
    raise SystemExit(run())
