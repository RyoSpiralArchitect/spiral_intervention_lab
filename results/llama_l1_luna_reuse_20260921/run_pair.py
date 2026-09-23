"""Repeat the sealed Luna pair with review reuse, never conditional answer hints."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
ORIGINAL = REPO / "results/llama_l1_luna_20260921/run_pair.py"
REFERENCE = REPO / "results/llama_l1_luna_20260921_r2/manifest.json"


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
    reference = json.loads(REFERENCE.read_text())
    if pair.sha(ORIGINAL) != reference["wrapper_sha256"]:
        raise ValueError("Original paired wrapper changed")
    expected = list(reference["argv"])
    # Only machine-local path spelling and a fresh output directory may differ.
    actual = sys.argv[1:]
    for flag in ("--log-dir", "--worker-model-path"):
        expected[expected.index(flag) + 1] = actual[actual.index(flag) + 1]
    if actual != expected:
        raise ValueError("Comparison settings differ from the reference pair")
    if Path(actual[actual.index("--log-dir") + 1]).resolve() != ROOT / "live":
        raise ValueError("Write this run's logs only inside its new evidence directory")
    pair.ROOT = ROOT
    original_write = pair.write
    original_cleanup = HookedTransformerWorkerRuntime.cleanup_expired
    torch.mps.set_per_process_memory_fraction(0.7)

    def annotated_write(name, payload):
        if name == "manifest.json":
            previous = reference["code_hashes"]
            current = payload["code_hashes"]
            payload["followup"] = {
                "reference_manifest": str(REFERENCE.relative_to(REPO)),
                "reference_manifest_sha256": pair.sha(REFERENCE),
                "runner_sha256": pair.sha(Path(__file__)),
                "changed_code_files": [key for key in sorted(set(previous) | set(current))
                                       if previous.get(key) != current.get(key)],
                "new_independent_controller_trajectory": True,
                "purpose": "observe closed-review reuse in a live controller loop",
                "conditional_completion_probe_invoked": False,
                "previous_run_or_debrief_added_to_controller_context": False,
                "worker_task_decoder_and_apply_limits_changed": False,
                "not_a_same_request_sequence_ablation": True,
            }
            payload["memory_housekeeping"] = {
                "mps_memory_fraction": 0.7,
                "method": "collect unreachable objects and release unoccupied MPS buffers after cleanup_expired",
                "matches_reference_allocator_policy": True,
                "paired_trace_contents_changed": False,
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
