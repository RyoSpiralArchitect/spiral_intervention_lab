"""Fresh bounded B0 screen for the matched plain/preview selector study."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from ..tasks.structural_rescue_generator import FAMILIES_V3, VERBS_V3, candidate_rows, generate
from .structural_rescue_benchmark import Meter
from .structural_rescue_screen import execute, select
from .structural_rescue_screen_v3 import manifest as prior_manifest


SEED = 2026111101
COUNT = 30
RUN = {"worker_model": "meta-llama/Llama-3.2-3B-Instruct", "device": "mps",
       "dtype": "float16", "mps_memory_fraction": 0.7}


def manifest(checkpoint: dict) -> dict:
    return {**prior_manifest(), "version": "structural_rescue_screen_v4", "seed": SEED,
            "count": COUNT, "candidates": candidate_rows(generate(SEED, COUNT, VERBS_V3, FAMILIES_V3)),
            "selected_split": "rescue_v4", "run": RUN, "disk_floor_gib": 10,
            "wall_seconds": 1800, "forward_caps": dict(Meter.FORWARD_CAPS),
            "checkpoint_sha256": checkpoint,
            "follow_up": "paired_b0_fixed_routed_controller_controller_b0_preview",
            "analysis_plan": {"primary": "paired_preview_vs_plain_on_reproduced_b0_failures",
                              "harm_check": "b0_reproduction_passes_kept_as_regression_stratum",
                              "screen_denominator": "all_30_in_order_including_unavailable_boundaries",
                              "stop_rule": "no_automatic_extension_if_no_failures",
                              "inference": "descriptive_same_generator_not_population_efficacy"}}


def manifest_bytes(checkpoint: dict) -> bytes:
    return (json.dumps(manifest(checkpoint), indent=2, sort_keys=True) + "\n").encode()


def selected_cases(screen_dir: Path):
    frozen = (screen_dir / "manifest.json").read_bytes()
    protocol = json.loads(frozen)
    checkpoint = protocol.get("checkpoint_sha256")
    if not checkpoint or frozen != manifest_bytes(checkpoint):
        raise ValueError("screen_manifest_differs_from_generator_v4")
    raw = (screen_dir / "results.json").read_bytes()
    result = json.loads(raw)
    if (result.get("status") != "complete"
            or result.get("manifest_sha256") != hashlib.sha256(frozen).hexdigest()
            or result.get("checkpoint_sha256_verified") != checkpoint):
        raise ValueError("screen_incomplete_or_unbound")
    rows = result.get("rows", [])
    if ([row["case_id"] for row in rows] != [case["case_id"] for case in protocol["candidates"]]
            or select(rows) != result.get("selected_case_ids")):
        raise ValueError("screen_rows_or_selection_mismatch")
    by_id = {case.case_id: case for case in generate(SEED, COUNT, VERBS_V3, FAMILIES_V3)}
    cases = tuple(replace(by_id[key], split="rescue_v4") for key in result["selected_case_ids"])
    return cases, {"screen_manifest_sha256": hashlib.sha256(frozen).hexdigest(),
                   "screen_results_sha256": hashlib.sha256(raw).hexdigest(),
                   "selected_case_ids": list(result["selected_case_ids"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path)
    for name, default in RUN.items():
        parser.add_argument("--" + name.replace("_", "-"), default=default,
                            type=float if isinstance(default, float) else str)
    args = parser.parse_args(argv)
    source = json.loads(args.checkpoint_manifest.read_text())
    checkpoint = source.get("checkpoint_sha256") or source.get("checkpoint_sha256_verified")
    if not checkpoint:
        parser.error("checkpoint manifest must contain expected checkpoint file hashes")
    return execute(args, seed=SEED, count=COUNT, verbs=VERBS_V3, families=FAMILIES_V3,
                   frozen=manifest_bytes(checkpoint))


if __name__ == "__main__":
    raise SystemExit(main())
