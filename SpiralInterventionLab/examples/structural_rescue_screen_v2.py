"""Second pre-registered B0 screen: extended role check plus restricted verbs.

Two v1-screen lessons are applied, both pre-registered before any model call:
the role check now binds double-object datives (role_binding change), and the
generator draws only verbs whose natural greedy output keeps the
`to <recipient>` boundary (send/take/bring; v1's give/hand/mail drifted to
double objects and never formed a boundary). The candidate count rises from 72
to 120 so the selection limit of 12 is reachable. --prepare freezes the
generator version, seed, count, selection rule and follow-up; --run decodes
each candidate once at the v2 boundary without edits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..tasks.structural_rescue import RescueCase
from ..tasks.structural_rescue_generator import (
    CONNECTIVES, DEADLINES, FAMILIES, GENERATOR_VERSION_V2, NAMES, OBJECTS, VERBS_V2,
    candidate_rows, generate,
)
from .structural_rescue_screen import DISK_FLOOR_GIB, SELECTION_LIMIT, SELECTION_RULE, execute


SEED = 20261106
COUNT = 120


def manifest() -> dict[str, Any]:
    return {"version": "structural_rescue_screen_v2", "generator": GENERATOR_VERSION_V2,
            "seed": SEED, "count": COUNT, "candidates": candidate_rows(generate(SEED, COUNT, VERBS_V2)),
            "pools": {"names": NAMES, "objects": OBJECTS, "verbs": VERBS_V2, "deadlines": DEADLINES,
                      "connectives": CONNECTIVES, "families": FAMILIES},
            "screen": "greedy_b0_from_the_v2_boundary_no_edits_no_controller",
            "boundary_rule": "first_natural_prefix_ending_in_to_first_source_recipient",
            "selection_rule": SELECTION_RULE, "selection_limit": SELECTION_LIMIT,
            "selected_split": "rescue_v2",
            "evaluator_changes_since_screen_v1":
                ["role_check_binds_double_object_dative_recipients",
                 "generator_verbs_restricted_to_send_take_bring"],
            "follow_up": "structural_rescue_v2_norm_relative_contract_datasheet_or_unchanged_contract_on_selected_cases",
            "analysis_plan": {"primary": "joint_success_count_per_policy_on_the_rescue_v2_split",
                              "comparison": "controller_vs_fixed_readout_paired_by_case",
                              "harm_check": "losses_vs_b0_on_the_v1_confirmation_split",
                              "b0_reproduction": "a_selected_case_where_b0_succeeds_is_reported_not_dropped",
                              "exploratory": "b0_failures_by_type_constraint_vs_role",
                              "inference": "descriptive_counts_only_no_significance_claims_at_n_le_12"},
            "max_words": 18, "max_generated_tokens": 64, "disk_floor_gib": DISK_FLOOR_GIB,
            "production_apply_allowed": False}


def manifest_bytes() -> bytes:
    return (json.dumps(manifest(), indent=2, sort_keys=True) + "\n").encode()


def selected_cases(screen_dir: Path) -> tuple[tuple[RescueCase, ...], dict[str, Any]]:
    """Rebuild the v2-selected cases and bind them to the frozen screen receipts."""
    frozen = manifest_bytes()
    if (screen_dir / "manifest.json").read_bytes() != frozen:
        raise ValueError("screen_manifest_differs_from_generator_v2")
    results_bytes = (screen_dir / "results.json").read_bytes()
    results = json.loads(results_bytes)
    if results.get("status") != "complete" or results.get("manifest_sha256") != hashlib.sha256(frozen).hexdigest():
        raise ValueError("screen_incomplete_or_unbound")
    by_id = {case.case_id: case for case in generate(SEED, COUNT, VERBS_V2)}
    cases = tuple(RescueCase(case_id, "rescue_v2", by_id[case_id].source, by_id[case_id].frames,
                             by_id[case_id].deadline) for case_id in results["selected_case_ids"])
    return cases, {"screen_manifest_sha256": hashlib.sha256(frozen).hexdigest(),
                   "screen_results_sha256": hashlib.sha256(results_bytes).hexdigest(),
                   "selected_case_ids": list(results["selected_case_ids"])}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--worker-model-path", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--mps-memory-fraction", type=float, default=0.85)
    args = parser.parse_args(argv)
    return execute(args, seed=SEED, count=COUNT, verbs=VERBS_V2, frozen=manifest_bytes())


if __name__ == "__main__":
    raise SystemExit(main())
