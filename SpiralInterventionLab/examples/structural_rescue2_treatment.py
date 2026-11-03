"""Freeze the whole rescue-2 treatment from the verified cap sweep, before screen 2 runs.

Reads the sweep's analysis (bound by hash), takes r* from the pre-declared cap
rule, binds the datasheets at r* and at 0.08, renders every controller prompt at
r*, and writes one JSON file that the screen-2 manifest embeds. After that file
exists nothing about the treatment may change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..runtime import reach_datasheet
from . import run_hygiene
from .structural_controller_sampling import ARMS, prompt
from .summarize_structural_table import R_STAR_CANDIDATES, R_STAR_DEFAULT, R_STAR_MARGIN

CONTROL_CAP = 0.08
SAMPLES = 3
DETERMINISTIC_ARMS = ("b0", "fixed", "fixed_readout", "random", "random_readout", "datasheet_rule",
                      "fixed_readout_at_control_cap", "datasheet_rule_at_control_cap")


def treatment(sweep_dir: Path) -> dict[str, Any]:
    analysis_bytes = (sweep_dir / "analysis_sweep.json").read_bytes()
    analysis = json.loads(analysis_bytes)
    if analysis["reproduction"]["mismatches"] or analysis["missing_pool_indices"] or any(analysis["invariants"].values()):
        raise ValueError("sweep_not_clean_enough_to_bind")
    cap = float(analysis["r_star"]["r_star"])
    sheets = {}
    for label, value in (("r_star", cap), ("control", CONTROL_CAP)):
        raw = (sweep_dir / f"datasheet_cap_{value:.2f}.json").read_bytes()
        sheet = json.loads(raw)
        reach_datasheet.verify(sheet, cap=value)
        if sheet["datasheet_sha256"] != analysis["datasheets"][f"{value:.2f}"]:
            raise ValueError("datasheet_differs_from_the_sweep_analysis")
        sheets[label] = {"cap": value, "file_sha256": hashlib.sha256(raw).hexdigest(),
                         "datasheet_sha256": sheet["datasheet_sha256"]}
    return {"version": "structural_rescue2_treatment_v1",
            "sweep_analysis_sha256": hashlib.sha256(analysis_bytes).hexdigest(),
            "cap_rule": {"candidates": list(R_STAR_CANDIDATES), "default": R_STAR_DEFAULT, "margin": R_STAR_MARGIN,
                         "result": analysis["r_star"]},
            "r_star": cap, "control_cap": CONTROL_CAP, "datasheets": sheets,
            "evaluator_sha256": run_hygiene.evaluator_sha256(),
            "table": {"caps": sorted({cap, CONTROL_CAP}), "patch_caps": [cap], "patch_scope": "all",
                      "packet_caps": [cap]},
            "deterministic_arms": list(DETERMINISTIC_ARMS),
            "datasheet_rule": "readouts_in_source_order_skip_ends_output_and_already_top_push_first_gap_below_"
                              "stratum_max_closure_n_ge_5_else_noop",
            "controller_arms": list(ARMS), "samples_per_arm": SAMPLES, "provider": "openai", "model": "gpt-5.6-luna",
            "opaque_case_ids": True,
            "system_prompt_sha256": {arm: hashlib.sha256(prompt(arm, cap).encode()).hexdigest() for arm in ARMS},
            "missing_data_rule": "intention_to_treat_error_or_invalid_choice_counts_as_noop_one_transport_retry",
            "primary": "joint_success_count_per_arm_on_the_rescue_split_evaluator_v2",
            "comparisons": ["controller_datasheet_vs_controller", "controller_datasheet_vs_datasheet_rule",
                            "controller_vs_fixed_readout", "controller_datasheet_vs_fixed_readout",
                            "controller_mechanics_vs_controller", "datasheet_rule_vs_fixed_readout",
                            "fixed_readout_vs_fixed_readout_at_control_cap",
                            "datasheet_rule_vs_datasheet_rule_at_control_cap"],
            "harm": "losses_vs_b0_on_the_harm_split; v1_confirmation_cases_reported_as_replay",
            "strata": ["b0_failure_type", "top1_class", "family"],
            "secondary": ["oracle_ceiling", "fresh_datasheet_coverage_and_rule_flip_accuracy",
                          "choice_agreement_with_oracle_and_datasheet_rule",
                          "blind_audit_of_every_distinct_b0_and_changed_output_by_a_fresh_subagent"],
            "support_pattern": "an_arm_helps_only_if_it_rescues_more_rescue_cases_than_its_comparator_"
                               "with_no_more_harm_split_losses",
            "inference": "descriptive_counts_only_no_significance_claims"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen = json.dumps(treatment(args.sweep_dir), indent=2, sort_keys=True) + "\n"
    args.output.write_text(frozen)
    print(json.dumps({"treatment_sha256": hashlib.sha256(frozen.encode()).hexdigest(), "path": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
