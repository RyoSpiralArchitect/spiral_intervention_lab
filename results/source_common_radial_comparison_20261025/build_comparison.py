"""Combine verified, separately scoped measurements without running a model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "llama": "source_common_radial_cpu32_retry1_20261025",
    "gpt2": "source_common_radial_gpt2_cpu32_20261025",
}


def main():
    result = {"kind": "same_visible_prefix_radial_comparison", "models": {},
        "production_apply_allowed": False, "controller_calls": 0,
        "cross_model_causal_attribution": False, "task_success_evaluated": False,
        "scope": "within_model_normal_cap_readout_diagnostics_not_component_contribution_percentages"}
    prefixes = []
    for model, directory in RUNS.items():
        base = ROOT / "results" / directory
        summary_path = base / "summary.json"
        summary = json.loads(summary_path.read_text())
        receipt = json.loads((base / "cpu32/measurement/trajectory.json").read_text())
        if summary["status"] != "verified" or receipt["status"] != "complete":
            raise ValueError("verified_complete_inputs_required")
        if summary["diagnostic_budget_used"] != 12 or len(receipt["records"]) != 10:
            raise ValueError("budget_or_condition_count_changed")
        if any(s["surface"]["alpha"] != .04 or s["surface"]["step_size"] != .08
               for s in summary["surfaces"]):
            raise ValueError("caps_changed")
        prefixes.append(summary["prefix"])
        result["models"][model] = {
            "summary": str(summary_path.relative_to(ROOT)),
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "prefix_origin": summary["prefix_origin"], "position": summary["position"],
            "token_identity": receipt["cache_alignment"]["tokens_identity"],
            "source_token_ids": summary["source_token_ids"],
            "model_execution": summary["model_execution"],
            "baseline_winner": summary["fixed_baseline_winner_piece"],
            "all_greedy_winners_unchanged": summary["all_greedy_winners_unchanged"],
            "all_captured_null_repeat_and_baseline_floors_zero": summary["all_captured_null_repeat_and_baseline_floors_zero"],
            "surfaces": summary["surfaces"], "geometry": summary["radial_geometry"],
            "diagnostic_budget_used": summary["diagnostic_budget_used"],
            "condition_replays": summary["recorded_condition_replay_count"],
            "captured_packet_preparation_forwards": sum(f["origin"]["kind"] == "packet_preparation"
                for r in receipt["records"] for f in r["frames"]),
            "actual_forwards": summary["total_forward_counts"],
        }
    if len(set(prefixes)) != 1:
        raise ValueError("visible_prefix_mismatch")
    result["same_visible_prefix"] = prefixes[0]
    output = Path(__file__).resolve().parent / "comparison.json"
    with output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(str(output))


if __name__ == "__main__":
    main()
