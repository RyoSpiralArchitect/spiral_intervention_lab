"""Recompute geometry and raw readout from retained vectors, without model calls."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

torch.set_default_device("cpu")
torch.set_num_threads(4)
from SpiralInterventionLab.runtime import clause_choice, clause_radial, clause_trajectory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    path = directory / "cpu32/measurement/trajectory.json"
    receipt = json.loads(path.read_text())
    supervision = json.loads((directory / "cpu32/supervision.json").read_text())
    if receipt["status"] != "complete" or supervision["status"] != "complete":
        raise ValueError("complete_attempt_required_no_partial_success")
    for relative, expected in receipt["code_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"analysis_code_drift:{relative}")
    artifacts = {str(path.relative_to(ROOT)): digest(path)}
    records = []
    for compact in receipt["records"]:
        artifact = path.parent / compact["vectors_artifact"]
        if digest(artifact) != compact["vectors_sha256"]:
            raise ValueError("raw_vector_hash_mismatch")
        raw = torch.load(artifact, map_location="cpu", weights_only=True)
        if clause_trajectory.compact_record(raw) != {k: v for k, v in compact.items() if k not in {"vectors_artifact", "vectors_sha256"}}:
            raise ValueError("compact_and_raw_receipt_disagree")
        records.append(raw)
        artifacts[str(artifact.relative_to(ROOT))] = digest(artifact)
    recomputed = clause_radial.summarize(records)
    if recomputed != receipt["comparison"]:
        raise ValueError("radial_summary_does_not_reproduce")
    choice = clause_choice.summarize(records, component_labels=clause_radial.LABELS)
    stored_choice = receipt["choice_comparison"]
    choice["fixed_baseline_winner_piece"] = stored_choice["fixed_baseline_winner_piece"]
    for actual, stored in zip(choice["surfaces"], stored_choice["surfaces"], strict=True):
        for row, other in zip(actual["conditions"], stored["conditions"], strict=True):
            row["edited_winner_pieces"] = other["edited_winner_pieces"]
    if choice != stored_choice:
        raise ValueError("raw_choice_summary_does_not_reproduce")
    tables = []
    for index, surface in enumerate(choice["surfaces"]):
        hook = surface["surface"]["cache_hook_name"]
        table = {"surface": surface["surface"], "conditions": []}
        for record, row in zip(records, surface["conditions"], strict=True):
            readouts = {r["token_id"]: r for r in row["readout_rows"]}
            delta = clause_trajectory._deltas(record, index, clause_trajectory.RAW_LOGITS)
            telemetry = record["result"]["operator_rows"][index]
            t = record["inputs"][index]["target"].double()
            written = clause_trajectory._deltas(record, index, hook)
            table["conditions"].append({"label": row["label"], "raw_greedy_token_ids": row["edited_winner_ids"],
                "comma_probability_deltas": readouts[11]["raw_probability_deltas"],
                "while_probability_deltas": readouts[1418]["raw_probability_deltas"],
                "comma_margin_vs_winner_deltas": readouts[11]["logit_margin_deltas_vs_fixed_winner"],
                "while_margin_vs_winner_deltas": readouts[1418]["logit_margin_deltas_vs_fixed_winner"],
                "comma_minus_while_logit_deltas": [float(v[11] - v[1418]) for v in delta],
                "actual_written_norms": [float(v.norm()) for v in written],
                "target_activation_norm_deltas": [float((t + v).norm() - t.norm()) for v in written],
                "safety_deltas": telemetry["safety_deltas"],
                "collapse_sharpening_signal": telemetry["collapse_sharpening_signal"]})
        tables.append(table)
    previous = json.loads((ROOT / "results/source_direction_controls_cpu32_20261024/cpu32/measurement/trajectory.json").read_text())
    identity_checks = {
        "same_prefix": receipt["observed_prefix"] == previous["observed_prefix"],
        "same_token_identity": receipt["cache_alignment"]["tokens_identity"] == previous["cache_alignment"]["tokens_identity"],
        "same_checkpoint": receipt["checkpoint_sha256_verified"] == previous["checkpoint_sha256_verified"],
        "same_first_seven_inputs": [r["inputs"] for r in receipt["records"][:7]] == [r["inputs"] for r in previous["records"][:7]],
        "same_first_seven_captured_vectors": [[f["vectors"] for f in r["frames"]] for r in receipt["records"][:7]]
            == [[f["vectors"] for f in r["frames"]] for r in previous["records"][:7]],
    }
    summary = {"status": "verified", "measurement_hashes": artifacts,
        "verification": "raw_hashes_compact_identities_and_full_numerical_summaries_recomputed_no_model_forward",
        "previous_cpu32_identity": identity_checks, "prefix": receipt["observed_prefix"],
        "fixed_baseline_winner_token_id": choice["fixed_baseline_winner_token_id"],
        "fixed_baseline_winner_piece": choice["fixed_baseline_winner_piece"],
        "all_greedy_winners_unchanged": all(not any(r["winner_changed_vs_fixed_baseline"]) for s in choice["surfaces"] for r in s["conditions"]),
        "surfaces": tables, "radial_geometry": recomputed["surfaces"],
        "all_captured_null_repeat_and_baseline_floors_zero": all(
            stage[key] == 0 for s in recomputed["trajectory"]["surfaces"] for stage in s["stages"]
            for key in ("baseline_drift_norm_max", "preparation_vs_baseline_norm_max",
                        "delta_repeat_difference_norm_max", "self_copy_delta_norm_max")),
        "diagnostic_budget_used": receipt["diagnostic_budget_used"],
        "diagnostic_interval_forward_counts": receipt["diagnostic_interval_forward_counts"],
        "total_forward_counts": receipt["forward_counts_total"],
        "recorded_condition_replay_count": receipt["recorded_condition_replay_count"],
        "elapsed_seconds": supervision["elapsed_seconds"], "controller_calls": 0,
        "production_apply_allowed": False, "semantic_safety": "unassessed",
        "scope": "one_boundary_geometric_direction_comparison_not_additive_attribution_or_ownership"}
    output = directory / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
