"""Verify retained artifacts and summarize raw readout without any model forward."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

torch.set_default_device("cpu")
torch.set_num_threads(4)
from SpiralInterventionLab.runtime import clause_choice, clause_direction_controls, clause_trajectory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = Path(__file__).resolve().parent
    receipt_path = directory / "cpu32/measurement/trajectory.json"
    receipt = json.loads(receipt_path.read_text())
    supervision = json.loads((directory / "cpu32/supervision.json").read_text())
    if receipt["status"] != "complete" or supervision["status"] != "complete":
        raise ValueError("complete_attempt_required_no_partial_success")
    for relative, expected in receipt["code_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"analysis_code_drift:{relative}")
    records, artifacts = [], {str(receipt_path.relative_to(ROOT)): digest(receipt_path)}
    for compact in receipt["records"]:
        path = receipt_path.parent / compact["vectors_artifact"]
        if digest(path) != compact["vectors_sha256"]:
            raise ValueError(f"raw_vector_hash_mismatch:{path.name}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        expected = {k: v for k, v in compact.items() if k not in {"vectors_artifact", "vectors_sha256"}}
        if clause_trajectory.compact_record(raw) != expected:
            raise ValueError("compact_and_raw_receipt_disagree")
        records.append(raw)
        artifacts[str(path.relative_to(ROOT))] = digest(path)
    recomputed = clause_direction_controls.summarize(records)
    if recomputed != receipt["comparison"]:
        raise ValueError("retained_direction_summary_does_not_reproduce")
    # Decoding is a display label; all numerical and token-ID fields must match.
    choice = clause_choice.summarize(records, component_labels=clause_direction_controls.LABELS)
    stored_choice = receipt["choice_comparison"]
    choice["fixed_baseline_winner_piece"] = stored_choice["fixed_baseline_winner_piece"]
    for actual, stored in zip(choice["surfaces"], stored_choice["surfaces"], strict=True):
        for row, other in zip(actual["conditions"], stored["conditions"], strict=True):
            row["edited_winner_pieces"] = other["edited_winner_pieces"]
    if choice != stored_choice:
        raise ValueError("raw_choice_summary_does_not_reproduce")
    table = []
    surface = choice["surfaces"][0]
    hook = surface["surface"]["cache_hook_name"]
    for record, row in zip(records, surface["conditions"], strict=True):
        readouts = {r["token_id"]: r for r in row["readout_rows"]}
        delta = clause_trajectory._deltas(record, 0, clause_trajectory.RAW_LOGITS)
        telemetry = record["result"]["operator_rows"][0]
        table.append({"label": row["label"], "raw_greedy_token_ids": row["edited_winner_ids"],
            "comma_probability_deltas": readouts[11]["raw_probability_deltas"],
            "while_probability_deltas": readouts[1418]["raw_probability_deltas"],
            "comma_margin_vs_winner_deltas": readouts[11]["logit_margin_deltas_vs_fixed_winner"],
            "while_margin_vs_winner_deltas": readouts[1418]["logit_margin_deltas_vs_fixed_winner"],
            "comma_minus_while_logit_deltas": [float(v[11] - v[1418]) for v in delta],
            "actual_written_norms": [float(v.norm()) for v in clause_trajectory._deltas(record, 0, hook)],
            "safety_deltas": telemetry["safety_deltas"],
            "collapse_sharpening_signal": telemetry["collapse_sharpening_signal"]})
    previous_path = ROOT / "results/source_clause_components_cpu32_20261024/cpu32/measurement/trajectory.json"
    previous = json.loads(previous_path.read_text())
    previous_common = previous["records"][4]
    shared_common = {"same_prefix": receipt["observed_prefix"] == previous["observed_prefix"],
        "same_token_identity": receipt["cache_alignment"]["tokens_identity"] == previous["cache_alignment"]["tokens_identity"],
        "same_checkpoint": receipt["checkpoint_sha256_verified"] == previous["checkpoint_sha256_verified"],
        "same_common_inputs": receipt["records"][4]["inputs"] == previous_common["inputs"],
        "same_common_captured_vectors": [f["vectors"] for f in receipt["records"][4]["frames"]]
            == [f["vectors"] for f in previous_common["frames"]]}
    summary = {"status": "verified", "measurement_hashes": artifacts,
        "verification": "raw_hashes_compact_identities_and_full_numerical_summaries_recomputed_no_model_forward",
        "previous_common_replay_identity": shared_common,
        "prefix": receipt["observed_prefix"], "surface": surface["surface"],
        "fixed_baseline_winner_token_id": choice["fixed_baseline_winner_token_id"],
        "fixed_baseline_winner_piece": choice["fixed_baseline_winner_piece"],
        "all_greedy_winners_unchanged": all(not any(r["winner_changed_vs_fixed_baseline"]) for r in surface["conditions"]),
        "conditions": table, "direction_controls": recomputed["surfaces"],
        "diagnostic_budget_used": receipt["diagnostic_budget_used"],
        "diagnostic_interval_forward_counts": receipt["diagnostic_interval_forward_counts"],
        "total_forward_counts": receipt["forward_counts_total"],
        "recorded_condition_replay_count": receipt["recorded_condition_replay_count"],
        "elapsed_seconds": supervision["elapsed_seconds"], "controller_calls": 0,
        "production_apply_allowed": False, "semantic_safety": "unassessed",
        "scope": "single_fixed_coordinate_control_pair_not_semantic_null_ensemble_or_source_ownership_proof"}
    output = directory / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(output), "conditions": table,
                      "previous_common_replay_identity": shared_common}, indent=2))


if __name__ == "__main__":
    main()
