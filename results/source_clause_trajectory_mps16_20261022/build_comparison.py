"""Join the labelled MPS capture and explicitly post-hoc CPU reanalysis."""
import hashlib
import json
from pathlib import Path


directory = Path(__file__).resolve().parent
root = directory.parents[1]
paths = {
    "cpu32": root / "results/source_clause_trajectory_cpu32_20261022/trajectory_reanalysis.json",
    "mps16": directory / "measurement/trajectory.json",
}
data = {name: json.loads(path.read_text()) for name, path in paths.items()}
output = directory / "comparison.json"
if output.exists():
    raise FileExistsError(output)
if data["cpu32"]["status"] != "reanalysis_complete" or data["mps16"]["status"] != "complete":
    raise ValueError("required analysis is incomplete")
if (data["cpu32"]["reference_match"]["bindings"] != data["mps16"]["reference_match"]["bindings"]
        or data["cpu32"]["cache_alignment"]["tokens_identity"] != data["mps16"]["cache_alignment"]["tokens_identity"]):
    raise ValueError("not matched input tokens and edit bindings")
result = {"scope": "local_geometry_and_readout_not_semantic_rescue_or_mediation",
          "backend_and_dtype_both_changed": True, "production_apply_allowed": False,
          "prefix": data["mps16"]["observed_prefix"], "numerics": {}}
for name, report in data.items():
    surface = report["comparison"]["surfaces"][0]
    stages = []
    for row in surface["stages"]:
        pair = row["pairwise"][0]
        if [pair["left"], pair["right"]] != ["source:11", "source:1418"]:
            raise ValueError("unexpected pair")
        stages.append({"stage": row["stage"], "source_pair": pair["repeats"],
                       "source_pair_centered": pair.get("centered_repeats"),
                       "baseline_drift_norm_max": row["baseline_drift_norm_max"],
                       "repeat_delta_difference_norm_max": row["delta_repeat_difference_norm_max"],
                       "self_copy_delta_norm_max": row["self_copy_delta_norm_max"],
                       "preparation_vs_baseline_norm_max": row["preparation_vs_baseline_norm_max"]})
    responses = [{"condition": record["label"], "rows": [
        {"piece": row["piece"], "token_id": row["token_id"], "probability_deltas": row["probability_deltas"],
         "rank_after": row["after"][0]["rank"], "greedy_after": row["after"][0]["is_greedy_token"]}
        for row in record["result"]["operator_rows"][0]["structure_response_rows"]]}
        for record in report["records"]]
    result["numerics"][name] = {
        "receipt_sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
        "analysis_method": report.get("analysis_method", "capture_time_call_site_tagging"),
        "analysis_caveat": report.get("analysis_caveat"), "model_execution": report.get("model_execution"),
        "surface": surface["surface"], "source_geometry": surface["source_geometry"],
        "source_minus_target_geometry": surface["source_minus_target_geometry"], "stage_comparisons": stages,
        "readout_responses": responses,
        "counts": {"declared_diagnostic_replays": report["declared_diagnostic_replay_count"],
                   "captured_condition_forwards": report["comparison"]["captured_forward_count"],
                   "captured_preparation_forwards": report["comparison"]["captured_preparation_forward_count"],
                   "diagnostic_slots": report["diagnostic_budget_used"],
                   "diagnostic_interval_forward_counts": report.get("diagnostic_interval_forward_counts"),
                   "whole_run_forward_counts": report.get("forward_counts_total")}}
output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({"output": str(output), "prefix": result["prefix"]}))
