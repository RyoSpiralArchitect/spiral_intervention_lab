"""Summarize retained JSON receipts; no model imports, forwards, or calls."""
import hashlib
import json
from pathlib import Path


directory = Path(__file__).resolve().parent
root = directory.parents[1]
path = directory / "trajectory_reanalysis.json"
reference_path = root / "results/source_clause_specificity_llama_20261021_aligned/specificity.json"
output = directory / "audit.json"
if output.exists():
    raise FileExistsError(output)
data = json.loads(path.read_text())
reference = json.loads(reference_path.read_text())
if data["status"] != "reanalysis_complete" or data["comparison"]["status"] != "ok":
    raise ValueError("cannot summarize an incomplete trajectory")

source_pair = [f"source:{tid}" for tid in data["source_token_ids"]]
surfaces = []
for surface in data["comparison"]["surfaces"]:
    stages = []
    for row in surface["stages"]:
        pair = next(p for p in row["pairwise"] if [p["left"], p["right"]] == source_pair)
        stages.append({"stage": row["stage"], "source_pair": pair["repeats"],
                       "source_pair_centered": pair.get("centered_repeats"),
                       "baseline_drift_norm_max": row["baseline_drift_norm_max"],
                       "repeat_delta_difference_norm_max": row["delta_repeat_difference_norm_max"],
                       "self_copy_delta_norm_max": row["self_copy_delta_norm_max"]})
    surfaces.append({"surface": surface["surface"], "source_geometry": surface["source_geometry"],
                     "source_minus_target_geometry": surface["source_minus_target_geometry"],
                     "capture_checks": surface["capture_checks"], "stage_comparisons": stages})

old = reference["measurements"][0]
old_rows = {f"source:{r['result']['candidate_binding']['target_token_id']}": r["result"]["operator_rows"]
            for r in old["operator_measurements"]}
old_rows.update({r["kind"]: r["operator_rows"] for r in old["specificity_controls"]})
readout = []
for record in data["records"]:
    for row in record["result"]["operator_rows"]:
        prior = next(r for r in old_rows[record["label"]] if r["surface"] == row["surface"])
        readout.append({"condition": record["label"], "surface": row["surface"],
                        "greedy_after": [r["piece"] for r in row["structure_response_rows"] if r["after"][0]["is_greedy_token"]],
                        "rows": [{"token_id": r["token_id"], "piece": r["piece"],
                                  "cpu32_probability_deltas": r["probability_deltas"],
                                  "mps16_probability_deltas": next(p["probability_deltas"]
                                      for p in prior["structure_response_rows"] if p["token_id"] == r["token_id"])}
                                 for r in row["structure_response_rows"]]})

audit = {"scope": "one_prefix_two_predeclared_sources_normal_caps_not_task_success_or_mediation",
         "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
         "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
         "numerical_comparison": "CPU float32 vs prior MPS float16; backend and dtype both changed",
         "analysis_method": data["analysis_method"], "analysis_caveat": data["analysis_caveat"],
         "prefix": data["observed_prefix"], "reference_match": data["reference_match"],
         "diagnostic_slots": data["diagnostic_budget_used"],
         "declared_diagnostic_replays": data["declared_diagnostic_replay_count"],
         "captured_condition_forwards": data["comparison"]["captured_forward_count"],
         "captured_condition_preparation_forwards": data["comparison"]["captured_preparation_forward_count"],
         "controller_calls": data["controller_calls"], "production_apply_allowed": False,
         "all_states_restored": all(r["result"]["state_restored"] for r in data["records"]),
         "all_caps_unchanged": all(r["result"]["caps_unchanged"] for r in data["records"]),
         "all_edited_hook_counts_one": all(t[0]["hook_call_count"] == 1
             for r in data["records"] for row in r["result"]["operator_rows"] for t in row["runtime_telemetry"]),
         "surface_geometry": surfaces, "matched_readout": readout}
output.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
print(json.dumps({k: audit[k] for k in ("diagnostic_slots", "captured_condition_forwards", "prefix")}))
