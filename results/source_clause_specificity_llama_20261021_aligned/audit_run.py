"""Rebuild the bounded control audit with only the Python standard library."""
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def build():
    raw_path = HERE / "specificity.json"
    raw = json.loads(raw_path.read_text())
    measurement, = raw["measurements"]
    sources = [r["result"] for r in measurement["operator_measurements"]]
    controls = measurement["specificity_controls"]
    results = sources + controls
    rows = [row for result in results for row in result["operator_rows"]]
    null_rows = next(c for c in controls if c["kind"] == "self_copy")["operator_rows"]
    alignment = raw["cache_alignment"]
    checks = {
        "completed": raw["status"] == "complete",
        "same_prefix_maintained": measurement["context_unchanged"] and alignment["generation_unchanged"],
        "all_measurements_ok": all(r["status"] == "ok" for r in results),
        "all_state_and_caps_checks_pass": all(r["state_restored"] and r["caps_unchanged"] for r in results),
        "null_displacement_zero": all(
            t["source_target_delta_norm"] == t["blend_delta_norm"] == 0
            for row in null_rows for repeat in row["runtime_telemetry"] for t in repeat),
        "one_hook_per_edit": all(t["hook_call_count"] == 1
                                for row in rows for repeat in row["runtime_telemetry"] for t in repeat),
        "all_pairs_repeatable_on_this_backend": all(
            row["baseline_repeat_variation"]["max_abs_logit_delta"] == 0
            and row["edited_repeat_variation"]["max_abs_logit_delta"] == 0 for row in rows),
        "comparison_complete": measurement["specificity_comparison"]["status"] == "ok",
        "production_closed": raw["production_apply_allowed"] is False
        and all(r["production_apply_allowed"] is False for r in results),
        "no_controller_calls_or_rollout_edits": raw["controller_calls"] == 0
        and measurement["rollout_edit_count"] == 0,
    }
    if not all(checks.values()):
        raise ValueError({k: v for k, v in checks.items() if not v})
    matrix = []
    for result in results:
        for row in result["operator_rows"]:
            matrix.append({
                "condition": result.get("kind") or result["candidate_binding"]["target_piece"],
                "surface": row["surface"], "candidate_fingerprint": row["candidate_fingerprint"],
                "context_fingerprint": row["context_fingerprint"],
                "response": [
                    {"token_id": r["token_id"], "piece": r["piece"],
                     "probability_deltas": r["probability_deltas"],
                     "rank_before": [v["rank"] for v in r["before"]],
                     "rank_after": [v["rank"] for v in r["after"]],
                     "gap_to_top1_before": [v["gap_to_top1"] for v in r["before"]],
                     "gap_to_top1_after": [v["gap_to_top1"] for v in r["after"]],
                     "greedy_after": [v["is_greedy_token"] for v in r["after"]]}
                    for r in row["structure_response_rows"]],
                "safety_deltas": row["safety_deltas"],
                "collapse_sharpening_signal": row["collapse_sharpening_signal"],
            })
    return {
        "kind": "source_clause_specificity_audit",
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "checks": checks, "model": raw["worker_model"], "device": raw["device"], "dtype": raw["dtype"],
        "prefix": measurement["prefix"], "cache_alignment": alignment,
        "physical_forwards": alignment["physical_replay_count"] + measurement["result"]["physical_replay_count"]
        + sum(r["physical_replay_count"] for r in results),
        "diagnostic_slots": alignment["diagnostic_call_cost"] + measurement["result"]["diagnostic_call_cost"]
        + sum(r["diagnostic_call_cost"] for r in results),
        "diagnostic_slots_left": measurement["diagnostic_budget_left_after_controls"],
        "matrix": matrix, "comparison": measurement["specificity_comparison"],
        "claim_limits": [
            "One prefix, one offered site, one cap; SOURCE ordering, not outcome selection.",
            "Own-source advantage over these controls is not semantic or causal-path certification.",
            "Radial control matches total displacement approximately, not the radial component.",
            "MPS/fp16 repeatability does not validate CPU, fp32 or another model.",
            "No full continuation, task score, semantic safety or production permission evaluated.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=True, allow_nan=False))
