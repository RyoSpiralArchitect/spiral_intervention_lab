"""Audit the retained normal-cap replay without loading a model or calling an API."""

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RAW = HERE / "normal_cap_structure_patch.json"
PRIOR = HERE.parent / "source_clause_diagnostic_llama_20261021/source_clause_readout.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    raw = json.loads(RAW.read_text())
    prior = json.loads(PRIOR.read_text())
    measurement, = raw["measurements"]
    observation = measurement["result"]
    old = next(m for m in prior["measurements"]
               if m["generated_tokens"] == measurement["generated_tokens"])
    results = [m["result"] for m in measurement["operator_measurements"]]
    rows = [row for result in results for row in result["operator_rows"]]
    telemetry = [entry for row in rows for repeat in row["runtime_telemetry"]
                 for entry in repeat]
    events = [json.loads(line) for line in (HERE / "runtime.stdout.log").read_text().splitlines()
              if line.startswith('{"event":')]
    end = next(event for event in events if event["event"] == "supervised_end")
    invariants = {
        "run_complete": raw["status"] == "complete" and end["returncode"] == 0,
        "no_resource_stop": end["stop_reason"] is None,
        "same_prior_prefix_and_observation": (
            measurement["prefix"] == old["prefix"]
            and observation["structure_readout_rows"] == old["result"]["structure_readout_rows"]
        ),
        "all_receipts_ok": observation["status"] == "ok"
        and all(result["status"] == "ok" for result in results),
        "context_and_caps_restored": measurement["context_unchanged"]
        and observation["state_restored"]
        and all(result["state_restored"] and result["caps_unchanged"] for result in results),
        "hooks_executed_once": bool(telemetry)
        and all(entry["hook_call_count"] == 1 for entry in telemetry),
        "paired_measurements": bool(rows)
        and all(len(row["paired_deltas"]) == 2 for row in rows),
        "normal_caps_no_override": raw["caps"] == "unchanged_runtime_defaults"
        and all(row["edit"]["op"]["kind"] == "activation_patch"
                and row["edit"]["op"]["mode"] == "blend"
                and row["edit"]["op"]["alpha"] <= min(0.04, row["surface"]["caps"]["max_alpha"])
                and row["edit"]["budget"]["step_size"] <= min(0.08, row["surface"]["caps"]["step_size"])
                and row["edit"]["budget"]["ttl_steps"] == 1 for row in rows),
        "production_closed": raw["production_apply_allowed"] is False
        and measurement["production_apply_allowed"] is False
        and observation["production_apply_allowed"] is False
        and all(result["production_apply_allowed"] is False
                and result["certified_for_apply"] is False
                and result["policy_candidate_ready"] is False for result in results)
        and all(row["production_apply_allowed"] is False
                and row["target_readout_effect_certified"] is False for row in rows),
        "no_controller_calls_or_rollout_edits": raw["controller_calls"] == 0
        and measurement["rollout_edit_count"] == 0,
        "harness_selected_not_controller": all(
            m["selection_origin"] == "source_order_diagnostic_harness_not_controller_choice"
            for m in measurement["operator_measurements"]),
    }
    if not all(invariants.values()):
        raise ValueError({name: value for name, value in invariants.items() if not value})

    matrix = []
    for result in results:
        binding = result["candidate_binding"]
        for row in result["operator_rows"]:
            target = next(response for response in row["structure_response_rows"]
                          if response["token_id"] == binding["target_token_id"])
            winners = [[response["token_id"] for response in row["structure_response_rows"]
                        if response["after"][i]["is_greedy_token"]] for i in range(2)]
            matrix.append({
                "target_piece": binding["target_piece"],
                "target_token_id": binding["target_token_id"],
                "source_token_index": binding["source_token_index"],
                "candidate_fingerprint": row["candidate_fingerprint"],
                "context_fingerprint": row["context_fingerprint"],
                "surface": row["surface"],
                "effect_role": row["effect_role"],
                "ownership_role": row["ownership_role"],
                "paired_deltas": row["paired_deltas"],
                "target_gap_to_top1_before": [value["gap_to_top1"] for value in target["before"]],
                "target_gap_to_top1_after": [value["gap_to_top1"] for value in target["after"]],
                "greedy_token_ids_after": winners,
                "greedy_unchanged_both_repeats": all(
                    winner == [observation["greedy_token_id"]] for winner in winners),
                "self_delta": row["self_delta"],
                "cross_delta": row["cross_delta"],
                "alignment_margin": row["alignment_margin"],
                "realized_lift_token_id": row["realized_lift_token_id"],
                "probability_response": [
                    {key: response[key] for key in (
                        "token_id", "piece", "probability_deltas", "repeat_probability_floor")}
                    for response in row["structure_response_rows"]
                ],
                "baseline_repeat_variation": row["baseline_repeat_variation"],
                "edited_repeat_variation": row["edited_repeat_variation"],
                "safety_deltas": row["safety_deltas"],
                "collapse_sharpening_signal": row["collapse_sharpening_signal"],
                "semantic_safety": row["semantic_safety"],
                "clause_correctness": row["clause_correctness"],
            })

    return {
        "kind": "normal_cap_source_clause_operator_audit",
        "raw_sha256": sha256(RAW),
        "prior_observation_sha256": sha256(PRIOR),
        "invariants": invariants,
        "worker_model": raw["worker_model"],
        "device": raw["device"],
        "dtype": raw["dtype"],
        "seed": raw["seed"],
        "prefix": measurement["prefix"],
        "generated_tokens": measurement["generated_tokens"],
        "supervisor_end": end,
        "minimum_recorded_free_disk_gib": min(event["disk_free_gib"] for event in events
                                             if "disk_free_gib" in event),
        "operator_requests": len(results),
        "operator_rows": len(rows),
        "physical_replays": observation["physical_replay_count"]
        + sum(result["physical_replay_count"] for result in results),
        "edited_replays": sum(result["edited_replay_count"] for result in results),
        "diagnostic_slots_spent": observation["diagnostic_call_cost"]
        + sum(result["diagnostic_call_cost"] for result in results),
        "diagnostic_slots_left": measurement["diagnostic_budget_left_after_operators"],
        "hook_records": len(telemetry),
        "logged_blend_delta_norm_range": [
            min(entry["blend_delta_norm"] for entry in telemetry),
            max(entry["blend_delta_norm"] for entry in telemetry),
        ],
        "all_step_size_saturated": all(entry["step_size_clip_saturated"] for entry in telemetry),
        "readout_response_matrix": matrix,
        "all_greedy_choices_unchanged": all(row["greedy_unchanged_both_repeats"] for row in matrix),
        "realized_lift_token_ids": sorted(set(row["realized_lift_token_id"] for row in matrix)),
        "controller_calls": raw["controller_calls"],
        "rollout_edit_count": measurement["rollout_edit_count"],
        "production_apply_allowed": raw["production_apply_allowed"],
        "claim_limits": [
            "One prefix and one already-offered site; not a layer or dose sweep.",
            "Targets were selected in SOURCE order by the harness, not by Luna.",
            "Probability deltas use CPU float64 arithmetic on recorded MPS/fp16 logits.",
            "Zero within-backend repeat variation is not CPU or cross-dtype validation.",
            "TransformerLens emitted an MPS numerical-correctness warning.",
            "Logged blend norms include fp16 rounding around the configured 0.08 cap.",
            "Shared comma response does not establish source-specific clause actuation.",
            "No full continuation, task-score uplift, or semantic safety was evaluated.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=True, allow_nan=False))
