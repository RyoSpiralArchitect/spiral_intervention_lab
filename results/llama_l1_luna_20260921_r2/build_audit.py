"""Summarize completed paired evidence; never infer task gain from a debrief."""
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def counts(values):
    return dict(Counter(str(value) for value in values))


def check_output(output, fixture):
    text = output.lower()
    return {
        "required_terms_present": [term for term in fixture["required_terms"] if term.lower() in text],
        "missing_required_terms": [term for term in fixture["required_terms"] if term.lower() not in text],
        "forbidden_terms_present": [term for term in fixture["forbidden_terms"] if term.lower() in text],
        "word_count": len(output.split()),
        "max_words": fixture["max_words"],
        "within_word_budget": len(output.split()) <= fixture["max_words"],
        "semantic_faithfulness_certified": False,
    }


def main():
    status = json.loads((ROOT / "run_status.json").read_text())
    if status["status"] != "complete" or status["cli_exit_code"] != 0:
        raise ValueError("A completed pair is required for an outcome audit")
    manifest = json.loads((ROOT / "manifest.json").read_text())
    identity = json.loads((ROOT / "baseline_identity_check.json").read_text())
    assert identity["prompt_output_score_match"]
    b0_rows = read_events(ROOT / "live/b0.jsonl")
    c1_rows = read_events(ROOT / "live/c1.jsonl")
    b0 = next(row for row in b0_rows if row["event"] == "episode_end")
    c1 = next(row for row in c1_rows if row["event"] == "episode_end")
    b0_prompt = next(row["prompt"] for row in b0_rows if row["event"] == "episode_start")
    c1_prompt = next(row["prompt"] for row in c1_rows if row["event"] == "episode_start")
    assert b0_prompt == c1_prompt == manifest["fixture"]["prompt"]
    commands = [row for row in c1_rows if row["event"] == "controller_command"]
    attempts = [row for row in c1_rows if row["event"] == "controller_provider_attempt"]
    diagnostics = [row for row in c1_rows if row["event"] == "controller_diagnostic_result"]
    charged_diagnostics = sum(
        int((row.get("budget_before") or {}).get("diagnostic_calls_left") or 0)
        - int((row.get("budget_after") or {}).get("diagnostic_calls_left") or 0)
        for row in diagnostics
    )
    charged_inspections = sum(
        int((row.get("budget_before") or {}).get("inspection_calls_left") or 0)
        - int((row.get("budget_after") or {}).get("inspection_calls_left") or 0)
        for row in diagnostics
    )
    usage = Counter()
    for row in attempts:
        u = row.get("usage") or {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            usage[field] += int(u.get(field) or 0)
        usage["cached_input_tokens"] += int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        usage["reasoning_output_tokens"] += int((u.get("output_tokens_details") or {}).get("reasoning_tokens") or 0)
    timeline = []
    observation = {}
    for row in c1_rows:
        if row["event"] == "controller_observation":
            observation = row
        elif row["event"] == "controller_command":
            command = row["command"]
            meta = command.get("meta") or {}
            timeline.append({
                "controller_step": row["step"],
                "observation_step": observation.get("step"),
                "generated_tail": observation.get("generated_tail"),
                "task_feedback": observation.get("task_feedback"),
                "decision": command["decision"],
                "selection_source": row.get("controller_selection_source"),
                "selected_bundle_key": row.get("controller_selected_bundle_key"),
                "diagnostic_request": meta.get("diagnostic_request"),
                "next_action": meta.get("next_action"),
                "blocked_by": meta.get("blocked_by"),
                "why_not_apply": row.get("controller_why_not_apply") or meta.get("why_not_apply"),
            })
    for row in (b0, c1):
        row["lexical_check"] = check_output(row["output"], manifest["fixture"])
    audit = {
        "scope": "one prespecified L1/review/seed0 pair, full-depth Llama, current working tree",
        "controller": "gpt-5.6-luna",
        "worker": {"layers": 28, "device": "mps", "dtype": "float16", "max_new_tokens": 64},
        "prompt_identity_match": True,
        "frozen_screen_b0_match": identity,
        "b0": b0, "c1": c1,
        "output_identical": b0["output"] == c1["output"],
        "score_delta": c1["score"] - b0["score"],
        "compiled_rollout_edit_count": sum(row["event"] == "compiled_edit" for row in c1_rows),
        "apply_command_count": sum(row["command"]["decision"] == "apply" for row in commands),
        "command_decisions": counts(row["command"]["decision"] for row in commands),
        "event_counts": counts(row["event"] for row in c1_rows),
        "provider_models": counts(row.get("model") for row in attempts),
        "provider_parse_results": counts(row.get("parse_ok") for row in attempts),
        "controller_usage_excluding_debrief_and_interrupted_attempt": dict(usage),
        "controller_timeline": timeline,
        "diagnostic_result_event_count_including_inspection": len(diagnostics),
        "diagnostic_count": charged_diagnostics,
        "inspection_count": charged_inspections,
        "diagnostic_count_scope": "budget charges, not independent physical replays",
        "diagnostics": [{key: row.get(key) for key in (
            "step", "diagnostic", "status", "budget_before", "budget_after", "next_evidence_needed",
            "focus_terms", "objective_bundle_key", "production_apply_allowed", "certified_for_apply",
            "target_piece_binding_seed_matrix_summary", "activation_patch_compile_preview_blocked_reason",
            "activation_patch_runtime_support_status",
        )} for row in diagnostics],
        "direct_readout_observations": [
            {"step": row["step"], "evidence": evidence}
            for row in diagnostics for evidence in row.get("evidence_rows", [])
            if evidence.get("operator_axis") == "entity_insertion_materialization"
        ],
        "matched_binding_observations": [
            {"step": row["step"], "evidence": evidence}
            for row in diagnostics for evidence in row.get("target_piece_binding_seed_matrix_rows", [])
        ],
        "memory_retry": manifest["memory_retry"],
        "policy_changed": False,
        "paired_trace_scope": "baseline trace content retained, completed worker caches released",
        "not_claimed": ["diagnostic evidence grants apply permission", "a debrief establishes causality",
                        "one fixture establishes generalization", "cached summaries are independent replays"],
    }
    debrief_path = ROOT / "live/post_run_debrief.json"
    if debrief_path.exists():
        debrief = json.loads(debrief_path.read_text())
        audit["debrief"] = {"scope": "qualitative only", "usage": debrief.get("usage"),
                            "status": (debrief.get("metadata") or {}).get("status"),
                            "reference_audit": debrief.get("reference_audit"),
                            "artifact": "live/post_run_debrief.json"}
    memory_rows = read_events(ROOT / "memory_boundaries.jsonl")
    audit["observed_memory_boundaries"] = {
        "count": len(memory_rows),
        "max_before_driver_bytes": max(row["before"]["mps_driver_bytes"] for row in memory_rows),
        "max_after_driver_bytes": max(row["after"]["mps_driver_bytes"] for row in memory_rows),
        "not_a_continuous_peak_measurement": True,
    }
    audit["artifact_hashes"] = {
        str(path.relative_to(ROOT)): {"sha256": sha(path), "bytes": path.stat().st_size}
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "audit.json" and "__pycache__" not in path.parts
    }
    (ROOT / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "b0", "c1", "score_delta", "command_decisions", "compiled_rollout_edit_count",
        "diagnostic_count", "provider_parse_results", "controller_usage_excluding_debrief_and_interrupted_attempt",
    )}, indent=2))


if __name__ == "__main__":
    main()
