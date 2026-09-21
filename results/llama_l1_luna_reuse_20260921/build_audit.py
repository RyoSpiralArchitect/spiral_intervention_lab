"""Audit a completed live pair separately from frozen and conditioned replays."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
REFERENCE = REPO / "results/llama_l1_luna_20260921_r2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    spec = importlib.util.spec_from_file_location("reference_audit", REFERENCE / "build_audit.py")
    reference_audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_audit)
    status, manifest = read(ROOT / "run_status.json"), read(ROOT / "manifest.json")
    if status["status"] != "complete" or status["cli_exit_code"] != 0:
        raise ValueError("Require a complete live pair, not partial evidence")
    assert read(ROOT / "baseline_identity_check.json")["prompt_output_score_match"]
    assert all(sha(REPO / path) == digest for path, digest in manifest["code_hashes"].items())
    traces = {label: events(ROOT / f"live/{label}.jsonl") for label in ("b0", "c1")}
    outcomes = {}
    for label, rows in traces.items():
        assert sum(r["event"] == "episode_start" for r in rows) == 1
        ends = [r for r in rows if r["event"] == "episode_end"]
        assert len(ends) == 1
        start = next(r for r in rows if r["event"] == "episode_start")
        assert start["prompt"] == manifest["fixture"]["prompt"]
        outcomes[label] = {**ends[0], "lexical_check": reference_audit.check_output(ends[0]["output"], manifest["fixture"])}
    rows = traces["c1"]
    diagnostics = [r for r in rows if r["event"] == "controller_diagnostic_result"]
    reused = [r for r in diagnostics if r.get("review_reused")]
    for r in reused:
        assert r["budget_before"] == r["budget_after"]
        assert r["physical_replay_count"] == r["new_measurement_count"] == 0
        assert r["production_apply_allowed"] is False and r["certified_for_apply"] is False
        assert r["evidence_scope"] == "historical_review_not_current_measurement"
    charges = Counter()
    for r in diagnostics:
        for field in ("diagnostic_calls_left", "inspection_calls_left"):
            before, after = r.get("budget_before", {}), r.get("budget_after", {})
            assert field in before and field in after
            delta = before[field] - after[field]
            assert delta in (0, 1)
            charges[field] += delta
    assert charges["diagnostic_calls_left"] <= 12 and charges["inspection_calls_left"] <= 4
    attempts = [r for r in rows if r["event"] == "controller_provider_attempt"]
    commands = [r for r in rows if r["event"] == "controller_command"]
    usage = Counter()
    for r in attempts:
        u = r.get("usage") or {}
        for field in ("input_tokens", "output_tokens", "total_tokens"):
            usage[field] += int(u.get(field) or 0)
        usage["cached_input_tokens"] += int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    prior = read(REFERENCE / "audit.json")
    identity_fields = ("measurement_context_id", "execution_id", "observable_id", "source_tensor_identity",
                       "activation_patch_step_size", "target_piece_token_id", "target_piece")
    metric_fields = ("target_piece_logit_delta", "target_piece_prob_delta", "target_top20_hit_delta",
                     "target_top20_threshold_gap_delta", "repeat_max_abs_logit_delta")

    def measured_rows(trace):
        return [{k: e.get(k) for k in (*identity_fields, *metric_fields)}
                for r in trace if r.get("target_piece_binding_seed_matrix_executed")
                for e in r.get("target_piece_binding_seed_matrix_rows", [])]

    current_measurements = measured_rows(rows)
    previous_measurements = measured_rows(events(REFERENCE / "live/c1.jsonl"))
    audit = {
        "status": "complete", "scope": "one fresh Luna trajectory; same worker/task/budgets, changed review-reuse interface",
        **outcomes,
        "output_identical": outcomes["b0"]["output"] == outcomes["c1"]["output"],
        "score_delta": outcomes["c1"]["score"] - outcomes["b0"]["score"],
        "compiled_rollout_edit_count": sum(r["event"] == "compiled_edit" for r in rows),
        "command_decisions": dict(Counter(r["command"]["decision"] for r in commands)),
        "event_counts": dict(Counter(r["event"] for r in rows)),
        "provider_models": dict(Counter(str(r.get("model")) for r in attempts)),
        "provider_parse_results": dict(Counter(str(r.get("parse_ok")) for r in attempts)),
        "controller_usage_excluding_debrief": dict(usage),
        "charged_diagnostic_count": charges["diagnostic_calls_left"],
        "charged_inspection_count": charges["inspection_calls_left"],
        "review_reuse_count": len(reused),
        "review_reuse_prefix_changed_count": sum(bool(r.get("prefix_changed_since_review")) for r in reused),
        "diagnostic_timeline": [{k: r.get(k) for k in (
            "step", "source", "diagnostic", "status", "objective_bundle_key", "review_reused",
            "recorded_step", "reviewed_at_step", "prefix_changed_since_review", "evidence_scope",
            "budget_before", "budget_after", "physical_replay_count", "new_measurement_count",
            "next_evidence_needed", "activation_patch_compile_preview_blocked_reason",
            "target_piece_binding_seed_matrix_summary", "production_apply_allowed", "certified_for_apply",
        ) if k in r} for r in diagnostics],
        "explicit_measurement_request_count": sum(r.get("diagnostic") == "matched_response_probe" for r in diagnostics),
        "matched_binding_repeat": {
            "previous_observable_count": len(previous_measurements),
            "current_observable_count": len(current_measurements),
            "identity_and_listed_metrics_exactly_match": bool(current_measurements) and current_measurements == previous_measurements,
            "current_measurements": current_measurements,
            "not_whole_term_or_task_success": True,
        },
        "previous_pair_comparison": {
            "manifest_sha256": sha(REFERENCE / "manifest.json"),
            "previous_charged_diagnostics": prior["diagnostic_count"],
            "previous_output": prior["c1"]["output"],
            "same_c1_output": prior["c1"]["output"] == outcomes["c1"]["output"],
            "causal_savings_established": False,
            "reason": "Independent provider trajectory; no matched request-sequence ablation",
        },
        "conditional_completion_probe_invoked": False,
        "production_policy_changed": False,
        "debrief_scope": "qualitative only, not causal evidence or next-run context",
    }
    debrief = read(ROOT / "live/post_run_debrief.json")
    audit["debrief"] = {k: debrief.get(k) for k in ("metadata", "reference_audit", "usage")}
    audit["artifact_hashes"] = {
        str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size}
        for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name != "audit.json"
        and "__pycache__" not in p.parts and p.suffix != ".log"
    }
    (ROOT / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: audit[k] for k in (
        "status", "b0", "c1", "score_delta", "command_decisions", "compiled_rollout_edit_count",
        "charged_diagnostic_count", "charged_inspection_count", "review_reuse_count",
        "review_reuse_prefix_changed_count", "explicit_measurement_request_count",
        "provider_models", "provider_parse_results", "controller_usage_excluding_debrief",
    )}, indent=2))


if __name__ == "__main__":
    main()
