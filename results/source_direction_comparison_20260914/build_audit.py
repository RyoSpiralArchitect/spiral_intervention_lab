"""Package measured evidence without rewriting the original live JSONL."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--supporting-artifacts", type=Path)
    parser.add_argument("--verify-replay", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    root = destination.parents[1]
    raw = args.source_jsonl.read_bytes()
    if args.source_jsonl.suffix == ".gz":
        raw = gzip.decompress(raw)
    (destination / "source_episode.jsonl.gz").write_bytes(gzip.compress(raw, mtime=0))
    events = [json.loads(line) for line in raw.splitlines()]
    report = json.loads((destination / "replay.json").read_text())
    rows = report["rows"]
    source_matrix = next(e for e in events if e.get("event") == "controller_diagnostic_result"
        and e.get("step") == report["source_log"]["controller_step"] and e.get("target_piece_binding_seed_matrix_executed"))
    source_executions = {r["execution_id"] for r in source_matrix["target_piece_binding_seed_matrix_rows"]}
    term_executions = {r["execution_id"] for r in rows if r["source_variant"] == "source_term_token"}
    centered_executions = {r["execution_id"] for r in rows if r["source_variant"] == "source_centered_pm1"}
    checks = {
        "complete": report["status"] == "dose_matched_response_complete" and not report["errors"],
        "eight_observables_four_executions_ten_simulations": len(rows) == 8 and report["observable_count"] == 8
            and report["new_measurement_count"] == 4 and report["physical_replay_count"] == 10,
        "source_log_hash_matches": hashlib.sha256(raw).hexdigest() == report["source_log"]["sha256"],
        "original_context_matches": report["measurement_context_id"] == source_matrix["target_piece_binding_seed_matrix_summary"]["measurement_context_id"],
        "term_execution_ids_match_original": term_executions == source_executions,
        "centered_execution_ids_are_new": centered_executions.isdisjoint(source_executions),
        "different_source_tensors": all(r["source_tensors_distinct"] for r in report["source_direction_comparisons"]),
        "no_edit_and_repeat_controls_zero": report["no_edit_max_abs_logit_delta"] == 0 and all(r["repeat_max_abs_logit_delta"] == 0 for r in rows),
        "state_restored": report["state_restored"] is True,
        "all_bindings_honored": all(r["target_piece_binding_requested_honored"] for r in rows),
        "hooks_called_once": all(r["activation_patch_hook_call_count"] == 1 for r in rows),
        "effective_dose_matches_within_1e_6": all(abs(r["activation_patch_blend_delta_norm"] - r["activation_patch_step_size"]) < 1e-6 for r in rows),
        "no_bound_top20_or_rank_improvement": all(r["bound_token_top20_hit_delta"] == 0 and r["target_rank_delta"] == 0 for r in rows),
        "no_candidate_ready_for_promotion": all(not r["review_eligible"] for r in report["promotion_readiness_by_observable"].values()),
        "no_permission_added_by_measurement": all(not r["production_apply_allowed"] and not r["certified_for_apply"] for r in rows),
    }
    audit = {"checks": checks, "all_checks_passed": all(checks.values()),
        "measurement_context_id": report["measurement_context_id"], "prefix": report["source_log"]["prefix"],
        "source_episode_apply_count": sum(e.get("command", {}).get("decision") == "apply" for e in events if e.get("event") == "controller_command"),
        "provider_calls_in_this_comparison": 0, "comparison_production_apply_count": 0,
        "observation_scope": "one_frozen_prefix_source_construction_not_task_success",
        "ownership_warning": "legacy composite ownership includes rank; not target mass certification",
        "rows": [{k: r[k] for k in ("source_variant", "activation_patch_step_size", "target_piece", "execution_id", "observable_id",
            "target_piece_logit_delta", "target_piece_prob_delta", "threshold20_logit_delta", "target_top20_threshold_gap_delta",
            "bound_token_top20_hit_delta", "actual_delta_class", "actuator_class", "self_delta", "cross_delta", "alignment_margin")}
            for r in rows]}
    if args.verify_replay:
        repeated = json.loads(args.verify_replay.read_text())
        audit["packaged_replay_verification"] = {
            "status": repeated["status"],
            "identical_rows": repeated["rows"] == rows,
            "identical_comparisons": repeated["source_direction_comparisons"] == report["source_direction_comparisons"],
            "identical_readiness": repeated["promotion_readiness_by_observable"] == report["promotion_readiness_by_observable"],
            "identical_context": repeated["measurement_context_id"] == report["measurement_context_id"],
            "additional_physical_replays": repeated["physical_replay_count"],
        }
        checks["packaged_replay_reproduced"] = all(value is True for key, value in audit["packaged_replay_verification"].items() if key.startswith("identical_"))
        audit["all_checks_passed"] = all(checks.values())
    (destination / "audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    if args.supporting_artifacts:
        for source, target in (("post_run_debrief.md", "source_episode_debrief.md"),
            ("post_run_debrief_input.json", "source_episode_debrief_input.json"),
            ("local_remeasurement_fixed.json", "preceding_seed_remeasurement.json"),
            ("evidence_audit.json", "preceding_seed_audit.json")):
            shutil.copyfile(args.supporting_artifacts / source, destination / target)
    runtime_files = ["SpiralInterventionLab/runtime/response_probe.py", "SpiralInterventionLab/runtime/response_promotion.py",
        "SpiralInterventionLab/runtime/worker.py", "SpiralInterventionLab/runtime/adapter.py",
        "SpiralInterventionLab/examples/replay_matched_response_probe.py"]
    manifest = {"source_log_decompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "files": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(destination.iterdir())
                  if p.is_file() and p.name != "manifest.json"},
        "worker_model_files": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                               for p in sorted(args.worker_model_path.iterdir()) if p.is_file()},
        "code_hashes_at_packaging_not_original_live_snapshot": {name: file_hash(root / name) for name in runtime_files},
        "interview_scope": "earlier_episode_qualitative_only_not_a_new_call_or_certification"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit["checks"]))
    if not audit["all_checks_passed"]:
        raise SystemExit("Evidence audit failed; preserve artifacts and investigate")


if __name__ == "__main__":
    main()
