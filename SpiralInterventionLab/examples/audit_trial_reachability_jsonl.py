"""Read-only admission-path audit. Missing instrumentation is not a failed gate."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(rows: list[dict]) -> dict:
    handoffs, offers, cards, measurements = {}, set(), {}, []
    counts, diagnostics = Counter(), Counter()
    first_prefix = None
    outcome = None
    for row in rows:
        event = row.get("event")
        if event == "episode_end":
            outcome = {k: row.get(k) for k in ("output", "score", "steps", "task_done")}
        if event == "controller_observation" and first_prefix is None:
            first_prefix = row.get("generated_tail")
        if event == "controller_command":
            counts["command_" + str(row.get("command", {}).get("decision"))] += 1
        if event == "compiled_edit" and row.get("kind") != "rollback":
            counts["compiled_rollout_edits"] += 1
        if event == "controller_prefix_hold":
            counts["prefix_hold_requests"] += 1
            counts["prefix_holds_accepted"] += row.get("accepted") is True
        if event in {"controller_observation", "controller_selection"}:
            hints = row.get("strategy_hints", {}) if event == "controller_observation" else row
            offer = hints.get("candidate_trial_offer") or {}
            if offer.get("trial_authorization_id"):
                offers.add(offer["trial_authorization_id"])
            for card in (hints.get("candidate_diagnostic_choices") or {}).get("cards", []):
                cards[card["candidate_id"]] = {k: card.get(k) for k in (
                    "candidate_id", "objective_bundle_key", "target_piece", "operator")}
        if event != "controller_diagnostic_result":
            continue
        diagnostics[str(row.get("diagnostic"))] += 1
        if row.get("diagnostic") == "candidate_action":
            evidence = row.get("evidence") or {}
            measurements.append({"candidate_id": row.get("candidate_id"),
                "parent_candidate_id": row.get("parent_candidate_id"),
                "action": row.get("executed_action"), "status": row.get("status"),
                "worker_step": evidence.get("worker_step"), "prefix": evidence.get("prefix"),
                "metrics": evidence.get("metrics"), "state_restored": evidence.get("state_restored"),
                "handoff_blockers": (row.get("candidate_trial_handoff") or {}).get("blocked_reasons")})
        if row.get("evidence_id") and row.get("diagnostic") in {
            "activation_patch_promotion_gate_review", "activation_patch_production_shadow_replay",
            "activation_patch_production_trial_gate_review",
        }:
            counts["evidence_id_review_results"] += 1
            if row.get("confirmation") is not None:
                counts["physical_confirmation_results"] += 1
        for report in [row.get("candidate_trial_handoff") or {}, *(row.get("candidate_trial_handoffs") or [])]:
            if report.get("evidence_id"):
                handoffs[report["evidence_id"]] = report

    blockers = Counter()
    checks_passed = 0
    available = 0
    for report in handoffs.values():
        blockers.update(report.get("blocked_reasons", []))
        checks = report.get("checks") or {}
        checks_passed += bool(checks) and all(v is True for v in checks.values())
        available += report.get("status") == "confirmation_available"
    return {
        "scope": "recorded_path_only_not_proof_of_operator_absence_or_controller_intent",
        "episode_complete": outcome is not None, "outcome": outcome,
        "first_controller_prefix": first_prefix,
        "stage_counts": {"recorded_handoffs": len(handoffs),
            "readout_checks_passed": checks_passed if handoffs else None,
            "confirmation_available": available if handoffs else None,
            "evidence_id_review_results": counts["evidence_id_review_results"],
            "physical_confirmation_results": counts["physical_confirmation_results"],
            "unique_trial_offers_seen": len(offers),
            "controller_apply_commands": counts["command_apply"],
            "controller_noop_commands": counts["command_noop"],
            "compiled_rollout_edits": counts["compiled_rollout_edits"]},
        "prefix_hold_requests": counts["prefix_hold_requests"],
        "prefix_holds_accepted": counts["prefix_holds_accepted"],
        "handoff_blocker_counts": dict(blockers), "diagnostic_counts": dict(diagnostics),
        "visible_candidates": list(cards.values()), "measurements": measurements,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.traces:
        content = path.read_bytes()
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        reports.append({"trace": str(path), "sha256": hashlib.sha256(content).hexdigest(), **summarize(rows)})
    # Do not overwrite a sealed run or an earlier analysis artifact.
    with args.output.open("x") as stream:
        json.dump({"schema": "trial_reachability_audit_v1", "reports": reports}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
