"""Deterministic public-event sampling for quarantined post-run interviews."""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .evidence_inspection import evidence_catalog
from .entity_candidate_review import NAME as ENTITY_REVIEW, blueprint_ids


def constraint_view(feedback: Any) -> dict[str, Any]:
    if not isinstance(feedback, Mapping):
        return {}
    keys = ("done", "partial_score", "constraint_violations", "missing_required_terms",
            "required_terms_present", "entity_recall_terms", "missing_keywords", "missing_summary_terms",
            "forbidden_terms_present", "budget_ok", "word_budget_score", "required_term_recall")
    return {k: feedback[k][:24] if isinstance(feedback[k], (list, tuple)) else feedback[k]
            for k in keys if k in feedback}


def _remaining(event: Mapping[str, Any], key: str) -> int | None:
    budget = event.get(key)
    value = budget.get("diagnostic_calls_left") if isinstance(budget, Mapping) else None
    return value if type(value) is int and value >= 0 else None


def _diagnostic_cost(event: Mapping[str, Any]) -> int | None:
    before, after = _remaining(event, "budget_before"), _remaining(event, "budget_after")
    if before is not None and after is not None and before >= after:
        return before - after
    cost = event.get("diagnostic_call_cost")
    if type(cost) is int and cost >= 0:
        return cost
    charged = event.get("diagnostic_budget_charged")
    return int(charged) if isinstance(charged, bool) else None


def _matches(value: Any, names: set[str], values: set[Any] | None = None, depth: int = 0) -> bool:
    if depth > 10:
        return False
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in names and (bool(child) if values is None else isinstance(child, (str, bool, int, float)) and child in values):
                return True
            if _matches(child, names, values, depth + 1):
                return True
    elif isinstance(value, list):
        return any(_matches(x, names, values, depth + 1) for x in value[:128])
    return False


def _event_view(event: Mapping[str, Any], event_id: str, *, preserve_negative: bool = False) -> dict[str, Any]:
    keys = ("event", "step", "steps", "diagnostic", "status", "output", "score", "task_done",
            "budget_before", "budget_after", "new_measurement_count", "physical_replay_count",
            "cached_measurement_count", "new_gate_fact_count", "controller_selection_source",
            "controller_selected_bundle_key", "controller_rejected_signals",
            "controller_round", "requested_action", "accepted", "blocked_reason",
            "generated_tokens_during_hold", "same_prefix_rounds_left", "budget_pool",
            "actual_delta_class", "actuator_class", "verdict")
    view = {"event_id": event_id, **{k: event[k] for k in keys if k in event}}
    for key, value in list(view.items()):
        if isinstance(value, str):
            view[key] = value[:350]
    feedback = constraint_view(event.get("task_feedback"))
    if feedback:
        view["task_feedback"] = feedback
        view["task_feedback_scope"] = event.get("task_feedback_scope", "observed_prefix_feedback")
    for key in ("task_feedback_status", "task_feedback_error_type"):
        if key in event:
            view[key] = event[key]
    if event.get("diagnostic") in {"target_entity_insertion_probe", ENTITY_REVIEW}:
        view["entity_discovery"] = {k: event[k] for k in (
            "probe_kind", "probe_refresh_id", "requested_terms", "focus_terms", "candidate_terms",
            "source_probe_id", "source_probe_recorded_step", "source_probe_is_current_prefix",
            "entity_candidate_review_input_id", "candidate_blueprint_count", "new_candidate_blueprint_count",
            "reused_candidate_blueprint_count", "review_reused", "review_outputs_unchanged",
            "candidate_novelty_scope", "diagnostic_call_cost", "diagnostic_budget_charge_reason",
            "model_forward_count", "evidence_scope", "next_evidence_needed", "production_apply_allowed",
            "blueprint_novelty_audit") if k in event}
        view["entity_discovery"]["blueprints"] = [{k: row[k] for k in (
            "candidate_key", "objective_term", "first_piece", "first_token_id",
            "source_provenance", "rank", "recommended_operator_families") if k in row}
            for row in (event.get("candidate_blueprints") or ())[:6] if isinstance(row, Mapping)]
    if isinstance(event.get("command"), Mapping):
        command = event["command"]
        meta = command.get("meta") if isinstance(command.get("meta"), Mapping) else {}
        view["command"] = {"decision": command.get("decision"), "edit_count": len(command.get("edits", ())),
                           "meta": {k: str(meta[k])[:250] for k in ("blocked_by", "why_not_apply", "micro_rationale", "next_action", "diagnostic_request", "generation_action") if k in meta}}
    # Preserve the admission failure separately from effect size. A positive
    # readout blocked by a changed cap must not be presented as a weak effect.
    reports = [event.get("candidate_trial_handoff"), *(event.get("candidate_trial_handoffs") or ())]
    handoffs = [{k: report[k] for k in ("evidence_id", "candidate_id", "status", "checks",
                 "blocked_reasons", "production_trial_allowed") if k in report}
                for report in reports if isinstance(report, Mapping)]
    if handoffs:
        view["candidate_trial_handoffs"] = handoffs[:4]
    if event.get("diagnostic") == "candidate_action":
        view["candidate_action"] = {k: event[k] for k in (
            "action_id", "candidate_id", "requested_action", "executed_action", "current_context_id",
            "parent_candidate_id", "candidate_derivation", "inherits_measurement",
            "measurement_context_id", "requested_position", "blocked_reason", "evidence_scope",
            "diagnostic_budget_charged", "production_apply_allowed") if k in event}
        evidence = event.get("evidence")
        if isinstance(evidence, Mapping):
            view["candidate_action"]["evidence"] = {k: evidence[k] for k in (
                "measurement_context_id", "measurement_position", "target_piece", "target_piece_token_id",
                "state_restored", "no_edit_control", "repeat_control", "continuation_identical",
                "measurement_origin", "error") if k in evidence}
            metrics = evidence.get("metrics") or {}
            view["candidate_action"]["evidence"]["metrics"] = {k: metrics[k] for k in (
                "target_top20_threshold_gap_delta", "target_piece_prob_delta", "target_piece_logit_delta",
                "target_rank_before", "target_rank_after", "bound_token_top20_hit_delta") if k in metrics}
    summary = event.get("target_piece_binding_seed_matrix_summary")
    if isinstance(summary, Mapping):
        view["matrix"] = {k: summary.get(k) for k in (
            "status", "measurement_mode", "measurement_context_id", "row_count", "new_measurement_count",
            "physical_replay_count", "state_restored", "no_edit_max_abs_logit_delta", "unavailable_reason")}
        invalid = [k for k, value in view["matrix"].items() if isinstance(value, float) and not math.isfinite(value)]
        for key in invalid:
            view["matrix"][key] = None
        if invalid:
            view["matrix"]["invalid_numeric_fields"] = invalid
    # An observation can repeat old evidence in its ledger. Do not present that
    # copy as a new measurement attached to the final task-feedback anchor.
    rows = evidence_catalog([event]) if event.get("event") in {
        "controller_diagnostic_result", "controller_effect_summary"} else []
    if rows:
        selected = [row for row in rows if row.get("execution_id")][:4]
        negative = next((row for row in rows if any(row.get(k) in {
            "harmful", "collapse_sharpener", "collapse_isomorphic", "wrong_direction", "cross_bound"
        } for k in ("actual_delta_class", "actuator_class", "verdict"))), None)
        if negative and negative not in selected:
            selected.append(negative)
        if negative and preserve_negative:
            selected = [negative, *(row for row in selected if row is not negative)]
        selected.extend(row for row in rows if row not in selected)
        selected = selected[:8]
        view["candidate_rows"] = [{k: r.get(k) for k in (
            "evidence_id", "execution_id", "operator_recipe_id", "seed_source", "activation_patch_step_size",
            "target_piece", "target_piece_logit_delta", "threshold20_logit_delta",
            "target_top20_threshold_gap_delta", "target_mass_delta", "target_top20_hit_delta",
            "target_piece_prob_delta", "bound_token_top20_hit_delta",
            "no_edit_max_abs_logit_delta", "repeat_max_abs_logit_delta", "measurement_focus_terms",
            "actual_delta_class", "actuator_class", "verdict", "actual_delta_class_scope", "bound_token_response", "evidence_scope") if k in r} for r in selected]
        view["candidate_rows_omitted"] = max(0, len(rows) - 8)
    return view


def event_anchored_log_digest(log_dir: str | Path | None, *, max_files: int = 16, max_chars: int = 20000) -> dict[str, Any]:
    base = Path(log_dir) if log_dir is not None else None
    paths = sorted(base.rglob("*.jsonl")) if base and base.exists() else []
    counts: Counter[str] = Counter()
    anchors: dict[str, tuple[str, int, dict[str, Any]]] = {}
    files = []
    total = 0
    action_counts: Counter[str] = Counter()
    action_statuses: Counter[str] = Counter()
    action_totals: Counter[str] = Counter()
    spend: dict[str, Counter[str]] = {}
    review_totals: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    for path in paths[:max_files]:
        file_hash = hashlib.sha256()
        relative = str(path.relative_to(base))
        local_counts: Counter[str] = Counter()
        seen_blueprints: set[str] = set()
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, 1):
                file_hash.update(raw)
                total += 1
                try:
                    event = json.loads(raw)
                except (ValueError, UnicodeError):
                    counts["<json_decode_error>"] += 1
                    continue
                if not isinstance(event, dict):
                    continue
                name = str(event.get("event", "<missing_event>"))
                if name == "episode_start":
                    seen_blueprints.clear()
                counts[name] += 1
                local_counts[name] += 1
                kinds = []
                if name == "controller_diagnostic_result":
                    diagnostic = str(event.get("diagnostic") or "unknown")
                    accounting = spend.setdefault(diagnostic, Counter())
                    accounting["result_count"] += 1
                    cost = _diagnostic_cost(event)
                    accounting["cost_unknown_count" if cost is None else "charged_calls"] += 1 if cost is None else cost
                    for key in ("physical_replay_count", "new_measurement_count"):
                        value = event.get(key)
                        if type(value) is int and value >= 0:
                            accounting[key] += value
                    before, after = _remaining(event, "budget_before"), _remaining(event, "budget_after")
                    if before is not None and before > 0 and after == 0:
                        kinds.append("diagnostic_budget_exhaustion")
                    if diagnostic == "target_entity_insertion_probe" and event.get("probe_kind") == "missing_term_refresh":
                        kinds.extend(("first_missing_term_refresh", "last_missing_term_refresh"))
                    if diagnostic == ENTITY_REVIEW:
                        ids = set(blueprint_ids(event))
                        new_count = len(ids - seen_blueprints)
                        seen_blueprints.update(ids)
                        event["blueprint_novelty_audit"] = {
                            "structurally_new_count": new_count, "reused_count": len(ids) - new_count,
                            "recorded_new_count": event.get("new_candidate_blueprint_count"),
                            "recorded_diagnostic_cost": cost,
                            "scope": "retrospective_structure_comparison_not_measurement_or_budget_refund",
                        }
                        review_totals["structurally_new_blueprint_count"] += new_count
                        review_totals["reused_blueprint_count"] += len(ids) - new_count
                        if ids and not new_count:
                            kinds.append("first_review_without_new_blueprints")
                            review_totals["charged_calls_without_new_blueprints"] += cost or 0
                        kinds.extend(("first_entity_candidate_review", "last_entity_candidate_review"))
                    if diagnostic == "operator_diagnostic_replay":
                        kinds.append("last_operator_replay")
                    if event.get("diagnostic") == "candidate_action":
                        action_counts[str(event.get("executed_action") or "not_executed")] += 1
                        action_statuses[str(event.get("status") or "unknown")] += 1
                        for key in ("new_measurement_count", "cached_measurement_count", "physical_replay_count"):
                            action_totals[key] += int(event.get(key) or 0)
                        action_totals["diagnostic_budget_charged"] += int(bool(event.get("diagnostic_budget_charged")))
                        kinds.append("first_candidate_action")
                        if event.get("status") == "measured":
                            kinds.extend(("first_current_prefix_measurement", "last_current_prefix_measurement"))
                        if event.get("status") in {"blocked", "incomplete", "held", "reviewed", "measurement_reused"}:
                            kinds.append("first_candidate_action_" + event["status"])
                    if _matches(event, {"materialized_candidate_count", "entity_operator_materialization_count", "activation_patch_candidate_pool", "activation_patch_blueprint_materialization_rows"}):
                        kinds.append("first_materialization")
                    if event.get("target_piece_binding_seed_matrix_executed"):
                        kinds.append("first_actual_matrix")
                    if _matches(event, {"actual_delta_class", "actuator_class", "verdict"}, {"harmful", "collapse_sharpener", "wrong_direction", "cross_bound"}):
                        kinds.append("first_negative_outcome")
                    if _matches(event, {"status"}, {"already_replayed"}):
                        kinds.append("first_cached_review")
                if name == "controller_effect_summary" and _matches(event, {"harmful", "harmful_count"}):
                    kinds.append("first_negative_outcome")
                if name == "episode_end":
                    kinds.append("termination")
                if constraint_view(event.get("task_feedback")):
                    kinds.append("last_task_feedback")
                if name == "controller_prefix_hold":
                    kinds.append("first_prefix_hold_accepted" if event.get("accepted") else "first_prefix_hold_denied")
                if not anchors:
                    kinds.append("first_event")
                for kind in kinds:
                    anchor_counts[kind] += 1
                    if kind not in anchors or kind == "termination" or kind.startswith("last_"):
                        anchors[kind] = (relative, line_number, event)
        files.append({"path": relative, "sha256": file_hash.hexdigest(), "event_counts": dict(local_counts)})
    anchor_views = []
    included = set()
    by_event_id: dict[str, dict[str, Any]] = {}
    for kind, (relative, line_number, event) in sorted(anchors.items(), key=lambda item: item[1][:2]):
        event_id = f"{relative}:L{line_number}"
        if event_id in by_event_id:
            by_event_id[event_id]["anchor_kinds"].append(kind)
            continue
        included.add(event_id)
        negative = anchors.get("first_negative_outcome")
        view = _event_view(event, event_id, preserve_negative=bool(negative and negative[:2] == (relative, line_number)))
        view["anchor_kind"] = kind
        view["anchor_kinds"] = [kind]
        by_event_id[event_id] = view
        view["related_events"] = []
        step = event.get("step")
        if step is not None:
            with (base / relative).open(encoding="utf-8") as handle:
                for n, raw in enumerate(handle, 1):
                    try:
                        other = json.loads(raw)
                    except ValueError:
                        continue
                    if not isinstance(other, dict) or other.get("step") != step or n >= line_number:
                        continue
                    if other.get("event") not in {"controller_diagnostic_request", "controller_command", "controller_selection"}:
                        continue
                    related_id = f"{relative}:L{n}"
                    included.add(related_id)
                    view["related_events"].append(_event_view(other, related_id))
                    view["related_events"] = view["related_events"][-3:]
            if view["related_events"]:
                view["related_event_scope"] = "nearest_preceding_same_step_context_not_transaction_identity"
        anchor_views.append(view)
    result = {"log_dir": str(base) if base else None, "jsonl_file_count": len(paths),
              "event_counts": dict(counts), "log_files": files, "controller_step_views_tail": [],
              "candidate_action_summary": {"executed_actions": dict(action_counts),
                  "statuses": dict(action_statuses), **dict(action_totals),
                  "scope": "public_diagnostic_results_not_rollout_apply_or_intervention_success"},
              "diagnostic_accounting_summary": {
                  "by_diagnostic": [{"diagnostic": name, **dict(values)} for name, values in sorted(spend.items())],
                  "blueprint_reviews": dict(review_totals),
                  "scope": "recorded_spend_and_structural_novelty_not_counterfactual_task_improvement"},
              "event_anchors": anchor_views,
              "coverage_manifest": {"selection": "deterministic_event_anchors", "total_event_lines": total,
                  "anchor_occurrence_counts": dict(anchor_counts),
                  "omitted_files": max(0, len(paths) - max_files), "character_budget": max_chars,
                  "interpretation": "Absent from this view does not imply absent from runtime."}}
    final_feedback = anchors.get("last_task_feedback")
    termination = anchors.get("termination")
    if final_feedback and (not termination or final_feedback[0] == termination[0]):
        relative, line_number, event = final_feedback
        result["final_constraint_report"] = {
            "event_id": f"{relative}:L{line_number}",
            "observed_at_step": event.get("step", event.get("steps")),
            "scope": event.get("task_feedback_scope", "last_observed_prefix_not_recomputed_at_termination"),
            "task_feedback": constraint_view(event.get("task_feedback")),
        }
    else:
        result["final_constraint_report"] = {"status": "not_recorded_in_scanned_events"}
    # Account for the manifest itself. Trim detail, not by positive score.
    while True:
        included = {v["event_id"] for v in anchor_views}
        included.update(r["event_id"] for v in anchor_views for r in v.get("related_events", ()))
        result["coverage_manifest"].update(included_event_ids=sorted(included),
            omitted_event_count=max(0, total - len(included)),
            omitted_file_metadata=max(0, min(len(paths), max_files) - len(files)),
            omitted_fields_policy="Only explicitly listed event/measurement/decision fields were exposed; raw provider text and hidden reasoning are excluded.")
        if len(json.dumps(result)) <= max_chars:
            return result
        removable = [v for v in anchor_views if v.get("related_events")]
        if removable:
            max(removable, key=lambda v: len(json.dumps(v)))["related_events"].pop()
            continue
        removable = [v for v in anchor_views if any("recommended_operator_families" in row
            for row in v.get("entity_discovery", {}).get("blueprints", ()))]
        if removable:
            for row in removable[-1]["entity_discovery"]["blueprints"]:
                row.pop("recommended_operator_families", None)
            removable[-1]["entity_discovery"]["blueprint_families_omitted"] = True
            continue
        removable = [v for v in anchor_views if len(v.get("candidate_rows", ())) > (
            1 if {"first_actual_matrix", "first_negative_outcome"}.intersection(v["anchor_kinds"]) else 0)]
        if removable:
            candidate = max(removable, key=lambda v: (len(v["candidate_rows"]), len(json.dumps(v))))
            candidate["candidate_rows"].pop()
            candidate["candidate_rows_omitted"] += 1
        elif len(files) > 1:
            files.pop()
        else:
            raise ValueError("debrief character budget too small for anchor metadata")


def audit_debrief_references(memo: str, digest: Mapping[str, Any]) -> dict[str, Any]:
    included = set(digest.get("coverage_manifest", {}).get("included_event_ids", ()))
    cited = set(re.findall(r"[\w./-]+\.jsonl:L\d+", memo))
    section = memo.split("## What would have made this easier?", 1)
    wishes = section[1].split("\n## ", 1)[0].splitlines() if len(section) > 1 else []
    wishes = [line for line in wishes if line.strip().startswith("-")]
    categories = ("missing_at_runtime", "present_but_not_shown", "shown_but_ambiguous", "unknown_from_supplied_evidence")
    return {"qualitative_only": True, "production_apply_allowed": False,
            "reference_check_scope": "checks_exposure_not_claim_truth",
            "cited_event_ids": sorted(cited), "unexposed_event_ids": sorted(cited - included),
            "wish_count": len(wishes),
            "uncited_wish_count": sum(not any(event_id in line for event_id in included) for line in wishes),
            "uncategorized_wish_count": sum(not any(category in line for category in categories) for line in wishes)}
