from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload
from SpiralInterventionLab.examples.run_iteration_pair import build_argv
from SpiralInterventionLab.runtime import candidate_handoff
from SpiralInterventionLab.runtime.loop import _build_controller_selection_report, _extract_diagnostic_requests
from SpiralInterventionLab.runtime.response_probe import select_probe_seeds


OBJECTIVE = "entity_insert:mira:source_body:weak_reachable"


def seed_row(*, objective: str = OBJECTIVE, effect: str = "rank_carrier") -> dict:
    return {"operator_recipe_id": "readout_escape|activation_patch|resid_pre|L11|mira",
        "operator_axis": "activation_patch_blueprint_materialization",
        "objective_bundle_key": objective, "intended_term": "Mira",
        "actual_delta_class": effect, "activation_patch_site": "resid_pre",
        "activation_patch_layer": 11, "activation_patch_alpha": 0.04,
        "activation_patch_source_localization": "source_term_token"}


def worker_with_pool(row: dict | None = None) -> SimpleNamespace:
    worker = SimpleNamespace(_steps=2, _segments=[], max_diagnostic_calls_per_run=12,
        _diagnostic_calls_used=1, _diagnostic_results=[{"evidence_rows": [row or seed_row()]}],
        _frozen_diagnostic_candidates={}, done=lambda: False, _collect_active_edits=lambda: [])
    candidate_handoff.reset(worker)
    return worker


def test_handoff_preflight_matches_probe_seed_rules_and_does_not_certify():
    worker = worker_with_pool()
    handoff = candidate_handoff.report(worker)
    assert handoff["state"] == "measurable"
    assert handoff["measured_card_count"] == 0
    assert handoff["production_apply_allowed"] is False
    assert "unverified" in handoff["preflight_scope"]
    offer = handoff["measurement_offers"][0]
    assert offer["objective_bundle_key"] == OBJECTIVE
    assert offer["request"] == {"diagnostic": "matched_response_probe",
        "objective_bundle_key": OBJECTIVE, "focus_term": "Mira",
        "comparison_axis": "source_localization", "dose_grid": [0.04],
        "candidate_ids": [seed_row()["operator_recipe_id"]]}
    selected, reason = select_probe_seeds([seed_row()], objective_bundle_key=OBJECTIVE,
        objective_term="Mira", comparison_axis="source_localization")
    assert reason is None and len(selected) == 2
    assert _compact_controller_payload({"strategy_hints": {"candidate_handoff": handoff}})[
        "strategy_hints"]["candidate_handoff"]["measurement_offers"][0]["request"] == offer["request"]


def test_handoff_unmeasurable_and_carded_are_not_offered_for_measurement():
    worker = worker_with_pool(seed_row(effect="collapse_sharpener"))
    assert candidate_handoff.report(worker)["state"] == "unmeasurable"
    assert candidate_handoff.report(worker)["measurement_offers"] == []
    worker._diagnostic_results = [{"evidence_rows": [seed_row()]}]
    worker._diagnostic_calls_used = 12
    exhausted = candidate_handoff.report(worker)
    assert exhausted["state"] == "unmeasurable"
    assert exhausted["blocked_reason"] == "diagnostic_budget_exhausted"
    worker._frozen_diagnostic_candidates = {"card": {"frozen": SimpleNamespace(objective=OBJECTIVE)}}
    assert candidate_handoff.report(worker)["state"] == "carded"


def test_materializable_seed_survives_result_display_window_eviction():
    worker = worker_with_pool()
    candidate_handoff.capture_seed_rows(worker, worker._diagnostic_results[0])
    worker._diagnostic_results = []
    handoff = candidate_handoff.report(worker)
    assert handoff["state"] == "measurable"
    assert handoff["measurement_offers"][0]["request"]["candidate_ids"] == [
        seed_row()["operator_recipe_id"]]
    assert candidate_handoff.seed_catalog(worker)[0]["operator_recipe_id"] == (
        seed_row()["operator_recipe_id"])
    for expected_count in (1, 2):
        deferred = {}
        candidate_handoff.record_diagnostic(worker, handoff,
            {"diagnostic": "objective_rotation_pipeline", "objective_bundle_key": OBJECTIVE},
            deferred, cost=1, source="controller")
        handoff = candidate_handoff.report(worker)
        assert handoff["state"] == "measurable"
        assert handoff["deferred_diagnostic_count"] == expected_count
    assert handoff["soft_opportunity_cost"] == 1


def test_already_present_required_term_is_control_only_not_a_progress_offer():
    worker = worker_with_pool()
    worker._last_task_feedback = {"required_terms_present": ["Mira"],
                                  "missing_required_terms": ["send", "budget", "Omar"]}
    handoff = candidate_handoff.report(worker)
    assert handoff["state"] == "unmeasurable"
    assert handoff["measurement_offers"] == []
    assert handoff["blocked_reason"] == "objective_term_already_present_control_only"
    assert handoff["soft_opportunity_cost"] == 0


def test_handoff_penalty_only_counts_charged_non_safety_diagnostic_at_same_prefix():
    worker = worker_with_pool()
    handoff = candidate_handoff.report(worker)
    safety = {}
    candidate_handoff.record_diagnostic(worker, handoff,
        {"diagnostic": "activation_patch_production_trial_gate_review"}, safety,
        cost=1, source="controller")
    assert safety["candidate_handoff_choice"] == "safety_or_card_action_exempt"
    uncharged = {}
    candidate_handoff.record_diagnostic(worker, handoff,
        {"diagnostic": "objective_rotation_pipeline"}, uncharged, cost=0, source="controller")
    assert not uncharged
    for count in (1, 2):
        result = {}
        candidate_handoff.record_diagnostic(worker, handoff,
            {"diagnostic": "objective_rotation_pipeline"}, result, cost=1, source="controller")
        assert result["candidate_handoff_deferred_diagnostic_count"] == count
    after = candidate_handoff.report(worker)
    assert after["deferred_diagnostic_count"] == 2
    assert after["soft_opportunity_cost"] == 1
    measurement = {"physical_replay_count": 4, "new_measurement_count": 2}
    candidate_handoff.record_diagnostic(worker, after, handoff["measurement_offers"][0]["request"],
        measurement, cost=1, source="controller")
    assert measurement["candidate_handoff_choice"] == "measurement_attempted"
    assert candidate_handoff.report(worker)["state"] == "unmeasurable"
    assert candidate_handoff.report(worker)["unavailable_reasons"][OBJECTIVE] == (
        "measurement_attempt_consumed_at_this_prefix")
    worker._segments = [SimpleNamespace(kind="output", token_ids=[12])]
    assert candidate_handoff.report(worker)["deferred_diagnostic_count"] == 0
    assert candidate_handoff.report(worker)["state"] == "measurable"


def test_invalid_or_different_probe_does_not_consume_the_offered_measurement():
    worker = worker_with_pool()
    handoff = candidate_handoff.report(worker)
    request = handoff["measurement_offers"][0]["request"]
    invalid = {"status": "invalid_request", "physical_replay_count": 0,
               "new_measurement_count": 0}
    candidate_handoff.record_diagnostic(worker, handoff,
        {**request, "dose_grid": [99.0]}, invalid, cost=0, source="controller")
    assert invalid["candidate_handoff_choice"] == "measurement_not_executed"
    assert candidate_handoff.report(worker)["state"] == "measurable"

    different = {"physical_replay_count": 4, "new_measurement_count": 2}
    candidate_handoff.record_diagnostic(worker, handoff,
        {**request, "candidate_ids": ["different-recipe"]}, different,
        cost=1, source="controller")
    assert different["candidate_handoff_choice"] == "different_measurement_executed"
    assert candidate_handoff.report(worker)["state"] == "measurable"

    measured = {"physical_replay_count": 4, "new_measurement_count": 2}
    candidate_handoff.record_diagnostic(worker, handoff, request, measured,
        cost=1, source="controller")
    assert measured["candidate_handoff_choice"] == "measurement_attempted"
    assert candidate_handoff.report(worker)["unavailable_reasons"][OBJECTIVE] == (
        "measurement_attempt_consumed_at_this_prefix")


def test_exact_offered_probe_without_replay_is_unavailable_only_at_this_prefix():
    worker = worker_with_pool()
    handoff = candidate_handoff.report(worker)
    unavailable = {"status": "no_cached_evidence", "physical_replay_count": 0,
                   "new_measurement_count": 0}
    candidate_handoff.record_diagnostic(worker, handoff,
        handoff["measurement_offers"][0]["request"], unavailable,
        cost=0, source="controller")
    assert unavailable["candidate_handoff_choice"] == "measurement_unavailable_at_this_prefix"
    report = candidate_handoff.report(worker)
    assert report["state"] == "unmeasurable"
    assert report["blocked_reason"] == "offered_measurement_no_physical_replay_at_this_prefix"
    worker._segments = [SimpleNamespace(kind="output", token_ids=[12])]
    assert candidate_handoff.report(worker)["state"] == "measurable"


def test_controller_logs_explicit_handoff_deferral_without_overriding_choice():
    worker = worker_with_pool()
    handoff = candidate_handoff.report(worker)
    packet = {"strategy_hints": {"candidate_handoff": handoff}}
    command = {"version": "0.1", "decision": "noop", "edits": [], "rollback_ids": [], "meta": {
        "diagnostic_request": {"diagnostic": "objective_rotation_pipeline", "objective_bundle_key": OBJECTIVE},
        "handoff_defer_reason": "Need an independent operator family before binding this seed.",
    }}
    report = _build_controller_selection_report(packet, command)
    assert report["candidate_handoff_choice"] == "deferred_for_other_diagnostic"
    assert report["candidate_handoff_defer_reason"] == command["meta"]["handoff_defer_reason"]
    assert report["candidate_handoff_defer_reason_source"] == "explicit_handoff_reason"
    command["meta"].pop("handoff_defer_reason")
    command["meta"]["micro_rationale"] = "Checking independent operator evidence."
    fallback = _build_controller_selection_report(packet, command)
    assert fallback["candidate_handoff_defer_reason"] == "Checking independent operator evidence."
    assert fallback["candidate_handoff_defer_reason_source"] == "general_micro_rationale"
    command["meta"]["diagnostic_request"] = handoff["measurement_offers"][0]["request"]
    command["meta"]["next_action"] = "request_objective_rotation_pipeline"
    assert _extract_diagnostic_requests(command, packet) == [handoff["measurement_offers"][0]["request"]]
    assert _build_controller_selection_report(packet, command)["candidate_handoff_choice"] == "measurement_requested"
    command["meta"]["diagnostic_request"] = {
        **handoff["measurement_offers"][0]["request"], "dose_grid": [0.16]}
    assert _build_controller_selection_report(packet, command)["candidate_handoff_choice"] == (
        "different_measurement_requested")


def test_iteration_pair_mode_is_cli_only_and_preserves_diagnostic_budget():
    argv = build_argv("gpt2_control", Path("/tmp/model"), Path("/tmp/run"),
                      device="mps", controller="gpt-5.6-luna", candidate_handoff_mode="soft")
    assert argv[argv.index("--candidate-handoff-mode") + 1] == "soft"
    assert argv[argv.index("--max-diagnostic-calls-per-run") + 1] == "12"
