from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload
from SpiralInterventionLab.examples.replay_candidate_handoff_choice import (
    _assert_handoff_only_diff,
    _load_recorded_seed,
)
from SpiralInterventionLab.examples.replay_candidate_seed_discovery import (
    load_early_reviews,
    select_frozen_remeasurement,
    summarize_discovery,
)
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


def test_source_body_blueprint_is_only_an_executable_discovery_option():
    worker = worker_with_pool()
    worker._diagnostic_results = [{"diagnostic": "entity_insertion_operator_candidate_review",
        "candidate_blueprints": [{"kind": "entity_insertion_candidate_blueprint",
            "candidate_key": OBJECTIVE, "objective_term": "Mira", "source_provenance": "source_body",
            "source_span": {"start": 10, "end": 12, "provenance_class": "source_body",
                            "span_kind": "exact_prompt_span"}}]}]
    worker.surface_catalog = [SimpleNamespace(allow_ops=("activation_patch",),
        target=SimpleNamespace(kind="activation", site="resid_pre", token=SimpleNamespace(mode="last")))]
    worker._last_task_feedback = {"required_terms_present": []}
    discovery = candidate_handoff.seed_discovery_report(worker)
    assert discovery["status"] == "available" and discovery["blueprint_count"] == 1
    assert discovery["options"][0]["request"] == {
        "diagnostic": "activation_patch_candidate_review", "bundle_key": OBJECTIVE,
        "objective_bundle_key": OBJECTIVE,
        "operator_recipe_expansion_mode": "activation_patch_candidate_review"}
    assert discovery["options"][0]["seed_status"] == "unmeasured_blueprint"
    assert discovery["production_apply_allowed"] is False
    assert candidate_handoff.report(worker)["state"] == "unmeasurable"

    worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run
    assert candidate_handoff.seed_discovery_report(worker)["status"] == "diagnostic_budget_exhausted"
    worker._diagnostic_calls_used = 1
    worker.surface_catalog = []
    assert candidate_handoff.seed_discovery_report(worker)["status"] == "no_live_activation_patch_surface"
    worker.surface_catalog = [SimpleNamespace(allow_ops=("activation_patch",),
        target=SimpleNamespace(kind="activation", site="resid_pre", token=SimpleNamespace(mode="last")))]
    worker._last_task_feedback = {"required_terms_present": ["Mira"]}
    assert candidate_handoff.seed_discovery_report(worker)["options"] == []
    worker._last_task_feedback = {"required_terms_present": []}
    worker._diagnostic_results.append({"diagnostic": "activation_patch_candidate_review",
                                       "objective_bundle_key": OBJECTIVE, "status": "ok"})
    assert candidate_handoff.seed_discovery_report(worker)["options"] == []


def test_activation_patch_review_next_action_alias_spends_only_one_diagnostic():
    command = {"decision": "noop", "meta": {
        "diagnostic_request": {"diagnostic": "activation_patch_candidate_review",
                               "bundle_key": OBJECTIVE, "objective_bundle_key": OBJECTIVE,
                               "operator_recipe_expansion_mode": "activation_patch_candidate_review"},
        "next_action": "request_activation_patch_candidate_review",
        "next_evidence_needed": "activation_patch_candidate_review",
        "objective_bundle_key": OBJECTIVE,
    }}
    requests = _extract_diagnostic_requests(command, {"strategy_hints": {}})
    assert len(requests) == 1
    assert requests[0]["operator_recipe_expansion_mode"] == "activation_patch_candidate_review"


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


def test_fixed_prefix_shadow_requires_recorded_binding_and_single_seed(tmp_path):
    import json

    path = tmp_path / "source.jsonl"
    row = seed_row()
    row["target_piece_binding_manifest"] = {"answer_prefix_tail": " In the case of"}
    path.write_text("\n".join(json.dumps(event) for event in (
        {"event": "episode_start", "prompt": "fixture"},
        {"event": "controller_diagnostic_result", "evidence_rows": [row]},
    )) + "\n")
    prompt, loaded = _load_recorded_seed(path, row["operator_recipe_id"], " In the case of")
    assert prompt == "fixture" and loaded == row
    from pytest import raises

    with raises(ValueError, match="different answer prefix"):
        _load_recorded_seed(path, row["operator_recipe_id"], " another prefix")
    with raises(ValueError, match="one episode prompt and one recorded seed"):
        _load_recorded_seed(path, "missing recipe", " In the case of")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "episode_start", "prompt": "fixture"}) + "\n")
    with raises(ValueError, match="one episode"):
        _load_recorded_seed(path, row["operator_recipe_id"], " In the case of")


def test_fixed_prefix_shadow_rejects_non_handoff_packet_drift():
    from pytest import raises

    off = {"telemetry": {"target_mass": 0.1}, "strategy_hints": {"available_next_diagnostics": []}}
    soft = {"telemetry": {"target_mass": 0.1}, "strategy_hints": {
        "candidate_handoff": {"state": "measurable"},
        "available_next_diagnostics": [{"diagnostic": "matched_response_probe"}]}}
    assert _assert_handoff_only_diff(off, soft, compact=False) == [
        "strategy_hints.available_next_diagnostics", "strategy_hints.candidate_handoff"]
    soft["telemetry"]["target_mass"] = 0.2
    with raises(ValueError, match="packet drift outside handoff"):
        _assert_handoff_only_diff(off, soft, compact=False)


def test_early_seed_discovery_requires_recorded_prefix_and_source_body_blueprint(tmp_path):
    import json
    from pytest import raises

    path = tmp_path / "source.jsonl"
    events = [
        {"event": "episode_start", "prompt": "fixture"},
        {"event": "controller_observation", "generated_tail": " In"},
        {"event": "controller_diagnostic_result", "diagnostic": "target_entity_insertion_probe"},
        {"event": "controller_diagnostic_result", "diagnostic": "entity_insertion_operator_candidate_review",
         "candidate_blueprints": [{"candidate_key": OBJECTIVE, "source_provenance": "source_body"}]},
    ]
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n")
    prompt, reviews = load_early_reviews(path, prefix=" In", objective=OBJECTIVE)
    assert prompt == "fixture" and len(reviews) == 2
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "controller_observation", "generated_tail": " In the"}) + "\n")
    with raises(ValueError, match="early prefix"):
        load_early_reviews(path, prefix=" In the", objective=OBJECTIVE)
    with raises(ValueError, match="early prefix"):
        load_early_reviews(path, prefix=" In the case", objective=OBJECTIVE)
    with raises(ValueError, match="source-body blueprint"):
        load_early_reviews(path, prefix=" In", objective="entity_insert:other")

    late_path = tmp_path / "late.jsonl"
    late_path.write_text("\n".join(json.dumps(event) for event in (
        {"event": "episode_start", "prompt": "fixture"},
        {"event": "controller_observation", "generated_tail": " In"},
        {"event": "controller_observation", "generated_tail": " In the"},
        events[2], events[3],
    )) + "\n")
    with raises(ValueError, match="early prefix"):
        load_early_reviews(late_path, prefix=" In", objective=OBJECTIVE)


def test_early_seed_discovery_keeps_measured_rows_distinct_from_permission():
    result = {"diagnostic": "activation_patch_candidate_review", "status": "ok",
              "diagnostic_call_cost": 1, "diagnostic_budget_charged": True,
              "activation_patch_blueprint_materialization_count": 2,
              "evidence_rows": [
                  {"objective_bundle_key": OBJECTIVE, "diagnostic_family": "activation_patch",
                   "activation_hook_call_count": 1, "actual_delta_class": "rank_carrier",
                   "target_mass_delta": 0.000001, "production_apply_allowed": False},
                  {"objective_bundle_key": OBJECTIVE, "diagnostic_family": "activation_patch",
                   "activation_hook_call_count": 0, "actual_delta_class": "materialization_failed"},
              ]}
    handoff = {"state": "measurable", "measurement_offers": [
        {"objective_bundle_key": OBJECTIVE, "seed_recipe_id": "recipe"}]}
    summary = summarize_discovery(result, handoff, objective=OBJECTIVE)
    assert summary["activation_patch_evidence_row_count"] == 2
    assert summary["activation_patch_hooked_row_count"] == 1
    assert summary["handoff_offer_recipe_ids"] == ["recipe"]
    assert summary["production_apply_allowed"] is False


def test_frozen_remeasurement_selects_exact_recipe_and_available_action():
    from pytest import raises

    worker = SimpleNamespace(_frozen_diagnostic_candidates={"card": {
        "frozen": SimpleNamespace(descriptor={"operator_recipe_id": "matched-a",
                                              "seed_operator_recipe_id": "recipe-a"})}})
    packet = {"strategy_hints": {"candidate_diagnostic_choices": {"cards": [{
        "candidate_id": "card", "objective_bundle_key": OBJECTIVE, "target_piece": " Mir",
        "actions": [{"action": "review_existing", "action_id": "review", "available": True},
                    {"action": "remeasure_current_prefix", "action_id": "measure", "available": True}],
    }]}}}
    assert select_frozen_remeasurement(worker, packet, objective=OBJECTIVE, recipe_id="recipe-a",
                                       target_piece=" Mir") == {
        "diagnostic": "candidate_action", "action_id": "measure"}
    with raises(ValueError, match="found 0"):
        select_frozen_remeasurement(worker, packet, objective=OBJECTIVE, recipe_id="recipe-b",
                                    target_piece=" Mir")
