import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from SpiralInterventionLab.examples.rebuild_post_run_debrief import rebuild
from SpiralInterventionLab.runtime.debrief import _event_view, event_anchored_log_digest
from SpiralInterventionLab.runtime.loop import _episode_task_feedback


def write_events(tmp_path, events):
    (tmp_path / "c1.jsonl").write_text("\n".join(json.dumps(row) for row in events) + "\n")


def review(step, term, before, after, *, input_id="new-input"):
    return {"event": "controller_diagnostic_result", "step": step,
        "diagnostic": "entity_insertion_operator_candidate_review",
        "status": "candidate_blueprints_ready", "entity_candidate_review_input_id": input_id,
        "candidate_terms": [term], "candidate_blueprint_count": 1, "new_candidate_blueprint_count": 1,
        "candidate_blueprints": [{"kind": "entity_insertion_candidate_blueprint",
            "objective_term": term, "first_piece": " " + term, "first_token_id": 4,
            "source_provenance": "source_body", "source_span": {"start": 4, "end": 5},
            "recommended_operator_families": ["source_body_activation_patch_term_token"]}],
        "new_measurement_count": 0, "physical_replay_count": 0,
        "budget_before": {"diagnostic_calls_left": before},
        "budget_after": {"diagnostic_calls_left": after}, "production_apply_allowed": False}


def anchors(digest):
    return {kind: row for row in digest["event_anchors"] for kind in row["anchor_kinds"]}


def test_debrief_keeps_discovery_exhaustion_and_constraints_without_positive_selection(tmp_path):
    events = [review(0, "sample", 12, 11)]
    events.extend({"event": "controller_command", "step": i, "command": {"decision": "noop"}} for i in range(4))
    events += [review(4, "sample", 4, 3, input_id="changed-feedback"),
        {"event": "controller_diagnostic_result", "step": 5, "diagnostic": "operator_diagnostic_replay",
         "actual_delta_class": "dead_actuator", "budget_before": {"diagnostic_calls_left": 3},
         "budget_after": {"diagnostic_calls_left": 2}, "physical_replay_count": 2},
        {"event": "controller_diagnostic_result", "step": 5, "diagnostic": "target_entity_insertion_probe",
         "probe_kind": "missing_term_refresh", "requested_terms": ["Selim"], "new_measurement_count": 1,
         "budget_before": {"diagnostic_calls_left": 2}, "budget_after": {"diagnostic_calls_left": 1}},
        review(6, "Selim", 1, 0),
        {"event": "controller_observation", "step": 17, "task_feedback": {"missing_required_terms": ["Selim"]}},
        {"event": "episode_end", "steps": 18, "task_done": False, "output": "sample report",
         "task_feedback": {"missing_required_terms": ["Selim"], "budget_ok": True},
         "task_feedback_scope": "final_output_task_feedback"}]
    write_events(tmp_path, events)
    original = (tmp_path / "c1.jsonl").read_bytes()
    digest = event_anchored_log_digest(tmp_path)
    kinds = anchors(digest)
    assert kinds["first_missing_term_refresh"]["entity_discovery"]["requested_terms"] == ["Selim"]
    assert kinds["last_entity_candidate_review"]["entity_discovery"]["candidate_terms"] == ["Selim"]
    assert kinds["diagnostic_budget_exhaustion"]["budget_after"]["diagnostic_calls_left"] == 0
    assert kinds["last_operator_replay"]["step"] == 5
    audit = kinds["first_review_without_new_blueprints"]["entity_discovery"]["blueprint_novelty_audit"]
    assert audit["structurally_new_count"] == 0 and audit["recorded_new_count"] == 1
    assert audit["recorded_diagnostic_cost"] == 1
    assert digest["diagnostic_accounting_summary"]["blueprint_reviews"] == {
        "structurally_new_blueprint_count": 2, "reused_blueprint_count": 1,
        "charged_calls_without_new_blueprints": 1}
    report = digest["final_constraint_report"]
    assert report["scope"] == "final_output_task_feedback"
    assert report["task_feedback"]["missing_required_terms"] == ["Selim"]
    assert report["event_id"] in digest["coverage_manifest"]["included_event_ids"]
    assert original == (tmp_path / "c1.jsonl").read_bytes()
    assert digest == event_anchored_log_digest(tmp_path)
    assert len(json.dumps(digest)) <= 20000


def test_debrief_labels_last_prefix_feedback_as_not_final_evaluation(tmp_path):
    write_events(tmp_path, [{"event": "controller_observation", "step": 1,
        "task_feedback": {"missing_required_terms": ["Selim"]}},
        {"event": "episode_end", "steps": 2, "output": "Selim", "task_done": True}])
    report = event_anchored_log_digest(tmp_path)["final_constraint_report"]
    assert report["scope"] == "last_observed_prefix_not_recomputed_at_termination"
    assert report["observed_at_step"] == 1
    # The sampler must not silently claim the pre-token failure describes the final output.
    assert report["task_feedback"]["missing_required_terms"] == ["Selim"]


def test_debrief_relates_nearest_preceding_round_not_first_round_or_later_command(tmp_path):
    events = [{"event": "controller_command", "step": 5, "command": {"decision": "noop", "meta": {"next_action": "old"}}},
        {"event": "controller_diagnostic_result", "step": 5, "diagnostic": "inspect_evidence"},
        {"event": "controller_command", "step": 5, "command": {"decision": "noop", "meta": {"next_action": "refresh"}}},
        {"event": "controller_selection", "step": 5},
        {"event": "controller_diagnostic_request", "step": 5, "diagnostic": "target_entity_insertion_probe"},
        {"event": "controller_diagnostic_result", "step": 5, "diagnostic": "target_entity_insertion_probe",
         "probe_kind": "missing_term_refresh"},
        {"event": "controller_command", "step": 5, "command": {"decision": "apply"}}]
    write_events(tmp_path, events)
    view = anchors(event_anchored_log_digest(tmp_path))["first_missing_term_refresh"]
    assert [row["event_id"] for row in view["related_events"]] == ["c1.jsonl:L3", "c1.jsonl:L4", "c1.jsonl:L5"]
    assert "not_transaction_identity" in view["related_event_scope"]


def test_debrief_keeps_bound_response_controls_distinct_from_aggregate_lift():
    row = {"operator_recipe_id": "r", "execution_id": "e", "actual_delta_class": "target_lift",
        "actual_delta_class_scope": "execution_canonical_binding_and_term_set", "target_mass_delta": 0.000023,
        "target_piece": " sample", "target_piece_prob_delta": 0.0, "target_piece_logit_delta": 0.0,
        "bound_token_top20_hit_delta": 0, "repeat_max_abs_logit_delta": 0.0,
        "no_edit_max_abs_logit_delta": 0.0, "bound_token_response": "within_repeat_variation"}
    view = _event_view({"event": "controller_diagnostic_result", "rows": [row]}, "c1.jsonl:L1")
    result = view["candidate_rows"][0]
    for key in row:
        assert result[key] == row[key]


def test_final_feedback_reuses_task_evaluator_without_reinterpreting_or_changing_score():
    seen = []
    feedback = {"missing_required_terms": ["Selim"], "budget_ok": True}
    env = SimpleNamespace(task_feedback=lambda output: seen.append(output) or deepcopy(feedback))
    result = _episode_task_feedback(env, "sample report")
    assert seen == ["sample report"]
    assert result["task_feedback"] == feedback and result["task_feedback_scope"] == "final_output_task_feedback"
    assert _episode_task_feedback(object(), "x")["task_feedback_status"] == "unavailable"
    def broken(output):
        raise ValueError("private details should not be serialized")
    assert _episode_task_feedback(SimpleNamespace(task_feedback=broken), "x") == {
        "task_feedback_status": "error", "task_feedback_error_type": "ValueError"}


def test_offline_rebuild_preserves_source_and_refuses_overwrite(tmp_path):
    logs = tmp_path / "live"
    logs.mkdir()
    write_events(logs, [review(0, "sample", 12, 11), review(4, "sample", 4, 3)])
    packet = {"qualitative_only": True, "do_not_autofeed_into_next_run": True,
        "log_digest": {"coverage_manifest": {"included_event_ids": []}}}
    source = logs / "post_run_debrief_input.json"
    source.write_text(json.dumps(packet))
    original = source.read_bytes()
    output = tmp_path / "rebuilt"
    comparison = rebuild(logs, output)
    assert comparison["provider_calls"] == comparison["worker_runs"] == 0
    assert comparison["source_input_unchanged"] and comparison["source_logs_unchanged"]
    assert comparison["diagnostic_accounting_summary"]["blueprint_reviews"]["charged_calls_without_new_blueprints"] == 1
    rebuilt = json.loads((output / "rebuilt_post_run_debrief_input.json").read_text())
    assert rebuilt["qualitative_only"] and rebuilt["do_not_autofeed_into_next_run"]
    assert rebuilt["offline_rebuild"]["provider_called"] is False
    with pytest.raises(FileExistsError):
        rebuild(logs, output)
    assert source.read_bytes() == original


def test_budget_trimming_keeps_transition_identity_controls_and_negative_evidence(tmp_path):
    candidate_rows = [{"operator_recipe_id": "recipe-" + str(i) + "x" * 100, "execution_id": str(i),
        "actual_delta_class": "target_lift", "actuator_class": "cross_bound", "target_piece": " sample",
        "target_piece_logit_delta": 0.0, "repeat_max_abs_logit_delta": 0.0,
        "no_edit_max_abs_logit_delta": 0.0, "target_mass_delta": 0.000023,
        "actual_delta_class_scope": "execution_canonical_binding_and_term_set"} for i in range(8)]
    events = [review(0, "sample", 12, 11),
        {"event": "controller_diagnostic_result", "step": 3, "rows": candidate_rows,
         "target_piece_binding_seed_matrix_executed": True},
        review(4, "sample", 4, 3),
        {"event": "controller_diagnostic_result", "step": 5, "diagnostic": "target_entity_insertion_probe",
         "probe_kind": "missing_term_refresh", "requested_terms": ["Selim"]},
        review(6, "Selim", 1, 0),
        {"event": "controller_observation", "step": 17, "task_feedback": {"missing_required_terms": ["Selim"]},
         "recent_diagnostic_results": [{"rows": candidate_rows}]},
        {"event": "episode_end", "steps": 18, "task_done": False}]
    write_events(tmp_path, events)
    full = event_anchored_log_digest(tmp_path)
    digest = event_anchored_log_digest(tmp_path, max_chars=10000)
    assert len(json.dumps(digest)) <= 10000
    assert len(json.dumps(full)) > len(json.dumps(digest))
    kinds = anchors(digest)
    assert kinds["last_entity_candidate_review"]["entity_discovery"]["candidate_terms"] == ["Selim"]
    assert kinds["diagnostic_budget_exhaustion"]["budget_after"]["diagnostic_calls_left"] == 0
    assert kinds["first_negative_outcome"]["candidate_rows"][0]["repeat_max_abs_logit_delta"] == 0
    assert not kinds["last_task_feedback"].get("candidate_rows")


def test_final_constraints_do_not_borrow_baseline_feedback_for_controller_run(tmp_path):
    (tmp_path / "b0.jsonl").write_text(json.dumps({"event": "episode_end", "task_done": True,
        "task_feedback": {"missing_required_terms": []}, "task_feedback_scope": "final_output_task_feedback"}))
    write_events(tmp_path, [{"event": "episode_end", "steps": 18, "task_done": False}])
    assert event_anchored_log_digest(tmp_path)["final_constraint_report"] == {
        "status": "not_recorded_in_scanned_events"}


def test_final_constraints_do_not_borrow_feedback_from_an_earlier_episode_in_the_same_file(tmp_path):
    write_events(tmp_path, [
        {"event": "episode_start", "prompt": "p"},
        {"event": "episode_end", "steps": 18, "task_done": True,
         "task_feedback": {"missing_required_terms": []}, "task_feedback_scope": "final_output_task_feedback"},
        {"event": "episode_start", "prompt": "p"},
        {"event": "episode_end", "steps": 18, "task_done": False},
    ])
    assert event_anchored_log_digest(tmp_path)["final_constraint_report"] == {
        "status": "not_recorded_in_scanned_events"}
