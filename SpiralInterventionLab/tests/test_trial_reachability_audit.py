from SpiralInterventionLab.examples.audit_trial_reachability_jsonl import summarize
from SpiralInterventionLab.runtime.debrief import _event_view


def test_missing_handoff_instrumentation_is_unknown_not_a_failed_gate():
    result = summarize([{"event": "controller_command", "command": {"decision": "noop"}}])
    assert not result["episode_complete"]
    assert result["stage_counts"]["readout_checks_passed"] is None
    assert result["stage_counts"]["confirmation_available"] is None
    assert result["stage_counts"]["compiled_rollout_edits"] == 0


def test_readout_success_is_not_confirmation_permission_or_execution():
    report = {"evidence_id": "obs:1", "checks": {"bound_target_lift": True, "same_context": True},
              "status": "blocked", "blocked_reasons": ["normal_trial_edit_differs_from_diagnostic"]}
    row = {"event": "controller_diagnostic_result", "diagnostic": "candidate_action",
           "candidate_trial_handoff": report}
    result = summarize([row, row, {"event": "compiled_edit", "kind": "rollback"}])
    assert result["stage_counts"]["recorded_handoffs"] == 1
    assert result["stage_counts"]["readout_checks_passed"] == 1
    assert result["stage_counts"]["confirmation_available"] == 0
    assert result["stage_counts"]["physical_confirmation_results"] == 0
    assert result["stage_counts"]["compiled_rollout_edits"] == 0
    assert result["handoff_blocker_counts"] == {"normal_trial_edit_differs_from_diagnostic": 1}


def test_debrief_keeps_cap_failure_distinct_from_positive_readout():
    report = {"evidence_id": "obs:1", "status": "blocked", "checks": {"bound_target_lift": True},
              "blocked_reasons": ["normal_trial_edit_differs_from_diagnostic"], "production_trial_allowed": False}
    view = _event_view({"event": "controller_diagnostic_result", "diagnostic": "candidate_action",
                       "candidate_trial_handoff": report}, "c1.jsonl:L1")
    assert view["candidate_trial_handoffs"] == [report]


def test_null_optional_handoff_list_is_uninstrumented():
    row = {"event": "controller_diagnostic_result", "diagnostic": "activation_patch_candidate_review",
           "candidate_trial_handoffs": None, "candidate_trial_handoff": None}
    assert "candidate_trial_handoffs" not in _event_view(row, "c1.jsonl:L1")
    assert summarize([row])["stage_counts"]["readout_checks_passed"] is None
