from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.runtime.response_probe import matched_response_probe, bound_metrics, logit_variation
from SpiralInterventionLab.runtime.response_promotion import response_review_readiness, review_response_evidence
from SpiralInterventionLab.runtime.evidence_inspection import inspect_evidence
from SpiralInterventionLab.runtime.debrief import event_anchored_log_digest, audit_debrief_references
from SpiralInterventionLab.runtime.worker import HookedTransformerWorkerRuntime, _normalize_controller_diagnostic_request
from SpiralInterventionLab.runtime.loop import _diagnostic_request_signature, _extract_diagnostic_requests
from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.examples.summarize_activation_patch_jsonl import summarize_activation_patch_jsonl


def fixture_probe():
    before = torch.linspace(5, -5, 40)
    objective = "entity_insert:a:source_body:weak_reachable"
    seeds = [{"objective_bundle_key": objective, "operator_recipe_id": name, "recipe_name": name,
              "activation_patch_seed_source": source, "activation_patch_site": "resid_pre",
              "activation_patch_layer": 1, "activation_patch_alpha": 0.04,
              "activation_patch_step_size": dose, "activation_patch_source_localization": "source_term_token"}
             for name, source, dose in [("direct", "direct_candidate", 0.04), ("observed", "observed_gap_carrier", 0.16)]]
    calls = []
    def binding(_logits, **kwargs):
        token = (kwargs.get("requested_binding") or {}).get("chosen_target_token_id", 30)
        return {"binding_id": f"binding:{token}", "chosen_target_token_id": token, "objective_term": "a",
                "candidate_target_piece_rows": [{"token_id": 30, "baseline_rank": 31}, {"token_id": 31, "baseline_rank": 32}]}
    def materialize(candidate, **kwargs):
        return {"id": "test_trial", "target": {"surface_id": "s1"}, "source": {"dtype": "vector", "expr": {"ref": {"scope": "runtime", "tensor": "hidden", "layer": 1, "token": {"mode": "last"}}}},
                "op": {"kind": "activation_patch", "alpha": candidate["alpha"], "mode": "blend"},
                "budget": {"ttl_steps": 1, "step_size": candidate["step_size"], "revertible": True}}
    def replay(edits, **kwargs):
        dose = edits[0]["budget"]["step_size"]
        calls.append(dose)
        after = before.clone()
        after[30] += dose
        after[31] -= dose
        kwargs["_measurement_capture"]["edited_logits"] = after
        return {"status": "ok", "actual_delta_class": "rank_carrier", "activation_patch_blend_delta_norm": dose}
    worker = SimpleNamespace(runtime_state=SimpleNamespace(),
        adapter=SimpleNamespace(read_ref=lambda ref, ctx: torch.ones(4)),
        build_controller_packet=lambda: {}, _simulate_decode=lambda **kw: {"first_logits": before.clone()},
        _resolve_target_piece_binding=binding, _activation_patch_trial_edit_from_candidate=materialize,
        replay_candidate_edits_actual_delta=replay,
        _first_token_target_readout_metrics=lambda before, after, **kw: {
            "target_piece_token_id": kw["target_piece_binding"]["chosen_target_token_id"],
            "target_piece_binding_id": kw["target_piece_binding"]["binding_id"],
            "target_mass_delta": 0.0, "target_top20_hit_delta": 0})
    return worker, seeds, calls, objective


def test_matched_probe_aliases_execution_and_measures_each_binding():
    worker, seeds, calls, objective = fixture_probe()
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:fixed"):
        report = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a")
    assert report["status"] == "dose_matched_response_complete"
    assert report["row_count"] == 8
    assert report["new_measurement_count"] == 2
    assert report["observable_count"] == 4
    assert report["physical_replay_count"] == 6
    assert calls == [0.04, 0.04, 0.16, 0.16]
    assert sum(r["execution_alias"] for r in report["rows"]) == 4
    assert all(r["target_piece_binding_requested_honored"] for r in report["rows"])
    assert all(not r["certified_for_apply"] and not r["production_apply_allowed"] for r in report["rows"])
    assert report["no_edit_max_abs_logit_delta"] == 0
    assert all(r["repeat_max_abs_logit_delta"] == 0 for r in report["rows"])
    assert report["cached_measurement_count"] == 2
    for seed in ("direct_candidate", "observed_gap_carrier"):
        assert {r["activation_patch_step_size"] for r in report["rows"] if r["seed_source"] == seed} == {0.04, 0.16}


@pytest.mark.parametrize("dose", [[True], [-1], [0.20], [], [0.04, 0.04], "0.04"])
def test_matched_probe_rejects_invalid_dose_before_execution(dose):
    worker, seeds, calls, objective = fixture_probe()
    result = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a", dose_grid=dose)
    assert result["status"] == "invalid_request"
    assert not calls


def test_matched_probe_rejects_changed_state_and_replay_errors():
    worker, seeds, calls, objective = fixture_probe()
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", side_effect=["before", "after"]):
        result = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a")
    assert result["row_count"] == 0
    assert result["unavailable_reason"] == "state_restoration_failed"
    worker.replay_candidate_edits_actual_delta = lambda *a, **kw: {"status": "error", "error": "bad edit"}
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="same"):
        result = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a")
    assert result["status"] == "incomplete" and result["errors"]
    assert not result["production_apply_allowed"]


def test_bound_gap_decomposes_falling_boundary_from_target_gain():
    before = torch.arange(40).float()
    after = before.clone()
    after[20:] -= 1
    after[0] -= 0.1
    m = bound_metrics(before, after, 0)
    assert m["target_piece_logit_delta"] < 0
    assert m["target_top20_threshold_gap_delta"] < 0
    assert m["target_top20_threshold_gap_delta"] == m["threshold20_logit_delta"] - m["target_piece_logit_delta"]


def test_control_excludes_identical_token_masks_but_not_nan_or_changed_masks():
    before = torch.tensor([1.0, 2.0, float("-inf")])
    control = logit_variation(before, before.clone())
    assert control["max_abs_logit_delta"] == 0.0 and control["masked_token_count"] == 1
    for after in (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([float("nan"), 2.0, float("-inf")])):
        with pytest.raises(ValueError):
            logit_variation(before, after)


def test_inspection_is_cached_bounded_and_gate_context_aware():
    row = {"execution_id": "e1", "observable_id": "o1", "operator_recipe_id": "r1",
           "actual_delta_class": "rank_carrier", "objective_bundle_key": "obj", "target_piece_logit_delta": 0.1}
    worker = SimpleNamespace(_diagnostic_results=[{"rows": [row, row]}], _inspection_seen_ids=set(), _inspection_gate_signature=None)
    result = inspect_evidence(worker, {"execution_id": "e1"}, {})
    assert len(result["rows"]) == 1
    assert result["new_measurement_count"] == 0 and result["physical_replay_count"] == 0
    assert result["cached_measurement_count"] == 1 and result["new_gate_fact_count"] == 0
    again = inspect_evidence(worker, {"execution_id": "e1"}, {})
    assert again["newly_visible_row_count"] == 0 and again["new_gate_fact_count"] == 0
    changed = inspect_evidence(worker, {"execution_id": "e1"}, {"diagnostic_budget_exhausted": True})
    assert changed["new_gate_fact_count"] == 1
    assert inspect_evidence(worker, {"limit": True}, {})["status"] == "invalid_request"
    assert inspect_evidence(worker, {"execution_id": "missing"}, {})["status"] == "no_matching_evidence"
    row["repeat_max_abs_logit_delta"] = float("nan")
    result = inspect_evidence(worker, {"execution_id": "e1"}, {})
    assert result["rows"][0]["repeat_max_abs_logit_delta"] is None
    assert result["rows"][0]["invalid_numeric_fields"] == ["repeat_max_abs_logit_delta"]
    json.dumps(result, allow_nan=False)


def test_debrief_reference_audit_does_not_certify_free_text():
    memo = "## What would have made this easier?\n- shown_but_ambiguous | c1.jsonl:L3 | metric: reason -> field\n- missing_signal: vague\n## Else\nSee c1.jsonl:L999"
    audit = audit_debrief_references(memo, {"coverage_manifest": {"included_event_ids": ["c1.jsonl:L3"]}})
    assert audit["unexposed_event_ids"] == ["c1.jsonl:L999"]
    assert audit["uncited_wish_count"] == 1 and audit["uncategorized_wish_count"] == 1
    assert not audit["production_apply_allowed"]


def test_inspection_remains_available_after_replay_budget_and_has_own_cap():
    worker = object.__new__(HookedTransformerWorkerRuntime)
    worker.max_diagnostic_calls_per_run = 1
    worker.diagnostic_result_window = 12
    worker._diagnostic_results = [{"diagnostic": "prior"}]
    worker._evidence_inspection_count = 0
    worker._pending_diagnostic_events = []
    worker._last_packet = {}
    worker._execute_controller_diagnostic_request = lambda request, **kw: {"diagnostic": request["diagnostic"], "physical_replay_count": 0}
    for i in range(4):
        result = worker.request_controller_diagnostics({"diagnostic": "inspect_evidence"})
        assert result[0]["budget_before"]["diagnostic_calls_left"] == 0
        assert result[0]["budget_after"]["inspection_calls_left"] == 3-i
    assert worker.request_controller_diagnostics({"diagnostic": "inspect_evidence"}) == []
    assert len(worker._diagnostic_results) == 1


def test_request_identity_and_compact_views_keep_inspection_parameters():
    request = {"diagnostic": "inspect_evidence", "execution_id": "e1", "limit": 3}
    assert _normalize_controller_diagnostic_request(request) == request
    assert _diagnostic_request_signature(request) != _diagnostic_request_signature({**request, "execution_id": "e2"})
    compact = _compact_controller_payload({"strategy_hints": {"evidence_inspection_catalog": [{"execution_id": "e1"}], "evidence_inspection_calls_left": 4}})
    assert compact["strategy_hints"]["evidence_inspection_calls_left"] == 4
    result = _compact_diagnostic_result({**request, "rows": [{"target_piece_logit_delta": 0.0}], "new_measurement_count": 0})
    assert result["rows"][0]["target_piece_logit_delta"] == 0.0
    requests = _extract_diagnostic_requests(
        {"meta": {"diagnostic_request": request, "objective_bundle_key": "different_objective"}},
        {"strategy_hints": {"diagnostic_frontier_bundle_key": "different_objective", "diagnostic_budget_exhausted": True}})
    assert requests == [request]


def test_distinct_doses_do_not_collapse_in_diagnostic_history():
    row = {"operator_recipe_id": "r1", "target_piece_binding_id": "binding:1", "activation_patch_step_size": 0.04}
    identity = HookedTransformerWorkerRuntime._diagnostic_evidence_row_identity
    assert identity(row) != identity({**row, "activation_patch_step_size": 0.16})


def test_blueprint_axis_is_direct_but_forced_seed_is_not_observed():
    worker, seeds, calls, objective = fixture_probe()
    seeds[0].pop("activation_patch_seed_source")
    seeds[0]["operator_axis"] = "activation_patch_blueprint_materialization"
    seeds[1]["activation_patch_forced_seed"] = True
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:fixed"):
        report = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a")
    assert report["seed_sources"] == ["direct_candidate"]
    assert report["row_count"] == 4


def test_explicit_reconfirmation_keeps_recorded_term_case_and_new_context():
    worker, seeds, calls, objective = fixture_probe()
    seeds[0]["intended_term"] = "A"
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:first"):
        first = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a")
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:second"):
        second = matched_response_probe(worker, first["rows"], objective_bundle_key=objective, objective_term="a")
    assert second["objective_term"] == "A"
    assert second["status"] == "dose_matched_response_complete"
    assert second["new_measurement_count"] == 2
    assert {r["execution_id"] for r in first["rows"]}.isdisjoint(r["execution_id"] for r in second["rows"])


def test_matrix_summary_preserves_two_doses_but_counts_alias_executions_once(tmp_path):
    worker, seeds, calls, objective = fixture_probe()
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:fixed"):
        report = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a")
    path = tmp_path / "matrix.jsonl"
    event = {"event": "controller_diagnostic_result", "target_piece_binding_seed_matrix_rows": report["rows"]}
    path.write_text(json.dumps(event, allow_nan=False) + "\n")
    summary = summarize_activation_patch_jsonl([path])
    matrix = summary["target_piece_binding_seed_matrices"][0]
    assert matrix["row_count"] == 8
    assert matrix["unique_execution_count"] == 2 and matrix["unique_observable_count"] == 4


def test_source_comparison_changes_construction_not_seed_label_or_dose(tmp_path):
    worker, seeds, calls, objective = fixture_probe()
    materialize = worker._activation_patch_trial_edit_from_candidate
    def variants(candidate, **kwargs):
        edit = materialize(candidate, **kwargs)
        edit["source"]["expr"]["ref"]["tensor"] = candidate["source_localization"]
        return edit
    worker._activation_patch_trial_edit_from_candidate = variants
    worker.adapter.read_ref = lambda ref, ctx: torch.ones(4) * (2 if ref["tensor"] == "source_term_token" else -1)
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:fixed"):
        result = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a", comparison_axis="source_localization")
    assert result["status"] == "dose_matched_response_complete"
    assert result["row_count"] == 8 and result["new_measurement_count"] == 4
    assert result["physical_replay_count"] == 10 and result["observable_count"] == 8
    assert len(result["source_direction_comparisons"]) == 4
    assert all(p["source_tensors_distinct"] and p["executions_distinct"] for p in result["source_direction_comparisons"])
    assert {r["seed_operator_recipe_id"] for r in result["rows"]} == {"direct"}
    assert {r["seed_source"] for r in result["rows"]} == {"direct_candidate"}
    assert len({r["operator_recipe_id"] for r in result["rows"]}) == 4
    assert {r["activation_patch_alpha"] for r in result["rows"]} == {0.04}
    assert not any(r["production_apply_allowed"] for r in result["rows"])
    log = tmp_path / "matrix.jsonl"
    log.write_text(json.dumps({"event": "controller_diagnostic_result", "target_piece_binding_seed_matrix_rows": result["rows"]}) + "\n")
    matrix = summarize_activation_patch_jsonl([log])["target_piece_binding_seed_matrices"][0]
    assert matrix["comparison_axes"] == ["source_localization"]
    assert matrix["source_variants"] == ["source_centered_pm1", "source_term_token"]
    assert matrix["unique_execution_count"] == 4 and len(matrix["cells"]) == 8


def test_source_comparison_reports_aliases_instead_of_independent_evidence():
    worker, seeds, calls, objective = fixture_probe()
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:fixed"):
        result = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a", comparison_axis="source_localization")
    assert result["new_measurement_count"] == 2
    assert all(not p["executions_distinct"] for p in result["source_direction_comparisons"])
    assert all(not any(p["paired_deltas"].values()) for p in result["source_direction_comparisons"])
    request = {"diagnostic": "matched_response_probe", "comparison_axis": "source_localization"}
    assert _normalize_controller_diagnostic_request(request) == request
    assert _diagnostic_request_signature(request) != _diagnostic_request_signature({**request, "comparison_axis": "seed_provenance"})
    assert _compact_diagnostic_result({"target_piece_binding_seed_matrix_summary": result})["target_piece_binding_seed_matrix_summary"]["comparison_axis"] == "source_localization"


def test_probe_rejects_silent_dose_clipping_and_invalid_comparison_axis():
    worker, seeds, calls, objective = fixture_probe()
    result = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a", comparison_axis="chosen_winner")
    assert result["status"] == "invalid_request" and not calls
    assert matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a", comparison_axis=[])["status"] == "invalid_request"
    materialize = worker._activation_patch_trial_edit_from_candidate
    def clipped(candidate, **kwargs):
        edit = materialize(candidate, **kwargs)
        edit["budget"]["step_size"] = 0.01
        return edit
    worker._activation_patch_trial_edit_from_candidate = clipped
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:fixed"):
        result = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a")
    assert result["status"] == "incomplete" and not calls
    assert all("materialized_alpha_or_dose_mismatch" in r["error"] for r in result["errors"])


def fixture_promotable_response():
    worker, seeds, calls, objective = fixture_probe()
    before = worker._simulate_decode()["first_logits"]
    def replay(edits, **kwargs):
        calls.append(edits[0]["budget"]["step_size"])
        after = before.clone()
        after[30] += 6.0
        kwargs["_measurement_capture"]["edited_logits"] = after
        return {"status": "ok", "actual_delta_class": "target_lift", "actuator_class": "self_actuator",
            "target_piece_token_id": 30, "target_piece_binding_id": "binding:30",
            "operator_recipe_id": "confirmed_fixture_recipe", "term_readout_deltas": {"a": {"lift_score": 0.2}, "b": {"lift_score": 0.0}},
            "self_delta": 0.2, "cross_delta": 0.0, "alignment_margin": 0.2, "realized_lift_bundle_key": objective,
            "target_mass_delta": 0.01, "target_top20_hit_delta": 1, "focus_rank_delta": 20,
            "activation_patch_hook_call_count": 1, "repeat_flag_delta": 0.0, "repetition_score_delta": 0.0,
            "entropy_delta": 0.0, "top1_margin_delta": 0.0, "required_term_recall_delta": 0.0,
            "required_term_span_progress_delta": 0.0}
    worker.replay_candidate_edits_actual_delta = replay
    worker._summarize_operator_recipe_bundle_ownership = HookedTransformerWorkerRuntime._summarize_operator_recipe_bundle_ownership.__get__(worker)
    materialize = worker._activation_patch_trial_edit_from_candidate
    def trial_edit(candidate, **kwargs):
        edit = materialize(candidate, **kwargs)
        edit["meta"] = {}
        return edit
    worker._activation_patch_trial_edit_from_candidate = trial_edit
    with patch("SpiralInterventionLab.runtime.response_probe.state_identity", return_value="context:fixed"):
        report = matched_response_probe(worker, seeds, objective_bundle_key=objective, objective_term="a", dose_grid=[0.04])
    worker._diagnostic_results = [report]
    for name in ("_activation_patch_candidate_review", "_activation_patch_runtime_support_probe", "_activation_patch_promotion_gate_review",
                 "_activation_patch_production_shadow_replay", "_activation_patch_production_trial_gate_review"):
        setattr(worker, name, getattr(HookedTransformerWorkerRuntime, name).__get__(worker))
    row = report["rows"][0]
    packet = {"strategy_hints": {"diagnostic_frontier_bundle_key": objective}, "budget": {
        "production_trial_edits_left_this_run": 1, "production_trial_alpha_left_total": 0.15,
        "production_trial_edit_cost_left_total": 0.15}}
    request = {"diagnostic": "activation_patch_production_trial_gate_review", "evidence_id": row["observable_id"], "objective_bundle_key": objective}
    return worker, row, request, packet, calls


@pytest.fixture(autouse=True)
def fixed_trial_context(monkeypatch):
    monkeypatch.setattr("SpiralInterventionLab.runtime.candidate_trial.state_identity", lambda w: "context:fixed")


def test_response_review_opens_only_bounded_trial_after_physical_confirmation():
    worker, row, request, packet, calls = fixture_promotable_response()
    assert response_review_readiness(row, "context:fixed")["review_eligible"]
    with patch("SpiralInterventionLab.runtime.response_promotion.state_identity", return_value="context:fixed"):
        result = review_response_evidence(worker, request, packet)
    assert result["production_trial_allowed"] is True
    assert result["physical_replay_count"] == 4
    assert result["confirmation"]["review_eligible"]
    assert not result["production_apply_allowed"] and not result["certified_for_apply"]
    assert result["production_trial_contract"]["ttl_steps"] == 1
    assert result["production_trial_candidate"]["apply_kind"] == "production_trial"
    compact = _compact_diagnostic_result(result)
    assert compact["production_trial_allowed"] is True
    assert compact["production_trial_candidate"]["trial_edit"] == result["production_trial_candidate"]["trial_edit"]
    assert compact["production_trial_contract"] == result["production_trial_contract"]
    assert compact["production_apply_allowed"] is False


@pytest.mark.parametrize("change,reason", [
    ({"measurement_context_id": "old"}, "same_context"),
    ({"target_piece_binding_variant": "alternate"}, "canonical_binding"),
    ({"target_piece_prob_delta": 0.000001, "bound_token_top20_hit_delta": 0}, "bound_target_lift"),
    ({"repeat_flag_delta": 1.0}, "nonregressing"),
    ({"repeat_max_abs_logit_delta": float("nan")}, "finite_controls_and_effects"),
    ({"alignment_margin": -1.0}, "owned_target_lift"),
    ({"state_restored": False}, "complete_restored_measurement"),
])
def test_response_review_blocks_unqualified_evidence_before_replay(change, reason):
    worker, row, request, packet, calls = fixture_promotable_response()
    row.update(change)
    before = len(calls)
    with patch("SpiralInterventionLab.runtime.response_promotion.state_identity", return_value="context:fixed"):
        result = review_response_evidence(worker, request, packet)
    assert reason in result["blocked_reasons"]
    assert not result["production_trial_allowed"] and len(calls) == before


def test_response_review_does_not_transfer_diagnostic_cap_to_trial_or_ignore_budget():
    worker, row, request, packet, calls = fixture_promotable_response()
    materialize = worker._activation_patch_trial_edit_from_candidate
    def clipped(candidate, **kwargs):
        edit = materialize(candidate, **kwargs)
        edit["budget"]["step_size"] = 0.03
        return edit
    worker._activation_patch_trial_edit_from_candidate = clipped
    with patch("SpiralInterventionLab.runtime.response_promotion.state_identity", return_value="context:fixed"):
        result = review_response_evidence(worker, request, packet)
    assert result["blocked_reasons"] == ["normal_trial_edit_differs_from_diagnostic"]
    worker._activation_patch_trial_edit_from_candidate = materialize
    packet["budget"]["production_trial_edits_left_this_run"] = 0
    with patch("SpiralInterventionLab.runtime.response_promotion.state_identity", return_value="context:fixed"):
        result = review_response_evidence(worker, request, packet)
    assert not result["production_trial_allowed"]
    assert "trial_budget_exhausted" in result["production_trial_blocked_reasons"]


def test_response_review_requires_physical_confirmation_not_cached_positive():
    worker, row, request, packet, calls = fixture_promotable_response()
    prior = worker.replay_candidate_edits_actual_delta
    def failed_confirmation(*args, **kwargs):
        result = prior(*args, **kwargs)
        return {**result, "actuator_class": "collapse_sharpener", "repeat_flag_delta": 1.0}
    worker.replay_candidate_edits_actual_delta = failed_confirmation
    with patch("SpiralInterventionLab.runtime.response_promotion.state_identity", return_value="context:fixed"):
        result = review_response_evidence(worker, request, packet)
    assert "physical_confirmation_failed" in result["blocked_reasons"]
    assert not result["production_trial_allowed"]


def test_response_review_rejects_confirmation_on_a_different_binding():
    worker, row, request, packet, calls = fixture_promotable_response()
    prior = worker.replay_candidate_edits_actual_delta
    def different_binding(*args, **kwargs):
        result = prior(*args, **kwargs)
        return {**result, "target_piece_token_id": 31, "target_piece_binding_id": "binding:31"}
    worker.replay_candidate_edits_actual_delta = different_binding
    with patch("SpiralInterventionLab.runtime.response_promotion.state_identity", return_value="context:fixed"):
        result = review_response_evidence(worker, request, packet)
    assert "canonical_binding" in result["confirmation"]["blocked_reasons"]
    assert not result["production_trial_allowed"]


def test_worker_evidence_review_uses_ledger_not_request_supplied_positive_scores():
    worker, row, request, packet, calls = fixture_promotable_response()
    worker._steps = 0
    row["actual_delta_class"] = "rank_carrier"
    row["target_piece_prob_delta"] = 0.000001
    row["bound_token_top20_hit_delta"] = 0
    request["activation_patch_shadow_actuator"] = {"promotable_to_candidate_compiler": True, "activation_patch_actuator_class": "self_actuator"}
    before = len(calls)
    with patch("SpiralInterventionLab.runtime.response_promotion.state_identity", return_value="context:fixed"):
        result = HookedTransformerWorkerRuntime._execute_controller_diagnostic_request(worker, request, source="controller", packet=packet)
    assert result["status"] == "blocked" and not result["production_trial_allowed"]
    assert "bound_target_lift" in result["blocked_reasons"]
    assert len(calls) == before


def test_debrief_keeps_early_matrix_and_negative_anchor_with_coverage(tmp_path: Path):
    events = [{"event": "controller_command", "step": i, "command": {"decision": "noop", "edits": []}}
              for i in range(50)]
    matrix = {"event": "controller_diagnostic_result", "step": 4, "target_piece_binding_seed_matrix_executed": True,
              "target_piece_binding_seed_matrix_rows": [{"operator_recipe_id": "r1", "actual_delta_class": "rank_carrier", "target_mass_delta": 0}]}
    events.insert(5, matrix)
    events.insert(7, {"event": "controller_diagnostic_result", "step": 5, "actual_delta_class": "collapse_sharpener"})
    events.append({"event": "controller_diagnostic_result", "step": 50, "status": "already_replayed"})
    events.append({"event": "episode_end", "steps": 51, "output": "the the", "task_done": False})
    path = tmp_path / "c1.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    digest = event_anchored_log_digest(tmp_path)
    kinds = {a["anchor_kind"]: a for a in digest["event_anchors"]}
    assert kinds["first_actual_matrix"]["step"] == 4
    assert "first_negative_outcome" in kinds and "first_cached_review" in kinds
    assert kinds["first_actual_matrix"]["candidate_rows"][0]["target_mass_delta"] == 0
    assert digest["coverage_manifest"]["omitted_event_count"] > 0
    assert kinds["first_actual_matrix"]["event_id"] in digest["coverage_manifest"]["included_event_ids"]
    assert len(json.dumps(digest)) <= 20000
    assert digest == event_anchored_log_digest(tmp_path)


def test_debrief_shares_anchor_identity_and_preserves_negative_row(tmp_path):
    rows = [{"operator_recipe_id": f"r{i}", "actual_delta_class": "neutral"} for i in range(10)]
    rows.append({"operator_recipe_id": "negative", "actual_delta_class": "collapse_sharpener"})
    event = {"event": "controller_diagnostic_result", "step": 4,
             "target_piece_binding_seed_matrix_executed": True, "rows": rows,
             "target_piece_binding_seed_matrix_summary": {"no_edit_max_abs_logit_delta": float("nan")}}
    (tmp_path / "c1.jsonl").write_text(json.dumps(event) + "\n")
    digest = event_anchored_log_digest(tmp_path)
    assert len(digest["event_anchors"]) == 1
    anchor = digest["event_anchors"][0]
    assert {"first_actual_matrix", "first_negative_outcome"}.issubset(anchor["anchor_kinds"])
    assert anchor["candidate_rows"][0]["actual_delta_class"] == "collapse_sharpener"
    assert anchor["matrix"]["invalid_numeric_fields"] == ["no_edit_max_abs_logit_delta"]
    json.dumps(digest, allow_nan=False)
