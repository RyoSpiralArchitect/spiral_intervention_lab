from copy import deepcopy
from unittest.mock import patch

import pytest

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.runtime import diagnostic_budget, entity_candidate_review, operation_cards
from SpiralInterventionLab.tests import test_controller_runtime as fixtures


NAME = entity_candidate_review.NAME


def probe(step=0, *, term="sample", rank=89, status="near_reachable"):
    return {
        "diagnostic": "target_entity_insertion_probe", "status": "ok", "recorded_step": step,
        "next_evidence_needed": NAME, "focus_terms": [term, "Ivo"],
        "term_readout_rows": [
            {"term": term, "first_piece": " " + term, "first_token_id": 1, "rank": rank,
             "source_provenance": "source_body", "readout_status": status, "span_progress": 0.0,
             "source_span": {"start": 0, "end": 1, "provenance_class": "source_body"}},
            {"term": "Ivo", "first_piece": " I", "rank": 6, "source_provenance": "source_body",
             "readout_status": "top20_reachable", "span_progress": 0.0},
        ],
    }


def worker_with_probe():
    worker = fixtures.TestWorkerRuntimeAndBaselines()._make_worker_runtime()
    worker.reset("p")
    worker.max_diagnostic_calls_per_run = 12
    worker.diagnostic_result_window = 12
    worker._last_task_feedback = {"missing_required_terms": ["sample", "Ivo", "Selim"]}
    worker._diagnostic_results = [probe()]
    return worker


def hints(worker):
    return worker._strategy_hints(control_phase_hint="entity_insertion")


def offers(hints_value):
    return [row for row in hints_value.get("available_next_diagnostics", ()) if row["diagnostic"] == NAME]


@pytest.mark.parametrize("value,expected", [(0, 0), (3, 3), (None, -1), (False, -1), (-1, -1), ("0", -1)])
def test_recorded_step_does_not_treat_zero_as_missing(value, expected):
    assert entity_candidate_review.recorded_step({"recorded_step": value}) == expected


def test_step_zero_review_disappears_from_frontier_and_operation_cards_without_replay():
    worker = worker_with_probe()
    initial_hints = hints(worker)
    assert offers(initial_hints)
    initial_cards = operation_cards.build_menu({"strategy_hints": initial_hints})[0]["cards"]
    card = next(card for card in initial_cards if card["diagnostic"] == NAME)
    assert card["cost"]["physical_replays_upper_bound"] == 0
    assert not card["production_apply_allowed"]
    before = diagnostic_budget.left(worker)
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("review cannot measure")):
        result = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
        assert result["recorded_step"] == result["source_probe_recorded_step"] == 0
        assert result["new_candidate_blueprint_count"] == 1
        assert result["diagnostic_call_cost"] == 1
        assert diagnostic_budget.left(worker) == before - 1
        after = hints(worker)
    assert not offers(after)
    assert after.get("diagnostic_frontier_request") != NAME
    assert after["entity_candidate_review_availability"]["review_reused"]
    assert any(row["diagnostic"] == NAME and row["reason"] == "unchanged_review_inputs"
               for row in after["blocked_next_diagnostics"])
    assert not any(card["diagnostic"] == NAME for card in operation_cards.build_menu(
        {"strategy_hints": after})[0]["cards"])
    assert not result["certified_for_apply"] and not result["production_apply_allowed"]


def test_legacy_step_zero_review_is_recognized_as_completed():
    worker = worker_with_probe()
    worker._diagnostic_results.append({"diagnostic": NAME, "recorded_step": 0,
        "candidate_blueprint_count": 1, "candidate_blueprints": []})
    assert not offers(hints(worker))


def test_same_evidence_at_later_prefix_is_not_a_new_measurement_or_charge():
    worker = worker_with_probe()
    result = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    for step in (1, 4):
        worker._steps = step
        again = worker.request_controller_diagnostics({"diagnostic": NAME, "reason": "review here again"})[0]
        assert again["review_reused"]
        assert again["entity_candidate_review_input_id"] == result["entity_candidate_review_input_id"]
        assert again["budget_before"] == again["budget_after"]
        assert again["new_measurement_count"] == again["physical_replay_count"] == 0
        assert not again["source_probe_is_current_prefix"]
        assert not offers(hints(worker))
    assert diagnostic_budget.used(worker) == 2  # original probe and first blueprint review
    assert len(worker._diagnostic_results) == 2


def test_empty_reviews_at_steps_5_6_7_preserve_budget_and_invalidate_old_blueprints():
    worker = worker_with_probe()
    worker.request_controller_diagnostics({"diagnostic": NAME})
    worker._diagnostic_calls_used = 9
    worker._last_task_feedback = {"missing_required_terms": ["Ivo", "Selim", "send", "report", "Yara"]}
    before = diagnostic_budget.left(worker)
    for step in (5, 6, 7):
        worker._steps = step
        with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("review cannot measure")):
            menu = hints(worker)
            result = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
        assert not offers(menu)
        assert result["status"] == "no_candidate_blueprints"
        assert result["diagnostic_call_cost"] == 0 and not result["diagnostic_budget_charged"]
        assert result["physical_replay_count"] == result["new_measurement_count"] == 0
        assert result["budget_before"] == result["budget_after"]
        assert result["source_probe_recorded_step"] == 0
        assert result["review_reused"] is (step > 5)
        assert diagnostic_budget.left(worker) == before == 3
    worker._latest_diagnostic_results = []
    assert worker._latest_entity_insertion_candidate_blueprints({}) == []
    assert len(worker._diagnostic_results) == 3  # keep the empty receipt, not repeated copies


def test_no_probe_or_all_terms_satisfied_is_free_and_cannot_reopen_satisfied_terms():
    worker = worker_with_probe()
    worker._diagnostic_results = []
    empty = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert empty["status"] == "no_target_entity_probe" and empty["diagnostic_call_cost"] == 0
    worker._diagnostic_results.append(probe())
    worker._last_task_feedback = {"missing_required_terms": []}
    satisfied = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert satisfied["status"] == "no_candidate_blueprints"
    assert all(row["review_decision"] == "already_satisfied" for row in satisfied["review_matrix"])
    assert diagnostic_budget.used(worker) == 1


def test_new_probe_and_relevant_feedback_change_reopen_without_permanent_veto():
    worker = worker_with_probe()
    first = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    original = deepcopy(worker._diagnostic_results[0])
    # A new observation can arrive at the same generated-token count.
    fresh = probe(rank=75)
    worker._diagnostic_results.append(fresh)
    worker._latest_diagnostic_results = [fresh]
    assert offers(hints(worker))
    second = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert second["entity_candidate_review_input_id"] != first["entity_candidate_review_input_id"]
    assert not second["review_reused"]
    assert second["candidate_blueprint_ids"] == first["candidate_blueprint_ids"]
    assert second["new_candidate_blueprint_count"] == second["diagnostic_call_cost"] == 0
    worker._last_task_feedback = {"missing_required_terms": ["Ivo"]}
    assert not offers(hints(worker))
    empty = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    worker._last_task_feedback = {"missing_required_terms": ["sample", "Ivo"]}
    assert offers(hints(worker))
    reopened = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert reopened["candidate_blueprint_count"] == 1
    assert not reopened["review_reused"]  # relevant feedback changed since the empty review
    assert reopened["diagnostic_call_cost"] == 0  # rediscovered, not a new blueprint
    assert empty["entity_candidate_review_input_id"] != reopened["entity_candidate_review_input_id"]
    assert original == worker._diagnostic_results[0]


def test_feedback_change_does_not_recharge_identical_blueprint_or_reoffer_review():
    worker = worker_with_probe()
    first = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    worker._steps = 4
    worker._last_task_feedback = {"missing_required_terms": ["sample", "Selim"]}
    before = diagnostic_budget.left(worker)
    preview = hints(worker)
    assert not offers(preview)
    assert any(row["reason"] == "unchanged_candidate_blueprints"
               for row in preview["blocked_next_diagnostics"] if row["diagnostic"] == NAME)
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("review cannot measure")):
        result = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert result["entity_candidate_review_input_id"] != first["entity_candidate_review_input_id"]
    assert not result["review_reused"] and result["review_outputs_unchanged"]
    assert result["candidate_blueprints"] == first["candidate_blueprints"]
    assert result["reused_candidate_blueprint_count"] == 1
    assert result["new_candidate_blueprint_count"] == result["diagnostic_call_cost"] == 0
    assert result["diagnostic_budget_charge_reason"] == "no_new_candidate_blueprints"
    assert result["new_measurement_count"] == result["physical_replay_count"] == 0
    assert diagnostic_budget.left(worker) == before
    assert len(worker._diagnostic_results) == 2  # not a fresh candidate generation for replay routing
    assert not result["production_apply_allowed"]
    compact = _compact_diagnostic_result(result)
    assert compact["candidate_blueprint_ids"] == result["candidate_blueprint_ids"]
    assert compact["review_outputs_unchanged"]


def test_blueprint_identity_preserves_binding_span_and_recipe_but_not_rank():
    worker = worker_with_probe()
    first = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    for field, value in (("first_piece", " Sample"), ("first_token_id", 2),
                         ("source_span", {"start": 2, "end": 3, "provenance_class": "source_body"}),
                         ("recommended_operator_families", ["different_family"])):
        row = deepcopy(first)
        row["candidate_blueprints"][0][field] = value
        entity_candidate_review.account_review(row, [first])
        assert row["new_candidate_blueprint_count"] == row["diagnostic_call_cost"] == 1
    row = deepcopy(first)
    row["candidate_blueprints"][0]["rank"] = 75
    entity_candidate_review.account_review(row, [first])
    assert row["new_candidate_blueprint_count"] == row["diagnostic_call_cost"] == 0


def test_subset_and_returned_blueprints_survive_receipt_eviction_but_not_episode_reset():
    worker = worker_with_probe()
    first = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    source_probe = probe()
    worker._diagnostic_results = [source_probe]
    worker._latest_diagnostic_results = []
    again = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert again["diagnostic_call_cost"] == 0
    assert again["reused_candidate_blueprint_count"] == 1
    pair = deepcopy(first)
    second = deepcopy(pair["candidate_blueprints"][0])
    second["objective_term"] = "Selim"
    pair["candidate_blueprints"].extend([second, deepcopy(second)])
    entity_candidate_review.account_review(pair, [first])
    assert pair["new_candidate_blueprint_count"] == pair["reused_candidate_blueprint_count"] == 1
    assert pair["diagnostic_call_cost"] == 1
    worker.reset("p")
    worker._diagnostic_results = [source_probe]
    worker._last_task_feedback = {"missing_required_terms": ["sample"]}
    restarted = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert restarted["new_candidate_blueprint_count"] == restarted["diagnostic_call_cost"] == 1


def test_same_batch_duplicate_review_is_only_charged_once():
    worker = worker_with_probe()
    results = worker.request_controller_diagnostics([{"diagnostic": NAME}, {"diagnostic": NAME}])
    assert [row["diagnostic_call_cost"] for row in results] == [1, 0]
    assert results[1]["review_reused"]
    assert diagnostic_budget.used(worker) == 2


def test_batch_probe_review_replay_uses_fresh_evidence_and_accounts_review_once():
    worker = worker_with_probe()
    fresh = probe(step=1, term="Selim")
    worker._steps = 1
    execute = worker._execute_controller_diagnostic_request
    seen_blueprints = []
    def dispatch(request, **kwargs):
        if request["diagnostic"] == "target_entity_insertion_probe":
            return deepcopy(fresh)
        if request["diagnostic"] == "operator_diagnostic_replay":
            seen_blueprints.extend(request["candidate_blueprints"])
            return {"diagnostic": "operator_diagnostic_replay", "status": "ok", "physical_replay_count": 1}
        return execute(request, **kwargs)
    worker._execute_controller_diagnostic_request = dispatch
    results = worker.request_controller_diagnostics([
        {"diagnostic": "target_entity_insertion_probe"}, {"diagnostic": NAME},
        {"diagnostic": "operator_diagnostic_replay"},
    ])
    assert results[1]["candidate_terms"] == ["Selim"]
    assert results[1]["source_probe_recorded_step"] == 1
    assert seen_blueprints[0]["objective_term"] == "Selim"
    assert diagnostic_budget.used(worker) == 4


def test_compact_payload_preserves_review_provenance_and_no_work_reason():
    worker = worker_with_probe()
    worker._last_task_feedback = {"missing_required_terms": ["Ivo"]}
    result = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    compact = _compact_diagnostic_result(result)
    assert compact["source_probe_recorded_step"] == 0
    assert compact["diagnostic_budget_charge_reason"] == "no_candidate_blueprints"
    assert compact["physical_replay_count"] == compact["new_measurement_count"] == 0
    assert not compact["production_apply_allowed"]
    packet = _compact_controller_payload({"strategy_hints": hints(worker)})
    assert packet["strategy_hints"]["entity_candidate_review_availability"]["review_reused"]


def test_budget_retained_free_receipts_do_not_spend_and_eviction_does_not_refund():
    worker = worker_with_probe()
    worker._diagnostic_calls_used = 5
    worker._diagnostic_results.extend([
        {"diagnostic_call_cost": 0, "diagnostic_budget_charged": False} for _ in range(10)])
    assert diagnostic_budget.used(worker) == 5
    worker._diagnostic_results = []
    assert diagnostic_budget.used(worker) == 5
    worker.reset("p")
    assert diagnostic_budget.used(worker) == 0
