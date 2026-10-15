from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.runtime import diagnostic_budget, entity_probe_refresh, operation_cards
from SpiralInterventionLab.runtime.loop import _diagnostic_request_signature, _extract_diagnostic_requests
from SpiralInterventionLab.tests.test_entity_candidate_review import NAME, hints, worker_with_probe


def exhausted_worker():
    worker = worker_with_probe()
    worker.max_generated_tokens = 64
    worker._steps = 5
    worker._last_task_feedback = {"missing_required_terms": ["Ivo", "Selim", "send", "report", "Yara", "before dusk"]}
    return worker


def refresh_request(worker):
    return worker._entity_probe_refresh_report()["request"]


def test_late_missing_terms_are_offered_without_replacing_existing_candidate_actions():
    worker = exhausted_worker()
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("preflight cannot measure")):
        h = hints(worker)
    refresh = h["entity_probe_refresh"]
    assert refresh["status"] == "refresh_available"
    assert refresh["probe_terms"] == ["Selim", "send", "report", "Yara"]
    h["candidate_diagnostic_choices"] = {"cards": [{"candidate_id": "existing_sample",
        "objective_bundle_key": "entity_insert:sample", "target_piece": " sample",
        "actions": [{"action": "review_existing", "action_id": "review_sample", "available": True}]}]}
    packet = {"strategy_hints": h}
    cards, _ = operation_cards.build_menu(packet)
    assert any(c["candidate_id"] == "existing_sample" and c["available"] for c in cards["cards"])
    card = next(row for row in cards["cards"] if row["diagnostic"] == entity_probe_refresh.NAME)
    assert card["available"] and not card["production_apply_allowed"]
    assert card["probe_terms"] == refresh["probe_terms"]
    command = {"decision": "noop", "meta": {"operation_id": card["operation_id"]}}
    assert _extract_diagnostic_requests(command, packet) == [refresh_request(worker)]
    assert _compact_controller_payload(packet)["strategy_hints"]["entity_probe_refresh"]["status"] == "refresh_available"


def test_refresh_executes_exact_terms_then_exposes_fresh_review_without_advancing():
    worker = exhausted_worker()
    request = refresh_request(worker)
    original = deepcopy(worker._diagnostic_results)
    tokens = worker._current_token_tensor().clone()
    before = diagnostic_budget.left(worker)
    with patch.object(worker.runtime_state, "run_with_cache", wraps=worker.runtime_state.run_with_cache) as forward:
        with patch.object(worker, "_token_rank_for_logits", return_value=90), patch.object(
            worker, "_prompt_term_spans", return_value=[{"start": 0, "end": 1, "provenance_class": "source_body"}]
        ):
            result = worker.request_controller_diagnostics(request)[0]
    assert forward.call_count == result["model_forward_count"] == result["new_measurement_count"] == 1
    assert result["physical_replay_count"] == 0  # readout observation, not an edited replay
    assert result["requested_terms"] == request["terms"]
    assert {r["term"] for r in result["term_readout_rows"]} == set(request["terms"])
    assert result["probe_refresh_id"] == request["probe_refresh_id"]
    assert result["readout_cache_restored"] and worker._steps == 5
    assert torch.equal(worker._current_token_tensor(), tokens)
    assert diagnostic_budget.left(worker) == before - 1
    assert worker._diagnostic_results[:len(original)] == original
    assert any(r["diagnostic"] == NAME for r in hints(worker)["available_next_diagnostics"])
    review = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert review["source_probe_recorded_step"] == 5
    assert set(review["candidate_terms"]) == set(request["terms"])
    assert not review["production_apply_allowed"] and not review["certified_for_apply"]
    compact = _compact_diagnostic_result(result)
    assert compact["requested_terms"] == request["terms"] and compact["model_forward_count"] == 1


@pytest.mark.parametrize("change", ["satisfied", "terms", "id"])
def test_stale_or_changed_request_is_free_and_does_not_probe(change):
    worker = exhausted_worker()
    request = refresh_request(worker)
    if change == "satisfied":
        worker._last_task_feedback = {"missing_required_terms": []}
    elif change == "terms":
        request["terms"] = ["sample"]
    else:
        request["probe_refresh_id"] = "stale"
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("stale cannot measure")):
        result = worker.request_controller_diagnostics(request)[0]
    assert result["status"] == "stale_refresh_request"
    assert result["diagnostic_call_cost"] == result["model_forward_count"] == result["new_measurement_count"] == 0
    assert result["budget_before"] == result["budget_after"]


def test_duplicate_refresh_in_batch_is_free_and_coverage_survives_eviction_until_reset():
    worker = exhausted_worker()
    worker._last_task_feedback = {"missing_required_terms": ["Selim"]}
    request = refresh_request(worker)
    results = worker.request_controller_diagnostics([request, request])
    assert [r["diagnostic_call_cost"] for r in results] == [1, 0]
    assert results[1]["status"] == "stale_refresh_request"
    worker._diagnostic_results = []
    worker._latest_diagnostic_results = []
    worker._steps = 10
    assert worker._entity_probe_refresh_report()["status"] == "all_missing_terms_already_probed"
    assert not refresh_request(worker)
    worker.reset("p")
    assert worker._entity_probe_attempted_terms == set()


def test_forward_error_is_an_attempt_not_positive_evidence_or_an_unbounded_retry():
    worker = exhausted_worker()
    worker._last_task_feedback = {"missing_required_terms": ["Selim"]}
    request = refresh_request(worker)
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=RuntimeError("readout unavailable")):
        result = worker.request_controller_diagnostics(request)[0]
    assert result["status"] == "error" and result["new_measurement_count"] == 0
    assert result["diagnostic_call_cost"] == result["model_forward_count"] == 1
    assert worker._entity_probe_refresh_report()["status"] == "all_missing_terms_already_probed"
    assert not result["production_apply_allowed"]


def test_existing_blueprints_and_exhausted_budget_do_not_offer_refresh():
    worker = worker_with_probe()
    assert worker._entity_probe_refresh_report()["status"] == "recorded_candidates_pending"
    worker = exhausted_worker()
    worker._diagnostic_calls_used = 12
    h = hints(worker)
    assert h["entity_probe_refresh"]["status"] == "diagnostic_budget_exhausted"
    assert not any(r["diagnostic"] == entity_probe_refresh.NAME for r in h.get("available_next_diagnostics", ()))
    assert any(r["diagnostic"] == entity_probe_refresh.NAME and r["reason"] == "diagnostic_budget_exhausted"
               for r in h["blocked_next_diagnostics"])
    assert worker.request_controller_diagnostics(refresh_request(worker)) == []


@pytest.mark.parametrize("failure", ["stale", "error"])
def test_unmeasured_refresh_receipts_do_not_replace_the_recorded_probe(failure):
    worker = worker_with_probe()
    recorded = worker._diagnostic_results[0]
    unmeasured = {"diagnostic": entity_probe_refresh.NAME, "recorded_step": 3,
                  "status": "stale_refresh_request" if failure == "stale" else "error"}
    worker._latest_diagnostic_results = [unmeasured]
    worker._diagnostic_results.append(dict(unmeasured))
    assert not entity_probe_refresh.is_recorded_probe(unmeasured)
    assert entity_probe_refresh.is_recorded_probe(recorded)
    review = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert review["status"] == "candidate_blueprints_ready"
    assert review["source_probe_recorded_step"] == recorded["recorded_step"]
    assert worker._entity_probe_refresh_report()["pending_blueprint_count"] == 0


def test_reviewed_blueprints_no_longer_block_refresh_for_unprobed_terms():
    worker = worker_with_probe()
    worker._last_task_feedback = {"missing_required_terms": ["sample", "Ivo", "Selim", "Yara"]}
    assert worker._entity_probe_refresh_report()["status"] == "recorded_candidates_pending"
    review = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert review["candidate_blueprint_count"] > 0
    refresh = worker._entity_probe_refresh_report()
    assert refresh["status"] == "refresh_available"
    assert refresh["pending_blueprint_count"] == 0
    assert "Yara" in refresh["probe_terms"]


def test_signatures_distinguish_refreshes_and_casefolded_coverage_is_not_reoffered():
    worker = exhausted_worker()
    first = refresh_request(worker)
    worker._diagnostic_results[0]["term_readout_rows"][0].update(readout_status="top20_reachable", rank=6)
    worker._last_task_feedback = {"missing_required_terms": ["SAMPLE", "ivo", "before dusk"]}
    second = refresh_request(worker)
    assert second["terms"] == ["before dusk"]
    assert _diagnostic_request_signature(first) != _diagnostic_request_signature(second)
    worker._last_task_feedback = {}
    assert worker._entity_probe_refresh_report()["status"] == "missing_terms_unknown"
