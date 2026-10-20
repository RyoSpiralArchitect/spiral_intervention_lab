from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.runtime import clause_structure, diagnostic_budget, operation_cards
from SpiralInterventionLab.runtime.loop import _diagnostic_request_signature, _extract_diagnostic_requests
from SpiralInterventionLab.runtime.response_probe import state_identity
from SpiralInterventionLab.tests.test_entity_candidate_review import hints, worker_with_probe


NAME = clause_structure.NAME


def records():
    return [
        {"position": i, "token_id": token, "piece": piece, "provenance_class": provenance,
         "segment_kind": "output" if provenance == "output" else "prompt"}
        for i, (token, piece, provenance) in enumerate([
            (3, "Use and; SOURCE:", "constraint_header"),
            (1, " before", "source_body"), (2, ",", "source_body"),
            (3, " while", "source_body"), (2, ",", "source_body"),
            (4, " Selim", "source_body"), (5, ".", "source_body"),
            (6, "ANSWER:;", "answer_prefix"), (4, " and", "output"),
        ])
    ]


def make_worker():
    worker = worker_with_probe()
    worker.clause_structure_diagnostics = True
    worker._append_output_token(1)
    worker._steps = 1
    worker._token_position_records = records
    return worker


def request(worker):
    return clause_structure.offer(worker)["request"]


def test_source_candidates_are_bound_to_actual_tokens_not_names_headers_or_winning_order():
    worker = make_worker()
    assert clause_structure.source_candidates(worker) == [
        {"token_id": 1, "piece": " before", "source_positions": [1], "source_provenance": "source_body"},
        {"token_id": 2, "piece": ",", "source_positions": [2, 4], "source_provenance": "source_body"},
        {"token_id": 3, "piece": " while", "source_positions": [3], "source_provenance": "source_body"},
        {"token_id": 5, "piece": ".", "source_positions": [6], "source_provenance": "source_body"},
    ]
    split = records()
    split[3]["piece"] = " but"
    split[4]["piece"] = "ter"
    worker._token_position_records = lambda: split
    assert 3 not in [r["token_id"] for r in clause_structure.source_candidates(worker)]


def test_opt_in_offer_preserves_existing_frontier_and_measurement_free_card_construction():
    worker = make_worker()
    worker.clause_structure_diagnostics = False
    original = hints(worker)
    assert "source_clause_structure" not in original
    worker.clause_structure_diagnostics = True
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("offer cannot measure")):
        current = hints(worker)
    for key in original:
        if key not in {"available_next_diagnostics"}:
            assert current[key] == original[key]
    assert current["available_next_diagnostics"][:-1] == original["available_next_diagnostics"]
    menu, _ = operation_cards.build_menu({"strategy_hints": current})
    card = menu["cards"][-1]
    assert card["diagnostic"] == NAME and card["available"]
    assert card["cost"]["physical_replays_upper_bound"] == 1
    command = {"decision": "noop", "meta": {"operation_id": card["operation_id"]}}
    assert _extract_diagnostic_requests(command, {"strategy_hints": current}) == [request(worker)]
    assert not card["production_apply_allowed"]
    compact = _compact_controller_payload({"strategy_hints": current})
    assert compact["strategy_hints"]["source_clause_structure"] == current["source_clause_structure"]


def test_dispatch_measures_once_restores_state_and_never_mints_operator_evidence():
    worker = make_worker()
    before = state_identity(worker)
    before_caps = deepcopy(worker._surface_catalog_raw)
    original_request = request(worker)
    slots = diagnostic_budget.left(worker)
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        result = worker.request_controller_diagnostics(original_request)[0]
    assert result["status"] == "ok"
    assert simulate.call_count == result["physical_replay_count"] == result["new_measurement_count"] == 1
    assert simulate.call_args.kwargs == {
        "max_new_tokens": 1, "top_k": 6, "score_candidate_text": False, "score_observer_check": False}
    assert result["edited_replay_count"] == 0
    assert result["state_restored"] and state_identity(worker) == before and worker._steps == 1
    assert worker._surface_catalog_raw == before_caps
    assert diagnostic_budget.left(worker) == slots - 1
    assert not result["certified_for_apply"] and not result["production_apply_allowed"]
    assert not worker._collect_active_edits()
    assert not worker._frozen_diagnostic_candidates
    compact = _compact_diagnostic_result(result)
    assert compact["structure_readout_rows"] == result["structure_readout_rows"]
    assert compact["structure_context_id"] == original_request["structure_context_id"]
    assert not compact["production_apply_allowed"]


def test_duplicate_and_evicted_receipts_never_repeat_work_but_a_new_prefix_can():
    worker = make_worker()
    first_request = request(worker)
    results = worker.request_controller_diagnostics([first_request, first_request])
    assert [r["diagnostic_call_cost"] for r in results] == [1, 0]
    assert results[1]["blocked_reason"] == "already_attempted_at_context"
    worker._diagnostic_results = []
    worker._latest_diagnostic_results = []
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("repeat cannot measure")):
        repeat = worker.request_controller_diagnostics(first_request)[0]
        current = hints(worker)
    assert repeat["diagnostic_call_cost"] == repeat["physical_replay_count"] == 0
    assert not any(r["diagnostic"] == NAME for r in current.get("available_next_diagnostics", []))
    worker._append_output_token(2)
    worker._steps += 1
    new_request = request(worker)
    assert _diagnostic_request_signature(first_request) != _diagnostic_request_signature(new_request)
    stale = worker.request_controller_diagnostics(first_request)[0]
    assert stale["status"] == "stale_structure_request" and stale["diagnostic_call_cost"] == 0
    assert worker.request_controller_diagnostics(new_request)[0]["status"] == "ok"
    worker.reset("p")
    assert not worker._clause_structure_attempted_contexts


@pytest.mark.parametrize("condition,status", [
    ("off", "disabled"), ("empty", "awaiting_nonempty_prefix"),
    ("no_source", "no_source_structure_tokens"), ("budget", "diagnostic_budget_exhausted"),
    ("terminal", "generation_finished"), ("active", "active_edits_present"),
])
def test_unavailable_probe_never_creates_an_executable_offer(condition, status):
    worker = make_worker()
    if condition == "off":
        worker.clause_structure_diagnostics = False
    elif condition == "empty":
        worker.reset("p")
    elif condition == "no_source":
        worker._token_position_records = lambda: []
    elif condition == "budget":
        worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run
    elif condition == "active":
        worker._collect_active_edits = lambda: [{"edit_id": "existing"}]
    else:
        for _ in range(worker.max_generated_tokens):
            worker._append_output_token(1)
    assert clause_structure.offer(worker)["status"] == status
    assert not any(r["diagnostic"] == NAME for r in hints(worker).get("available_next_diagnostics", []))
    if condition == "active":
        with patch.object(worker, "_simulate_decode", side_effect=AssertionError("must be unedited")):
            result = worker.request_controller_diagnostics(request(worker))[0]
        assert result["blocked_reason"] == "active_edits_present" and result["diagnostic_call_cost"] == 0


def test_failed_measurement_is_charged_once_as_attempt_not_positive_evidence():
    worker = make_worker()
    req = request(worker)
    with patch.object(worker, "_simulate_decode", return_value=None) as simulate:
        result, duplicate = worker.request_controller_diagnostics([req, req])
    assert simulate.call_count == result["physical_replay_count"] == result["diagnostic_call_cost"] == 1
    assert result["status"] == "error" and result["new_measurement_count"] == 0
    assert result["state_restored"] and duplicate["diagnostic_call_cost"] == 0
    assert "structure_readout_rows" not in result


@pytest.mark.parametrize("change", ["mask", "source", "patch", "guardrails", "decoder"])
def test_context_binding_changes_with_source_or_readout_state_without_advancing_tokens(change):
    worker = make_worker()
    old = request(worker)
    if change == "mask":
        worker.allowed_token_ids = (1, 2)
    elif change == "source":
        new_records = records()
        new_records[3]["token_id"] = 4
        worker._token_position_records = lambda: new_records
    elif change == "patch":
        worker._collect_active_edits = lambda: [{"edit_id": "new_active_patch"}]
    elif change == "guardrails":
        worker._decoder_guardrail_constraints = lambda: {"max_words": 9, "forbidden_terms": ("while",)}
    else:
        worker.decoder_control_mode = "loop_aware_constraint"
        worker._no_progress_steps = 2
    assert request(worker)["structure_context_id"] != old["structure_context_id"]
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("stale cannot run")):
        result = worker.request_controller_diagnostics(old)[0]
    assert result["status"] == "stale_structure_request" and result["diagnostic_call_cost"] == 0


def test_drift_fails_closed_instead_of_returning_measurement_as_restored():
    worker = make_worker()
    def mutate(**kwargs):
        worker._append_output_token(1)
        return {"first_logits": torch.zeros(7)}
    with patch.object(worker, "_simulate_decode", side_effect=mutate):
        with pytest.raises(RuntimeError, match="changed_worker_state"):
            worker.request_controller_diagnostics(request(worker))


def test_ties_masks_and_invalid_logits_do_not_claim_greedy_win_or_emit_infinity():
    candidates = [{"token_id": i} for i in (0, 1, 2)]
    rows, winner = clause_structure.readout_rows(torch.tensor([3., 3., -float("inf")]), candidates)
    assert winner == 0 and rows[0]["is_greedy_token"] and not rows[1]["is_greedy_token"]
    assert rows[1]["rank"] == 1
    assert rows[2]["status"] == "masked_by_decode_constraints"
    assert rows[2]["rank"] is rows[2]["gap_to_top1"] is None
    assert rows[2]["prob"] == 0 and not rows[2]["top20_hit"]
    for logits in ([float("nan"), 0, 1], [float("inf"), 0, 1], [-float("inf")] * 3):
        with pytest.raises(ValueError, match="invalid_structure_readout"):
            clause_structure.readout_rows(torch.tensor(logits), candidates)
    with pytest.raises(ValueError, match="outside_readout_vocabulary"):
        clause_structure.readout_rows(torch.tensor([1., 2.]), candidates)


def test_cli_is_opt_in_and_request_without_current_id_is_free():
    import inspect
    from SpiralInterventionLab.examples import digit_transform_e2e as e2e
    from SpiralInterventionLab.examples.digit_transform_e2e import _build_parser
    args = ["--provider", "openai", "--controller-model", "test", "--worker-model", "gpt2"]
    assert not _build_parser().parse_args(args).clause_structure_diagnostics
    assert _build_parser().parse_args([*args, "--clause-structure-diagnostics"]).clause_structure_diagnostics
    worker = make_worker()
    result = worker.request_controller_diagnostics({"diagnostic": NAME})[0]
    assert result["status"] == "stale_structure_request" and result["physical_replay_count"] == 0
    for builder in (e2e.build_hooked_transformer_worker_runtime, e2e.run_digit_transform_experiment,
                    e2e.run_digit_transform_c1_only_experiment, e2e.run_digit_transform_sweep):
        assert inspect.signature(builder).parameters["clause_structure_diagnostics"].default is False


def test_legacy_request_is_exclusive_and_not_retargeted_to_stale_frontier():
    worker = make_worker()
    req = request(worker)
    packet = {"strategy_hints": {
        "diagnostic_frontier_canonical_request": {"diagnostic": "compare_extra_operator_diagnostics",
            "operator_recipe_expansion_mode": "readout_steering_deepening"},
        "diagnostic_frontier_bundle_key": "entity_insert:sample",
    }}
    command = {"decision": "noop", "meta": {
        "diagnostic_request": req, "next_action": "request_operator_diagnostic",
        "operator_recipe_expansion_mode": "readout_steering_deepening",
        "controller_memory": {"diagnostic_request": "operator_diagnostic_replay"},
    }}
    assert _extract_diagnostic_requests(command, packet) == [req]


def test_local_harness_dispatch_has_no_controller_selection_or_forced_token_claim():
    from SpiralInterventionLab.examples.replay_source_clause_diagnostic import measure_current_prefix
    worker = make_worker()
    result = measure_current_prefix(worker)
    assert result["result"]["status"] == "ok"
    assert result["selection_origin"] == "diagnostic_harness_not_controller_choice"
    assert result["result"]["source"] == "diagnostic_harness"
    assert result["context_unchanged"] and result["rollout_edit_count"] == 0
    assert not result["production_apply_allowed"]


def test_source_receipt_reader_excludes_future_outcomes_and_rejects_conflicting_prefixes(tmp_path):
    import json
    from SpiralInterventionLab.examples.replay_source_clause_diagnostic import source_prefixes
    path = tmp_path / "source.jsonl"
    rows = [
        {"event": "episode_start", "prompt": "SOURCE:abc ANSWER:"},
        {"event": "controller_observation", "step": 8, "generated_tail": "abc",
         "hypothetical_rescue": "must not reach diagnostic"},
        {"event": "episode_end", "output": "future answer", "score": 1.0},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert source_prefixes(path) == ("SOURCE:abc ANSWER:", {8: "abc"})
    path.write_text("\n".join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(ValueError, match="incomplete"):
        source_prefixes(path)
    rows.insert(2, {"event": "controller_observation", "step": 8, "generated_tail": "changed"})
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="differs"):
        source_prefixes(path)
