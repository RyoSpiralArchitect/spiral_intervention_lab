from copy import deepcopy
import json

import pytest

from SpiralInterventionLab.controllers.base import ControllerProviderResponse
from SpiralInterventionLab.examples.source_radial_choice import (
    compare_offline, recorded_command, run_choice, validate_offline_reference, validate_selected_execution,
    validate_source_execution,
)
from SpiralInterventionLab.examples.replay_source_clause_trajectory import main
from SpiralInterventionLab.runtime import clause_radial, clause_trajectory, diagnostic_budget, operation_cards
from SpiralInterventionLab.tests.test_clause_components import STAGES
from SpiralInterventionLab.tests.test_clause_radial import multi_source_worker


class ChoiceProvider:
    provider_name = "fake"
    model_name = "test-model"

    def __init__(self, *, pair=(3, 2), component="common_orthogonal", mode="select"):
        self.pair, self.component, self.mode = pair, component, mode
        self.calls = []

    def complete(self, request):
        self.calls.append(request)
        if self.mode == "fail":
            raise RuntimeError("provider unavailable")
        meta = {}
        if self.mode == "select":
            card = next(c for c in request.payload["operation_menu"]["cards"]
                        if c["diagnostic"] == clause_radial.NAME and c["component"] == self.component
                        and tuple(s["target_token_id"] for s in c["source_pair"]) == self.pair)
            meta["operation_id"] = card["operation_id"]
        elif self.mode == "source":
            card = next(c for c in request.payload["operation_menu"]["cards"]
                        if c["diagnostic"] == "source_clause_activation_patch_probe" and c["available"])
            meta["operation_id"] = card["operation_id"]
        command = {"version": "0.1", "decision": "noop", "edits": [], "meta": meta}
        return ControllerProviderResponse(provider="fake", model="test-model",
            text="invalid" if self.mode == "invalid" else json.dumps(command), usage={})


@pytest.fixture
def history():
    worker = multi_source_worker()
    return [clause_trajectory.compact_record(r)
            for r in clause_radial.measure(worker, [3, 2], stages=STAGES)]


def choose(provider, history, **kwargs):
    worker, report, records = multi_source_worker(), {}, []
    before = diagnostic_budget.used(worker)
    run_choice(worker, provider, report, historical=history, save=lambda: None,
               record=records.append, stages=STAGES, **kwargs)
    assert report["packet_unchanged"] and report["context_unchanged"]
    assert not report["rollout_edit_count"] and not worker._frozen_diagnostic_candidates
    assert diagnostic_budget.used(worker) - before == report["diagnostic_slots_used"]
    return report, records


def test_live_choice_measures_exact_third_pair_without_historical_outcome_hints(history):
    provider = ChoiceProvider()
    report, records = choose(provider, history)
    assert report["status"] == "complete"
    assert len(provider.calls) == report["provider_calls_attempted"] == 1
    assert report["diagnostics_dispatched"] == report["diagnostic_slots_used"] == 1
    assert report["radial_recipe_count"] == 12
    assert report["selected_execution_comparison"]["status"] == "matched"
    assert report["offline_comparison"]["status"] == "matched"
    assert report["offline_comparison"]["readout_and_execution_exact_match"]
    assert [c["target_token_id"] for c in report["selected_recipe"]["source_pair"]] == [3, 2]
    assert records[0]["result"]["physical_replay_count"] == 4
    assert "historical" not in json.dumps(provider.calls[0].payload)
    assert not report["historical_outcomes_visible_to_controller"]
    assert not report["production_apply_allowed"]


def test_another_pair_is_executed_but_not_called_an_offline_match(history):
    report, records = choose(ChoiceProvider(pair=(1, 3), component="common_parallel"), history)
    assert report["selected_execution_comparison"]["status"] == "matched"
    assert report["offline_comparison"]["status"] == "not_covered"
    assert report["offline_comparison"]["reason"] == "selected_pair_not_in_offline_matrix"
    assert len(records) == 1


def test_non_radial_request_is_preserved_as_a_choice_not_mislabeled_abstention():
    provider = ChoiceProvider(mode="source")
    report, records = choose(provider, [])
    assert report["status"] == "non_radial_selection_not_dispatched"
    assert report["dispatch_block_reason"] == "outside_predeclared_radial_only_scope"
    assert report["selected_operation_card"]["diagnostic"] == "source_clause_activation_patch_probe"
    assert report["selected_operation_card"]["available"]
    assert report["selected_radial_operation_card"] is None
    assert report["resolved_requests"][0]["diagnostic"] == "source_clause_activation_patch_probe"
    assert not records and len(provider.calls) == 1
    assert report["diagnostics_dispatched"] == report["diagnostic_slots_used"] == 0


def test_source_enabled_scope_dispatches_the_chosen_single_token_not_a_radial_replacement(history):
    provider = ChoiceProvider(mode="source")
    report, records = choose(provider, history, allow_source_cards=True)
    assert report["status"] == "complete"
    assert len(provider.calls) == report["diagnostics_dispatched"] == report["diagnostic_slots_used"] == 1
    assert report["selected_source_candidate"]["target_piece"] == " before"
    assert records[0]["result"]["physical_replay_count"] == 4
    assert records[0]["result"]["edited_replay_count"] == 2
    assert report["selected_execution_comparison"]["status"] == "matched"
    assert report["offline_comparison"]["status"] == "not_covered"
    assert report["offline_comparison"]["reason"] == "exact_source_candidate_not_in_offline_matrix"


def test_exact_recorded_choice_replays_without_provider_or_rebinding():
    prior, _ = choose(ChoiceProvider(mode="source"), [])
    original = deepcopy(prior)
    report, records = choose(None, [], allow_source_cards=True, recorded_choice=prior)
    assert prior == original
    assert report["status"] == "complete" and report["recorded_packet_match"]
    assert report["provider_call_limit"] == report["provider_calls_attempted"] == 0
    assert report["controller_trace"] is None
    assert report["selection_origin"] == "recorded_controller_choice_replay"
    assert report["command"] == prior["command"]
    assert report["resolved_requests"] == prior["resolved_requests"]
    assert report["selected_execution_comparison"]["status"] == "matched"
    assert records[0]["result"]["source"] == "recorded_controller_choice_replay"


@pytest.mark.parametrize("field", ["packet", "menu", "raw", "command", "request"])
def test_recorded_response_integrity_is_checked_without_any_measurement(field):
    prior, _ = choose(ChoiceProvider(mode="source"), [])
    changed = deepcopy(prior)
    if field == "packet":
        changed["packet"]["step"] += 1
    elif field == "menu":
        changed["operation_menu"]["cards"].pop()
    elif field == "raw":
        changed["controller_trace"]["attempts"][0]["raw_json_object"]["meta"]["operation_id"] = "different"
    elif field == "command":
        changed["command"]["meta"]["operation_id"] = "different"
    else:
        changed["resolved_requests"][0]["structure_operator_id"] = "different"
    with pytest.raises(ValueError, match="recorded_choice"):
        recorded_command(changed, prior["packet"])


def test_recorded_choice_rejects_a_new_prefix_even_if_the_piece_still_exists():
    prior, _ = choose(ChoiceProvider(mode="source"), [])
    worker = multi_source_worker("pacb!")
    before = diagnostic_budget.used(worker)
    with pytest.raises(ValueError, match="recorded_choice_packet_mismatch"):
        run_choice(worker, None, {}, historical=[], save=lambda: None, record=lambda r: pytest.fail("must not measure"),
                   stages=STAGES, allow_source_cards=True, recorded_choice=prior)
    assert diagnostic_budget.used(worker) == before
    assert not worker._clause_patch_attempted_ids


@pytest.mark.parametrize("field", ["source_index", "alpha", "source_hash", "hook", "permission"])
def test_source_execution_audit_rejects_wrong_operator(field):
    report, records = choose(ChoiceProvider(mode="source"), [], allow_source_cards=True)
    item = records[0]
    measured = deepcopy(item["result"])
    row = measured["operator_rows"][0]
    context = row["context_fingerprint"]
    if field == "source_index":
        row["edit"]["source"]["expr"]["ref"]["token"]["value"] += 1
    elif field == "alpha":
        row["edit"]["op"]["alpha"] *= 2
    elif field == "source_hash":
        row["source_tensor_identity"]["sha256"] = "wrong"
    elif field == "hook":
        row["runtime_telemetry"][0][0]["hook_call_count"] = 0
    else:
        measured["production_apply_allowed"] = True
    result = validate_source_execution(report["selected_source_candidate"], measured, item["inputs"],
                                       report["expected_edits"], context)
    assert result["status"] == "mismatch"


def test_source_offline_comparison_requires_the_exact_single_candidate():
    first, records = choose(ChoiceProvider(mode="source"), [], allow_source_cards=True)
    historical = [clause_trajectory.compact_record(r) for r in records]
    second, _ = choose(ChoiceProvider(mode="source"), historical, allow_source_cards=True)
    assert second["offline_comparison"]["status"] == "matched"
    assert second["offline_comparison"]["readout_and_execution_exact_match"]


@pytest.mark.parametrize("mode,status", [("noop", "no_radial_selection"),
                                        ("invalid", "controller_error"), ("fail", "controller_error")])
def test_no_selection_or_invalid_response_never_forces_a_diagnostic_or_retries(mode, status):
    provider = ChoiceProvider(mode=mode)
    report, records = choose(provider, [])
    assert report["status"] == status
    assert len(provider.calls) == 1
    assert not records
    assert report["diagnostics_dispatched"] == report["diagnostic_slots_used"] == 0


def test_legacy_receipt_keeps_unbound_label_while_exact_raw_measurements_can_match(history):
    legacy = deepcopy(history)
    for item in legacy:
        for obj in (item["result"], item.get("component_definition", {}), *item["result"].get("operator_rows", [])):
            for key in ("source_pair_id", "component_recipe_id", "recipe", "source_pair"):
                obj.pop(key, None)
    original = deepcopy(legacy)
    report, _ = choose(ChoiceProvider(), legacy)
    comparison = report["offline_comparison"]
    assert comparison["status"] == "matched" and comparison["historical_binding_status"] == "legacy_unbound"
    assert legacy == original


@pytest.mark.parametrize("field", ["legs", "expression", "dose", "context", "readout"])
def test_receipt_drift_is_separated_from_actual_card_execution(history, field):
    changed = deepcopy(history)
    if field == "legs":
        changed[0]["inputs"][0]["source"]["sha256"] = "different-leg-even-if-same-mean"
    elif field == "expression":
        changed[-1]["component_definition"]["source_expressions"][0] = {"fn": "other"}
    elif field == "dose":
        changed[-1]["result"]["operator_rows"][0]["edit"]["budget"]["step_size"] *= 2
    elif field == "context":
        changed[-1]["result"]["operator_rows"][0]["context_fingerprint"] = "stale"
    else:
        changed[-1]["result"]["operator_rows"][0]["structure_response_rows"][0]["probability_deltas"] = [9., 9.]
    report, _ = choose(ChoiceProvider(), changed)
    assert report["selected_execution_comparison"]["status"] == "matched"
    if field == "readout":
        assert report["offline_comparison"]["status"] == "matched"
        assert not report["offline_comparison"]["readout_and_execution_exact_match"]
    else:
        assert report["offline_comparison"]["status"] == "mismatch"


@pytest.mark.parametrize("field", ["alpha", "step_size", "surface", "hook_count", "permission"])
def test_selected_card_validation_rejects_changed_operation(history, field):
    report, records = choose(ChoiceProvider(), history)
    measured = deepcopy(records[0]["result"])
    row = measured["operator_rows"][0]
    if field == "alpha":
        row["edit"]["op"]["alpha"] *= 2
    elif field == "step_size":
        row["edit"]["budget"]["step_size"] *= 2
    elif field == "surface":
        row["edit"]["target"]["surface_id"] = "other"
    elif field == "hook_count":
        row["runtime_telemetry"][0][0]["hook_call_count"] = 0
    else:
        row["production_apply_allowed"] = True
    assert validate_selected_execution(report["selected_recipe"], measured, records[0]["inputs"])["status"] == "mismatch"


def test_reference_provenance_and_numerics_checked_before_a_model_or_api_call():
    reference = {"kind": "source_common_radial_controls", "status": "complete",
                 "production_apply_allowed": False, "rollout_apply_count": 0, "records": [{}] * 10,
                 **{key: key for key in ("profile", "device", "dtype", "torch_threads", "position", "expected_prefix",
                                         "source_token_ids", "source_sha256", "caps", "diagnostic_budget_limit")}}
    run = deepcopy(reference)
    validate_offline_reference(reference, run)
    for key in ("dtype", "source_sha256", "expected_prefix", "caps"):
        changed = {**run, key: "different"}
        with pytest.raises(ValueError, match=key):
            validate_offline_reference(reference, changed)


def test_mislabelled_card_is_rejected_before_provider_or_dispatch(monkeypatch):
    worker, provider = multi_source_worker(), ChoiceProvider()
    original = operation_cards.build_menu

    def wrong_menu(packet):
        menu, requests = original(packet)
        card = next(c for c in menu["cards"] if c["diagnostic"] == clause_radial.NAME)
        card["component"] = "incorrectly_labelled_component"
        return menu, requests

    monkeypatch.setattr(operation_cards, "build_menu", wrong_menu)
    before = diagnostic_budget.used(worker)
    with pytest.raises(ValueError, match="card_label_differs"):
        run_choice(worker, provider, {}, historical=[], save=lambda: None, record=lambda r: None, stages=STAGES)
    assert not provider.calls
    assert diagnostic_budget.used(worker) == before


@pytest.mark.parametrize("flags", [["--controller-radial-choice"], ["--offline-radial-receipt", "missing"]])
def test_live_mode_requires_explicit_offline_reference_and_radial_flags(flags):
    args = ["--source-jsonl", "missing", "--source-manifest", "missing", "--reference-receipt", "missing",
            "--worker-model-path", "missing", "--position", "8", "--source-token-ids", "11,1418",
            "--output-dir", "must-not-be-created"]
    with pytest.raises(SystemExit) as exc:
        main([*args, *flags])
    assert exc.value.code == 2


@pytest.mark.parametrize("floor", ["9.9", "0", "nan", "inf", "-1"])
def test_disk_floor_cannot_be_disabled_with_a_small_or_nonfinite_value(floor):
    args = ["--source-jsonl", "missing", "--source-manifest", "missing", "--reference-receipt", "missing",
            "--worker-model-path", "missing", "--position", "8", "--source-token-ids", "11,1418",
            "--output-dir", "must-not-be-created", "--min-free-disk-gib", floor]
    with pytest.raises(SystemExit) as exc:
        main(args)
    assert exc.value.code == 2
