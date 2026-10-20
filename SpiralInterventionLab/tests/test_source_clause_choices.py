import json
from copy import deepcopy

import pytest

from SpiralInterventionLab.controllers.base import ControllerProviderResponse
from SpiralInterventionLab.examples.compare_source_clause_choices import (
    NAME, SCHEDULE, load_snapshot, prepare_report, run_choices, stable_hash, visibility_packets,
)
from SpiralInterventionLab.examples.replay_source_clause_diagnostic import measure_current_prefix
from SpiralInterventionLab.runtime import operation_cards
from SpiralInterventionLab.tests.test_clause_structure import make_worker


def snapshot(tmp_path):
    worker = make_worker()
    measurement = measure_current_prefix(worker, export_controller_packets=True)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({
        "kind": "source_clause_diagnostic_replay", "status": "complete",
        "measurements": [measurement],
    }))
    return worker, path, measurement["controller_comparison"]["packets"]


def test_export_changes_only_source_offer_and_no_old_request_or_worker_authority(tmp_path):
    worker, path, packets = snapshot(tmp_path)
    row, loaded = load_snapshot(path)
    assert loaded == packets
    off_menu, off_requests = operation_cards.build_menu(packets["off"])
    on_menu, on_requests = operation_cards.build_menu(packets["offered"])
    assert all(card["diagnostic"] != NAME for card in off_menu["cards"])
    assert list(off_requests.values()) == [
        req for req in on_requests.values() if req["diagnostic"] != NAME]
    assert on_menu["cards"][-1]["diagnostic"] == NAME
    assert row["result"]["diagnostic_call_cost"] == 1
    assert not worker._collect_active_edits()
    assert not worker._frozen_diagnostic_candidates
    observed = packets["observed"]["strategy_hints"]["source_clause_structure"]
    assert observed["status"] == "already_attempted_at_context"
    assert not any(card["diagnostic"] == NAME and card["available"]
                   for card in operation_cards.build_menu(packets["observed"])[0]["cards"])


def test_snapshot_rejects_unrelated_drift_even_with_rehashed_input(tmp_path):
    _, path, _ = snapshot(tmp_path)
    data = json.loads(path.read_text())
    comparison = data["measurements"][0]["controller_comparison"]
    comparison["packets"]["off"]["strategy_hints"]["invented_recommendation"] = "pick comma"
    comparison["packet_sha256"]["off"] = stable_hash(comparison["packets"]["off"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unrelated"):
        load_snapshot(path)


def test_snapshot_rejects_corruption_and_measurement_leakage(tmp_path):
    _, path, packets = snapshot(tmp_path)
    before = deepcopy(packets["offered"])
    before["structure_readout_rows"] = [{"rank": 1}]
    with pytest.raises(ValueError, match="pre-observation"):
        visibility_packets(before, packets["observed"])
    data = json.loads(path.read_text())
    data["measurements"][0]["controller_comparison"]["packet_sha256"]["off"] = "wrong"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        load_snapshot(path)


class FakeProvider:
    provider_name = "fake"
    model_name = "test-model"

    def __init__(self, *, invalid=False, fail=False):
        self.calls = []
        self.invalid = invalid
        self.fail = fail

    def complete(self, request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("provider unavailable")
        menu = request.payload["operation_menu"]
        source = next((card for card in menu["cards"] if card["diagnostic"] == NAME), None)
        command = {"decision": "noop", "edits": [], "meta": {}}
        if source is not None:
            command["meta"]["operation_id"] = source["operation_id"]
        return ControllerProviderResponse(
            provider="fake", model="test-model", text="invalid" if self.invalid else json.dumps(command),
            usage={"input_tokens": 10, "output_tokens": 5},
        )


def test_comparison_is_six_stateless_choices_without_dispatch_or_packet_mutation(tmp_path):
    worker, path, packets = snapshot(tmp_path)
    report = prepare_report(path, "test-model")
    provider = FakeProvider()
    budgets = worker._diagnostic_calls_used
    original = deepcopy(packets)
    run_choices(report, packets, provider, lambda: None)
    assert len(provider.calls) == report["provider_calls_attempted"] == 6
    assert report["status"] == "complete"
    assert [(row["repeat"], row["condition"]) for row in report["choices"]] == list(SCHEDULE)
    assert [row["source_clause_selected"] for row in report["choices"]] == [
        False, True, True, False, False, False]
    assert len({request.system_prompt for request in provider.calls}) == 1
    assert all(row["packet_unchanged"] for row in report["choices"])
    assert packets == original
    assert worker._diagnostic_calls_used == budgets
    assert report["diagnostics_dispatched"] == report["rollout_edits_applied"] == 0
    assert not report["production_apply_executed"]
    assert report["inputs"]["observed"]["provider_payload"]["recent_diagnostic_results"]


def test_invalid_first_attempt_is_preserved_without_repair_calls(tmp_path):
    _, path, packets = snapshot(tmp_path)
    report = prepare_report(path, "test-model")
    provider = FakeProvider(invalid=True)
    run_choices(report, packets, provider, lambda: None)
    assert len(provider.calls) == 6
    assert all(row["status"] == "invalid_command" for row in report["choices"])
    assert all(len(row["trace"]["attempts"]) == 1 for row in report["choices"])


def test_provider_error_stops_instead_of_spending_the_remaining_calls(tmp_path):
    _, path, packets = snapshot(tmp_path)
    report = prepare_report(path, "test-model")
    provider = FakeProvider(fail=True)
    with pytest.raises(RuntimeError):
        run_choices(report, packets, provider, lambda: None)
    assert len(provider.calls) == report["provider_calls_attempted"] == 1
    assert report["status"] == "provider_error"
    assert report["choices"][0]["error_type"] == "RuntimeError"
