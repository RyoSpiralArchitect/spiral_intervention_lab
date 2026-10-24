from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.examples.replay_source_clause_trajectory import main
from SpiralInterventionLab.runtime import clause_radial as radial, clause_trajectory, diagnostic_budget, operation_cards
from SpiralInterventionLab.runtime.compiler import compile_expr
from SpiralInterventionLab.runtime.loop import _diagnostic_request_signature, _extract_diagnostic_requests
from SpiralInterventionLab.runtime.response_probe import state_identity
from SpiralInterventionLab.tests.test_clause_components import STAGES, component_worker
from SpiralInterventionLab.tests.test_entity_candidate_review import hints


def test_dsl_adds_target_back_and_splits_shrink_from_nonradial_direction():
    values = {"a": torch.tensor([0., 4., 0.]), "b": torch.tensor([2., 2., 0.]),
              "t": torch.tensor([4., 0., 0.])}
    ctx = SimpleNamespace(adapter=SimpleNamespace(read_ref=lambda ref, ctx: values[ref["name"]]))
    exprs = radial.expressions(*[{"ref": {"name": name}} for name in ("a", "b", "t")])
    sources = {k: compile_expr(expr)(ctx) for k, expr in exprs.items()}
    assert torch.equal(sources["common_parallel"], torch.tensor([1., 0., 0.]))
    assert torch.equal(sources["common_orthogonal"], torch.tensor([4., 3., 0.]))
    assert torch.equal(sources["common_negative"], torch.tensor([7., -3., 0.]))
    t = values["t"]
    assert torch.equal(sources["common_parallel"] + sources["common_orthogonal"] - 2*t, sources["common"] - t)
    shape = radial.geometry(sources["common"], t)
    assert shape["radial_coefficient"] == -0.75 and shape["radial_direction"] == "shrink"
    assert shape["raw_reconstruction_error_norm"] == shape["parallel_orthogonal_dot"] == 0
    assert radial.geometry(t * 2, t)["radial_direction"] == "expand"


@pytest.mark.parametrize("bad", ["zero", "nan", "shape"])
def test_invalid_basis_is_not_renormalized_into_a_signal(bad):
    target, common = torch.tensor([1., 0.]), torch.tensor([1., 2.])
    if bad == "zero":
        target.zero_()
    elif bad == "nan":
        common[0] = float("nan")
    else:
        common = common[:1]
    with pytest.raises(ValueError):
        radial.geometry(common, target)


def test_current_controller_cards_have_compact_dsl_cost_and_exclusive_request():
    worker = component_worker()
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("offer is not a forward")):
        offered = radial.offer(worker)
        packet = {"strategy_hints": hints(worker)}
    assert offered["status"] == "available"
    assert [r["component"] for r in offered["recipes"]] == list(radial.FORMULAS)
    recipes = {r["component_recipe_id"]: r for r in offered["recipes"]}
    menu, _ = operation_cards.build_menu(packet)
    cards = [c for c in menu["cards"] if c["diagnostic"] == radial.NAME]
    assert len(cards) == 4
    for card in cards:
        assert card["cost"]["physical_replays_upper_bound"] == 4
        assert card["cost"]["diagnostic_slots_upper_bound"] == 1
        assert not card["production_apply_allowed"]
        recipe = recipes[card["component_recipe_id"]]
        assert card["dsl_formula"] == recipe["dsl_formula"]
        command = {"decision": "noop", "meta": {"operation_id": card["operation_id"]}}
        assert _extract_diagnostic_requests(command, packet) == [recipe["request"]]
        legacy = {"decision": "noop", "meta": {"diagnostic_request": recipe["request"],
                  "next_action": "request_operator_diagnostic", "operator_recipe_expansion_mode": "readout_steering_deepening"}}
        assert _extract_diagnostic_requests(legacy, packet) == [recipe["request"]]
    assert len({_diagnostic_request_signature(r["request"]) for r in recipes.values()}) == 4
    compact = _compact_controller_payload(packet)["strategy_hints"]["source_common_radial"]
    assert compact["recipes"][0]["geometry"] and compact["recipes"][0]["source_pair"]
    assert "source_expressions" not in compact["recipes"][0]


def test_runtime_replays_two_selected_recipes_charges_once_each_and_grants_no_permission():
    worker = component_worker()
    before, used = state_identity(worker), diagnostic_budget.used(worker)
    recipes = radial.offer(worker)["recipes"]
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        results = worker.request_controller_diagnostics([recipes[1]["request"], recipes[2]["request"], recipes[1]["request"]])
    assert simulate.call_count == 8
    assert all("policy_override" not in c.kwargs for c in simulate.call_args_list)
    assert diagnostic_budget.used(worker) == used + 2
    assert [r["diagnostic_call_cost"] for r in results] == [1, 1, 0]
    assert results[-1]["blocked_reason"] == "already_attempted_at_context"
    assert state_identity(worker) == before and not worker._collect_active_edits()
    assert not worker._frozen_diagnostic_candidates
    for result in results[:2]:
        assert result["status"] == "ok" and result["new_measurement_count"] == 1
        assert result["selection_origin"] == "controller_selected_diagnostic_id"
        assert not result["production_apply_allowed"] and not result["certified_for_apply"]
        row = result["operator_rows"][0]
        assert row["edit"]["op"] == {"kind": "activation_patch", "mode": "blend", "alpha": 0.04}
        assert row["edit"]["budget"]["step_size"] == 0.08
        assert row["component_geometry"] and row["structure_response_rows"]
        assert "ownership_role" not in row and not row["target_readout_effect_certified"]
        compact = _compact_diagnostic_result(result)
        assert compact["operator_rows"][0]["component_geometry"] == row["component_geometry"]
        assert compact["recipe"]["component_recipe_id"] == result["component_recipe_id"]


@pytest.mark.parametrize("change", ["prefix", "caps", "tensor", "budget", "active", "off"])
def test_stale_or_unavailable_request_cannot_measure_or_broaden_caps(change):
    worker = component_worker()
    request = radial.offer(worker)["recipes"][0]["request"]
    if change == "prefix":
        worker._append_output_token(2)
    elif change == "caps":
        s = worker.surface_catalog[0]
        worker.surface_catalog[0] = replace(s, caps=replace(s.caps, step_size=0.02))
    elif change == "tensor":
        worker.runtime_state.last_cache[STAGES[0]][0, -1, 0] += 0.2
    elif change == "budget":
        worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run
    elif change == "active":
        worker._collect_active_edits = lambda: [{"id": "active"}]
    else:
        worker.clause_structure_operator_diagnostics = False
    used = diagnostic_budget.used(worker)
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("unavailable")):
        result = radial.execute(worker, request, source="test")
    assert result["status"] == "unavailable" and result["diagnostic_call_cost"] == result["physical_replay_count"] == 0
    assert diagnostic_budget.used(worker) == used


def test_failed_physical_attempt_is_charged_once_and_reset_clears_attempt_ids():
    worker = component_worker()
    request = radial.offer(worker)["recipes"][0]["request"]
    used = diagnostic_budget.used(worker)
    with patch.object(worker, "_simulate_decode", return_value=None):
        result, duplicate = worker.request_controller_diagnostics([request, request])
    assert result["status"] == "error" and result["physical_replay_count"] == 1
    assert result["new_measurement_count"] == 0 and diagnostic_budget.used(worker) == used + 1
    assert duplicate["diagnostic_call_cost"] == 0
    worker.reset("p")
    assert not worker._clause_patch_attempted_ids


def test_offline_ten_condition_matrix_keeps_original_controls_budget_and_raw_audit():
    worker = component_worker()
    used, before = diagnostic_budget.used(worker), state_identity(worker)
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        records = radial.measure(worker, [1, 3], stages=STAGES)
    assert simulate.call_count == 40
    assert diagnostic_budget.used(worker) == used + 10
    assert state_identity(worker) == before
    assert [r["label"] for r in records] == radial.plan([1, 3])["condition_order"]
    summary = radial.summarize(records)
    assert len(summary["surfaces"][0]["controls"]) == 3
    assert not summary["production_apply_allowed"]
    assert not any(r["available"] for r in radial.offer(worker)["recipes"])
    forged = deepcopy(records)
    forged[-1]["component_definition"]["geometry"][0]["radial_coefficient"] = 500
    with pytest.raises(ValueError, match="geometry_mismatch"):
        radial.summarize(forged)
    with pytest.raises(ValueError):
        clause_trajectory.summarize(records)


def test_offline_budget_preflight_and_incompatible_cli_are_free():
    worker = component_worker()
    worker._diagnostic_calls_used = worker.max_diagnostic_calls_per_run - 9
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("no forwards")):
        with pytest.raises(ValueError, match="insufficient_budget"):
            radial.measure(worker, [1, 3], stages=STAGES)
    args = ["--source-jsonl", "missing", "--source-manifest", "missing", "--reference-receipt", "missing",
            "--worker-model-path", "missing", "--position", "8", "--source-token-ids", "11,1418", "--output-dir", "missing"]
    for extra in (["--radial-component-controls"], ["--radial-component-controls", "--decompose-components", "--permuted-direction-controls"]):
        with pytest.raises(SystemExit) as error:
            main([*args, *extra])
        assert error.value.code == 2
