from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from SpiralInterventionLab.bridge.controller_clients import _compact_controller_payload, _compact_diagnostic_result
from SpiralInterventionLab.examples.replay_source_clause_trajectory import main
from SpiralInterventionLab.runtime import (
    clause_actuation, clause_radial as radial, clause_specificity, clause_structure,
    clause_trajectory, diagnostic_budget, operation_cards,
)
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


def multi_source_worker(prompt="pacb"):
    worker = component_worker()
    worker.reset(prompt)
    worker._append_output_token(1)
    worker._steps = 1
    worker.runtime_state.run_with_cache(worker._current_token_tensor())
    pieces = {0: "SOURCE:", 1: " before", 3: ",", 2: " while", 4: ";"}
    records = [{"position": i, "token_id": tid, "piece": pieces[tid],
                "provenance_class": "constraint_header" if i == 0 else "source_body",
                "segment_kind": "prompt"} for i, tid in enumerate(worker._current_token_tensor().flatten().tolist()[:-1])]
    worker._token_position_records = lambda: records
    observed = worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])[0]
    assert observed["status"] == "ok"
    return worker


def recipe_for(worker, token_ids=(3, 2), component="common_orthogonal"):
    return next(r for r in radial.offer(worker)["recipes"] if r["component"] == component
                and tuple(c["target_token_id"] for c in r["source_pair"]) == token_ids)


def test_explicit_pairs_are_bounded_source_order_choices_visible_in_compact_cards():
    worker = multi_source_worker("pacb!")
    used = diagnostic_budget.used(worker)
    with patch.object(worker.runtime_state, "run_with_cache", side_effect=AssertionError("offers cannot measure")):
        offered = radial.offer(worker)
        packet = {"strategy_hints": hints(worker)}
    assert offered["source_candidate_count"] == 4
    assert offered["excluded_source_candidate_count"] == 1
    assert len(offered["recipes"]) == radial.MAX_RECIPES == 12
    assert [tuple(c["target_token_id"] for c in r["source_pair"]) for r in offered["recipes"][::4]] == [
        (1, 3), (1, 2), (3, 2)]
    assert diagnostic_budget.used(worker) == used
    compact = _compact_controller_payload(packet)["strategy_hints"]["source_common_radial"]
    assert len(compact["recipes"]) == 12  # The final pair must not be lost in prompt dieting.
    cards = [c for c in operation_cards.build_menu(packet)[0]["cards"] if c["diagnostic"] == radial.NAME]
    for card, recipe in zip(cards, offered["recipes"], strict=True):
        for key in ("component_recipe_id", "source_pair_id", "source_pair", "component", "surfaces", "structure_context_id"):
            assert card[key] == recipe[key]
        assert card["cost"]["diagnostic_slots_upper_bound"] == 1
        assert card["cost"]["physical_replays_upper_bound"] == 4
        assert not card["production_apply_allowed"]


def test_third_pair_card_measures_that_pair_not_the_first_global_anchor():
    worker = multi_source_worker()
    recipe = recipe_for(worker)
    packet = {"strategy_hints": hints(worker)}
    card = next(c for c in operation_cards.build_menu(packet)[0]["cards"]
                if c.get("component_recipe_id") == recipe["component_recipe_id"])
    requests = _extract_diagnostic_requests({"decision": "noop", "meta": {"operation_id": card["operation_id"]}}, packet)
    assert requests == [recipe["request"]]
    used, before = diagnostic_budget.used(worker), state_identity(worker)
    with patch.object(worker, "_simulate_decode", wraps=worker._simulate_decode) as simulate:
        result = worker.request_controller_diagnostics(requests)[0]
    assert simulate.call_count == 4 and diagnostic_budget.used(worker) == used + 1
    assert all("policy_override" not in c.kwargs for c in simulate.call_args_list)
    assert result["status"] == "ok" and result["source_pair_id"] == recipe["source_pair_id"]
    row = result["operator_rows"][0]
    assert row["edit"]["source"]["expr"] == recipe["source_expressions"][0]
    assert [c["source_token_index"] for c in row["source_pair"]] == [2, 3]
    anchor = clause_actuation.offer(worker)["candidates"][1]
    expected_id = clause_specificity.expression_control_id(
        recipe["component"], anchor, recipe["source_expressions"])
    assert result["control_id"] == expected_id
    assert _compact_diagnostic_result(result)["recipe"]["source_pair_id"] == recipe["source_pair_id"]
    assert state_identity(worker) == before and not worker._collect_active_edits()
    assert not worker._frozen_diagnostic_candidates and not result["production_apply_allowed"]
    assert recipe_for(worker, (1, 3))["available"]
    duplicate = worker.request_controller_diagnostics(requests)[0]
    assert duplicate["blocked_reason"] == "already_attempted_at_context"
    assert duplicate["diagnostic_call_cost"] == duplicate["physical_replay_count"] == 0


def test_offline_replay_and_controller_bind_the_same_pair_expression_and_inputs():
    offline, controller = multi_source_worker(), multi_source_worker()
    offered = {r["component"]: r for r in radial.offer(controller)["recipes"]
               if tuple(c["target_token_id"] for c in r["source_pair"]) == (3, 2)}
    saved = []
    records = radial.measure(offline, [3, 2], stages=STAGES, record=lambda r: saved.append(deepcopy(r)))
    assert radial.summarize(records)["source_pair_binding_status"] == "verified_receipt_binding"
    for record in saved:
        if record["label"] not in radial.FORMULAS:
            continue
        expected = offered[record["label"]]
        assert record["result"]["component_recipe_id"] == expected["component_recipe_id"]
        assert record["component_definition"]["source_pair_id"] == expected["source_pair_id"]
        assert record["result"]["recipe"]["source_input_identities"] == expected["source_input_identities"]
    result = controller.request_controller_diagnostics(offered["common_orthogonal"]["request"])[0]
    assert result["control_id"] == saved[-1]["result"]["control_id"]
    assert result["operator_rows"][0]["candidate_fingerprint"] == saved[-1]["result"]["operator_rows"][0]["candidate_fingerprint"]
    assert not recipe_for(offline)["available"]
    assert recipe_for(offline, (1, 3))["available"]


def test_same_piece_at_another_occurrence_is_not_the_recorded_source_pair():
    worker = multi_source_worker("pacbc")
    old = recipe_for(worker)
    records = worker._token_position_records()
    records[2]["provenance_class"] = "misc_prompt"
    used = diagnostic_budget.used(worker)
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("stale pair cannot measure")):
        result = worker.request_controller_diagnostics(old["request"])[0]
    assert result["physical_replay_count"] == result["diagnostic_call_cost"] == 0
    assert diagnostic_budget.used(worker) == used
    worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])
    fresh = recipe_for(worker, (2, 3))
    assert fresh["source_pair"][1]["target_piece"] == old["source_pair"][0]["target_piece"]
    assert fresh["source_pair"][1]["source_token_index"] == 4
    assert fresh["source_pair_id"] != old["source_pair_id"]


def test_source_legs_are_bound_even_when_their_mean_does_not_change():
    worker = multi_source_worker()
    cache = worker.runtime_state.last_cache[STAGES[0]]
    cache[0, 2] = torch.tensor([4., 2., 0.])
    cache[0, 3] = torch.tensor([2., 4., 0.])
    before = recipe_for(worker, component="common")
    cache[0, 2, 0] += 1
    cache[0, 3, 0] -= 1
    after = recipe_for(worker, component="common")
    assert before["input_identities"] == after["input_identities"]
    assert before["source_input_identities"] != after["source_input_identities"]
    assert before["source_pair_id"] != after["source_pair_id"]
    with patch.object(worker, "_simulate_decode", side_effect=AssertionError("old raw binding cannot measure")):
        result = worker.request_controller_diagnostics(before["request"])[0]
    assert result["physical_replay_count"] == result["diagnostic_call_cost"] == 0


@pytest.mark.parametrize("field", ["source_pair_id", "component_recipe_id", "source_pair"])
def test_reanalysis_rejects_receipt_pair_or_recipe_relabeling(field):
    records = radial.measure(multi_source_worker(), [3, 2], stages=STAGES)
    records[-1]["result"]["recipe"][field] = "another binding"
    with pytest.raises(ValueError, match="recipe_binding_mismatch"):
        radial.summarize(records)


def test_legacy_receipts_remain_unbound_not_retroactively_promoted_to_current_ids():
    records = radial.measure(component_worker(), [1, 3], stages=STAGES)
    for record in records:
        result = record["result"]
        for key in ("recipe", "source_pair_id", "component_recipe_id"):
            result.pop(key, None)
        for holder in [record.get("component_definition", {}), *result["operator_rows"]]:
            for key in ("source_pair_id", "source_pair", "component_recipe_id"):
                holder.pop(key, None)
    summary = radial.summarize(records)
    assert summary["source_pair_binding_status"] == "legacy_unbound"
    assert not summary["production_apply_allowed"]
    records[-1]["result"]["operator_rows"][0]["source_pair_id"] = "partially retained binding"
    with pytest.raises(ValueError, match="recipe_binding_mismatch"):
        radial.summarize(records)
