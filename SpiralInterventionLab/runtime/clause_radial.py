"""Current-state radial/nonradial controls expressed in the existing blend DSL.

The controller can select a bounded diagnostic recipe, not an apply candidate.
Orthogonality describes raw displacement, not semantics or additive effects
after independent clipping and nonlinear downstream computation.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch

from . import clause_actuation, clause_components, clause_specificity, clause_trajectory, diagnostic_budget
from .compiler import StepContext, compile_expr
from .response_probe import identity, state_identity, tensor_identity


NAME = "source_common_radial_probe"
MAX_PAIR_SOURCES = 3
CONTROLS = ("common_negative", "common_parallel", "common_orthogonal")
LABELS = (*clause_components.COMPONENTS, *CONTROLS)
FORMULAS = {
    "common": "C = mean(S_left, S_right); source = C; displacement = C - T",
    "common_negative": "source = T - (C - T)",
    "common_parallel": "source = T + project_parallel(C - T, T)",
    "common_orthogonal": "source = T + project_orthogonal(C - T, T)",
}
MAX_RECIPES = (MAX_PAIR_SOURCES * (MAX_PAIR_SOURCES - 1) // 2) * len(FORMULAS)
BINDING_KEYS = ("structure_context_id", "structure_operator_id", "target_token_id", "target_piece",
                "source_token_index", "source_provenance", "source_localization",
                "source_occurrence_rule", "observation_id")


def expressions(left: dict, right: dict, current: dict) -> dict[str, dict]:
    common = {"fn": "mean", "args": [left, right]}
    delta = {"fn": "sub", "args": [common, current]}
    return deepcopy({
        "common": common,
        "common_negative": {"fn": "sub", "args": [current, delta]},
        **{name: {"fn": "add", "args": [current, {"fn": projection, "arg": delta, "basis": current}]}
           for name, projection in (("common_parallel", "project_parallel"),
                                    ("common_orthogonal", "project_orthogonal"))},
    })


def geometry(common: torch.Tensor, target: torch.Tensor) -> dict:
    common, target = common.detach().cpu(), target.detach().cpu()
    if (common.ndim != 1 or common.shape != target.shape or not torch.isfinite(common).all()
            or not torch.isfinite(target).all() or float(target.double().norm()) == 0):
        raise ValueError("nonfinite_misaligned_or_zero_current_activation")
    delta, t = common.double() - target.double(), target.double()
    coefficient = float(torch.dot(delta, t) / torch.dot(t, t))
    parallel = coefficient * t
    orthogonal = delta - parallel
    return {"reference": "current_last_activation_not_semantic_basis", "radial_coefficient": coefficient,
            "radial_direction": "shrink" if coefficient < 0 else "expand" if coefficient > 0 else "zero",
            "common_displacement_norm": float(delta.norm()), "parallel_norm": float(parallel.norm()),
            "orthogonal_norm": float(orthogonal.norm()),
            "raw_reconstruction_error_norm": float((delta - parallel - orthogonal).norm()),
            "parallel_orthogonal_dot": float(torch.dot(parallel, orthogonal))}


def _prepare(worker: Any, candidates: Sequence[Mapping[str, Any]]) -> dict:
    if len(candidates) != 2 or candidates[0]["surfaces"] != candidates[1]["surfaces"]:
        raise ValueError("two_source_bindings_with_identical_surfaces_required")
    cached = getattr(worker.runtime_state, "last_tokens", None)
    if not isinstance(cached, torch.Tensor) or not torch.equal(cached, worker._current_token_tensor()):
        raise ValueError("cache_not_aligned_to_current_prefix")
    ctx = StepContext(packet={}, runtime_state=worker.runtime_state, adapter=worker.adapter,
                      traces={}, stats={}, active_edits={})
    prepared = {name: {"expressions": [], "inputs": [], "geometry": [], "source_input_identities": []}
                for name in FORMULAS}
    for surface in candidates[0]["surfaces"]:
        refs = [clause_actuation.materialize(worker, c, surface)["source"]["expr"] for c in candidates]
        current = deepcopy(refs[0])
        current["ref"]["token"] = {"mode": "last"}
        target = compile_expr(current)(ctx).detach().cpu().clone()
        source_ids = dict(zip(("left", "right", "target"),
            [*(tensor_identity(compile_expr(ref)(ctx)) for ref in refs), tensor_identity(target)], strict=True))
        exprs = expressions(*refs, current)
        common = compile_expr(exprs["common"])(ctx).detach().cpu().clone()
        shape = geometry(common, target)
        for name, expr in exprs.items():
            value = compile_expr(expr)(ctx).detach().cpu().clone()
            if value.shape != target.shape or not torch.isfinite(value).all():
                raise ValueError("nonfinite_or_misaligned_component")
            prepared[name]["expressions"].append(expr)
            prepared[name]["inputs"].append({"source": value, "target": target.clone()})
            prepared[name]["geometry"].append(shape)
            prepared[name]["source_input_identities"].append(source_ids)
    return prepared


def _recipe(pair: Sequence[Mapping[str, Any]], name: str, values: Mapping[str, Any]) -> dict:
    """One binding contract for offline replay and controller-selected replay."""
    binding = {"structure_context_id": pair[0]["structure_context_id"],
               "source_pair": [{k: c[k] for k in BINDING_KEYS} for c in pair],
               "surfaces": pair[0]["surfaces"],
               "source_input_identities": values["source_input_identities"]}
    descriptor = {**binding, "source_pair_id": identity("clause-pair:", binding),
                  "component": name, "source_expressions": values["expressions"],
                  "input_identities": [{k: tensor_identity(v) for k, v in item.items()}
                                       for item in values["inputs"]]}
    return {**descriptor, "component_recipe_id": identity("clause-radial:", descriptor),
            "dsl_formula": FORMULAS[name], "geometry": values["geometry"],
            "production_apply_allowed": False}


def _bind_result(result: dict, recipe: Mapping[str, Any]) -> None:
    result.update(source_pair_id=recipe["source_pair_id"],
                  component_recipe_id=recipe["component_recipe_id"], recipe=deepcopy(dict(recipe)))
    for index, row in enumerate(result.get("operator_rows", [])):
        if (row["source_tensor_identity"] != recipe["input_identities"][index]["source"]
                or row["surface"] != recipe["surfaces"][index]
                or row["edit"]["source"]["expr"] != recipe["source_expressions"][index]):
            raise RuntimeError("measured_component_differs_from_bound_recipe")
        row.update(source_pair_id=recipe["source_pair_id"],
                   source_pair=deepcopy(recipe["source_pair"]),
                   component_recipe_id=recipe["component_recipe_id"],
                   component_geometry=recipe["geometry"][index],
                   target_readout_effect_certified=False)


def offer(worker: Any) -> dict:
    offered = clause_actuation.offer(worker)
    result = {"status": offered["status"], "structure_context_id": offered["structure_context_id"],
              "recipes": [], "pair_rule": "all_pairs_of_first_three_exact_source_tokens_in_source_order_not_utility",
              "source_candidate_limit": MAX_PAIR_SOURCES,
              "source_candidate_count": len(offered["candidates"]),
              "excluded_source_candidate_count": max(0, len(offered["candidates"]) - MAX_PAIR_SOURCES),
              "blocked_pairs": [],
              "evidence_scope": "geometric_direction_test_not_semantic_ownership_or_apply_permission",
              "diagnostic_only": True, "production_apply_allowed": False}
    if offered["status"] not in {"available", "already_attempted_at_context"}:
        return result
    candidates = offered["candidates"][:MAX_PAIR_SOURCES]
    if len(candidates) < 2:
        return {**result, "status": "two_exact_source_tokens_required"}
    for pair in combinations(candidates, 2):
        try:
            prepared = _prepare(worker, pair)
        except (ValueError, KeyError, IndexError) as exc:
            result["blocked_pairs"].append({"source_pair": [{k: c[k] for k in BINDING_KEYS} for c in pair],
                                            "blocked_reason": str(exc)})
            continue
        for name, values in prepared.items():
            recipe = _recipe(pair, name, values)
            zero = any(torch.equal(v["source"], v["target"]) for v in values["inputs"])
            attempted = getattr(worker, "_clause_patch_attempted_ids", set())
            control_id = clause_specificity.expression_control_id(name, pair[0], values["expressions"])
            blocked = "degenerate_component" if zero else "already_attempted_at_context" if (
                recipe["component_recipe_id"] in attempted or control_id in attempted) else None
            result["recipes"].append({**recipe, "available": blocked is None, "blocked_reason": blocked,
                "physical_replays_upper_bound": 2 + 2 * len(pair[0]["surfaces"]),
                "request": {"diagnostic": NAME, "component_recipe_id": recipe["component_recipe_id"]}})
    if not result["recipes"]:
        return {**result, "status": "unavailable", "blocked_reason": "no_preparable_source_pair"}
    result["status"] = "available" if any(r["available"] for r in result["recipes"]) else "no_unmeasured_component"
    return result


def execute(worker: Any, request: Mapping[str, Any], *, source: str) -> dict:
    offered = offer(worker)
    recipe = next((r for r in offered["recipes"] if r["component_recipe_id"] == request.get("component_recipe_id")), None)
    report = {"diagnostic": NAME, "source": source, "recorded_step": worker._steps,
              "structure_context_id": offered["structure_context_id"],
              "component_recipe_id": request.get("component_recipe_id"),
              "status": "unavailable", "diagnostic_call_cost": 0, "physical_replay_count": 0,
              "new_measurement_count": 0, "evidence_scope": offered["evidence_scope"],
              "diagnostic_only": True, "production_apply_allowed": False,
              "certified_for_apply": False, "policy_candidate_ready": False}
    if recipe is None or not recipe["available"]:
        return {**report, "blocked_reason": recipe["blocked_reason"] if recipe else "unknown_or_stale_component_recipe"}
    # The request is an ID, not a new expression, dose, target, or policy override.
    worker._clause_patch_attempted_ids.add(recipe["component_recipe_id"])
    anchor = next(c for c in clause_actuation.offer(worker)["candidates"]
                  if c["structure_operator_id"] == recipe["source_pair"][0]["structure_operator_id"])
    measured = clause_specificity.measure_expression_control(
        worker, recipe["component"], anchor, recipe["source_expressions"], charge_budget=False)
    result = {**report, **measured, "recipe": recipe, "selection_origin": "controller_selected_diagnostic_id",
              "new_measurement_count": len(measured["operator_rows"])}
    _bind_result(result, recipe)
    return result


def plan(token_ids: Sequence[int]) -> dict:
    base = clause_components.plan(token_ids)
    return {**base, "condition_order": [*base["condition_order"], *CONTROLS],
            "diagnostic_slots_for_conditions": 10,
            "displacement_definitions": {**base["displacement_definitions"],
                "common_negative": "-(C - T)", "common_parallel": "project_parallel(C - T, T)",
                "common_orthogonal": "project_orthogonal(C - T, T)"},
            "control_scope": "radial_relative_to_current_T_not_semantic_subspace",
            "projection_contract": "native_DSL_projection_plus_T_report_rounding_against_fp64_geometry"}


def measure(worker: Any, token_ids: Sequence[int], *, stages: Sequence[str] | None = None,
            record: Any = None) -> list[dict]:
    plan(token_ids)
    if diagnostic_budget.left(worker) < 10:
        raise ValueError("insufficient_budget_for_ten_radial_conditions")
    offered = clause_actuation.offer(worker)
    pair = [next((c for c in offered["candidates"] if c["target_token_id"] == tid and c["available"]), None)
            for tid in token_ids]
    if any(c is None for c in pair):
        raise ValueError("requested_source_pair_not_currently_offered")
    prepared = _prepare(worker, pair)
    recipes = {name: _recipe(pair, name, values) for name, values in prepared.items()}
    if any(torch.equal(v["source"], v["target"]) for name in CONTROLS for v in prepared[name]["inputs"]):
        raise ValueError("degenerate_radial_component")
    tokens = worker._current_token_tensor().detach().clone()
    stages = list(stages) if stages is not None else clause_trajectory.stage_names(worker.runtime_state.model.cfg.n_layers)
    before, step, caps = state_identity(worker), worker._steps, repr(worker.surface_catalog)

    def bind_and_record(item: dict) -> None:
        if item["label"] in recipes:
            recipe = recipes[item["label"]]
            _bind_result(item["result"], recipe)
            item["component_definition"].update(source_pair_id=recipe["source_pair_id"],
                                                component_recipe_id=recipe["component_recipe_id"])
        if record is not None:
            record(item)

    records = clause_components.measure(worker, token_ids, stages=stages, record=bind_and_record)
    for name in CONTROLS:
        with clause_trajectory.capture_probe(worker, tokens, stages) as frames:
            result = clause_specificity.measure_expression_control(worker, name, pair[0], prepared[name]["expressions"])
        item = {"label": name, "kind": "component", "result": result, "frames": frames,
                "inputs": prepared[name]["inputs"], "component_definition": {
                    "source_token_ids": list(token_ids), "source_expressions": prepared[name]["expressions"],
                    "dsl_formula": FORMULAS[name], "geometry": prepared[name]["geometry"]}}
        records.append(item)
        bind_and_record(item)
        if result["status"] != "ok":
            raise ValueError(f"radial_measurement_failed:{name}:{result.get('error', result.get('blocked_reason'))}")
        if before != state_identity(worker) or worker._steps != step or repr(worker.surface_catalog) != caps:
            raise RuntimeError("radial_measurement_changed_worker_state")
    return records


def summarize(records: Sequence[dict]) -> dict:
    trajectory = clause_trajectory.summarize(records, component_labels=LABELS)
    token_ids = [r["result"]["candidate_binding"]["target_token_id"] for r in records[:2]]
    if [r["label"] for r in records] != plan(token_ids)["condition_order"]:
        raise ValueError("radial_condition_order_differs_from_plan")
    result = {"status": "ok", "plan": plan(token_ids), "trajectory": trajectory,
              "components": clause_components.summarize(records[:7]), "surfaces": [],
              "production_apply_allowed": False, "target_readout_effect_certified": False}
    bound_records = [r for r in records if r["label"] in FORMULAS]
    has_binding = any(key in holder for r in bound_records
                      for holder in (r["result"], r["component_definition"], *r["result"]["operator_rows"])
                      for key in ("recipe", "source_pair_id", "component_recipe_id"))
    result["source_pair_binding_status"] = "verified_receipt_binding" if has_binding else "legacy_unbound"
    if has_binding:
        pair = [r["result"]["candidate_binding"] for r in records[:2]]
        source_ids, exprs, shapes = [], [], []
        for index in range(len(trajectory["surfaces"])):
            inputs = [r["inputs"][index] for r in records[:2]]
            source_ids.append({"left": tensor_identity(inputs[0]["source"]),
                               "right": tensor_identity(inputs[1]["source"]),
                               "target": tensor_identity(inputs[0]["target"])})
            refs = [r["result"]["operator_rows"][index]["edit"]["source"]["expr"] for r in records[:2]]
            current = deepcopy(refs[0])
            current["ref"]["token"] = {"mode": "last"}
            exprs.append(expressions(*refs, current))
            shapes.append(geometry(records[4]["inputs"][index]["source"], inputs[0]["target"]))
        for item in bound_records:
            expected = _recipe(pair, item["label"], {
                "expressions": [e[item["label"]] for e in exprs], "inputs": item["inputs"],
                "geometry": shapes, "source_input_identities": source_ids})
            measured = item["result"]
            if (any(measured.get("recipe", {}).get(k) != v for k, v in expected.items())
                    or any(holder.get(k) != expected[k] for holder in (
                        measured, item["component_definition"], *measured["operator_rows"])
                           for k in ("source_pair_id", "component_recipe_id"))
                    or any(row.get("source_pair") != expected["source_pair"] for row in measured["operator_rows"])):
                raise ValueError("radial_recipe_binding_mismatch")
    for index, surface in enumerate(trajectory["surfaces"]):
        common = records[4]["inputs"][index]
        shape = geometry(common["source"], common["target"])
        t = common["target"].double()
        d = common["source"].double() - t
        parallel = shape["radial_coefficient"] * t
        ideal = dict(zip(CONTROLS, (-d, parallel, d - parallel), strict=True))
        entry = {"surface": surface["surface"], "geometry": shape, "controls": []}
        for item in records[7:]:
            inputs, row = item["inputs"][index], item["result"]["operator_rows"][index]
            refs = [r["result"]["operator_rows"][index]["edit"]["source"]["expr"] for r in records[:2]]
            current = deepcopy(refs[0])
            current["ref"]["token"] = {"mode": "last"}
            expected_expr = expressions(*refs, current)[item["label"]]
            if (not torch.equal(inputs["target"], common["target"])
                    or row["source_tensor_identity"] != tensor_identity(inputs["source"])
                    or item["component_definition"]["geometry"][index] != shape
                    or item["component_definition"]["source_expressions"][index] != expected_expr
                    or row["edit"]["source"]["expr"] != expected_expr):
                raise ValueError("radial_input_identity_or_geometry_mismatch")
            delta = inputs["source"].double() - t
            written = clause_trajectory._deltas(item, index, surface["surface"]["cache_hook_name"])
            entry["controls"].append({"label": item["label"],
                "source_displacement_norm": float(delta.norm()),
                "source_rounding_error_vs_ideal_norm": float((delta - ideal[item["label"]]).norm()),
                "actual_written_delta_norms": [float(v.norm()) for v in written],
                "safety_deltas": row["safety_deltas"], "collapse_sharpening_signal": row["collapse_sharpening_signal"],
                "production_apply_allowed": False})
        result["surfaces"].append(entry)
    return result
