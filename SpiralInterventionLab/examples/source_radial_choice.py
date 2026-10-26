"""One neutral controller choice, exact dispatch, then offline receipt comparison.

Historical outcomes are never included in the controller packet. Uncovered
pairs, non-radial choices and abstention are retained, not repaired into a match.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any

from ..bridge.controller_clients import ProviderControllerClient, _payload_for_provider
from ..runtime import clause_actuation, clause_radial, clause_trajectory, diagnostic_budget, operation_cards, prefix_control
from ..runtime.compiler import StepContext, compile_expr
from ..runtime.loop import _extract_diagnostic_requests
from ..runtime.response_probe import identity, state_identity, tensor_identity
from ..runtime.schema import ControllerCommand
from .compare_source_clause_choices import stable_hash


def validate_offline_reference(reference: dict, run: dict) -> None:
    if (reference.get("kind") != "source_common_radial_controls"
            or reference.get("status") != "complete"
            or reference.get("production_apply_allowed") is not False
            or reference.get("rollout_apply_count") != 0):
        raise ValueError("requires_complete_diagnostic_only_radial_reference")
    for key in ("profile", "device", "dtype", "torch_threads", "position", "expected_prefix",
                "source_token_ids", "source_sha256", "caps", "diagnostic_budget_limit"):
        if key not in reference or reference[key] != run[key]:
            raise ValueError(f"offline_reference_context_differs:{key}")
    if len(reference.get("records", [])) != 10:
        raise ValueError("incomplete_offline_radial_reference")


def compare_offline(recipe: dict, measured: dict, historical: list[dict]) -> dict:
    """Compare raw bindings first; never assign current IDs to legacy evidence."""
    report = {"status": "not_covered", "production_apply_allowed": False,
              "historical_binding_status": "legacy_unbound",
              "scope": "operation_identity_and_repeated_readout_not_task_success"}
    if len(historical) < 2:
        return {**report, "reason": "offline_source_pair_missing"}
    pair = [r.get("result", {}).get("candidate_binding") for r in historical[:2]]
    if any(not isinstance(c, dict) for c in pair):
        return {**report, "reason": "offline_source_pair_missing"}
    keys = clause_radial.BINDING_KEYS
    old_pair = [{key: c.get(key) for key in keys} for c in pair]
    if old_pair != recipe["source_pair"]:
        return {**report, "reason": "selected_pair_not_in_offline_matrix", "offline_source_pair": old_pair}
    peers = [r for r in historical if r.get("label") == recipe["component"]]
    if len(peers) != 1:
        return {**report, "reason": "offline_component_missing_or_ambiguous"}
    old = peers[0]
    old_result = old["result"]
    if old_result.get("component_recipe_id"):
        report["historical_binding_status"] = "receipt_binding_present"
    if old_result.get("status") != "ok" or measured.get("status") != "ok":
        return {**report, "status": "invalid_measurement", "reason": "measurement_not_ok"}
    rows, old_rows = measured.get("operator_rows", []), old_result.get("operator_rows", [])
    if not rows or len(rows) != len(old_rows) or len(rows) != len(recipe["surfaces"]):
        return {**report, "status": "mismatch", "reason": "surface_count_differs"}
    inputs = [r.get("inputs", []) for r in historical[:2]]
    if any(len(items) != len(rows) for items in inputs):
        return {**report, "status": "mismatch", "reason": "raw_source_inputs_missing"}
    source_inputs = [{"left": left["source"], "right": right["source"], "target": left["target"]}
                     for left, right in zip(*inputs, strict=True)]
    checks = {
        "raw_source_legs_and_target": source_inputs == recipe["source_input_identities"]
            and all(a["target"] == b["target"] for a, b in zip(*inputs, strict=True)),
        "compiled_inputs": old.get("inputs") == recipe["input_identities"],
        "source_expressions": old.get("component_definition", {}).get("source_expressions")
            == recipe["source_expressions"],
        "control_id": old_result.get("control_id") == measured.get("control_id"),
        "restored_and_caps_unchanged": all(r.get("state_restored") is True and r.get("caps_unchanged") is True
                                            for r in (old_result, measured)),
    }
    if old_result.get("component_recipe_id"):
        checks["recorded_binding_ids"] = all(old_result.get(k) == recipe[k]
                                               for k in ("component_recipe_id", "source_pair_id"))
    for key in ("surface", "edit", "context_fingerprint", "candidate_fingerprint", "source_tensor_identity"):
        checks[key] = all(a.get(key) == b.get(key) for a, b in zip(old_rows, rows, strict=True))
    response_checks = {key: all(a.get(key) == b.get(key) for a, b in zip(old_rows, rows, strict=True))
                       for key in ("structure_response_rows", "runtime_telemetry", "baseline_repeat_variation",
                                   "edited_repeat_variation", "safety_deltas")}
    return {**report, "status": "matched" if all(checks.values()) else "mismatch",
            "identity_checks": checks, "readout_and_execution_exact_checks": response_checks,
            "readout_and_execution_exact_match": all(response_checks.values()),
            "failed_identity_checks": [k for k, v in checks.items() if not v]}


def validate_selected_execution(recipe: dict, result: dict, inputs: list[dict]) -> dict:
    rows = result.get("operator_rows", [])
    checks = {
        "measurement_ok": result.get("status") == "ok",
        "binding_ids": all(result.get(k) == recipe[k] for k in ("source_pair_id", "component_recipe_id")),
        "recipe": result.get("recipe") == recipe,
        "restored": result.get("state_restored") is True and result.get("caps_unchanged") is True,
        "no_permission": result.get("production_apply_allowed") is False,
        "row_count": len(rows) == len(recipe["surfaces"]) and bool(rows),
        "compiled_inputs": [{k: tensor_identity(v) for k, v in r.items()} for r in inputs]
            == recipe["input_identities"],
    }
    checks["actual_rows"] = checks["row_count"] and all(
        row.get("source_pair_id") == recipe["source_pair_id"]
        and row.get("component_recipe_id") == recipe["component_recipe_id"]
        and row.get("surface") == recipe["surfaces"][i]
        and row.get("source_tensor_identity") == recipe["input_identities"][i]["source"]
        and row.get("edit", {}).get("source", {}).get("expr") == recipe["source_expressions"][i]
        and row.get("edit", {}).get("target") == {"surface_id": recipe["surfaces"][i]["surface_id"]}
        and row.get("edit", {}).get("op") == {"kind": "activation_patch", "mode": "blend",
                                               "alpha": recipe["surfaces"][i]["alpha"]}
        and row.get("edit", {}).get("budget") == {"ttl_steps": 1, "revertible": True,
                "step_size": recipe["surfaces"][i]["step_size"], "norm_clip": recipe["surfaces"][i]["norm_clip"]}
        and row.get("production_apply_allowed") is False
        and len(row.get("runtime_telemetry", [])) == 2
        and all(len(repeat) == 1 and repeat[0].get("hook_call_count") == 1
                and repeat[0].get("op") == "activation_patch"
                and repeat[0].get("edit_id") == row["edit"]["id"]
                and repeat[0].get("selected_token_count") == 1
                and 0 <= repeat[0].get("blend_delta_norm", -1) <= recipe["surfaces"][i]["step_size"] + 1e-6
                for repeat in row.get("runtime_telemetry", [])) for i, row in enumerate(rows))
    return {"status": "matched" if all(checks.values()) else "mismatch", "checks": checks}


def recorded_command(choice: dict, packet: dict) -> ControllerCommand:
    """No ID rebinding: accept only an intact choice of the exact current packet."""
    old_packet = choice["packet"]
    if stable_hash(old_packet) != choice.get("packet_sha256") or stable_hash(packet) != choice["packet_sha256"]:
        raise ValueError("recorded_choice_packet_mismatch")
    menu, _ = operation_cards.build_menu(old_packet)
    if menu != choice.get("operation_menu"):
        raise ValueError("recorded_choice_menu_mismatch")
    trace = choice.get("controller_trace") or {}
    attempts = trace.get("attempts", [])
    if (choice.get("provider_calls_attempted") != 1 or len(attempts) != 1
            or attempts[0].get("parse_ok") is not True or trace.get("operation_menu") != menu):
        raise ValueError("recorded_choice_has_no_single_verified_provider_response")
    command = ControllerCommand.from_dict(choice["command"])
    for field in ("raw_json_object", "normalized_payload"):
        if asdict(ControllerCommand.from_dict(attempts[0][field])) != asdict(command):
            raise ValueError("recorded_choice_command_changed")
    operation_cards.validate_card_command(packet, asdict(command))
    if _extract_diagnostic_requests(command, packet) != choice.get("resolved_requests"):
        raise ValueError("recorded_choice_request_changed")
    return command


def validate_source_execution(candidate: dict, result: dict, inputs: list[dict], edits: list[dict], context: str) -> dict:
    rows = result.get("operator_rows", [])
    binding = {k: v for k, v in candidate.items() if k not in {"request", "available"}}
    checks = {
        "measurement_ok": result.get("status") == "ok",
        "candidate_binding": result.get("candidate_binding") == binding,
        "operator_id": result.get("structure_operator_id") == candidate["structure_operator_id"],
        "restored": result.get("state_restored") is True and result.get("caps_unchanged") is True,
        "no_permission": all(result.get(k) is False for k in ("production_apply_allowed", "certified_for_apply", "policy_candidate_ready")),
        "row_count": bool(rows) and len(rows) == len(candidate["surfaces"]) == len(edits) == len(inputs),
    }
    checks["actual_rows"] = checks["row_count"] and all(
        row.get("edit") == edits[i] and row.get("surface") == candidate["surfaces"][i]
        and row.get("context_fingerprint") == context
        and row.get("source_tensor_identity") == tensor_identity(inputs[i]["source"])
        and row.get("candidate_fingerprint") == identity("clause-operator:", [
            candidate, edits[i], tensor_identity(inputs[i]["source"]), context])
        and row.get("production_apply_allowed") is False
        and row.get("target_readout_effect_certified") is False
        and len(row.get("runtime_telemetry", [])) == 2
        and all(len(repeat) == 1 and repeat[0].get("hook_call_count") == 1
                and repeat[0].get("op") == "activation_patch" and repeat[0].get("edit_id") == edits[i]["id"]
                and repeat[0].get("selected_token_count") == 1
                and 0 <= repeat[0].get("blend_delta_norm", -1) <= candidate["surfaces"][i]["step_size"] + 1e-6
                for repeat in row.get("runtime_telemetry", [])) for i, row in enumerate(rows))
    return {"status": "matched" if all(checks.values()) else "mismatch", "checks": checks}


def compare_source_offline(candidate: dict, measured: dict, inputs: list[dict], historical: list[dict]) -> dict:
    binding = {k: v for k, v in candidate.items() if k not in {"request", "available"}}
    peers = [r for r in historical if r.get("result", {}).get("candidate_binding") == binding]
    base = {"scope": "single_source_not_a_radial_pair", "production_apply_allowed": False}
    if len(peers) != 1:
        return {**base, "status": "not_covered", "reason": "exact_source_candidate_not_in_offline_matrix"}
    old = peers[0]
    rows, old_rows = measured.get("operator_rows", []), old["result"].get("operator_rows", [])
    checks = {
        "successful_restored_measurements": all(r.get("status") == "ok" and r.get("state_restored") is True
            and r.get("caps_unchanged") is True for r in (measured, old["result"])),
        "compiled_inputs": old.get("inputs") == [{k: tensor_identity(v) for k, v in row.items()} for row in inputs],
        "row_count": bool(rows) and len(rows) == len(old_rows),
    }
    for key in ("edit", "surface", "context_fingerprint", "candidate_fingerprint", "source_tensor_identity"):
        checks[key] = checks["row_count"] and all(a.get(key) == b.get(key) for a, b in zip(rows, old_rows))
    response_keys = ("structure_response_rows", "runtime_telemetry", "baseline_repeat_variation",
                     "edited_repeat_variation", "safety_deltas")
    return {**base, "status": "matched" if all(checks.values()) else "mismatch", "identity_checks": checks,
            "readout_and_execution_exact_match": checks["row_count"] and all(
                all(a.get(k) == b.get(k) for k in response_keys) for a, b in zip(rows, old_rows))}


def run_choice(worker: Any, provider: Any, report: dict, *, historical: list[dict],
               save: Any, record: Any, stages: list[str] | None = None,
               allow_source_cards: bool = False, recorded_choice: dict | None = None) -> None:
    """At most one API attempt and one diagnostic dispatch; no generation commit."""
    before, step, used = state_identity(worker), worker._steps, diagnostic_budget.used(worker)
    packet = _payload_for_provider(prefix_control.annotate(
        worker.build_controller_packet(), rounds_left=2, terminal=worker.done()), packet_view="full")
    menu, _ = operation_cards.build_menu(packet)
    recipes = {r["component_recipe_id"]: r for r in clause_radial.offer(worker)["recipes"] if r["available"]}
    cards = [c for c in menu["cards"] if c["diagnostic"] == clause_radial.NAME and c["available"]]
    if not cards or {c["component_recipe_id"] for c in cards} != set(recipes):
        raise ValueError("radial_cards_missing_or_truncated")
    for card in cards:
        recipe = recipes[card["component_recipe_id"]]
        if any(card.get(key) != recipe[key] for key in (
                "source_pair_id", "source_pair", "component", "dsl_formula", "structure_context_id", "surfaces")):
            raise ValueError("card_label_differs_from_recipe")
    report.update(status="prepared", packet=packet, packet_sha256=stable_hash(packet), operation_menu=menu,
                  provider_call_limit=1, provider_calls_attempted=0, diagnostics_dispatched=0,
                  historical_outcomes_visible_to_controller=False, production_apply_allowed=False,
                  scope="single_choice_binding_check_not_controller_efficacy", radial_recipe_count=len(recipes),
                  harness_dispatch_scope=[clause_radial.NAME, clause_actuation.NAME] if allow_source_cards else [clause_radial.NAME],
                  selection_origin="recorded_controller_choice_replay" if recorded_choice is not None else "live_controller_choice")
    client = None if recorded_choice is not None else ProviderControllerClient(
        provider, action_view="cards", packet_view="compact", max_attempts=1)
    if recorded_choice is not None:
        report.update(provider_call_limit=0, recorded_choice_sha256=stable_hash(recorded_choice))
    save()
    try:
        if recorded_choice is not None:
            command = recorded_command(recorded_choice, packet)
            report["recorded_packet_match"] = True
        else:
            report.update(status="requesting_controller", provider_calls_attempted=1)
            save()
            try:
                command = client.invoke(packet)
            except Exception as exc:
                report.update(status="controller_error", error_type=type(exc).__name__)
                return
        report["command"] = asdict(command)
        if client is not None and client.latest_trace()["operation_menu"] != menu:
            raise ValueError("provider_menu_differs_from_frozen_cards")
        if before != state_identity(worker) or worker._steps != step or stable_hash(packet) != report["packet_sha256"]:
            raise ValueError("controller_changed_prefix_or_packet")
        requests = _extract_diagnostic_requests(command, packet)
        report["resolved_requests"] = requests
        operation_id = command.meta.get("operation_id")
        selected_card = next((c for c in menu["cards"] if c["operation_id"] == operation_id), None)
        selected = next((c for c in cards if c["operation_id"] == operation_id), None)
        report["selected_operation_card"] = selected_card
        report["selected_radial_operation_card"] = selected
        if command.decision == "apply" or command.edits:
            report["status"] = "apply_not_dispatched_by_diagnostic_harness"
            return
        if selected_card is not None and selected_card["diagnostic"] == clause_actuation.NAME and allow_source_cards:
            if len(requests) != 1 or requests[0]["diagnostic"] != clause_actuation.NAME:
                raise ValueError("source_card_request_mismatch")
            candidate = next((c for c in clause_actuation.offer(worker)["candidates"]
                              if c["request"] == requests[0] and c["available"]), None)
            if candidate is None or selected_card.get("target_piece") != candidate["target_piece"]:
                raise ValueError("source_card_binding_mismatch")
            edits = [clause_actuation.materialize(worker, candidate, s) for s in candidate["surfaces"]]
            ctx = StepContext(packet=packet, runtime_state=worker.runtime_state, adapter=worker.adapter,
                              traces={}, stats={}, active_edits={})
            inputs = [{"source": compile_expr(edit["source"]["expr"])(ctx).detach().cpu().clone(),
                       "target": worker.runtime_state.last_cache[s["cache_hook_name"]][0, -1].detach().cpu().clone()}
                      for edit, s in zip(edits, candidate["surfaces"], strict=True)]
            report.update(status="measuring_selected_source_card", selected_source_candidate=deepcopy(candidate),
                          expected_edits=deepcopy(edits), diagnostics_dispatched=1)
            save()
            stages = stages if stages is not None else clause_trajectory.stage_names(worker.runtime_state.model.cfg.n_layers)
            with clause_trajectory.capture_probe(worker, worker._current_token_tensor().detach().clone(), stages) as frames:
                results = worker.request_controller_diagnostics(requests, source=report["selection_origin"], packet=packet)
            if len(results) != 1:
                raise ValueError("expected_one_selected_source_receipt")
            measured = results[0]
            record({"label": f"source:{candidate['target_token_id']}", "kind": "source",
                    "result": measured, "inputs": inputs, "frames": frames})
            report["selected_execution_comparison"] = validate_source_execution(candidate, measured, inputs, edits, before)
            report["offline_comparison"] = compare_source_offline(candidate, measured, inputs, historical)
            report["status"] = "complete" if report["selected_execution_comparison"]["status"] == "matched" else "execution_mismatch"
            return
        if selected_card is not None and selected_card["diagnostic"] != clause_radial.NAME:
            report.update(status="non_radial_selection_not_dispatched",
                          dispatch_block_reason="outside_predeclared_source_and_radial_scope" if allow_source_cards
                          else "outside_predeclared_radial_only_scope")
            return
        if selected is None or len(requests) != 1 or requests[0] != recipes[selected["component_recipe_id"]]["request"]:
            report["status"] = "no_radial_selection"
            return
        recipe = deepcopy(recipes[selected["component_recipe_id"]])
        ctx = StepContext(packet=packet, runtime_state=worker.runtime_state, adapter=worker.adapter,
                          traces={}, stats={}, active_edits={})
        inputs = [{"source": compile_expr(expr)(ctx).detach().cpu().clone(),
                   "target": worker.runtime_state.last_cache[surface["cache_hook_name"]][0, -1].detach().cpu().clone()}
                  for expr, surface in zip(recipe["source_expressions"], recipe["surfaces"], strict=True)]
        report.update(status="measuring_selected_card", selected_recipe=recipe, diagnostics_dispatched=1)
        save()
        tokens = worker._current_token_tensor().detach().clone()
        stages = stages if stages is not None else clause_trajectory.stage_names(worker.runtime_state.model.cfg.n_layers)
        with clause_trajectory.capture_probe(worker, tokens, stages) as frames:
            results = worker.request_controller_diagnostics(requests, source=report["selection_origin"], packet=packet)
        if len(results) != 1:
            raise ValueError("expected_one_selected_recipe_receipt")
        measured = results[0]
        item = {"label": recipe["component"], "kind": "component", "result": measured, "inputs": inputs,
                "frames": frames, "component_definition": {
                    "source_token_ids": [c["target_token_id"] for c in recipe["source_pair"]],
                    **{k: recipe[k] for k in ("source_pair_id", "component_recipe_id", "source_expressions", "dsl_formula")}}}
        record(item)
        report["selected_execution_comparison"] = validate_selected_execution(recipe, measured, inputs)
        report["offline_comparison"] = compare_offline(recipe, measured, historical)
        report["status"] = "complete" if report["selected_execution_comparison"]["status"] == "matched" else "execution_mismatch"
    finally:
        report["controller_trace"] = client.latest_trace() if client is not None else None
        report["context_unchanged"] = before == state_identity(worker) and step == worker._steps
        report["packet_unchanged"] = stable_hash(packet) == report["packet_sha256"]
        report["diagnostic_slots_used"] = diagnostic_budget.used(worker) - used
        report["rollout_edit_count"] = len(worker._collect_active_edits())
        save()
        if not report["context_unchanged"] or not report["packet_unchanged"] or report["rollout_edit_count"]:
            raise ValueError("choice_measurement_changed_prefix_packet_or_active_edits")
