"""Bounded, dose-matched measurements; no selection or apply authority."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .compiler import StepContext, compile_expr


def identity(prefix: str, value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return prefix + hashlib.sha256(encoded.encode()).hexdigest()[:24]


def tensor_identity(value: torch.Tensor) -> dict[str, Any]:
    data = value.detach().cpu().contiguous()
    return {"shape": list(data.shape), "dtype": str(data.dtype),
            "sha256": hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}


def state_identity(worker: Any) -> str:
    state = worker.runtime_state
    return identity("context:", {
        "tokens": tensor_identity(worker._current_token_tensor()),
        "model": str(getattr(getattr(state, "model", None), "cfg", "unknown")),
        "codec": type(worker.codec).__qualname__,
        "patches": worker._collect_active_edits(),
        "last_logits": tensor_identity(state.last_logits) if isinstance(getattr(state, "last_logits", None), torch.Tensor) else None,
        "cache": {str(k): tensor_identity(v) for k, v in (getattr(state, "last_cache", None) or {}).items()},
    })


class ReadoutUnavailable(ValueError):
    def __init__(self, reason: str, details: dict[str, Any]):
        super().__init__(reason)
        self.details = details


def bound_metrics(before: torch.Tensor, after: torch.Tensor, token_id: int) -> dict[str, Any]:
    # Audit arithmetic runs on CPU: MPS does not support float64. Model replay
    # stays on its configured device; casting cannot recover lost precision.
    before, after = before.detach().cpu().flatten().double(), after.detach().cpu().flatten().double()
    width = min(20, before.numel())
    details = {"readout_stage": "post_constraints_and_decoder_control", "target_token_id": token_id,
        "before_finite_count": int(torch.isfinite(before).sum()), "after_finite_count": int(torch.isfinite(after).sum()),
        "before_target_masked": bool(torch.isneginf(before[token_id])),
        "after_target_masked": bool(torch.isneginf(after[token_id]))}
    if torch.isnan(before).any() or torch.isnan(after).any() or torch.isposinf(before).any() or torch.isposinf(after).any():
        raise ReadoutUnavailable("nonfinite_bound_token_or_top20_readout", details)
    if details["before_target_masked"] or details["after_target_masked"]:
        raise ReadoutUnavailable("bound_target_masked_in_decode_readout", details)
    if details["before_finite_count"] < width or details["after_finite_count"] < width:
        raise ReadoutUnavailable("insufficient_finite_tokens_for_top20_readout", details)
    threshold_before = float(before.topk(width).values[-1])
    threshold_after = float(after.topk(width).values[-1])
    logit_delta = float(after[token_id] - before[token_id])
    threshold_delta = threshold_after - threshold_before
    return {
        "target_piece_logit_delta": logit_delta,
        "target_piece_prob_delta": float(after.softmax(0)[token_id] - before.softmax(0)[token_id]),
        "threshold20_logit_delta": threshold_delta,
        "target_top20_threshold_gap_delta": threshold_delta - logit_delta,
        "target_rank_before": int((before > before[token_id]).sum()) + 1,
        "target_rank_after": int((after > after[token_id]).sum()) + 1,
        "target_rank_delta": int((before > before[token_id]).sum() - (after > after[token_id]).sum()),
        "bound_token_top20_hit_delta": int(token_id in after.topk(width).indices) - int(token_id in before.topk(width).indices),
    }


def logit_variation(before: torch.Tensor, after: torch.Tensor) -> dict[str, Any]:
    before, after = before.detach().cpu().double(), after.detach().cpu().double()
    if before.shape != after.shape:
        raise ValueError("control_logit_shape_changed")
    if torch.isnan(before).any() or torch.isnan(after).any() or torch.isposinf(before).any() or torch.isposinf(after).any():
        raise ValueError("control_nonfinite_logits")
    masked = torch.isneginf(before)
    if not torch.equal(masked, torch.isneginf(after)):
        raise ValueError("control_token_mask_changed")
    finite = ~masked
    if not finite.any():
        raise ValueError("control_has_no_finite_logits")
    return {"max_abs_logit_delta": float((before[finite] - after[finite]).abs().max()),
            "masked_token_count": int(masked.sum()), "finite_token_count": int(finite.sum()),
            "mask_identical": True}


def validate_doses(values: Any) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 2:
        raise ValueError("dose_grid must contain one or two supported doses")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or float(x) not in (0.04, 0.16) for x in values):
        raise ValueError("dose_grid supports only 0.04 and 0.16")
    if len(set(values)) != len(values):
        raise ValueError("duplicate dose")
    return tuple(float(x) for x in values)


def with_ownership(worker: Any, replay: Mapping[str, Any], objective: str, term: str) -> dict[str, Any]:
    """Reuse the runtime ownership classifier on this execution's term deltas."""
    deltas = replay.get("term_readout_deltas")
    if not isinstance(deltas, Mapping) or not deltas:
        return {**replay, "actuator_class": "unknown", "ownership_scope": "missing_term_readout_deltas"}
    terms = {objective: term}
    for other in deltas:
        if str(other).casefold() != term.casefold():
            terms[f"entity_insert:{str(other).casefold()}"] = str(other)
    summaries = worker._summarize_operator_recipe_bundle_ownership(
        [{**replay, "intended_bundle_key": objective}], bundle_term_by_key=terms)
    if not summaries:
        return {**replay, "actuator_class": "unknown", "ownership_scope": "missing_recipe_identity"}
    ownership = summaries[0]
    return {**replay, **{k: ownership[k] for k in ("self_delta", "cross_delta", "alignment_margin",
        "actuator_class", "realized_lift_bundle_key")}, "ownership_scope": "execution_measured_term_set"}


def select_probe_seeds(seeds: Sequence[Mapping[str, Any]], *, objective_bundle_key: str,
                       objective_term: str, candidate_ids: Sequence[str] = (),
                       comparison_axis: str = "seed_provenance") -> tuple[list[dict[str, Any]], str | None]:
    """Preflight recorded seeds; binding and state restoration still require replay."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for raw in seeds:
        if not isinstance(raw, Mapping) or raw.get("objective_bundle_key", raw.get("bundle_key")) != objective_bundle_key:
            continue
        if raw.get("operator_axis") == "target_piece_binding_seed_matrix" and not (
            raw.get("execution_id") and raw.get("source_tensor_identity")
        ):
            continue
        if candidate_ids and raw.get("operator_recipe_id") not in candidate_ids:
            continue
        source = raw.get("activation_patch_seed_source") or raw.get("seed_source")
        if raw.get("activation_patch_forced_seed") or raw.get("activation_patch_seed_discovery"):
            continue
        if not source and raw.get("operator_axis") == "activation_patch_blueprint_materialization":
            source = "direct_candidate"
        if not source and raw.get("operator_axis") in {"activation_patch_local_step_size_sweep", "activation_patch_cap_release_response_curve"}:
            source = "observed_gap_carrier"
        if source not in {"direct_candidate", "observed_gap_carrier"}:
            continue
        if raw.get("activation_patch_site") not in {"resid_pre", "resid_post", "mlp_out"} or raw.get("activation_patch_layer") is None:
            continue
        alpha = raw.get("activation_patch_alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha <= 0.15:
            continue
        if raw.get("actual_delta_class") in {"harmful", "collapse_sharpener", "collapse_isomorphic", "replay_error"}:
            continue
        source_term = str(raw.get("intended_term") or raw.get("objective_term") or objective_term)
        if source_term.casefold() != objective_term.casefold():
            continue
        groups.setdefault(str(source), []).append({**raw, "seed_source": source})
    selected = [max(groups[s], key=lambda r: -float(r.get("target_top20_threshold_gap_delta") or 0.0))
                for s in ("direct_candidate", "observed_gap_carrier") if s in groups]
    if not selected:
        return [], "no_materializable_activation_patch_seed"
    if comparison_axis == "source_localization":
        # One recorded anchor fixes every axis other than source construction.
        anchor = selected[0]
        selected = []
        for localization in ("source_term_token", "source_centered_pm1"):
            recipe_id = identity("source_variant:", [anchor.get("operator_recipe_id"),
                objective_bundle_key, anchor["activation_patch_site"], anchor["activation_patch_layer"],
                anchor["activation_patch_alpha"], localization])
            selected.append({**anchor, "seed_recipe_name": anchor.get("recipe_name"),
                "seed_operator_recipe_id": anchor.get("operator_recipe_id"),
                "recipe_name": f"matched_{localization}", "operator_recipe_id": recipe_id,
                "source_variant": localization, "source_variant_origin": "controlled_localization_variant",
                "activation_patch_source_localization": localization,
                "activation_patch_contrast_mode": "none", "activation_patch_contrast_scale": 0.0,
                "activation_patch_stealer_term": None, "activation_patch_stealer_bundle_key": None})
    controls = ("activation_patch_site", "activation_patch_layer", "activation_patch_alpha")
    if comparison_axis == "seed_provenance":
        controls += ("activation_patch_source_localization",)
    if any(tuple(r.get(k) for k in controls) != tuple(selected[0].get(k) for k in controls) for r in selected[1:]):
        return [], "seed_site_layer_alpha_localization_mismatch"
    return selected, None


def matched_response_probe(worker: Any, seeds: Sequence[Mapping[str, Any]], *,
                           objective_bundle_key: str, objective_term: str,
                           dose_grid: Any = (0.04, 0.16), candidate_ids: Any = (),
                           comparison_axis: str = "seed_provenance") -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "unavailable", "operator_axis": "target_piece_binding_seed_matrix",
        "measurement_mode": "matched_response_probe", "rows": [], "row_count": 0,
        "comparison_axis": comparison_axis,
        "objective_bundle_key": objective_bundle_key, "diagnostic_only": True,
        "production_apply_allowed": False, "certified_for_apply": False, "policy_candidate_ready": False,
        "physical_replay_count": 0, "new_measurement_count": 0, "cached_measurement_count": 0,
    }
    try:
        if not isinstance(comparison_axis, str) or comparison_axis not in {"seed_provenance", "source_localization"}:
            raise ValueError("comparison_axis must be seed_provenance or source_localization")
        doses = validate_doses(dose_grid)
        if not isinstance(candidate_ids, (list, tuple)) or len(candidate_ids) > 2 or any(not isinstance(x, str) or not 1 <= len(x) <= 512 for x in candidate_ids):
            raise ValueError("candidate_ids must contain at most two recipe IDs")
    except ValueError as exc:
        return {**report, "status": "invalid_request", "unavailable_reason": str(exc)}
    selected, unavailable_reason = select_probe_seeds(seeds, objective_bundle_key=objective_bundle_key,
        objective_term=objective_term, candidate_ids=candidate_ids, comparison_axis=comparison_axis)
    if unavailable_reason:
        return {**report, "unavailable_reason": unavailable_reason}
    report["requested_objective_term"] = objective_term
    objective_term = str(selected[0].get("intended_term") or selected[0].get("objective_term") or objective_term)
    report["objective_term"] = objective_term
    context_id = state_identity(worker)
    report["measurement_context_id"] = context_id
    report["dose_grid"] = list(doses)
    baseline = worker._simulate_decode(max_new_tokens=1, top_k=6, score_candidate_text=False, score_observer_check=False)
    report["physical_replay_count"] += 1
    if not isinstance(baseline, Mapping) or not isinstance(baseline.get("first_logits"), torch.Tensor):
        return {**report, "unavailable_reason": "baseline_failed"}
    before = baseline["first_logits"]
    canonical = worker._resolve_target_piece_binding(before, focus_terms=(objective_term,), preferred_term=objective_term)
    if not canonical:
        return {**report, "unavailable_reason": "no_binding"}
    alternate_rows = [r for r in canonical.get("candidate_target_piece_rows", ()) if r.get("token_id") != canonical.get("chosen_target_token_id")]
    if not alternate_rows:
        return {**report, "unavailable_reason": "no_alternate_binding"}
    alternate = worker._resolve_target_piece_binding(before, focus_terms=(objective_term,), preferred_term=objective_term,
        requested_binding={"objective_term": objective_term, "chosen_target_token_id": min(alternate_rows, key=lambda r: (r.get("baseline_rank", 10**9), r["token_id"]))["token_id"]})
    bindings = [("canonical", canonical), ("alternate", alternate)]
    if not alternate:
        return {**report, "unavailable_reason": "alternate_binding_failed"}
    null = worker._simulate_decode(max_new_tokens=1, top_k=6, score_candidate_text=False, score_observer_check=False)
    report["physical_replay_count"] += 1
    if not isinstance(null, Mapping) or not isinstance(null.get("first_logits"), torch.Tensor):
        return {**report, "unavailable_reason": "null_control_failed"}
    try:
        report["no_edit_control"] = logit_variation(before, null["first_logits"])
    except ValueError as exc:
        return {**report, "unavailable_reason": str(exc)}
    report["no_edit_max_abs_logit_delta"] = report["no_edit_control"]["max_abs_logit_delta"]
    cache: dict[str, tuple[dict[str, Any], torch.Tensor, float]] = {}
    alias_executions = 0
    matrix_id = identity("tpbsm:", [context_id, objective_bundle_key, doses, selected, [b["binding_id"] for _, b in bindings]])
    errors = []
    for seed in selected:
        for dose in doses:
            candidate = {
                "objective_bundle_key": objective_bundle_key, "actuator_bundle_key": objective_bundle_key,
                "bundle_key": objective_bundle_key, "objective_term": objective_term,
                "layer": seed["activation_patch_layer"], "site": seed["activation_patch_site"],
                "alpha": seed["activation_patch_alpha"], "step_size": dose,
                "source_localization": seed["activation_patch_source_localization"], "patch_mode": "blend",
                "contrast_mode": seed.get("activation_patch_contrast_mode"),
                "contrast_scale": seed.get("activation_patch_contrast_scale"),
                "stealer_term": seed.get("activation_patch_stealer_term"),
                "stealer_bundle_key": seed.get("activation_patch_stealer_bundle_key"),
                "recipe_name": seed.get("recipe_name"), "operator_recipe_id": seed.get("operator_recipe_id"),
            }
            # The seed label can encode a different cap. Preserve lineage while
            # naming the executable recipe by its actual controlled parameters.
            candidate["seed_operator_recipe_id"] = seed.get("operator_recipe_id")
            candidate["operator_recipe_id"] = identity("matched_recipe:", candidate)
            candidate["recipe_name"] = (
                f"matched_{candidate['site']}_l{candidate['layer']}_{candidate['source_localization']}"
                f"_alpha={candidate['alpha']:.4f}_step_size={dose:.4f}")
            try:
                edit = worker._activation_patch_trial_edit_from_candidate(candidate, trial_contract={
                    "max_alpha": max(0.08, float(candidate["alpha"])), "norm_clip": 1.0,
                    "trial_budget_class": "diagnostic_only", "allow_step_size_cap_release": True,
                    "max_step_size": dose, "production_trial_followup_allowed": False,
                })
                if edit is None:
                    raise ValueError("materialization_failed")
                if edit["op"].get("alpha") != candidate["alpha"] or edit["budget"].get("step_size") != dose:
                    raise ValueError("materialized_alpha_or_dose_mismatch")
                for surface in getattr(worker, "surface_catalog", ()):
                    if surface.surface_id == edit["target"].get("surface_id"):
                        if surface.target.site != candidate["site"] or surface.target.layer != candidate["layer"]:
                            raise ValueError("materialized_site_layer_mismatch")
                edit.update(bundle_key=objective_bundle_key, focus_feature=objective_term, phase_objective="readout_escape")
                packet = worker.build_controller_packet()
                ctx = StepContext(packet=packet, runtime_state=worker.runtime_state, adapter=worker.adapter, traces={}, stats={}, active_edits={})
                source_tensor = compile_expr(dict(edit["source"]["expr"]))(ctx)
                if not torch.isfinite(source_tensor).all():
                    raise ValueError("nonfinite_source_tensor")
                execution_id = identity("exec:", [context_id, edit["target"], edit["op"], edit["budget"], tensor_identity(source_tensor)])
                alias = execution_id in cache
                alias_executions += int(alias)
                if not alias:
                    captured: dict[str, Any] = {}
                    replay = worker.replay_candidate_edits_actual_delta([edit], max_new_tokens=1, top_k=6,
                        score_candidate_text=False, max_edits_per_step_override=1,
                        intended_bundle_key=objective_bundle_key, intended_term=objective_term,
                        ownership_terms=(objective_term,), target_piece_binding=canonical,
                        _baseline_snapshot=baseline, _measurement_capture=captured)
                    report["physical_replay_count"] += 1
                    if replay.get("status") != "ok" or "edited_logits" not in captured:
                        raise ValueError(replay.get("error") or "replay_failed")
                    replay = with_ownership(worker, replay, objective_bundle_key, objective_term)
                    repeated: dict[str, Any] = {}
                    repeat = worker.replay_candidate_edits_actual_delta([edit], max_new_tokens=1, top_k=6,
                        score_candidate_text=False, max_edits_per_step_override=1,
                        intended_bundle_key=objective_bundle_key, intended_term=objective_term,
                        target_piece_binding=canonical, _baseline_snapshot=baseline, _measurement_capture=repeated)
                    report["physical_replay_count"] += 1
                    if repeat.get("status") != "ok" or "edited_logits" not in repeated:
                        raise ValueError("repeat_control_failed")
                    after = captured["edited_logits"]
                    noise = logit_variation(after, repeated["edited_logits"])["max_abs_logit_delta"]
                    cache[execution_id] = (replay, after, noise)
                replay, after, noise = cache[execution_id]
                for variant, binding in bindings:
                    token_id = int(binding["chosen_target_token_id"])
                    metrics = worker._first_token_target_readout_metrics(before, after, focus_terms=(objective_term,), preferred_term=objective_term, target_piece_binding=binding)
                    row = {k: v for k, v in replay.items() if k.startswith("activation_patch_") or k in (
                        "entropy_delta", "repeat_flag_delta", "top1_margin_delta", "repetition_score_delta", "actual_delta_class",
                        "actuator_class", "self_delta", "cross_delta", "alignment_margin", "realized_lift_bundle_key",
                        "ownership_scope", "term_readout_deltas",
                        "required_term_recall_delta", "required_term_span_progress_delta", "semantic_progress_delta")}
                    row.update(metrics)
                    row.update(bound_metrics(before, after, token_id))
                    tolerance = max(noise, report["no_edit_max_abs_logit_delta"])
                    logit_delta = row["target_piece_logit_delta"]
                    row.update({
                        "status": "ok", "operator_axis": "target_piece_binding_seed_matrix", "diagnostic_family": "activation_patch",
                        "evidence_kind": "target_piece_binding_replay",
                        "objective_bundle_key": objective_bundle_key, "bundle_key": objective_bundle_key, "intended_term": objective_term,
                        "recipe_name": candidate["recipe_name"], "operator_recipe_id": candidate["operator_recipe_id"],
                        "seed_recipe_name": seed.get("seed_recipe_name") or seed.get("recipe_name"),
                        "seed_operator_recipe_id": seed.get("seed_operator_recipe_id") or seed.get("operator_recipe_id"),
                        "seed_source": seed.get("seed_source"),
                        "source_variant": seed.get("source_variant"), "source_variant_origin": seed.get("source_variant_origin"),
                        "comparison_axis": comparison_axis,
                        "activation_patch_site": candidate["site"], "activation_patch_layer": candidate["layer"],
                        "activation_patch_alpha": candidate["alpha"], "activation_patch_step_size": dose,
                        "activation_patch_source_localization": candidate["source_localization"],
                        "target_piece_binding_seed_matrix_id": matrix_id, "target_piece_binding_variant": variant,
                        "target_piece_binding_requested_honored": metrics.get("target_piece_token_id") == token_id,
                        "execution_id": execution_id, "observable_id": identity("obs:", [execution_id, binding["binding_id"]]),
                        "measurement_context_id": context_id, "execution_alias": alias,
                        "seed_reference_observable_id": seed.get("observable_id"),
                        "repeat_max_abs_logit_delta": noise, "evidence_scope": "frozen_prefix_diagnostic",
                        "no_edit_max_abs_logit_delta": report["no_edit_max_abs_logit_delta"],
                        "candidate_descriptor": dict(candidate),
                        "measured_edit": {key: edit[key] for key in ("target", "op", "source", "budget")},
                        "actual_delta_class_scope": "execution_canonical_binding_and_term_set",
                        "measurement_focus_terms": [objective_term],
                        "bound_token_response": ("positive_above_repeat_variation" if logit_delta > tolerance
                            else "negative_beyond_repeat_variation" if logit_delta < -tolerance else "within_repeat_variation"),
                        "target_piece_binding_manifest": dict(binding), "source_tensor_identity": tensor_identity(source_tensor),
                        "diagnostic_only": True, "production_apply_allowed": False, "certified_for_apply": False, "policy_candidate_ready": False,
                    })
                    report["rows"].append(row)
            except Exception as exc:
                errors.append({"recipe_id": candidate.get("operator_recipe_id"), "dose": dose, "error": f"{type(exc).__name__}:{exc}"})
    restored = context_id == state_identity(worker)
    report.update(status="dose_matched_response_complete" if not errors and restored else "incomplete",
        row_count=len(report["rows"]), target_piece_binding_seed_matrix_id=matrix_id,
        seed_sources=list(dict.fromkeys(r["seed_source"] for r in report["rows"])),
        state_restored=restored, errors=errors, new_measurement_count=len(cache),
        cached_measurement_count=alias_executions,
        aliased_observation_count=sum(int(r["execution_alias"]) for r in report["rows"]),
        observable_count=len({r["observable_id"] for r in report["rows"]}),
        dominant_logit_response_axis="not_inferred_from_dose_confounded_aggregate",
        term_mass_is_binding_invariant_by_definition=True)
    if not restored:
        report["rows"] = []
        report["row_count"] = 0
        report["unavailable_reason"] = "state_restoration_failed"
    for row in report["rows"]:
        row["measurement_complete"] = report["status"] == "dose_matched_response_complete"
        row["state_restored"] = restored
    if comparison_axis == "source_localization":
        report["source_direction_comparisons"] = source_direction_comparisons(report["rows"])
    return report


def source_direction_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Paired differences, never a winner/permission decision."""
    groups: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(k) for k in ("measurement_context_id", "target_piece_binding_id",
            "activation_patch_site", "activation_patch_layer", "activation_patch_alpha", "activation_patch_step_size"))
        groups.setdefault(key, {})[str(row.get("source_variant"))] = row
    comparisons = []
    for pair in groups.values():
        term, centered = pair.get("source_term_token"), pair.get("source_centered_pm1")
        if term is None or centered is None:
            continue
        comparisons.append({"dose": term["activation_patch_step_size"], "target_piece": term.get("target_piece"),
            "target_piece_binding_variant": term["target_piece_binding_variant"],
            "measurement_context_id": term["measurement_context_id"],
            "reference_observable_id": term["observable_id"], "variant_observable_id": centered["observable_id"],
            "source_tensors_distinct": term["source_tensor_identity"] != centered["source_tensor_identity"],
            "executions_distinct": term["execution_id"] != centered["execution_id"],
            "difference_direction": "centered_minus_term",
            "paired_deltas": {k: centered[k] - term[k] for k in ("target_piece_logit_delta", "target_piece_prob_delta",
                "threshold20_logit_delta", "target_top20_threshold_gap_delta", "bound_token_top20_hit_delta")},
            "production_apply_allowed": False})
    return comparisons
