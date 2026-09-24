"""Probe activation-patch seed discovery at a recorded early entity blueprint.

Runs one diagnostic in quarantine at the fixed prefix. It does not invoke a
controller, register a rollout edit, or grant production permission.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from ..runtime import candidate_handoff
from .digit_transform_e2e import (
    build_hooked_transformer_worker_runtime,
    create_readout_analyzer,
    create_task_env,
    load_worker_model,
)


def load_early_reviews(path: Path, *, prefix: str, objective: str) -> tuple[str, list[dict[str, Any]]]:
    prompt: str | None = None
    reviews: list[dict[str, Any]] = []
    prefix_observed = False
    observed_tail: str | None = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("event") == "episode_start":
                if prompt is not None:
                    raise ValueError("source JSONL must contain one episode")
                prompt = str(event["prompt"])
            if event.get("event") == "controller_observation":
                observed_tail = event.get("generated_tail")
                prefix_observed |= observed_tail == prefix
            if event.get("event") != "controller_diagnostic_result" or observed_tail != prefix:
                continue
            diagnostic = event.get("diagnostic")
            if (not reviews and diagnostic == "target_entity_insertion_probe") or (
                len(reviews) == 1 and diagnostic == "entity_insertion_operator_candidate_review"
            ):
                reviews.append(dict(event))
    if prompt is None or not prefix_observed or len(reviews) != 2:
        raise ValueError("source episode lacks the early prefix and paired entity probe/review")
    if [row.get("diagnostic") for row in reviews] != [
        "target_entity_insertion_probe", "entity_insertion_operator_candidate_review"
    ]:
        raise ValueError("entity diagnostic provenance/order mismatch")
    blueprints = reviews[-1].get("candidate_blueprints")
    if not isinstance(blueprints, list) or not any(
        isinstance(row, Mapping) and row.get("candidate_key") == objective
        and row.get("source_provenance") == "source_body" for row in blueprints
    ):
        raise ValueError("no recorded source-body blueprint for the requested objective")
    return prompt, reviews


def summarize_discovery(result: Mapping[str, Any], handoff: Mapping[str, Any],
                        *, objective: str) -> dict[str, Any]:
    matrix = result.get("target_piece_binding_seed_matrix_summary") or {}
    rows = [row for row in result.get("evidence_rows", ()) if isinstance(row, Mapping)
            and row.get("objective_bundle_key") == objective
            and row.get("diagnostic_family") == "activation_patch"]
    measured_rows = [row for row in rows if int(row.get("activation_hook_call_count") or 0) > 0]
    matching_offers = [offer for offer in handoff.get("measurement_offers", ())
                       if isinstance(offer, Mapping) and offer.get("objective_bundle_key") == objective]
    return {
        "diagnostic": result.get("diagnostic"), "status": result.get("status"),
        "charged_diagnostic_calls": int(result.get("diagnostic_call_cost", 1))
        if result.get("diagnostic_budget_charged") else 0,
        "blueprint_materialization_count": result.get("activation_patch_blueprint_materialization_count"),
        "activation_patch_evidence_row_count": len(rows),
        "activation_patch_hooked_row_count": len(measured_rows),
        "matched_response_status": matrix.get("status"),
        "matched_response_physical_replay_count": matrix.get("physical_replay_count"),
        "review_physical_replay_lower_bound": len(measured_rows) + int(matrix.get("physical_replay_count") or 0),
        "actual_delta_class_counts": {label: sum(str(row.get("actual_delta_class")) == label for row in rows)
                                      for label in sorted({str(row.get("actual_delta_class")) for row in rows})},
        "first_rows": [{key: row.get(key) for key in (
            "operator_recipe_id", "activation_patch_site", "activation_patch_source_localization",
            "actual_delta_class", "activation_hook_call_count", "target_mass_delta",
            "target_top20_hit_delta", "target_top20_threshold_gap_delta",
            "target_piece", "target_piece_binding_id", "production_apply_allowed")}
            for row in rows[:6]],
        "handoff_state": handoff.get("state"),
        "handoff_measured_card_count": handoff.get("measured_card_count"),
        "handoff_pool_row_count": handoff.get("pool_row_count"),
        "handoff_offer_count_for_objective": len(matching_offers),
        "handoff_offer_recipe_ids": [offer.get("seed_recipe_id") for offer in matching_offers],
        "handoff_blocked_reason": handoff.get("blocked_reason"),
        "handoff_unavailable_reason_for_objective": (
            handoff.get("unavailable_reasons") or {}).get(objective),
        "production_apply_allowed": False,
    }


def select_frozen_remeasurement(worker: Any, packet: Mapping[str, Any], *, objective: str,
                                recipe_id: str, target_piece: str) -> dict[str, Any]:
    choices = (packet.get("strategy_hints") or {}).get("candidate_diagnostic_choices") or {}
    matches = []
    for card in choices.get("cards", ()):
        if (not isinstance(card, Mapping) or card.get("objective_bundle_key") != objective
                or card.get("target_piece") != target_piece):
            continue
        entry = (getattr(worker, "_frozen_diagnostic_candidates", {}) or {}).get(card.get("candidate_id"))
        frozen = entry.get("frozen") if isinstance(entry, Mapping) else None
        if frozen is None or recipe_id not in {
            frozen.descriptor.get("operator_recipe_id"), frozen.descriptor.get("seed_operator_recipe_id")
        }:
            continue
        for action in card.get("actions", ()):
            if action.get("action") == "remeasure_current_prefix" and action.get("available") is True:
                matches.append({"diagnostic": "candidate_action", "action_id": action["action_id"]})
    if len(matches) != 1:
        raise ValueError(f"expected one executable frozen recipe/piece match, found {len(matches)}")
    return matches[0]


def card_inventory(worker: Any, packet: Mapping[str, Any]) -> list[dict[str, Any]]:
    choices = (packet.get("strategy_hints") or {}).get("candidate_diagnostic_choices") or {}
    registry = getattr(worker, "_frozen_diagnostic_candidates", {}) or {}
    cards = []
    for card in choices.get("cards", ()):
        if not isinstance(card, Mapping):
            continue
        entry = registry.get(card.get("candidate_id")) or {}
        frozen = entry.get("frozen")
        cards.append({"candidate_id": card.get("candidate_id"),
                      "objective_bundle_key": card.get("objective_bundle_key"),
                      "recipe_id": frozen.descriptor.get("operator_recipe_id") if frozen else None,
                      "seed_recipe_id": frozen.descriptor.get("seed_operator_recipe_id") if frozen else None,
                      "site": frozen.descriptor.get("site") if frozen else None,
                      "layer": frozen.descriptor.get("layer") if frozen else None,
                      "step_size": frozen.descriptor.get("step_size") if frozen else None,
                      "target_piece": card.get("target_piece"),
                      "actions": [{"action": action.get("action"), "available": action.get("available"),
                                   "blocked_reason": action.get("blocked_reason")}
                                  for action in card.get("actions", ()) if isinstance(action, Mapping)]})
    return cards


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--worker-model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", default="entity_insert:mira:source_body:weak_reachable")
    parser.add_argument("--prefix", default=" In")
    parser.add_argument("--prefix-steps", type=int, default=1)
    parser.add_argument("--followup-prefix")
    parser.add_argument("--followup-prefix-steps", type=int)
    parser.add_argument("--followup-recipe-id")
    parser.add_argument("--followup-target-piece")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.prefix_steps < 1:
        parser.error("prefix-steps must be positive")
    followup_args = (args.followup_prefix, args.followup_prefix_steps,
                     args.followup_recipe_id, args.followup_target_piece)
    if any(value is not None for value in followup_args) and not all(
        value is not None for value in followup_args):
        parser.error("all followup options must be set together")
    if args.followup_prefix_steps is not None and args.followup_prefix_steps <= args.prefix_steps:
        parser.error("followup-prefix-steps must exceed prefix-steps")
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    model_path = args.worker_model_path.expanduser().resolve(strict=True)
    source_prompt, early_reviews = load_early_reviews(source_path, prefix=args.prefix,
                                                      objective=args.objective)

    import torch

    torch.set_default_device("cpu")
    torch.set_grad_enabled(False)
    if args.device == "mps":
        torch.mps.set_per_process_memory_fraction(0.7)
    task = create_task_env("constrained_rewrite")
    prompt = task.reset(args.seed)
    if prompt != source_prompt:
        raise ValueError("task fixture differs from recorded episode")
    model = load_worker_model("gpt2", model_path=model_path, device=args.device,
                              dtype="float32", hf_offline=True, mps_mode="conservative")
    worker = build_hooked_transformer_worker_runtime(
        model, task, seed=args.seed, activation_surface_profile="activation_patch_expanded",
        max_diagnostic_calls_per_run=12, diagnostic_result_window=12,
        candidate_handoff_mode="soft", readout_sidecar_analyzer=create_readout_analyzer("sae_scaffold"),
        readout_analyzer_rerank_mode="apply")
    worker.reset(prompt)
    for _ in range(args.prefix_steps):
        worker.step()
    if worker.final_text() != args.prefix:
        raise ValueError(f"generated prefix drifted: {worker.final_text()!r}")
    worker._diagnostic_results = [dict(row) for row in early_reviews]
    worker._diagnostic_calls_used = sum(int(row.get("diagnostic_budget_charged") is True)
                                        for row in early_reviews)
    packet = worker.build_controller_packet()
    before = candidate_handoff.report(worker)
    early_hints = packet.get("strategy_hints") or {}
    discovery_options = [dict(item) for item in early_hints.get("available_next_diagnostics", ())
                         if isinstance(item, Mapping)
                         and item.get("diagnostic") == "activation_patch_candidate_review"]
    request = {"diagnostic": "activation_patch_candidate_review", "bundle_key": args.objective,
               "objective_bundle_key": args.objective,
               "operator_recipe_expansion_mode": "activation_patch_candidate_review"}
    result_rows = worker.request_controller_diagnostics(request, source="forced_diagnostic_shadow", packet=packet)
    if len(result_rows) != 1:
        raise ValueError(f"expected one diagnostic result, got {len(result_rows)}")
    result = result_rows[0]
    after = candidate_handoff.report(worker)
    after_packet = worker.build_controller_packet()
    frozen_recipes = sorted({str(entry["frozen"].descriptor.get("operator_recipe_id") or "")
                             for entry in worker._frozen_diagnostic_candidates.values()
                             if entry.get("frozen") is not None})
    report = {
        "kind": "early_activation_patch_seed_discovery_diagnostic_shadow",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": str(source_path),
        "source_jsonl_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "worker_model_path": str(model_path),
        "model_config_sha256": hashlib.sha256((model_path / "config.json").read_bytes()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "seed": args.seed, "prefix": args.prefix, "prefix_steps": args.prefix_steps,
        "objective_bundle_key": args.objective,
        "recorded_predecessor_diagnostics": [row["diagnostic"] for row in early_reviews],
        "diagnostic_budget_left_before": packet["telemetry"]["diagnostic_call_budget_left"],
        "handoff_state_before": before["state"],
        "early_seed_discovery_status": early_hints.get("candidate_seed_discovery_status"),
        "early_seed_discovery_options": discovery_options,
        "canonical_diagnostic_frontier_before": early_hints.get("diagnostic_frontier_request"),
        "discovery": summarize_discovery(result, after, objective=args.objective),
        "frozen_recipe_ids": frozen_recipes,
        "discovery_card_inventory": card_inventory(worker, after_packet),
        "diagnostic_budget_left_after": result["budget_after"]["diagnostic_calls_left"],
        "comparison_scope": "forced_early_diagnostic_only_not_controller_choice_or_rollout_outcome",
        "production_edit_applied": False,
    }
    if args.followup_prefix is not None:
        for _ in range(args.followup_prefix_steps - args.prefix_steps):
            worker.step()
        if worker.final_text() != args.followup_prefix:
            raise ValueError(f"followup prefix drifted: {worker.final_text()!r}")
        followup_packet = worker.build_controller_packet()
        report["followup_card_inventory"] = card_inventory(worker, followup_packet)
        try:
            followup_request = select_frozen_remeasurement(
                worker, followup_packet, objective=args.objective, recipe_id=args.followup_recipe_id,
                target_piece=args.followup_target_piece)
        except ValueError as exc:
            report["followup"] = {"status": "unavailable", "blocked_reason": str(exc),
                                  "physical_replay_count": 0, "production_apply_allowed": False}
        else:
            followup_result_rows = worker.request_controller_diagnostics(
                followup_request, source="forced_diagnostic_shadow", packet=followup_packet)
            if len(followup_result_rows) != 1:
                raise ValueError("expected exactly one followup diagnostic result")
            followup = followup_result_rows[0]
            evidence = followup.get("evidence") or {}
            metrics = evidence.get("promotion_evidence") or {}
            report["followup"] = {
                "prefix": args.followup_prefix, "prefix_steps": args.followup_prefix_steps,
                "recipe_id": args.followup_recipe_id, "candidate_id": followup.get("candidate_id"),
                "status": followup.get("status"), "executed_action": followup.get("executed_action"),
                "new_measurement_count": followup.get("new_measurement_count"),
                "physical_replay_count": followup.get("physical_replay_count"),
                "diagnostic_budget_charged": followup.get("diagnostic_budget_charged"),
                "state_restored": evidence.get("state_restored"),
                "target_piece": metrics.get("target_piece") or evidence.get("target_piece"),
                "target_piece_logit_delta": metrics.get("target_piece_logit_delta"),
                "target_piece_prob_delta": metrics.get("target_piece_prob_delta"),
                "target_top20_threshold_gap_delta": metrics.get("target_top20_threshold_gap_delta"),
                "target_rank_before": metrics.get("target_rank_before"),
                "target_rank_after": metrics.get("target_rank_after"),
                "bound_token_top20_hit_delta": metrics.get("bound_token_top20_hit_delta"),
                "production_apply_allowed": False,
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "discovery": report["discovery"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
