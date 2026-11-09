"""Norm-relative contract rerun of the one-boundary structural rescue pilot.

Same cases, boundary rule and restored diagnostic branches as v1. Absolute caps
are replaced, for declared edits only, by the opt-in contract: one edit, one
token, at most r of the edited residual norm, structural source tokens only.
Freeze with --prepare, then execute with --run and the printed manifest hash.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from ..controllers.factory import create_controller_provider
from ..runtime import clause_specificity, clause_structure, structural_contract
from ..runtime.compiler import StepContext, compile_expr
from ..runtime.policy import validate_command_against_packet
from ..runtime.response_probe import state_identity, tensor_identity
from ..tasks.structural_rescue import (
    CASES, StructuralRescueEnv, _COMMON_ENGLISH_VERBS, benchmark_manifest)
from ..tasks.structural_rescue_generator import candidate_rows
from .digit_transform_e2e import (
    build_default_activation_surface_catalog, build_hooked_transformer_worker_runtime, load_worker_model,
)
from .replay_orphan_blueprint_ceiling import _release_memory, verify_checkpoint
from .structural_rescue_benchmark import Meter, aggregate, capture_sources, choose, readout, unavailable_policies
from .structural_rescue_screen import DISK_FLOOR_GIB, load_selected_cases


POLICIES = ("b0", "fixed", "fixed_readout", "random", "controller")
CLAUSE_FORM_POLICIES = ("b0", "fixed", "fixed_readout", "fixed_clause_form", "random", "controller")
VERB_SLOT_POLICIES = ("b0", "fixed", "fixed_readout", "fixed_clause_form",
                      "fixed_verb_slot", "random", "controller")
RELATIVE_WRITE_CAP = structural_contract.DEFAULT_RELATIVE_WRITE_CAP
# Half-precision rounding of the clipped write; telemetry reports the realized norm.
CAST_TOLERANCE = 2e-3
BASE_PROMPT = """Select at most one bounded internal intervention recipe at this current worker prefix.
Return strict JSON only: {{"decision":"noop"}} or {{"decision":"probe","recipe_id":"<offered ID>"}}.
You receive generated text, next-token readout, and a catalog of source-bound recipes. Each recipe is
one edit for one token: a readout push toward one source punctuation or connective token, or a blend
toward the states of two source structural tokens. Every write is clipped to {cap:g} of the
edited residual norm, and names and required terms are never targets.
Choose a recipe only if it plausibly helps the worker preserve independent clauses. You may choose noop.
Do not output an answer, a token to force, new expressions, or any other fields.
This selects one restored diagnostic continuation, not production permission.
Role annotations, reference continuations and candidate outcomes are unavailable.
"""
SYSTEM_PROMPT = BASE_PROMPT.format(cap=RELATIVE_WRITE_CAP)
DATASHEET_PROMPT_ADD = """Readout recipes whose target was measured in a calibration cap sweep also carry a
reach_estimate: the measured winner-gap closure per unit of relative write, and its
estimated logit closure at this cap. Recipes without a reach_estimate were not covered
by that sweep. A recipe whose estimate range tops out below its target token's
gap_to_top1 cannot flip the first token; prefer a reachable recipe over an unreachable one.
"""


# Gate for clause_form starters: true only when the source's second clause is
# the generator's participial template ("with X -ing"). A word match, not a
# grammar check - prepositional "with" or bare participles would mislabel.
CLAUSE_FORM_APPLICABILITY_RULE = r"regex:\bwith\b"


def clause_form_applies(source: str) -> bool:
    return bool(re.search(CLAUSE_FORM_APPLICABILITY_RULE.removeprefix("regex:"), source))
CLAUSE_FORM_PROMPT_ADD = """The catalog may also offer clause_form recipes: a readout push toward a
finite-clause starter token that is not required to appear in the source.
A clause_form starter biases the continuation toward a finite clause ("while X
verbs") where a plain comma invites a participle ("with X -ing"); it carries
applies_to_this_case=true only when the source's second clause is participial
("with X -ing"). Prefer it only when applies_to_this_case is true and a comma
push would leave the next clause non-finite.
"""
VERB_SLOT_PROMPT_ADD = """The catalog may also offer verb_slot recipes: a readout push toward the
inflected required-action token, applied not at the boundary but at the b0
continuation's substituted-verb slot (the recipe replays the observed b0
prefix first, then pushes one token). It is offered only where the b0
continuation already shows a generic verb occupying the required-action
slot; prefer it there over a boundary push, which cannot reach the slot.
"""


def system_prompt(datasheet: bool = False, cap: float = RELATIVE_WRITE_CAP,
                  clause_form: bool = False, verb_slot: bool = False) -> str:
    return (BASE_PROMPT.format(cap=cap) + (DATASHEET_PROMPT_ADD if datasheet else "")
            + (CLAUSE_FORM_PROMPT_ADD if clause_form else "")
            + (VERB_SLOT_PROMPT_ADD if verb_slot else ""))


# How a verb_slot recipe is located: simulate the b0 continuation, find the
# first generic verb token after the second frame's agent, replay the b0
# prefix up to it, and push the inflected required action for one token.
VERB_SLOT_RULE = ("simulate_b0_continuation__first_common_verb_after_frame2_agent__"
                  "pre_roll_b0_prefix__push_<action>s_single_token_ttl1")
VERB_SLOT_ACTION_FORMS = ("action_forms_lemma_plus_s_inflection_only")


def manifest(case_set: dict[str, Any] | None = None, datasheet: bool = False,
             cap: float = RELATIVE_WRITE_CAP, clause_form: bool = False,
             verb_slot: bool = False) -> dict[str, Any]:
    """The frozen v1 cases by default; screen-selected cases carry their binding."""
    clause_form = clause_form or verb_slot
    v1 = benchmark_manifest()
    cases = v1["cases"] if case_set is None else candidate_rows(case_set["cases"])
    selection = {} if case_set is None else {"case_set": case_set["binding"]}
    sheet = ({"controller_datasheet": {"readout_closure_slope": dict(structural_contract.READOUT_CLOSURE_SLOPE),
                                       "reach_basis": structural_contract.REACH_BASIS,
                                       "unknown_pieces": "no_estimate"}}
             if datasheet else {})
    cf = {"clause_form_starters": [" while_finite_clause_starter"],
          "clause_form_applicability_rule": CLAUSE_FORM_APPLICABILITY_RULE} if clause_form else {}
    vs = ({"verb_slot_rule": VERB_SLOT_RULE,
          "verb_slot_action_forms": list(VERB_SLOT_ACTION_FORMS),
          "verb_slot_allowlist": "inflected_required_action_token_ids_added_to_readout_token_ids"}
          if verb_slot else {})
    return {"version": "structural_rescue_v2_norm_relative_contract" + ("_datasheet" if datasheet else "")
                       + ("_clauseform" if clause_form else "") + ("_verbslot" if verb_slot else "")
                       + (f"_cap{cap:g}" if cap != RELATIVE_WRITE_CAP else ""), **cf, **vs,
            "cases": cases, **sheet,
            "confirmation_status": (v1["confirmation_status"] if case_set is None
                                    else "selected_by_preregistered_b0_screen"), **selection,
            "relation_to_v1": "same_cases_boundary_and_patch_directions; doses_replaced_by_contract; "
                              "readout_recipes_added; pairs_with_protected_lexemes_dropped",
            "contract": {"id": structural_contract.CONTRACT, "relative_write_cap": cap,
                         "ttl_steps": 1, "edits_per_branch": 1, "site": "resid_pre", "token": "last",
                         "guard": "per_position_hook_output_minus_input_vs_input_norm",
                         "cast_tolerance": CAST_TOLERANCE,
                         "structural_lexemes": sorted(clause_structure.SOURCE_LEXEMES),
                         "protected_rule": "structural_lexemes_inside_required_terms",
                         "source_rule": "first_three_eligible_source_tokens_in_source_order"
                           + ("_plus_clause_form_starters" if clause_form else ""),
                         "patch_layer_rule": "n_layers//2", "patch_components": list(structural_contract.PATCH_COMPONENTS),
                         "patch_alpha": structural_contract.PATCH_ALPHA,
                         "readout_layer_rule": "n_layers-1", "readout_alpha_rule": "cap*||h_boundary||"},
            "max_words": v1["max_words"], "max_generated_tokens": v1["max_generated_tokens"],
            "boundary_rule": v1["boundary_rule"],
            "policies": list(VERB_SLOT_POLICIES if verb_slot
                             else CLAUSE_FORM_POLICIES if clause_form else POLICIES), "policy_seeds": [17],
            "evaluator": {"role_check_scope":
                          "surface_heuristic_nearest_agent_to_or_double_object_recipient_not_grammar",
                          "note": "double_object_binding_added_2026_11_06_alters_unknowns_vs_violations"},
            "fixed_rule": "common_orthogonal_of_first_eligible_pair",
            "fixed_readout_rule": "readout_of_first_eligible_source_token",
            "random_rule": "sha256(case_id,seed)_mod_catalog_size",
            "calibration_matrix": "every_catalog_recipe_on_calibration_cases",
            "diagnostic_forward_cap_per_policy": 24, "continuation_forward_cap_per_policy": 64,
            "max_controller_calls_per_case": 1, "max_controller_output_tokens": 1600,
            "controller_system_prompt_sha256": hashlib.sha256(system_prompt(datasheet, cap, clause_form, verb_slot).encode()).hexdigest(),
            "baseline_repeats_before_each_branch": 2, "disk_floor_gib": DISK_FLOOR_GIB,
            "protocol_deviations": ["disk_floor_6_gib_not_v1_10_gib_user_directed_2026_11_03_swap_headroom"],
            "apply_scope": "restored_contract_diagnostic_branch_only", "production_apply_allowed": False}


def manifest_bytes(case_set: dict[str, Any] | None = None, datasheet: bool = False,
                   cap: float = RELATIVE_WRITE_CAP, clause_form: bool = False,
                   verb_slot: bool = False) -> bytes:
    return (json.dumps(manifest(case_set, datasheet, cap, clause_form, verb_slot),
                       indent=2, sort_keys=True) + "\n").encode()


def required_terms(task: StructuralRescueEnv) -> tuple[str, ...]:
    return tuple(task._SCENARIOS[0]["required_terms"])


def verb_slot_plan(worker: Any, task: StructuralRescueEnv,
                   max_new_tokens: int = 40) -> dict[str, Any] | None:
    """Simulate the free (b0) continuation and locate the substituted verb.

    Returns the slot index inside the continuation, the b0 prefix token ids up
    to it, the substituted piece, and the required-action target token, or None
    when the continuation shows no generic verb after the second frame's agent.
    """
    sim = worker._simulate_decode(max_new_tokens=max_new_tokens, top_k=6,
                                  score_candidate_text=False, score_observer_check=False)
    if sim is None:
        return None
    cont_ids = sim["continuation_token_ids"]
    agent = str(task.case.frames[1][0])
    agent_idx = next((i for i, t in enumerate(cont_ids)
                      if agent in worker.codec.decode([t])), None)
    if agent_idx is None:
        return None
    slot = next((i for i in range(agent_idx + 1, len(cont_ids))
                 if worker.codec.decode([cont_ids[i]]).strip().casefold()
                 in _COMMON_ENGLISH_VERBS), None)
    if slot is None:
        return None
    lemma = str(task.case.frames[1][1])
    piece = " " + lemma + "s"
    target_ids = worker.codec.encode(piece).detach().reshape(-1).tolist()
    if len(target_ids) != 1:
        return None
    return {"apply_at_step": slot, "pre_roll_token_ids": cont_ids[:slot],
            "substituted_piece": worker.codec.decode([cont_ids[slot]]),
            "target_piece": piece, "target_token_id": int(target_ids[0])}


def catalog_for(worker: Any, task: StructuralRescueEnv, n_layers: int,
                cap: float = RELATIVE_WRITE_CAP, clause_form: bool = False,
                verb_slot: bool = False) -> tuple[list[dict], Any, list[dict]]:
    sources = structural_contract.eligible_sources(worker, required_terms(task))
    last = structural_contract.surface(worker, n_layers - 1)
    starters = (structural_contract.clause_form_recipes(worker, last, cap=cap)
                if clause_form else [])
    verb_recipes = []
    if verb_slot:
        plan = verb_slot_plan(worker, task)
        if plan is not None:
            verb_recipes = structural_contract.verb_slot_recipe(
                worker, last, cap=cap, token_id=plan["target_token_id"],
                piece=plan["target_piece"], apply_at_step=plan["apply_at_step"],
                pre_roll_token_ids=plan["pre_roll_token_ids"],
                substituted_piece=plan["substituted_piece"])
            for r in verb_recipes:
                r["verb_slot"].update(target_piece=plan["target_piece"])
    policy = structural_contract.policy_for(
        worker, required_terms(task), cap=cap,
        extra_readout_token_ids=([r["sources"][0]["token_id"] for r in starters]
                                 + [r["sources"][0]["token_id"] for r in verb_recipes]))
    middle = structural_contract.surface(worker, n_layers // 2)
    participial = clause_form_applies(task.case.source)
    for r in starters:
        r["applies_to_this_case"] = participial
    recipes = (structural_contract.patch_recipes(worker, sources, middle, cap=cap)
               + structural_contract.readout_recipes(worker, sources, last, cap=cap)
               + starters + verb_recipes)
    return recipes, policy, sources


def fixed_ids(recipes: list[dict]) -> dict[str, str | None]:
    patch = next((r for r in recipes if r["kind"] == "patch" and r["component"] == "common_orthogonal"), None)
    push = next((r for r in recipes if r["kind"] == "readout"), None)
    starter = next((r for r in recipes if r["kind"] == "clause_form"), None)
    verb = next((r for r in recipes if r["kind"] == "verb_slot"), None)
    return {"fixed": patch and patch["recipe_id"], "fixed_readout": push and push["recipe_id"],
            "fixed_clause_form": starter and starter["recipe_id"],
            "fixed_verb_slot": verb and verb["recipe_id"]}


def reach_estimate(recipe: dict) -> dict | None:
    """Calibration-measured gap closure for a readout target, or None if unmeasured."""
    if recipe["kind"] not in ("readout", "clause_form"):
        return None
    slope = structural_contract.READOUT_CLOSURE_SLOPE.get(recipe["sources"][0]["piece"])
    if slope is None:
        return None
    cap = recipe["relative_write_cap"]
    return {"slope_logits_per_unit_cap": dict(slope),
            "estimated_gap_closure": {"mean": slope["mean"] * cap,
                                      "min": slope["min"] * cap, "max": slope["max"] * cap},
            "extrapolated": cap > 0.12,
            "basis": structural_contract.REACH_BASIS}


def compact_packet(worker: Any, recipes: list[dict], case_id: str, observation: dict,
                   datasheet: bool = False, cap: float = RELATIVE_WRITE_CAP,
                   clause_form: bool = False, verb_slot: bool = False) -> dict:
    keys = ("recipe_id", "kind", "component", "sources", "surface", "relative_write_cap",
            "requested_relative_write", "expected_relative_write")
    offered = []
    for r in recipes:
        row = {**{k: r[k] for k in keys}, **({"dsl_formula": r["dsl_formula"], "geometry": r["geometry"]}
                                             if r["kind"] == "patch" else {})}
        estimate = reach_estimate(r) if datasheet else None
        if estimate is not None:
            row["reach_estimate"] = estimate
        if r["kind"] == "clause_form":
            row["form_hint"] = ("finite_clause_starter: biases the continuation toward a "
                                "finite verb clause where a comma push leaves 'with <name> -ing'")
            row["applies_to_this_case"] = r.get("applies_to_this_case", False)
        if r["kind"] == "verb_slot":
            row["verb_slot"] = dict(r["verb_slot"])
            row["apply_at_step"] = r["apply_at_step"]
            row["applies_to_this_case"] = r.get("applies_to_this_case", True)
        offered.append(row)
    return {"version": "structural_rescue_choice_v2" + ("_datasheet" if datasheet else "")
                       + ("_clauseform" if clause_form else "")
                       + ("_verbslot" if verb_slot else ""),
            "case_id": case_id,
            "generated_tail": worker.final_text(), "structure_readout": observation["structure_readout_rows"],
            "recipes": offered,
            "contract": {"id": structural_contract.CONTRACT, "relative_write_cap": cap,
                         "ttl_steps": 1, "max_edits": 1},
            "budget": {"max_edits": 1, "diagnostic_forwards": 24, "continuation_forwards": 64},
            "production_apply_allowed": False}


def select(policy: str, packet: dict, *, provider: Any, fixed: dict[str, str | None],
           datasheet: bool = False, cap: float = RELATIVE_WRITE_CAP,
           clause_form: bool = False, verb_slot: bool = False) -> tuple[str | None, dict]:
    if policy in fixed:
        return choose("fixed", packet, fixed_recipe_id=fixed[policy])
    return choose(policy, packet, provider=provider,
                  system_prompt=system_prompt(datasheet, cap, clause_form, verb_slot))


def branch(worker: Any, task: StructuralRescueEnv, recipe: dict | None, meter: Meter, policy: Any,
           *, cap: float = RELATIVE_WRITE_CAP) -> tuple[dict, Any]:
    if recipe is not None and recipe.get("relative_write_cap") != cap:
        raise ValueError("recipe_cap_differs_from_branch_cap")
    before, caps = state_identity(worker), repr(worker.surface_catalog)
    verb_recipe = recipe is not None and recipe.get("kind") == "verb_slot"
    baseline_raw = []
    # For a delayed edit the baseline logits must sit at the same position:
    # replay the b0 prefix unedited so baseline and edited vectors align.
    if verb_recipe:
        meter.boundary_length += recipe["apply_at_step"]
    try:
        meter.phase, meter.capture_edited = "branch", False
        for _ in range(2):
            meter.raw_first = None
            baseline = worker._simulate_decode(
                max_new_tokens=recipe["apply_at_step"] + 1 if verb_recipe else 1, top_k=6,
                pre_roll_token_ids=recipe["pre_roll_token_ids"] if verb_recipe else None,
                score_candidate_text=False, score_observer_check=False)
            if baseline is None or meter.raw_first is None:
                # Name the cause (a forward or disk limit, a runtime error) instead of hiding it.
                raise RuntimeError(f"repeated_baseline_failed: {worker._last_simulate_decode_error}")
            baseline_raw.append(meter.raw_first)
    finally:
        if verb_recipe:
            meter.boundary_length -= recipe["apply_at_step"]
    command = structural_contract.command(recipe) if recipe is not None else None
    if command:
        packet = worker.build_controller_packet()
        validate_command_against_packet(command, packet, policy=policy)
        ctx = StepContext(packet=packet, runtime_state=worker.runtime_state, adapter=worker.adapter,
                          traces={}, stats={}, active_edits={})
        if tensor_identity(compile_expr(command["edits"][0]["source"]["expr"])(ctx)) != recipe["input_identity"]:
            raise ValueError("recipe_source_changed_before_branch")
    meter.phase, meter.capture_edited, meter.raw_first = "branch", command is not None, None
    simulated = worker._simulate_decode(max_new_tokens=64 - len(worker._output_token_ids()), top_k=6,
        command=command, policy_override=policy if command else None,
        apply_at_step=recipe["apply_at_step"] if verb_recipe else 0,
        pre_roll_token_ids=recipe["pre_roll_token_ids"] if verb_recipe else None,
        score_candidate_text=False, score_observer_check=False)
    if simulated is None:
        raise RuntimeError(worker._last_simulate_decode_error)
    text = worker.codec.decode(worker._output_token_ids() + simulated["continuation_token_ids"])
    restored = before == state_identity(worker) and caps == repr(worker.surface_catalog)
    if not restored or worker._collect_active_edits():
        raise RuntimeError("diagnostic_branch_failed_to_restore")
    telemetry = simulated["edit_runtime_telemetry"]
    if command:
        if len(telemetry) != 1 or telemetry[0].get("guard_call_count") != 1:
            raise RuntimeError("one_token_contract_hook_not_confirmed")
        if not telemetry[0].get("relative_write_norm", float("inf")) <= cap + CAST_TOLERANCE:
            raise RuntimeError("contract_write_exceeded_cap")
    return {"status": "ok", **task.evaluate(text), "state_restored": restored,
            "scheduled_apply_at_step": recipe["apply_at_step"] if verb_recipe else 0,
            "baseline_repeat_max_abs_logit_delta": float((baseline_raw[0] - baseline_raw[1]).abs().max()),
            "baseline_logits_sha256": [tensor_identity(v)["sha256"] for v in baseline_raw],
            "command": command, "recipe_kind": recipe and recipe["kind"],
            "recipe_component": recipe and recipe["component"], "runtime_telemetry": telemetry,
            "first_token_id": simulated["continuation_token_ids"][0],
            "forwards": deepcopy(meter.counts), "production_apply_allowed": False}, meter.raw_first


def record_branch(worker, task, recipe, meter, destination: dict, policy,
                  *, cap: float = RELATIVE_WRITE_CAP) -> Any:
    """Retain attempted work even if simulation or a contract check fails."""
    destination["status"] = "running"
    try:
        outcome, raw = branch(worker, task, recipe, meter, policy, cap=cap)
        destination.update(outcome)
        return raw
    except Exception as exc:
        destination.update(status="error", error_type=type(exc).__name__, error=str(exc)[:300],
                           forwards=deepcopy(meter.counts), production_apply_allowed=False)
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--worker-model-path", type=Path)
    parser.add_argument("--checkpoint-manifest", type=Path)
    parser.add_argument("--worker-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--controller-provider", choices=("openai",), default="openai")
    parser.add_argument("--controller-model", default="gpt-5.6-luna")
    parser.add_argument("--policies", default=None)
    parser.add_argument("--case-ids")
    parser.add_argument("--screen-dir", type=Path, help="run the cases a frozen B0 screen selected")
    parser.add_argument("--datasheet", action="store_true",
                        help="offer calibration-measured readout reach estimates to the controller")
    parser.add_argument("--cap", type=float, default=RELATIVE_WRITE_CAP,
                        help="relative write cap; non-default values get a separate manifest version")
    parser.add_argument("--verb-slot", action="store_true",
                        help="also offer b0-located substituted-verb pushes; implies --clause-form")
    parser.add_argument("--clause-form", action="store_true",
                        help="offer a finite-clause starter recipe even when absent from the source")
    parser.add_argument("--wall-seconds", type=int, default=3600)
    parser.add_argument("--mps-memory-fraction", type=float, default=0.7)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    case_set = None
    cases = CASES
    if args.screen_dir is not None:
        cases, binding = load_selected_cases(args.screen_dir.expanduser().resolve())
        case_set = {"cases": cases, "binding": binding}
    if not 0 < args.cap <= 0.25:
        raise ValueError("cap must be in (0, 0.25]")
    clause_form = args.clause_form or args.verb_slot
    frozen = manifest_bytes(case_set, datasheet=args.datasheet, cap=args.cap,
                            clause_form=clause_form, verb_slot=args.verb_slot)
    digest = hashlib.sha256(frozen).hexdigest()
    if args.prepare:
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_bytes(frozen)
        print(json.dumps({"manifest_sha256": digest, "output_dir": str(output)}))
        return 0
    if args.manifest_sha256 != digest or (output / "manifest.json").read_bytes() != frozen:
        raise ValueError("predeclared_manifest_hash_required_and_must_match")
    if args.worker_model_path is None or args.checkpoint_manifest is None:
        parser.error("--run requires local model and checkpoint manifest")
    default_policies = (VERB_SLOT_POLICIES if args.verb_slot
                        else CLAUSE_FORM_POLICIES if clause_form else POLICIES)
    policies = (args.policies or ",".join(default_policies)).split(",")
    case_ids = args.case_ids.split(",") if args.case_ids else [c.case_id for c in cases]
    allowed = set(default_policies)
    if (set(policies) - allowed or policies[0] != "b0"
            or len(set(policies)) != len(policies) or len(set(case_ids)) != len(case_ids)
            or set(case_ids) - {c.case_id for c in cases}):
        raise ValueError("unknown or duplicate policy/case, or missing b0")
    if not 60 <= args.wall_seconds <= 7200:
        raise ValueError("wall_seconds must be between 60 and 7200")
    if not 0 < args.mps_memory_fraction <= 0.85:
        raise ValueError("mps_memory_fraction must be >0 and <=0.85")
    result_path = output / "results.json"
    for path in (result_path, output / "execution_plan.json"):
        if path.exists():
            raise FileExistsError(path)
    code_hashes = capture_sources(output)
    execution = {"manifest_sha256": digest, "device": args.device, "dtype": args.dtype,
                 "datasheet": args.datasheet, "cap": args.cap, "clause_form": clause_form,
                 "verb_slot": args.verb_slot,
                 "policies": policies, "case_ids": case_ids, "wall_seconds": args.wall_seconds,
                 "mps_memory_fraction": args.mps_memory_fraction,
                 "controller_model": args.controller_model if "controller" in policies else None,
                 "code_sha256": code_hashes}
    (output / "execution_plan.json").write_text(json.dumps(execution, indent=2))
    started = time.monotonic()
    splits = tuple(dict.fromkeys(c.split for c in cases))
    report = {"status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": digest, "checkpoint_sha256_verified": None,
              "device": args.device, "dtype": args.dtype, "surface_catalog": [],
              "policies_requested": policies, "case_ids_requested": case_ids, "cases": [],
              "scope": "restored_contract_branch_pilot_not_production_or_population_efficacy",
              "production_apply_allowed": False}

    def persist():
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["aggregate"] = aggregate(report["cases"], policies=policies, splits=splits)
        temp = result_path.with_suffix(".tmp")
        temp.write_text(json.dumps(report, indent=2, default=str) + "\n")
        temp.replace(result_path)

    persist()
    worker = None
    try:
        checkpoint = json.loads(args.checkpoint_manifest.read_text())["checkpoint_sha256"]
        report["checkpoint_sha256_verified"] = verify_checkpoint(args.worker_model_path, checkpoint)
        import torch
        torch.set_default_device("cpu")
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        if args.device == "mps":
            torch.mps.set_per_process_memory_fraction(args.mps_memory_fraction)
        model = load_worker_model(args.worker_model, model_path=args.worker_model_path, device=args.device,
                                  dtype=args.dtype, hf_offline=True, mps_mode="conservative")
        n_layers = model.cfg.n_layers
        layers = sorted({n_layers // 2, n_layers - 1})
        catalog = [s for s in build_default_activation_surface_catalog(model, layers=layers)
                   if s["target"].get("site") == "resid_pre" and s["target"]["token"]["mode"] == "last"]
        if sorted(s["target"]["layer"] for s in catalog) != layers:
            raise ValueError("contract_resid_surfaces_unavailable")
        report["surface_catalog"] = catalog
        provider = None
        if "controller" in policies:
            from openai import OpenAI
            provider = create_controller_provider(args.controller_provider, model=args.controller_model,
                                                   client=OpenAI(timeout=90, max_retries=0))
        for case in cases:
            if case.case_id not in case_ids:
                continue
            task = StructuralRescueEnv(case)
            prompt = task.reset(0)
            worker = build_hooked_transformer_worker_runtime(model, task, seed=0, surface_catalog=catalog,
                clause_structure_diagnostics=True, clause_structure_operator_diagnostics=True,
                max_diagnostic_calls_per_run=12)
            worker.reset(prompt)
            meter = Meter(worker, deadline=started + args.wall_seconds, disk_path=output,
                          disk_floor_gib=DISK_FLOOR_GIB)
            row = {"case_id": case.case_id, "split": case.split, "policies": {}, "calibration_matrix": []}
            row["setup_forwards"] = meter.counts
            report["cases"].append(row)
            with meter.install():
                recipient = case.frames[0][3]
                while not worker.done() and len(worker._output_token_ids()) < 32:
                    worker.step()
                    if re.search(r"\bto\s+" + re.escape(recipient) + r"$", worker.final_text()):
                        break
                boundary = bool(re.search(r"\bto\s+" + re.escape(recipient) + r"$", worker.final_text())) and not worker.done()
                row.update(prefix=worker.final_text(), boundary_found=boundary,
                           setup_forwards=meter.counts, tokens_identity=tensor_identity(worker._current_token_tensor()))
                if not boundary:
                    meter.reset("branch")
                    meter.boundary_length = worker._current_token_tensor().shape[-1]
                    if worker.done():
                        baseline = {"status": "ok", **task.evaluate(worker.final_text()), "command": None,
                                    "forwards": {}, "production_apply_allowed": False}
                    else:
                        baseline = {}
                        row["fallback_result"] = baseline
                        record_branch(worker, task, None, meter, baseline, None)
                    row["shared_fallback_forwards"] = deepcopy(baseline["forwards"])
                    row["policies"] = unavailable_policies(baseline, policies)
                    row["comparison_eligible"] = False
                    persist()
                    continue
                meter.boundary_length = worker._current_token_tensor().shape[-1]
                clause_specificity.align_cache(worker)
                observation = worker.request_controller_diagnostics(clause_structure.offer(worker)["request"])[0]
                if observation["status"] != "ok":
                    raise RuntimeError("structure_observation_failed")
                recipes, policy, sources = catalog_for(worker, task, n_layers, cap=args.cap,
                                                       clause_form=clause_form, verb_slot=args.verb_slot)
                fixed = fixed_ids(recipes)
                packet = compact_packet(worker, recipes, case.case_id, observation,
                                        datasheet=args.datasheet, cap=args.cap,
                                        clause_form=clause_form, verb_slot=args.verb_slot)
                (output / (case.case_id + "_observation.json")).write_text(json.dumps(packet, indent=2))
                row.update(comparison_eligible=True, shared_setup_forwards=deepcopy(meter.counts),
                           eligible_sources=[{k: s[k] for k in ("token_id", "piece", "source_positions")}
                                             for s in sources],
                           contract_allowlist={"readout_token_ids": list(policy.write_contract.readout_token_ids),
                                               "source_positions": list(policy.write_contract.source_positions)},
                           eligible_recipe_count=len(recipes), recipe_ids=[r["recipe_id"] for r in recipes],
                           fixed_recipe_ids=fixed)
                baseline_raw = None
                token_ids = [r["token_id"] for r in observation["structure_readout_rows"]]
                for name in policies:
                    meter.check_limits()
                    selected_id, trace = select(name, packet, provider=provider, fixed=fixed,
                                                datasheet=args.datasheet, cap=args.cap,
                                                clause_form=clause_form, verb_slot=args.verb_slot)
                    recipe = next((r for r in recipes if r["recipe_id"] == selected_id), None)
                    meter.reset("diagnostic")
                    outcome = {"selection": trace, "selected_recipe_id": selected_id}
                    row["policies"][name] = outcome
                    raw = record_branch(worker, task, recipe, meter, outcome, policy, cap=args.cap)
                    if name == "b0":
                        baseline_raw = raw
                    outcome.update(selection=trace, selected_recipe_id=selected_id,
                                   readout=readout(baseline_raw, raw, token_ids))
                    outcome["natural_branch_changed"] = (
                        outcome["output"] != row["policies"]["b0"]["output"]
                        if outcome.get("scheduled_apply_at_step")
                        else outcome["first_token_id"] != row["policies"]["b0"]["first_token_id"])
                    persist()
                    print(json.dumps({"case": case.case_id, "policy": name, "joint_success": outcome["joint_success"],
                        "score": outcome["legacy_score"], "selection": trace["status"],
                        "recipe": [outcome["recipe_kind"], outcome["recipe_component"]],
                        "output": outcome["output"]}), flush=True)
                if case.split == "calibration":
                    for recipe in recipes:
                        meter.reset("diagnostic")
                        outcome = {"kind": recipe["kind"], "component": recipe["component"],
                                   "sources": recipe["sources"], "recipe_id": recipe["recipe_id"]}
                        row["calibration_matrix"].append(outcome)
                        raw = record_branch(worker, task, recipe, meter, outcome, policy, cap=args.cap)
                        outcome["readout"] = readout(baseline_raw, raw, token_ids)
                        persist()
                        print(json.dumps({"case": case.case_id, "matrix": [recipe["kind"], recipe["component"],
                            [s["piece"] for s in recipe["sources"]]], "joint_success": outcome["joint_success"],
                            "output": outcome["output"]}), flush=True)
            worker.runtime_state.clear_edits()
            # Each case's worker keeps a full activation cache on the device.
            worker.runtime_state.last_cache = None
            _release_memory()
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, error=str(exc))
        persist()
        raise
    finally:
        if worker is not None:
            worker.runtime_state.clear_edits()
    persist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
