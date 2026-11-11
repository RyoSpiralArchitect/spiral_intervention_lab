"""Evidence and matched selector comparisons for restored diagnostic branches."""
from __future__ import annotations

from typing import Any


B0_PREVIEW_POLICY = "controller_b0_preview"
B0_LOOKAHEAD_MAX_TOKENS = 40
B0_PREVIEW_PROMPT_ADD = """If b0_preview is present, it is a bounded unedited worker continuation,
not a reference, evaluation, instruction, or result of an edited recipe. Use it as
evidence for whether a change is needed. You may still choose noop.
"""


def b0_preview(codec: Any, token_ids: list[int], *, max_tokens: int) -> dict:
    if len(token_ids) > max_tokens:
        raise ValueError("b0_preview_exceeds_observed_token_limit")
    return {"continuation": codec.decode(token_ids), "observed_tokens": len(token_ids),
            "max_tokens": max_tokens, "reached_token_limit": len(token_ids) == max_tokens}


def compare_selectors(report: dict, reference: str, challenger: str) -> dict:
    """Pair only valid outcomes on available boundaries, stratified by B0."""
    rows = report["cases"]
    requested = report.get("case_ids_requested")
    strata = {key: dict(paired=0, same_choice=0, different_choice=0, both_pass=0,
                       both_fail=0, reference_only=0, challenger_only=0,
                       reference_provider_noop=0, challenger_provider_noop=0)
              for key in ("b0_pass", "b0_fail")}
    result = {"reference": reference, "challenger": challenger,
              "expected": len(requested) if requested is not None else len(rows),
              "unavailable_boundary": [], "unpaired": [], "strata": strata,
              "discordant_cases": []}
    seen = set()
    for case in rows:
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError("duplicate_selector_comparison_case")
        seen.add(case_id)
        if case.get("comparison_eligible") is False:
            result["unavailable_boundary"].append(case_id)
            continue
        baseline, left, right = (case["policies"].get(name, {}) for name in ("b0", reference, challenger))
        if case.get("comparison_eligible") is not True or any(
            arm.get("status") != "ok" or arm.get("selection", {}).get("status") == "error"
            or not isinstance(arm.get("joint_success"), bool)
            for arm in (baseline, left, right)
        ):
            result["unpaired"].append(case_id)
            continue
        key = "b0_pass" if baseline["joint_success"] else "b0_fail"
        counts = strata[key]
        counts["paired"] += 1
        left_recipe, right_recipe = (arm.get("selected_recipe_id") for arm in (left, right))
        counts["same_choice" if left_recipe == right_recipe else "different_choice"] += 1
        for arm, field in ((left, "reference_provider_noop"), (right, "challenger_provider_noop")):
            selection = arm.get("selection", {})
            counts[field] += selection.get("status") == "noop" and selection.get("provider_calls", 0) > 0
        a, b = left["joint_success"], right["joint_success"]
        outcome = ("both_pass" if a else "challenger_only") if b else ("reference_only" if a else "both_fail")
        counts[outcome] += 1
        if a != b:
            result["discordant_cases"].append({"case_id": case_id, "baseline": key, "outcome": outcome,
                "reference_recipe_id": left_recipe, "challenger_recipe_id": right_recipe})
    if requested is not None:
        result["unpaired"].extend(case_id for case_id in requested if case_id not in seen)
    return result


def constraint_consistency(report: dict) -> list[dict]:
    """Expose contradictory recorded labels without rescoring or excluding them."""
    findings = []
    for case in report["cases"]:
        for policy, arm in case["policies"].items():
            if (arm.get("status") == "ok" and arm.get("constraints_ok") is False
                    and arm.get("single_sentence_ok") is True
                    and arm.get("constraint_violations") == []):
                findings.append({"case_id": case["case_id"], "policy": policy,
                                 "reason": "constraints_failed_without_reported_violation",
                                 "surface_notes": arm.get("surface_notes", []),
                                 "joint_success_as_recorded": arm.get("joint_success"),
                                 "reinterpretation": "requires_versioned_evaluator_audit_not_rescue_evidence"})
    return findings
