"""Calibrated reach of the structural contract's actuators, for a controller.

A datasheet says, from calibration prompts only, how far one recipe kind moves
the next-token readout at one relative cap. For a readout push toward a source
token it records the logit gap the push closed between that token and the
unedited top token, separately for prompts whose top token is itself an
eligible source token and for the rest; pushes whose target was already the top
token are counted apart, since they cannot close anything. For patches it
records how often the first token changed at all, at the cap it was measured.
It holds no per-case outcomes, reference continuations or role annotations; the
cases that built it are named only in its provenance, never in a packet.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from statistics import median
from typing import Any


VERSION = "structural_reach_datasheet_v2"
MIN_N = 5
# Fixed-point grid of the fp16 logits near the boundary top logit (about 16-32).
QUANTUM = 1 / 64
ENDS_OUTPUT = frozenset({".", "!", "?"})


def top1_class(case: Mapping[str, Any]) -> str:
    eligible = {int(s["token_id"]) for s in case["eligible_sources"]}
    return "source_token" if int(case["b0_winner_id"]) in eligible else "other"


def table_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per measured push in first-token table cases (ok cases with a B0 branch)."""
    rows = []
    for case in cases:
        if case.get("status") != "ok" or "b0" not in case:
            continue
        winner, first = int(case["b0_winner_id"]), int(case["b0"]["first_token_id"])
        klass = top1_class(case)
        for push in case["pushes"]:
            row = {"case_id": case["case_id"], "kind": push["kind"], "cap": float(push["cap"]), "top1_class": klass,
                   "first_token_changed": int(push["first_token_id"]) != first,
                   "third_token": int(push["first_token_id"]) not in (first, int(push["sources"][0]["token_id"]))
                   if push["kind"] == "readout" else None}
            if push["kind"] == "readout":
                token = int(push["sources"][0]["token_id"])
                target = next(r for r in push["readout"]["rows"] if int(r["token_id"]) == token)
                gap, edited = target["baseline_gap_to_fixed_winner"], target["edited_gap_to_fixed_winner"]
                row.update(token_id=token, piece=push["sources"][0]["piece"], already_top=token == winner,
                           baseline_gap=gap, edited_gap=edited, closure=gap - edited,
                           tie=abs(edited) <= QUANTUM + 1e-12,
                           target_became_next=int(push["first_token_id"]) == token)
            rows.append(row)
    return rows


def _stats(values: Sequence[float]) -> dict[str, Any]:
    values = sorted(float(v) for v in values)
    return {"n": len(values), "min": values[0], "median": median(values), "max": values[-1]}


def build(rows: Sequence[Mapping[str, Any]], *, cap: float, provenance: Mapping[str, Any],
          patch_rows: Sequence[Mapping[str, Any]] = (), patch_cap: float | None = None,
          exclude_case: str | None = None) -> dict[str, Any]:
    """Readout stats at exactly ``cap`` per (token, top1 class); patch stats at ``patch_cap``."""
    mine = [r for r in rows if r["kind"] == "readout" and abs(r["cap"] - cap) < 1e-12 and r["case_id"] != exclude_case]
    readouts: dict[str, Any] = {}
    for token, klass in sorted({(int(r["token_id"]), r["top1_class"]) for r in mine}):
        group = [r for r in mine if int(r["token_id"]) == token and r["top1_class"] == klass]
        measurable = [r for r in group if not r["already_top"]]
        readouts[f"{token}|{klass}"] = {
            "token_id": token, "piece": group[0]["piece"], "top1_class": klass,
            "closure_logits": _stats([r["closure"] for r in measurable]) if measurable else None,
            "already_top": len(group) - len(measurable),
            "target_became_next": sum(r["target_became_next"] for r in measurable),
            "third_token_changes": sum(bool(r["third_token"]) for r in measurable)}
    patches = [r for r in patch_rows if r["kind"] == "patch" and r["case_id"] != exclude_case
               and patch_cap is not None and abs(r["cap"] - patch_cap) < 1e-12]
    sheet = {"version": VERSION, "cap": float(cap), "min_n": MIN_N, "readout": readouts,
             "patch": {"cap_measured": patch_cap, "n": len(patches),
                       "first_token_changes": sum(r["first_token_changed"] for r in patches)},
             "provenance": {**dict(provenance), "excluded_case": exclude_case}}
    sheet["datasheet_sha256"] = digest(sheet)
    return sheet


def digest(sheet: Mapping[str, Any]) -> str:
    body = {k: v for k, v in sheet.items() if k != "datasheet_sha256"}
    return hashlib.sha256((json.dumps(body, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def verify(sheet: Mapping[str, Any], *, cap: float) -> None:
    if sheet.get("version") != VERSION or sheet.get("datasheet_sha256") != digest(sheet):
        raise ValueError("datasheet_unbound_or_tampered")
    if abs(float(sheet["cap"]) - float(cap)) > 1e-12:
        raise ValueError("datasheet_cap_differs_from_contract_cap")


def reach(gap: float | None, closure: Mapping[str, Any] | None, *, is_top: bool) -> str:
    if is_top:
        return "already_top"
    if closure is None or closure["n"] < MIN_N or gap is None:
        return "uncalibrated"
    if gap < closure["min"]:
        return "likely"
    return "borderline" if gap < closure["max"] else "unlikely"


def recipe_entry(recipe: Mapping[str, Any], readout_rows: Sequence[Mapping[str, Any]], sheet: Mapping[str, Any],
                 *, case_top1_class: str) -> dict[str, Any]:
    """What a controller sees about one recipe; deterministic in the case's own readout and the sheet."""
    if recipe["kind"] == "patch":
        patch = sheet["patch"]
        return {"cap_measured": patch["cap_measured"], "first_token_changes": patch["first_token_changes"],
                "n": patch["n"]}
    source = recipe["sources"][0]
    token = int(source["token_id"])
    current = next((r for r in readout_rows if int(r["token_id"]) == token), None)
    gap = None if current is None or current.get("gap_to_top1") is None else float(current["gap_to_top1"])
    is_top = bool(current and current.get("is_greedy_token"))
    entry = sheet["readout"].get(f"{token}|{case_top1_class}")
    closure = None if entry is None else entry["closure_logits"]
    shown = None if closure is None or closure["n"] < MIN_N else {
        "n": closure["n"], **{k: round(closure[k], 2) for k in ("min", "median", "max")}}
    return {"cap": sheet["cap"], "target_gap_to_top1": None if gap is None else round(gap, 2),
            "target_is_top1": is_top, "top1_class": case_top1_class, "closure_logits": shown or "uncalibrated",
            "reach": reach(gap, closure, is_top=is_top),
            "ends_output": str(source["piece"]).strip() in ENDS_OUTPUT}


def rule_choice(recipes: Sequence[Mapping[str, Any]], entries: Sequence[Mapping[str, Any]]) -> str | None:
    """The deterministic datasheet rule: first readout, in source order, that may change the next token.

    Skips targets that end the answer and targets already on top; pushes the first
    whose reach is likely or borderline (gap below the stratum's max closure).
    """
    for recipe, entry in zip(recipes, entries, strict=True):
        if recipe["kind"] == "readout" and not entry["ends_output"] and entry["reach"] in ("likely", "borderline"):
            return recipe["recipe_id"]
    return None
