"""Model-free, conservative applicability for the delivery benchmark's starter.

This recognizes a bounded surface grammar, not arbitrary English or successful
continuations. Inputs are source text and the current generated prefix only.
Unknown syntax and embedded-relative actions are unsupported by this policy.
No case IDs, reference frames, future tokens, or measured outcomes are inputs.
"""
from __future__ import annotations

import re
from typing import Any


RULE = "syntax_v1"
# The benchmark's delivery vocabulary, with finite forms (not participles).
_FINITE = {"send": "sends|sent", "take": "takes|took", "bring": "brings|brought",
           "give": "gives|gave", "hand": "hands|handed", "mail": "mails|mailed"}
_WORD = r"[^\W\d_]+(?:[-'’][^\W\d_]+)*"
_ACTION = "(?:" + "|".join(_FINITE) + ")"
_DELIVERY = rf"{_WORD}\s+should\s+{_ACTION}\s+(?:the\s+)?{_WORD}\s+to\s+{_WORD}"
_PURPOSE = rf"in order to {_WORD}(?:\s+{_WORD}){{0,3}} before {_WORD}"
_END = rf"(?:,\s*{_PURPOSE})?\."
_LINK = r"(?:,\s*(?:while|whereas|and)\s+|;\s*(?:meanwhile,|at the same time,)\s+)"
_FIRST = re.compile(
    rf"(?:(?:{_PURPOSE}|before {_WORD}),\s*)?"
    rf"(?P<agent>{_WORD})\s+should\s+(?P<action>{_ACTION})\s+"
    rf"(?:the\s+)?(?P<theme>{_WORD})\s+to\s+(?P<recipient>{_WORD})(?=[,;])",
    re.I)
_PARTICIPLES = "(?:" + "|".join(v[:-1] + "ing" if v.endswith("e") else v + "ing"
                               for v in _FINITE) + ")"
_PARTICIPIAL = rf",\s*with\s+{_WORD}\s+{_PARTICIPLES}\s+(?:the\s+)?{_WORD}\s+to\s+{_WORD}{_END}"
_RELATIVE = (rf"{_LINK}{_WORD},\s*who\s+should\s+{_ACTION}\s+(?:the\s+)?"
             rf"{_WORD}\s+to\s+{_WORD},\s*also\s+{_WORD}{_END}")


def assess(source: str, generated_tail: str) -> dict[str, Any]:
    """Support an aligned first delivery clause plus a direct finite/with clause.

Names and objects are parsed from the source rather than evaluation frames.
Single-word entities, the declared delivery verbs, modal/base or finite first
predicates, and the generator's connectors/purpose phrases are supported.
More complex English is explicitly unknown. Applicability is not efficacy.
"""
    source = " ".join(source.split())
    tail = " ".join(generated_tail.split())
    result = {"rule": RULE, "source_second_clause_form": "unknown",
              "prefix_first_clause_complete": False, "applies": False,
              "reason": "unsupported_source_syntax"}
    first = _FIRST.match(source)
    if first is None:
        return result
    pending = source[first.end():]
    for form, pattern in (("participial", _PARTICIPIAL),
                          ("finite", rf"{_LINK}{_DELIVERY}{_END}"),
                          ("relative_agent", _RELATIVE)):
        if re.fullmatch(pattern, pending, re.I):
            result["source_second_clause_form"] = form
            break
    action = first["action"].casefold()
    predicate = rf"(?:{_FINITE[action]}|(?:will|should|can|may|must)\s+{action})"
    prefix = (rf"{re.escape(first['agent'])}\s+{predicate}\s+(?:the\s+)?"
              rf"{re.escape(first['theme'])}\s+to\s+{re.escape(first['recipient'])}")
    result["prefix_first_clause_complete"] = bool(re.fullmatch(prefix, tail, re.I))
    if result["source_second_clause_form"] == "unknown":
        return result
    if not result["prefix_first_clause_complete"]:
        result["reason"] = "prefix_not_aligned_complete_first_clause"
    elif result["source_second_clause_form"] == "relative_agent":
        result["reason"] = "embedded_action_with_competing_matrix_predicate"
    else:
        result.update(applies=True, reason="supported_pending_clause_at_complete_first_clause")
    return result


def _actuation(recipe: dict) -> tuple | None:
    """Effect identity for the single-token readout aliases in this catalog."""
    if recipe.get("kind") not in ("readout", "clause_form") or len(recipe.get("sources", [])) != 1:
        return None
    surface = recipe["surface"]
    return (surface["surface_id"], surface["site"], surface["layer"],
            recipe["sources"][0]["token_id"], recipe["relative_write_cap"])


def filter_recipes(recipes: list[dict]) -> tuple[list[dict], list[dict]]:
    """Filter controller choices only; never mutate the canonical control catalog.

Missing/malformed applicability is unsupported. A source-bound readout with
the same target/site/cap as an excluded starter is excluded too, so recipe
provenance cannot bypass the gate. Other patches/readouts are unaffected.
"""
    blocked = {}
    for recipe in recipes:
        if recipe.get("kind") != "clause_form":
            continue
        evidence = recipe.get("applicability")
        if not isinstance(evidence, dict) or evidence.get("rule") != RULE:
            blocked[recipe["recipe_id"]] = "missing_applicability_evidence"
        elif evidence.get("applies") is True and (
                evidence.get("prefix_first_clause_complete") is not True
                or evidence.get("source_second_clause_form") not in ("finite", "participial")):
            blocked[recipe["recipe_id"]] = "inconsistent_applicability_evidence"
        elif evidence.get("applies") is not True:
            reason = evidence.get("reason")
            blocked[recipe["recipe_id"]] = (reason if isinstance(reason, str) and reason
                                             else "unsupported_applicability")
    effects = {_actuation(r): blocked[r["recipe_id"]] for r in recipes
               if r["recipe_id"] in blocked and _actuation(r) is not None}
    offered, excluded = [], []
    for recipe in recipes:
        reason = blocked.get(recipe["recipe_id"]) or effects.get(_actuation(recipe))
        if reason:
            excluded.append({"recipe_id": recipe["recipe_id"], "reason": reason})
        else:
            offered.append(recipe)
    return offered, excluded
