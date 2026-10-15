"""Surface-level role-binding check for rewrite tasks with explicit role frames.

A frame says who does what to which object for whom, e.g. Selim sends the report
to Yara. The check binds each frame's action to the nearest preceding agent name
(possessives such as ``Selim's`` own an object, they do not act), its theme to the
first mention after the action, and its recipient to the first ``to <name>``
after the theme. It does not parse grammar or score fluency; a role-correct
sentence can still be ungrammatical.
"""
from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

_WORD = re.compile(r"[A-Za-z]+(?:'s)?")
_IRREGULAR_FORMS = {"send": ("sent",), "take": ("took", "taken"), "bring": ("brought",)}


def action_forms(action: str) -> set[str]:
    base = action.casefold()
    forms = {base, base + "s", base + "es", base + "ing", *_IRREGULAR_FORMS.get(base, ())}
    if base.endswith("e"):
        forms.add(base[:-1] + "ing")
    return forms


def _words(text: str) -> list[tuple[str, bool]]:
    """Lowercased words with a possessive flag."""
    words = []
    for match in _WORD.finditer(text):
        word = match.group(0).casefold()
        possessive = word.endswith("'s")
        words.append((word[:-2] if possessive else word, possessive))
    return words


def check_role_frames(output: str, frames: Sequence[Sequence[str]]) -> dict[str, Any]:
    words = _words(output)
    agents = {str(frame[0]).casefold() for frame in frames}
    names = agents | {str(frame[3]).casefold() for frame in frames}
    reports, violations = [], []
    for agent, action, theme, recipient in frames:
        agent_key, theme_key, recipient_key = agent.casefold(), theme.casefold(), recipient.casefold()
        report: dict[str, Any] = {"agent": agent, "action": action, "theme": theme, "recipient": recipient,
                                  "bound_agent": None, "bound_recipient": None}
        forms = action_forms(action)
        action_index = next((i for i, (word, _) in enumerate(words) if word in forms), None)
        if action_index is None:
            report["status"] = "action_missing"
            reports.append(report)
            continue
        bound_agent = next((word for word, possessive in reversed(words[:action_index])
                            if word in agents and not possessive), None)
        theme_index = next((i for i in range(action_index + 1, len(words)) if words[i][0] == theme_key), None)
        bound_recipient = None
        if theme_index is not None:
            for i in range(theme_index + 1, len(words) - 1):
                if words[i][0] == "to" and words[i + 1][0] in names:
                    bound_recipient = words[i + 1][0]
                    break
        report.update(bound_agent=bound_agent, bound_recipient=bound_recipient,
                      theme_follows_action=theme_index is not None)
        problems = []
        if bound_agent is not None and bound_agent != agent_key:
            problems.append(f"{action}: agent {bound_agent}, expected {agent_key}")
        if bound_recipient is not None and bound_recipient != recipient_key:
            problems.append(f"{theme}: recipient {bound_recipient}, expected {recipient_key}")
        complete = bound_agent is not None and theme_index is not None and bound_recipient is not None
        report["status"] = "violated" if problems else "bound" if complete else "partially_bound"
        violations.extend(problems)
        reports.append(report)
    return {
        "frames": reports,
        "role_violations": violations,
        "role_binding_ok": not violations and all(row["status"] == "bound" for row in reports),
        "role_check_scope": "surface_heuristic_nearest_agent_and_to_recipient_not_grammar",
    }
