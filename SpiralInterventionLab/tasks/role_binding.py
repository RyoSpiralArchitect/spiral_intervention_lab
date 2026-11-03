"""Surface-level role-binding check for rewrite tasks with explicit role frames.

A frame says who does what to which object for whom, e.g. Selim sends the report
to Yara. The check binds each frame's action to the nearest preceding agent name
(possessives such as ``Selim's`` own an object, they do not act), its theme to the
first mention after the action, and its recipient to the first ``to <name>``
after the theme. It does not parse grammar or score fluency; a role-correct
sentence can still be ungrammatical.

With ``v2=True`` (the structural rescue evaluator v2) a known name right after
the action, followed by the theme, is the recipient of a double-object dative
(``Yusuf gives Wanda the badge``) unless a ``to <name>`` follows the theme too; a
name right after ``to`` or after an action form is an object, never an agent;
curly apostrophes count as possessives; regular and irregular pasts of hand,
mail and give are actions; and each frame report also carries the word indices
the clause-join check needs. The default keeps the original reports unchanged.
"""
from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

_WORD = re.compile(r"[A-Za-z]+(?:'s)?")
_IRREGULAR_FORMS = {"send": ("sent",), "take": ("took", "taken"), "bring": ("brought",)}
_V2_PAST_FORMS = {"give": ("gave", "given"), "hand": ("handed",), "mail": ("mailed",)}
DETERMINERS = frozenset({"the", "a", "an", "this", "that", "his", "her", "their", "its", "my", "our", "your"})
AUXILIARIES = frozenset({"am", "is", "are", "was", "were", "be", "been", "will", "would", "shall", "should",
                         "can", "could", "may", "might", "must", "do", "does", "did", "has", "have", "had"})


def action_forms(action: str) -> set[str]:
    base = action.casefold()
    forms = {base, base + "s", base + "es", base + "ing", *_IRREGULAR_FORMS.get(base, ())}
    if base.endswith("e"):
        forms.add(base[:-1] + "ing")
    return forms


def action_forms_v2(action: str) -> set[str]:
    return action_forms(action) | set(_V2_PAST_FORMS.get(action.casefold(), ()))


def normalize_v2(text: str) -> str:
    """Curly apostrophes as straight ones; same length, so spans still index the original."""
    return text.replace("\u2019", "'")


def _words(text: str) -> list[tuple[str, bool]]:
    """Lowercased words with a possessive flag."""
    words = []
    for match in _WORD.finditer(text):
        word = match.group(0).casefold()
        possessive = word.endswith("'s")
        words.append((word[:-2] if possessive else word, possessive))
    return words


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the words ``_words`` returns, index for index."""
    return [match.span() for match in _WORD.finditer(text)]


def _clause_end(words: list[tuple[str, bool]], start: int, *, agents: set[str], other_actions: set[str],
                keep: int | None = None) -> int:
    """First index after ``start`` that opens another clause: a new subject or another frame's action.

    A name right after ``to`` is a recipient and a possessive owns an object, so
    neither opens a clause. Nor does ``keep``, a double-object recipient.
    """
    for i in range(start + 1, len(words)):
        if i == keep:
            continue
        word, possessive = words[i]
        if word in other_actions:
            return i
        if word in agents and not possessive and words[i - 1][0] != "to":
            return i
    return len(words)


def check_role_frames(output: str, frames: Sequence[Sequence[str]], *, v2: bool = False) -> dict[str, Any]:
    forms_of = action_forms_v2 if v2 else action_forms
    words = _words(normalize_v2(output) if v2 else output)
    agents = {str(frame[0]).casefold() for frame in frames}
    names = agents | {str(frame[3]).casefold() for frame in frames}
    every_action = set().union(*(forms_of(str(frame[1])) for frame in frames))
    reports, violations = [], []
    for agent, action, theme, recipient in frames:
        other_actions = set().union(*(forms_of(str(frame[1])) for frame in frames
                                      if str(frame[1]).casefold() != action.casefold()))
        agent_key, theme_key, recipient_key = agent.casefold(), theme.casefold(), recipient.casefold()
        report: dict[str, Any] = {"agent": agent, "action": action, "theme": theme, "recipient": recipient,
                                  "bound_agent": None, "bound_recipient": None}
        forms = forms_of(action)
        action_index = next((i for i, (word, _) in enumerate(words) if word in forms), None)
        if action_index is None:
            report["status"] = "action_missing"
            reports.append(report)
            continue
        # v2: a name right after "to" or after an action form is an object, not a subject.
        agent_index = next((i for i in reversed(range(action_index))
                            if words[i][0] in agents and not words[i][1]
                            and not (v2 and i > 0 and (words[i - 1][0] == "to" or words[i - 1][0] in every_action))),
                           None)
        bound_agent = None if agent_index is None else words[agent_index][0]
        dative = (v2 and action_index + 1 < len(words)
                  and words[action_index + 1][0] in names and not words[action_index + 1][1])
        clause_end = _clause_end(words, action_index, agents=agents, other_actions=other_actions,
                                 keep=action_index + 1 if dative else None)
        theme_index = next((i for i in range(action_index + 1, clause_end) if words[i][0] == theme_key), None)
        recipient_index = None
        conflict = False
        if theme_index is not None:
            to_index = next((i + 1 for i in range(theme_index + 1, min(clause_end, len(words) - 1))
                             if words[i][0] == "to" and words[i + 1][0] in names), None)
            if dative and theme_index > action_index + 1:
                # A dative with a later "to <name>" names two recipients; leave it unbound.
                conflict = to_index is not None
                recipient_index = None if conflict else action_index + 1
            else:
                recipient_index = to_index
        bound_recipient = None if recipient_index is None else words[recipient_index][0]
        report.update(bound_agent=bound_agent, bound_recipient=bound_recipient,
                      theme_in_action_clause=theme_index is not None)
        if v2:
            between = words[agent_index + 1:action_index] if agent_index is not None else []
            report.update(agent_index=agent_index, action_index=action_index, theme_index=theme_index,
                          recipient_index=recipient_index, recipient_conflict=conflict,
                          recipient_form=None if recipient_index is None
                          else "dative" if recipient_index == action_index + 1 else "to",
                          nonfinite=words[action_index][0].endswith("ing")
                          and not any(w in AUXILIARIES for w, _ in between),
                          bare_theme=theme_index is not None and theme_index > 0
                          and words[theme_index - 1][0] not in DETERMINERS and not words[theme_index - 1][1])
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
        "role_check_scope": ("surface_heuristic_nearest_agent_to_or_dative_recipient_not_grammar" if v2
                             else "surface_heuristic_nearest_agent_and_to_recipient_not_grammar"),
    }
