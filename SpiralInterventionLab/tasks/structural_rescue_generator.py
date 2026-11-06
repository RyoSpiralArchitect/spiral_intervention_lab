"""Seeded two-clause delivery cases for a pre-registered baseline-failure screen.

The generator, seed, count and selection rule are frozen before any model sees
a candidate. Names, objects and deadlines are disjoint from the v1 cases, so a
selected case cannot be a renamed calibration case.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from .structural_rescue import CASES, RescueCase, StructuralRescueEnv


GENERATOR_VERSION = "structural_rescue_generator_v1"
NAMES = ("Alma", "Boris", "Celia", "Edda", "Felix", "Greta", "Henrik", "Ilse", "Jonas", "Kira", "Lars",
         "Marta", "Nikos", "Olga", "Pablo", "Rosa", "Stefan", "Tilda", "Ugo", "Vesna", "Wanda", "Yusuf")
OBJECTS = ("ledger", "badge", "manual", "ticket", "voucher", "drawing", "proposal", "receipt",
           "blueprint", "letter", "permit", "roster")
# Verbs whose surface forms the role check recognizes, so roles stay checkable.
VERBS = ("send", "take", "bring", "hand", "give", "mail")
# v1 showed give/hand/mail drifting to double-object outputs ("Wanda gives Lars
# the roster"), which never forms the `to <recipient>` boundary. v2 restricts
# the pool to verbs whose natural greedy output keeps that boundary.
GENERATOR_VERSION_V2 = "structural_rescue_generator_v2"
VERBS_V2 = ("send", "take", "bring")
DEADLINES = (("finish", "before dusk"), ("be ready", "before noon"), ("start", "before lunch"),
             ("close", "before midnight"), ("leave", "before dawn"), ("eat", "before supper"))
CONNECTIVES = (", while", ", whereas", "; meanwhile,", ", and", "; at the same time,")
FAMILIES = ("purpose_first", "deadline_first", "purpose_last")
# v3 adds families whose source syntax the simple screen outputs struggle to
# compress into one sentence: a participial second clause and an agent-relative.
GENERATOR_VERSION_V3 = "structural_rescue_generator_v3"
VERBS_V3 = ("send", "take", "bring", "give")
FAMILIES_V3 = FAMILIES + ("participle_second", "relative_agent")


def _clause(agent: str, verb: str, obj: str, recipient: str) -> str:
    return f"{agent} should {verb} the {obj} to {recipient}"


def _source(family: str, goal: str, deadline: str, first: str, connective: str, second: str) -> str:
    if family == "purpose_first":
        return f"In order to {goal} {deadline}, {first}{connective} {second}."
    if family == "deadline_first":
        return f"{deadline[0].upper() + deadline[1:]}, {first}{connective} {second}."
    return f"{first}{connective} {second}, in order to {goal} {deadline}."


def _gerund(verb: str) -> str:
    return verb[:-1] + "ing" if verb.endswith("e") else verb + "ing"


def _source_v3(family: str, goal: str, deadline: str, first: str, connective: str,
               second: tuple[str, str, str, str]) -> str:
    agent_c, verb_c, object_c, recipient_d = second
    if family == "participle_second":
        return (f"In order to {goal} {deadline}, {first}, "
                f"with {agent_c} {_gerund(verb_c)} the {object_c} to {recipient_d}.")
    # relative_agent
    return (f"{deadline[0].upper() + deadline[1:]}, {first}{connective} "
            f"{agent_c}, who should {verb_c} the {object_c} to {recipient_d}, also finishes.")


def generate(seed: int, count: int, verbs: tuple[str, ...] = VERBS,
             families: tuple[str, ...] = FAMILIES) -> tuple[RescueCase, ...]:
    """Families rotate in order; every other slot is drawn from one seeded stream."""
    used = {word for case in CASES for frame in case.frames for word in frame}
    if used & set(NAMES) or used & set(OBJECTS):
        raise ValueError("generator_pools_overlap_frozen_cases")
    if len(verbs) < 2 or set(verbs) - set(VERBS):
        raise ValueError("verbs_must_be_at_least_two_forms_the_role_check_recognizes")
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        family = families[index % len(families)]
        agent_a, recipient_b, agent_c, recipient_d = rng.sample(NAMES, 4)
        object_a, object_c = rng.sample(OBJECTS, 2)
        verb_a, verb_c = rng.sample(verbs, 2)
        goal, deadline = rng.choice(DEADLINES)
        connective = rng.choice(CONNECTIVES)
        if family in ("participle_second", "relative_agent"):
            source = _source_v3(family, goal, deadline,
                                _clause(agent_a, verb_a, object_a, recipient_b), connective,
                                (agent_c, verb_c, object_c, recipient_d))
        else:
            source = _source(family, goal, deadline, _clause(agent_a, verb_a, object_a, recipient_b),
                             connective, _clause(agent_c, verb_c, object_c, recipient_d))
        cases.append(RescueCase(f"gen_{index:03d}", "rescue_candidate", source,
                                ((agent_a, verb_a, object_a, recipient_b), (agent_c, verb_c, object_c, recipient_d)),
                                deadline))
    return tuple(cases)


def candidate_rows(cases: tuple[RescueCase, ...]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        prompt = StructuralRescueEnv(case).reset(0)
        rows.append({"case_id": case.case_id, "split": case.split, "source": case.source,
                     "frames_evaluation_only": case.frames, "deadline": case.deadline,
                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    return rows


def digest(value: Any) -> str:
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
