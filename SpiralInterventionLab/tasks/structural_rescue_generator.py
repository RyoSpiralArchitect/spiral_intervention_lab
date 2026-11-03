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
DEADLINES = (("finish", "before dusk"), ("be ready", "before noon"), ("start", "before lunch"),
             ("close", "before midnight"), ("leave", "before dawn"), ("eat", "before supper"))
CONNECTIVES = (", while", ", whereas", "; meanwhile,", ", and", "; at the same time,")
FAMILIES = ("purpose_first", "deadline_first", "purpose_last")


def _clause(agent: str, verb: str, obj: str, recipient: str) -> str:
    return f"{agent} should {verb} the {obj} to {recipient}"


def _source(family: str, goal: str, deadline: str, first: str, connective: str, second: str) -> str:
    if family == "purpose_first":
        return f"In order to {goal} {deadline}, {first}{connective} {second}."
    if family == "deadline_first":
        return f"{deadline[0].upper() + deadline[1:]}, {first}{connective} {second}."
    return f"{first}{connective} {second}, in order to {goal} {deadline}."


def generate(seed: int, count: int) -> tuple[RescueCase, ...]:
    """Families rotate in order; every other slot is drawn from one seeded stream."""
    used = {word for case in CASES for frame in case.frames for word in frame}
    if used & set(NAMES) or used & set(OBJECTS):
        raise ValueError("generator_pools_overlap_frozen_cases")
    rng = random.Random(seed)
    cases = []
    for index in range(count):
        family = FAMILIES[index % len(FAMILIES)]
        agent_a, recipient_b, agent_c, recipient_d = rng.sample(NAMES, 4)
        object_a, object_c = rng.sample(OBJECTS, 2)
        verb_a, verb_c = rng.sample(VERBS, 2)
        goal, deadline = rng.choice(DEADLINES)
        connective = rng.choice(CONNECTIVES)
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
