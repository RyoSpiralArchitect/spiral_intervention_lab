"""Frozen structural-rescue pilot; role annotations are evaluation-only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .language_tasks import SpiralConstrainedRewriteEnv, _strip_output, _word_count
from .role_binding import _words, action_forms, action_forms_v2, check_role_frames, normalize_v2, word_spans


EVALUATOR_V2 = "structural_rescue_eval_v2"
# Words that join two clauses without changing what the source says: its clauses
# are simultaneous, independent tasks (while, whereas, and, meanwhile).
JOINING_WORDS = frozenset({"and", "while", "whilst", "whereas", "but", "yet", "although", "though", "when", "as"})
# Joining words that add a condition, a cause, an order or an alternative.
MEANING_CHANGING_WORDS = frozenset({"or", "nor", "so", "because", "since", "after", "before", "until", "once",
                                    "unless", "if"})
FAULTY_JOINS = frozenset({"comma_splice", "run_on", "dangling_internal", "meaning_changing_join",
                          "double_connective", "interleaved"})
# A preposition or article still needs its complement, so it cannot sit right before
# punctuation ("... to Olga before, Boris ..."); nor can it, or a connective, end the sentence.
DANGLING_INTERNAL_WORDS = frozenset({"before", "after", "to", "the", "a", "an", "of", "for", "with", "by",
                                     "from", "at", "in", "on"})
DANGLING_FINAL_WORDS = DANGLING_INTERNAL_WORDS | {"and", "or", "but", "while", "whereas"}
_PUNCTUATION = (",", ";", ":", "-", "\u2013", "\u2014")
_DEADLINE = "<deadline>"


def classify_gap(gap: str, deadline: str) -> tuple[str, list[str]]:
    """Join between two clauses, from the text between them.

    The deadline phrase is a marker, not a joining word, and commas next to it
    belong to it. A joining word counts only after the last remaining comma, so
    "after lunch, Tilda mails" is still a splice.
    """
    marked = re.sub(re.escape(deadline), f" {_DEADLINE} ", gap, flags=re.I)
    tokens = re.findall(rf"{re.escape(_DEADLINE)}|[A-Za-z]+|[,;:\u2013\u2014-]", marked)
    for word, mark in zip(tokens, tokens[1:]):
        if word.casefold() in DANGLING_INTERNAL_WORDS and mark in _PUNCTUATION:
            return "dangling_internal", []
    kept = [t for i, t in enumerate(tokens) if t != _DEADLINE and not (
        t == "," and ((i > 0 and tokens[i - 1] == _DEADLINE) or (i + 1 < len(tokens) and tokens[i + 1] == _DEADLINE)))]
    if ";" in kept:
        return "semicolon", []
    if ":" in kept:
        return "colon", []
    if any(t in ("-", "\u2013", "\u2014") for t in kept):
        return "dash", []
    last_comma = max((i for i, t in enumerate(kept) if t == ","), default=-1)
    after = {t.casefold() for t in kept[last_comma + 1:] if t.isalpha()}
    if after & JOINING_WORDS:
        return "conjunction", sorted(after & JOINING_WORDS)
    if after & MEANING_CHANGING_WORDS:
        return "meaning_changing_join", sorted(after & MEANING_CHANGING_WORDS)
    return ("comma_splice" if "," in tokens else "run_on"), []


def clause_joins(output: str, frame_reports: list[dict[str, Any]], deadline: str) -> list[dict[str, Any]]:
    """How each clause, in output order, is joined to the one before it.

    The gap runs from the previous clause's last role word to this clause's
    agent. A subordinator before the first clause ("While Kira sends ..., Tilda
    mails ...") makes a bare comma correct and a second connective wrong, and a
    non-finite clause ("with Tilda mailing ...") is an adjunct, not a splice. Only
    fully bound frames from the v2 role check carry the indices this needs; other
    outputs fail the role check anyway.
    """
    output = normalize_v2(output)
    bound = sorted((f for f in frame_reports if f.get("status") == "bound" and f.get("agent_index") is not None),
                   key=lambda f: f["agent_index"])
    spans = word_spans(output)
    joins = []
    if bound:
        lead = re.sub(re.escape(deadline), " ", output[:spans[bound[0]["agent_index"]][0]], flags=re.I)
        lead_words = {w.casefold() for w in re.findall(r"[A-Za-z]+", lead)}
    for index, (previous, current) in enumerate(zip(bound, bound[1:])):
        start = spans[max(previous["theme_index"], previous["recipient_index"])][1]
        end = spans[current["agent_index"]][0]
        row = {"previous_agent": previous["agent"], "agent": current["agent"], "gap": output[start:end]}
        if end <= start:
            # The later agent opens inside the earlier clause.
            joins.append({**row, "gap": "", "kind": "interleaved", "joining_words": []})
            continue
        kind, words = classify_gap(row["gap"], deadline)
        if index == 0 and lead_words & (JOINING_WORDS | MEANING_CHANGING_WORDS) - {"and", "but", "yet", "or", "nor"}:
            if lead_words & MEANING_CHANGING_WORDS:
                kind, words = "meaning_changing_join", sorted(lead_words & MEANING_CHANGING_WORDS)
            elif kind in ("comma_splice", "run_on"):
                kind, words = "subordinate_first", sorted(lead_words & JOINING_WORDS)
            elif kind not in FAULTY_JOINS:
                kind = "double_connective"
        if kind in ("comma_splice", "run_on") and (previous.get("nonfinite") or current.get("nonfinite")):
            kind = "nonfinite_adjunct"
        joins.append({**row, "kind": kind, "joining_words": words})
    return joins


@dataclass(frozen=True)
class RescueCase:
    case_id: str
    split: str
    source: str
    frames: tuple[tuple[str, str, str, str], ...]
    deadline: str


CASES = (
    RescueCase("cal_delivery", "calibration",
        "In order to be ready before noon, Mira should send the budget to Omar, while Leena should bring the contract to Priya.",
        (("Mira", "send", "budget", "Omar"), ("Leena", "bring", "contract", "Priya")), "before noon"),
    RescueCase("cal_review", "calibration",
        "In order to finish before dusk, Nora should take the sample to Ivo, while Selim should send the report to Yara.",
        (("Nora", "take", "sample", "Ivo"), ("Selim", "send", "report", "Yara")), "before dusk"),
    RescueCase("cal_workshop", "calibration",
        "In order to start before lunch, Ada should bring the cable to Niko, while Rami should send the diagram to Vera.",
        (("Ada", "bring", "cable", "Niko"), ("Rami", "send", "diagram", "Vera")), "before lunch"),
    RescueCase("confirm_names_a", "confirmation",
        "In order to finish before dusk, Elise should take the parcel to Hugo, while Tomas should send the invoice to Lena.",
        (("Elise", "take", "parcel", "Hugo"), ("Tomas", "send", "invoice", "Lena")), "before dusk"),
    RescueCase("confirm_names_b", "confirmation",
        "In order to be ready before noon, Iris should send the schedule to Luca, while Pavel should bring the folder to Mina.",
        (("Iris", "send", "schedule", "Luca"), ("Pavel", "bring", "folder", "Mina")), "before noon"),
    RescueCase("confirm_order_a", "confirmation",
        "In order to finish before dusk, Tomas should send the invoice to Lena, while Elise should take the parcel to Hugo.",
        (("Tomas", "send", "invoice", "Lena"), ("Elise", "take", "parcel", "Hugo")), "before dusk"),
    RescueCase("confirm_order_b", "confirmation",
        "In order to be ready before noon, Pavel should bring the folder to Mina, while Iris should send the schedule to Luca.",
        (("Pavel", "bring", "folder", "Mina"), ("Iris", "send", "schedule", "Luca")), "before noon"),
    RescueCase("confirm_paraphrase_a", "confirmation",
        "Before lunch, Hana should bring the sketch to Emil; meanwhile, Dario should send the notice to Cora.",
        (("Hana", "bring", "sketch", "Emil"), ("Dario", "send", "notice", "Cora")), "before lunch"),
    RescueCase("confirm_paraphrase_b", "confirmation",
        "Before dusk, Bruno should take the packet to Anya, whereas Nadia should send the memo to Leon.",
        (("Bruno", "take", "packet", "Anya"), ("Nadia", "send", "memo", "Leon")), "before dusk"),
)


class StructuralRescueEnv(SpiralConstrainedRewriteEnv):
    """Same lexical score and prompt contract as the ladder, with explicit case IDs."""
    def __init__(self, case: RescueCase):
        super().__init__()
        self.case = case
        self.task_id = "structural_rescue_v1:" + case.case_id
        terms = tuple(value for frame in case.frames for value in frame) + (case.deadline,)
        self._SCENARIOS = ({"source_text": case.source, "required_terms": terms,
                            "forbidden_terms": ("in order to", "should"), "max_words": 18},)

    def stop_checker(self, output: str) -> bool:
        text = _strip_output(output)
        return text.endswith((".", "!", "?")) and _word_count(text) >= 3

    def worker_runtime_kwargs(self) -> dict[str, Any]:
        return {**super().worker_runtime_kwargs(), "max_generated_tokens": 64,
                "max_observer_checks_per_run": 0}

    def _role_status(self, output: str, roles: dict[str, Any], forms: Any = action_forms) -> tuple[str, list[str]]:
        words = re.findall(r"[A-Za-z]+(?:'s)?", output.casefold())
        unknown = []
        for frame, report in zip(self.case.frames, roles["frames"], strict=True):
            if sum(word in forms(frame[1]) for word in words) > 1:
                unknown.append("repeated_action:" + frame[1])
            if report["status"] in {"action_missing", "partially_bound"}:
                unknown.append("incomplete_frame:" + frame[1])
        # The nearest-agent heuristic cannot establish subjects of passives or
        # relative clauses. Preserve uncertainty instead of promoting a lexical pass.
        if re.search(r"\b(who|whom|whose|which|by)\b", output, re.I):
            unknown.append("unsupported_clause_or_voice")
        return "fail" if roles["role_violations"] else "unknown" if unknown else "pass", unknown

    def evaluate(self, output: str) -> dict[str, Any]:
        feedback = self.task_feedback(output)
        roles = check_role_frames(output, self.case.frames)
        status, unknown = self._role_status(output, roles)
        text = output.strip()
        sentence_ok = (text.endswith((".", "!", "?"))
                       and len(re.findall(r"[.!?](?:\s|$)", text)) == 1
                       and "\n" not in text)
        constraints_ok = bool(self.done(output)) and sentence_ok
        return {"output": output, "legacy_score": self.score(output),
                "constraint_violations": feedback["constraint_violations"],
                "constraints_ok": constraints_ok, "single_sentence_ok": sentence_ok,
                "role_status": status, "role_unknown_reasons": unknown,
                "role_report": roles, "joint_success": constraints_ok and status == "pass",
                "evaluation_scope": "lexical_constraints_plus_conservative_surface_role_heuristic"}

    def evaluate_v2(self, output: str) -> dict[str, Any]:
        """v1 plus datives, clause joins and dangling words (see ``clause_joins``).

        v1 stays the default for every earlier runner and receipt; its verdict is
        kept under ``evaluation_v1``. Articles are not checked: a bare theme
        ("sends drawing to Ilse") is reported, not failed.
        """
        v1 = self.evaluate(output)
        normalized = normalize_v2(output)
        roles = check_role_frames(output, self.case.frames, v2=True)
        status, unknown = self._role_status(normalized, roles, forms=action_forms_v2)
        joins = clause_joins(output, roles["frames"], self.case.deadline) if status == "pass" else None
        faulty = [j["kind"] for j in joins or () if j["kind"] in FAULTY_JOINS]
        words = _words(normalized)
        dangling = words[-1][0] if words and not words[-1][1] and words[-1][0] in DANGLING_FINAL_WORDS else None
        well_formed = not faulty and dangling is None
        return {**v1, "role_status": status, "role_unknown_reasons": unknown, "role_report": roles,
                "clause_joins": joins or [], "clause_join_ok": None if joins is None else not faulty,
                "dangling_final_word": dangling, "well_formed_ok": well_formed,
                "bare_themes": [f["theme"] for f in roles["frames"] if f.get("bare_theme")],
                "joint_success": v1["constraints_ok"] and status == "pass" and well_formed,
                "evaluator_version": EVALUATOR_V2,
                "evaluation_v1": {key: v1[key] for key in ("joint_success", "role_status", "role_unknown_reasons")},
                "evaluation_scope": "lexical_constraints_surface_roles_with_datives_clause_joins_and_dangling_words;"
                                    "articles_not_checked"}


def benchmark_manifest() -> dict[str, Any]:
    cases = []
    for case in CASES:
        task = StructuralRescueEnv(case)
        prompt = task.reset(0)
        cases.append({"case_id": case.case_id, "split": case.split, "source": case.source,
                      "frames_evaluation_only": case.frames, "deadline": case.deadline,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    return {"version": "structural_rescue_v1", "cases": cases,
            "confirmation_status": "predeclared_before_this_pilot_not_a_population_sample",
            "max_words": 18, "max_generated_tokens": 64,
            "boundary_rule": "first_natural_prefix_ending_in_to_first_source_recipient",
            "policy_seeds": [17], "diagnostic_forward_cap_per_policy": 24,
            "continuation_forward_cap_per_policy": 64,
            "components": ["common", "common_parallel", "common_orthogonal"],
            "fixed_rule": "common_orthogonal_of_first_source_pair",
            "random_rule": "sha256(case_id,seed)_mod_catalog_size",
            "max_controller_calls_per_case": 1, "max_controller_output_tokens": 1600,
            "baseline_repeats_before_each_branch": 2,
            "apply_scope": "restored_normal_cap_diagnostic_branch_only",
            "production_apply_allowed": False}


def manifest_bytes() -> bytes:
    return (json.dumps(benchmark_manifest(), indent=2, sort_keys=True) + "\n").encode()
