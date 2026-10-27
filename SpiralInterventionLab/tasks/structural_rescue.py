"""Frozen structural-rescue pilot; role annotations are evaluation-only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from .language_tasks import SpiralConstrainedRewriteEnv, _strip_output, _word_count
from .role_binding import action_forms, check_role_frames


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

    def evaluate(self, output: str) -> dict[str, Any]:
        feedback = self.task_feedback(output)
        roles = check_role_frames(output, self.case.frames)
        words = re.findall(r"[A-Za-z]+(?:'s)?", output.casefold())
        unknown = []
        for frame, report in zip(self.case.frames, roles["frames"], strict=True):
            if sum(word in action_forms(frame[1]) for word in words) > 1:
                unknown.append("repeated_action:" + frame[1])
            if report["status"] in {"action_missing", "partially_bound"}:
                unknown.append("incomplete_frame:" + frame[1])
        # The nearest-agent heuristic cannot establish subjects of passives or
        # relative clauses. Preserve uncertainty instead of promoting a lexical pass.
        if re.search(r"\b(who|whom|whose|which|by)\b", output, re.I):
            unknown.append("unsupported_clause_or_voice")
        status = "fail" if roles["role_violations"] else "unknown" if unknown else "pass"
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
