"""Prespecified lexical difficulty ladder, separate from the legacy benchmark.

The three cases share a template intentionally: this is a local capability
screen, not a broad generalization benchmark. No reference answers are exposed
to the worker, controller, semantic observer, or candidate compiler.
"""
from __future__ import annotations

import hashlib
from typing import Any

from .language_tasks import SpiralConstrainedRewriteEnv, _strip_output, _word_count
from .semantic_critic import SemanticCritic


REWRITE_LADDER_TASKS = tuple(f"constrained_rewrite_l{level}" for level in (1, 2, 3))
# Inherited Random(seed) selection covers each of the three cases once.
REWRITE_LADDER_SEEDS = (1, 0, 5)
REWRITE_LADDER_MAX_TOKENS = 64


class SpiralRewriteLadderEnv(SpiralConstrainedRewriteEnv):
    _CASES = (
        ("delivery", "In order to be ready before noon, Mira should send the budget to Omar, "
         "while Leena should bring the contract to Priya.",
         ("Mira", "send", "budget", "Omar", "Leena", "bring", "contract", "Priya", "before noon")),
        ("review", "In order to finish before dusk, Nora should take the sample to Ivo, "
         "while Selim should send the report to Yara.",
         ("Nora", "take", "sample", "Ivo", "Selim", "send", "report", "Yara", "before dusk")),
        ("workshop", "In order to start before lunch, Ada should bring the cable to Niko, "
         "while Rami should send the diagram to Vera.",
         ("Ada", "bring", "cable", "Niko", "Rami", "send", "diagram", "Vera", "before lunch")),
    )
    _AXES = {1: "required_term_load", 2: "word_budget_compression", 3: "lexical_exclusion"}

    def __init__(self, *, level: int, semantic_critic: SemanticCritic | None = None) -> None:
        if isinstance(level, bool) or not isinstance(level, int) or level not in self._AXES:
            raise ValueError("rewrite ladder level must be 1, 2, or 3")
        super().__init__(semantic_critic=semantic_critic)
        self.level = level
        self.task_id = f"spiral_constrained_rewrite_l{level}_v1"
        forbidden = ("in order to", "should") + (("the", "will", "must") if level == 3 else ())
        self._SCENARIOS = tuple(
            {"source_text": source, "required_terms": terms, "forbidden_terms": forbidden,
             "max_words": 18 if level == 1 else 12}
            for _, source, terms in self._CASES
        )

    def benchmark_manifest(self) -> dict[str, Any]:
        episode = self._episode()
        case_id = next(case_id for case_id, source, _ in self._CASES if source == episode.source_text)
        return {
            "benchmark_version": "rewrite_ladder_v1", "task_id": self.task_id,
            "case_id": case_id, "difficulty_level": self.level,
            "added_difficulty_axis": self._AXES[self.level],
            "seed": episode.seed,
            "prompt_sha256": hashlib.sha256(episode.prompt.encode()).hexdigest(),
            "source_sha256": hashlib.sha256(episode.source_text.encode()).hexdigest(),
            "required_term_count": len(episode.required_terms),
            "forbidden_term_count": len(episode.forbidden_terms),
            "max_words": episode.max_words, "max_generated_tokens": REWRITE_LADDER_MAX_TOKENS,
            "scoring_contract": "legacy_case_insensitive_substring_terms_and_whitespace_word_count",
            "semantic_preservation_certified": False,
        }

    def task_feedback(self, output: str) -> dict[str, Any]:
        return {**super().task_feedback(output), "rewrite_ladder": self.benchmark_manifest()}

    def stop_checker(self, output: str) -> bool:
        # Do not turn a word-budget violation into artificial decode starvation.
        candidate = _strip_output(output)
        return candidate.endswith((".", "!", "?")) and _word_count(candidate) >= 3

    def worker_runtime_kwargs(self) -> dict[str, Any]:
        return {**super().worker_runtime_kwargs(), "max_generated_tokens": REWRITE_LADDER_MAX_TOKENS}
