from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DigitTransformEpisode:
    seed: int
    source_text: str
    target_text: str
    prompt: str


class SpiralDigitTransformEnv:
    """Seeded exact-match task: reverse a digit string, then increment modulo 10."""

    task_id = "spiral_digit_transform_v0"
    goal_hint = "apply a deterministic digit-string rewrite and emit only the transformed digits"
    constraints = (
        "output digits only",
        "return only the transformed string",
        "keep the answer length equal to the source length",
    )

    def __init__(
        self,
        *,
        min_digits: int = 4,
        max_digits: int = 7,
    ) -> None:
        if min_digits <= 0:
            raise ValueError("min_digits must be positive")
        if max_digits < min_digits:
            raise ValueError("max_digits must be >= min_digits")
        self.min_digits = int(min_digits)
        self.max_digits = int(max_digits)
        self.current_episode: DigitTransformEpisode | None = None

    def reset(self, seed: int) -> str:
        rng = random.Random(seed)
        length = rng.randint(self.min_digits, self.max_digits)
        source_text = "".join(str(rng.randrange(10)) for _ in range(length))
        target_text = self.transform_digits(source_text)
        prompt = (
            "You are solving a deterministic rewrite task.\n"
            "Take the SOURCE digit string, reverse it, then increment each digit by 1 modulo 10.\n"
            "Return only the transformed digit string.\n"
            f"SOURCE: {source_text}\n"
            "ANSWER:"
        )
        self.current_episode = DigitTransformEpisode(
            seed=int(seed),
            source_text=source_text,
            target_text=target_text,
            prompt=prompt,
        )
        return prompt

    def score(self, output: str) -> float:
        return 1.0 if self._normalize_output(output) == self._episode().target_text else 0.0

    def done(self, output: str) -> bool:
        return self._normalize_output(output) == self._episode().target_text

    def task_feedback(self, output: str) -> dict[str, Any]:
        episode = self._episode()
        candidate = self._normalize_output(output)
        prefix_length = self._matching_prefix_length(candidate, episode.target_text)
        done = candidate == episode.target_text
        violations: list[str] = []

        if candidate and not candidate.isdigit():
            violations.append("non_digit_output")
        if len(candidate) > len(episode.target_text):
            violations.append("output_too_long")

        if done:
            progress_label = "progressing"
        elif not candidate:
            progress_label = "stalled"
        elif prefix_length == len(candidate):
            progress_label = "progressing"
        else:
            progress_label = "regressing"

        return {
            "done": done,
            "partial_score": prefix_length / len(episode.target_text),
            "progress_label": progress_label,
            "constraint_violations": violations,
        }

    def stop_checker(self, output: str) -> bool:
        episode = self._episode()
        candidate = self._normalize_output(output)
        return len(candidate) >= len(episode.target_text)

    def worker_runtime_kwargs(self) -> dict[str, Any]:
        episode = self._episode()
        return {
            "task_id": self.task_id,
            "goal_hint": self.goal_hint,
            "constraints": self.constraints,
            "max_generated_tokens": len(episode.target_text),
            "decode_constraint": "digits_only",
            "stop_checker": self.stop_checker,
            "task_feedback_fn": self.task_feedback,
        }

    @staticmethod
    def transform_digits(source_text: str) -> str:
        transformed = []
        for digit in reversed(source_text):
            if not digit.isdigit():
                raise ValueError("source_text must contain digits only")
            transformed.append(str((int(digit) + 1) % 10))
        return "".join(transformed)

    def _episode(self) -> DigitTransformEpisode:
        if self.current_episode is None:
            raise RuntimeError("reset(seed=...) must be called before scoring or feedback")
        return self.current_episode

    @staticmethod
    def _matching_prefix_length(candidate: str, target: str) -> int:
        for index, (left, right) in enumerate(zip(candidate, target)):
            if left != right:
                return index
        return min(len(candidate), len(target))

    @staticmethod
    def _normalize_output(output: str) -> str:
        return str(output).strip()
