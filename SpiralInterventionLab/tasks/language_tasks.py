from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence


def _strip_output(output: str) -> str:
    return str(output).strip()


def _compact_whitespace(text: str) -> str:
    return " ".join(str(text).strip().split())


def _word_count(text: str) -> int:
    compact = _compact_whitespace(text)
    if not compact:
        return 0
    return len(compact.split(" "))


def _contains_term(text: str, term: str) -> bool:
    return _compact_whitespace(term).lower() in _compact_whitespace(text).lower()


def _matching_prefix_length(candidate: str, target: str) -> int:
    for index, (left, right) in enumerate(zip(candidate, target)):
        if left != right:
            return index
    return min(len(candidate), len(target))


def _progress_label(*, done: bool, candidate: str, partial_score: float) -> str:
    if done:
        return "progressing"
    if not candidate:
        return "stalled"
    if partial_score > 0.0:
        return "progressing"
    return "regressing"


def _digits_only_candidate(output: str, limit: int) -> str:
    digits = "".join(char for char in str(output) if char.isdigit())
    return digits[:limit]


def _fraction(matched: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return matched / total


@dataclass(frozen=True)
class SentenceOrderingEpisode:
    seed: int
    ordered_events: tuple[str, ...]
    displayed_events: tuple[str, ...]
    target_text: str
    prompt: str


class SpiralSentenceOrderingEnv:
    task_id = "spiral_sentence_ordering_v0"
    goal_hint = "recover the chronological order of the listed events and emit only the order digits"
    constraints = (
        "output digits only",
        "return exactly three digits",
        "each digit must refer to one listed event position",
    )

    _EVENT_SETS: tuple[tuple[str, str, str], ...] = (
        (
            "Nora found the missing key under the mat.",
            "Nora unlocked the shed.",
            "Nora carried the toolbox inside.",
        ),
        (
            "The chef chopped the herbs.",
            "The chef stirred them into the soup.",
            "The chef served the soup to the table.",
        ),
        (
            "Ari charged the camera battery.",
            "Ari photographed the sunrise from the hill.",
            "Ari sent the best photo to the group chat.",
        ),
        (
            "Leah drafted the meeting notes.",
            "Leah shared the notes with her team.",
            "Leah archived the final version in the project folder.",
        ),
    )

    def __init__(self) -> None:
        self.current_episode: SentenceOrderingEpisode | None = None

    def reset(self, seed: int) -> str:
        rng = random.Random(seed)
        ordered_events = self._EVENT_SETS[rng.randrange(len(self._EVENT_SETS))]
        shuffled_positions = [0, 1, 2]
        rng.shuffle(shuffled_positions)
        displayed_events = tuple(ordered_events[index] for index in shuffled_positions)
        display_position_for_ordered_event = {
            ordered_index: shuffled_positions.index(ordered_index) + 1 for ordered_index in range(len(ordered_events))
        }
        target_text = "".join(str(display_position_for_ordered_event[index]) for index in range(len(ordered_events)))
        prompt = (
            "You are solving a sentence ordering task.\n"
            "Read the listed events and recover their chronological order.\n"
            "Output only a 3-digit string using the item numbers.\n"
            "Example: if item 2 happens first, then item 1, then item 3, output 213.\n"
            f"1. {displayed_events[0]}\n"
            f"2. {displayed_events[1]}\n"
            f"3. {displayed_events[2]}\n"
            "ANSWER:"
        )
        self.current_episode = SentenceOrderingEpisode(
            seed=int(seed),
            ordered_events=tuple(ordered_events),
            displayed_events=displayed_events,
            target_text=target_text,
            prompt=prompt,
        )
        return prompt

    def score(self, output: str) -> float:
        target = self._episode().target_text
        candidate = _digits_only_candidate(output, len(target))
        matches = sum(1 for left, right in zip(candidate, target) if left == right)
        return _fraction(matches, len(target))

    def done(self, output: str) -> bool:
        target = self._episode().target_text
        return _digits_only_candidate(output, len(target)) == target

    def task_feedback(self, output: str) -> dict[str, Any]:
        target = self._episode().target_text
        candidate = _digits_only_candidate(output, len(target))
        violations: list[str] = []
        raw = _strip_output(output)
        if raw and any(not char.isdigit() for char in raw):
            violations.append("non_digit_output")
        if len(candidate) > len(target):
            violations.append("output_too_long")
        partial_score = self.score(output)
        return {
            "done": self.done(output),
            "partial_score": partial_score,
            "progress_label": _progress_label(done=self.done(output), candidate=candidate, partial_score=partial_score),
            "constraint_violations": violations,
        }

    def stop_checker(self, output: str) -> bool:
        return len(_digits_only_candidate(output, len(self._episode().target_text))) >= len(self._episode().target_text)

    def worker_runtime_kwargs(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_hint": self.goal_hint,
            "constraints": self.constraints,
            "max_generated_tokens": 3,
            "decode_constraint": "digits_only",
            "stop_checker": self.stop_checker,
            "task_feedback_fn": self.task_feedback,
        }

    def _episode(self) -> SentenceOrderingEpisode:
        if self.current_episode is None:
            raise RuntimeError("reset(seed=...) must be called before scoring or feedback")
        return self.current_episode


@dataclass(frozen=True)
class EntailmentReasoningEpisode:
    seed: int
    premise: str
    hypothesis: str
    label_digit: str
    reason_digit: str
    prompt: str


class SpiralEntailmentReasoningEnv:
    task_id = "spiral_entailment_reasoning_v0"
    goal_hint = "judge the premise-hypothesis relation and emit the compact label/reason code"
    constraints = (
        "output digits only",
        "return exactly two digits",
        "first digit is the relation label, second digit is the reason code",
    )

    _SCENARIOS: tuple[dict[str, str], ...] = (
        {
            "premise": "Mira placed the blue notebook on the kitchen table.",
            "hypothesis": "The blue notebook is on the kitchen table.",
            "label_digit": "0",
            "reason_digit": "0",
        },
        {
            "premise": "Theo locked the studio before leaving for lunch.",
            "hypothesis": "Theo left the studio unlocked.",
            "label_digit": "1",
            "reason_digit": "1",
        },
        {
            "premise": "Rina mailed the contract before noon.",
            "hypothesis": "The contract arrived before noon.",
            "label_digit": "2",
            "reason_digit": "2",
        },
        {
            "premise": "Omar watered the plants after the meeting ended.",
            "hypothesis": "Omar watered the plants after the meeting.",
            "label_digit": "0",
            "reason_digit": "0",
        },
    )

    def __init__(self) -> None:
        self.current_episode: EntailmentReasoningEpisode | None = None

    def reset(self, seed: int) -> str:
        rng = random.Random(seed)
        scenario = self._SCENARIOS[rng.randrange(len(self._SCENARIOS))]
        prompt = (
            "You are solving a compact entailment reasoning task.\n"
            "Output exactly two digits.\n"
            "First digit: 0=entailment, 1=contradiction, 2=neutral.\n"
            "Second digit: 0=same fact, 1=negated fact, 2=missing detail.\n"
            "Return only the two digits, with no spaces.\n"
            f"Premise: {scenario['premise']}\n"
            f"Hypothesis: {scenario['hypothesis']}\n"
            "ANSWER:"
        )
        self.current_episode = EntailmentReasoningEpisode(
            seed=int(seed),
            premise=scenario["premise"],
            hypothesis=scenario["hypothesis"],
            label_digit=scenario["label_digit"],
            reason_digit=scenario["reason_digit"],
            prompt=prompt,
        )
        return prompt

    def score(self, output: str) -> float:
        candidate = _digits_only_candidate(output, 2)
        episode = self._episode()
        label_score = 0.6 if len(candidate) >= 1 and candidate[0] == episode.label_digit else 0.0
        reason_score = 0.4 if len(candidate) >= 2 and candidate[1] == episode.reason_digit else 0.0
        return label_score + reason_score

    def done(self, output: str) -> bool:
        candidate = _digits_only_candidate(output, 2)
        episode = self._episode()
        return candidate == f"{episode.label_digit}{episode.reason_digit}"

    def task_feedback(self, output: str) -> dict[str, Any]:
        candidate = _digits_only_candidate(output, 2)
        raw = _strip_output(output)
        violations: list[str] = []
        if raw and any(not char.isdigit() for char in raw):
            violations.append("non_digit_output")
        if len(candidate) > 2:
            violations.append("output_too_long")
        partial_score = self.score(output)
        return {
            "done": self.done(output),
            "partial_score": partial_score,
            "progress_label": _progress_label(done=self.done(output), candidate=candidate, partial_score=partial_score),
            "constraint_violations": violations,
        }

    def stop_checker(self, output: str) -> bool:
        return len(_digits_only_candidate(output, 2)) >= 2

    def worker_runtime_kwargs(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal_hint": self.goal_hint,
            "constraints": self.constraints,
            "max_generated_tokens": 2,
            "decode_constraint": "digits_only",
            "stop_checker": self.stop_checker,
            "task_feedback_fn": self.task_feedback,
        }

    def _episode(self) -> EntailmentReasoningEpisode:
        if self.current_episode is None:
            raise RuntimeError("reset(seed=...) must be called before scoring or feedback")
        return self.current_episode


@dataclass(frozen=True)
class ConstrainedRewriteEpisode:
    seed: int
    source_text: str
    required_terms: tuple[str, ...]
    forbidden_terms: tuple[str, ...]
    max_words: int
    prompt: str


class SpiralConstrainedRewriteEnv:
    task_id = "spiral_constrained_rewrite_v0"
    goal_hint = "rewrite the sentence to satisfy the required terms, banned terms, and word budget"
    constraints = (
        "output one rewritten sentence only",
        "keep the required terms",
        "avoid the forbidden terms",
        "stay within the word budget",
    )

    _SCENARIOS: tuple[dict[str, Any], ...] = (
        {
            "source_text": "Please note that the lantern by the garden gate is broken and needs a fresh bulb tonight.",
            "required_terms": ("lantern", "broken", "bulb"),
            "forbidden_terms": ("please", "note", "tonight"),
            "max_words": 8,
        },
        {
            "source_text": "In order to stay on schedule, Mira should send the budget draft to Omar before lunch.",
            "required_terms": ("Mira", "send", "budget", "Omar"),
            "forbidden_terms": ("in order to", "should"),
            "max_words": 9,
        },
        {
            "source_text": "The support team would really appreciate a quick update about the login outage before noon.",
            "required_terms": ("support", "update", "login", "outage"),
            "forbidden_terms": ("really", "quick"),
            "max_words": 10,
        },
    )

    def __init__(self) -> None:
        self.current_episode: ConstrainedRewriteEpisode | None = None

    def reset(self, seed: int) -> str:
        rng = random.Random(seed)
        scenario = self._SCENARIOS[rng.randrange(len(self._SCENARIOS))]
        prompt = (
            "You are solving a constrained rewrite task.\n"
            "Rewrite the source into one shorter sentence.\n"
            f"Keep these terms: {', '.join(scenario['required_terms'])}\n"
            f"Do not use these terms: {', '.join(scenario['forbidden_terms'])}\n"
            f"Use at most {scenario['max_words']} words.\n"
            "Return only the rewritten sentence.\n"
            f"SOURCE: {scenario['source_text']}\n"
            "ANSWER:"
        )
        self.current_episode = ConstrainedRewriteEpisode(
            seed=int(seed),
            source_text=scenario["source_text"],
            required_terms=tuple(scenario["required_terms"]),
            forbidden_terms=tuple(scenario["forbidden_terms"]),
            max_words=int(scenario["max_words"]),
            prompt=prompt,
        )
        return prompt

    def score(self, output: str) -> float:
        candidate = _compact_whitespace(output)
        if not candidate:
            return 0.0
        episode = self._episode()
        required_score, forbidden_score, budget_score = self._component_scores(candidate)
        return round((0.55 * required_score) + (0.25 * forbidden_score) + (0.20 * budget_score), 6)

    def done(self, output: str) -> bool:
        candidate = _compact_whitespace(output)
        if not candidate:
            return False
        episode = self._episode()
        return (
            all(_contains_term(candidate, term) for term in episode.required_terms)
            and all(not _contains_term(candidate, term) for term in episode.forbidden_terms)
            and _word_count(candidate) <= episode.max_words
        )

    def task_feedback(self, output: str) -> dict[str, Any]:
        candidate = _compact_whitespace(output)
        episode = self._episode()
        violations: list[str] = []
        missing_required = [term for term in episode.required_terms if candidate and not _contains_term(candidate, term)]
        forbidden_present = [term for term in episode.forbidden_terms if candidate and _contains_term(candidate, term)]
        required_score, forbidden_score, budget_score = self._component_scores(candidate)
        required_present = [term for term in episode.required_terms if candidate and _contains_term(candidate, term)]
        budget_ok = bool(candidate) and _word_count(candidate) <= episode.max_words
        if not candidate:
            violations.append("empty_output")
        if missing_required:
            violations.append("missing_required_terms")
        if forbidden_present:
            violations.append("forbidden_terms_present")
        if candidate and _word_count(candidate) > episode.max_words:
            violations.append("over_word_budget")
        partial_score = self.score(output)
        return {
            "done": self.done(output),
            "partial_score": partial_score,
            "progress_label": _progress_label(done=self.done(output), candidate=candidate, partial_score=partial_score),
            "required_term_recall": required_score,
            "required_terms_present": required_present,
            "forbidden_term_clean": forbidden_score,
            "budget_ok": budget_ok,
            "word_budget_score": budget_score,
            "constraint_violations": violations,
            "missing_required_terms": missing_required,
            "forbidden_terms_present": forbidden_present,
        }

    def stop_checker(self, output: str) -> bool:
        candidate = _strip_output(output)
        if not candidate:
            return False
        word_count = _word_count(candidate)
        if candidate.endswith((".", "!", "?")) and word_count >= 3:
            return True
        return word_count >= (self._episode().max_words + 3)

    def worker_runtime_kwargs(self) -> dict[str, Any]:
        episode = self._episode()
        return {
            "task_id": self.task_id,
            "goal_hint": self.goal_hint,
            "constraints": self.constraints,
            "max_generated_tokens": max(18, episode.max_words * 2),
            "stop_checker": self.stop_checker,
            "task_feedback_fn": self.task_feedback,
        }

    def _episode(self) -> ConstrainedRewriteEpisode:
        if self.current_episode is None:
            raise RuntimeError("reset(seed=...) must be called before scoring or feedback")
        return self.current_episode

    def _component_scores(self, candidate: str) -> tuple[float, float, float]:
        episode = self._episode()
        required_score = _fraction(
            sum(1 for term in episode.required_terms if _contains_term(candidate, term)),
            len(episode.required_terms),
        )
        forbidden_score = _fraction(
            sum(1 for term in episode.forbidden_terms if not _contains_term(candidate, term)),
            len(episode.forbidden_terms),
        )
        word_count = _word_count(candidate)
        if word_count == 0:
            budget_score = 0.0
        elif word_count <= episode.max_words:
            budget_score = 1.0
        else:
            overflow = word_count - episode.max_words
            budget_score = max(0.0, 1.0 - (overflow / max(1, episode.max_words)))
        return required_score, forbidden_score, budget_score


@dataclass(frozen=True)
class StructuredSummaryEpisode:
    seed: int
    note_text: str
    required_summary_terms: tuple[str, ...]
    required_keywords: tuple[str, ...]
    max_summary_words: int
    prompt: str


class SpiralStructuredSummaryEnv:
    task_id = "spiral_structured_summary_v0"
    goal_hint = "write a compact structured summary with the required summary line and keyword line"
    constraints = (
        "output exactly two labeled lines",
        "line one starts with summary:",
        "line two starts with keywords:",
        "keep the summary concise",
    )

    _SCENARIOS: tuple[dict[str, Any], ...] = (
        {
            "note_text": (
                "The museum closed early after a water leak in the west gallery. "
                "Staff guided visitors outside while repairs began."
            ),
            "required_summary_terms": ("museum", "leak", "closed"),
            "required_keywords": ("museum", "leak"),
            "max_summary_words": 10,
        },
        {
            "note_text": (
                "Tara missed the morning train, borrowed a bike, and still arrived before the meeting started."
            ),
            "required_summary_terms": ("Tara", "bike", "meeting"),
            "required_keywords": ("bike", "meeting"),
            "max_summary_words": 10,
        },
        {
            "note_text": (
                "The server reboot fixed the login issue, but the team kept the incident channel open for monitoring."
            ),
            "required_summary_terms": ("server", "login", "monitoring"),
            "required_keywords": ("login", "monitoring"),
            "max_summary_words": 11,
        },
    )

    def __init__(self) -> None:
        self.current_episode: StructuredSummaryEpisode | None = None

    def reset(self, seed: int) -> str:
        rng = random.Random(seed)
        scenario = self._SCENARIOS[rng.randrange(len(self._SCENARIOS))]
        prompt = (
            "You are solving a structured summary task.\n"
            "Write exactly two lines.\n"
            "Line 1: summary: <one short sentence>\n"
            "Line 2: keywords: <keyword1>, <keyword2>\n"
            f"Keep the summary to at most {scenario['max_summary_words']} words.\n"
            "Return only those two lines.\n"
            f"NOTE: {scenario['note_text']}\n"
            "ANSWER:"
        )
        self.current_episode = StructuredSummaryEpisode(
            seed=int(seed),
            note_text=scenario["note_text"],
            required_summary_terms=tuple(scenario["required_summary_terms"]),
            required_keywords=tuple(scenario["required_keywords"]),
            max_summary_words=int(scenario["max_summary_words"]),
            prompt=prompt,
        )
        return prompt

    def score(self, output: str) -> float:
        parsed = self._parse_output(output)
        summary_text = parsed["summary"]
        keyword_items = parsed["keywords"]
        format_score = 0.5 * float(bool(summary_text)) + 0.5 * float(bool(keyword_items))
        summary_term_score = _fraction(
            sum(1 for term in self._episode().required_summary_terms if _contains_term(summary_text, term)),
            len(self._episode().required_summary_terms),
        ) if summary_text else 0.0
        keyword_score = _fraction(
            sum(1 for term in self._episode().required_keywords if term.lower() in keyword_items),
            len(self._episode().required_keywords),
        ) if keyword_items else 0.0
        if not summary_text:
            budget_score = 0.0
        else:
            summary_words = _word_count(summary_text)
            if summary_words <= self._episode().max_summary_words:
                budget_score = 1.0
            else:
                overflow = summary_words - self._episode().max_summary_words
                budget_score = max(0.0, 1.0 - (overflow / max(1, self._episode().max_summary_words)))
        return round((0.20 * format_score) + (0.35 * summary_term_score) + (0.35 * keyword_score) + (0.10 * budget_score), 6)

    def done(self, output: str) -> bool:
        parsed = self._parse_output(output)
        summary_text = parsed["summary"]
        keyword_items = parsed["keywords"]
        return (
            bool(summary_text)
            and bool(keyword_items)
            and all(_contains_term(summary_text, term) for term in self._episode().required_summary_terms)
            and all(term.lower() in keyword_items for term in self._episode().required_keywords)
            and _word_count(summary_text) <= self._episode().max_summary_words
            and len(keyword_items) == 2
        )

    def task_feedback(self, output: str) -> dict[str, Any]:
        parsed = self._parse_output(output)
        summary_text = parsed["summary"]
        keyword_items = parsed["keywords"]
        violations: list[str] = []
        missing_summary_terms = [term for term in self._episode().required_summary_terms if summary_text and not _contains_term(summary_text, term)]
        missing_keywords = [term for term in self._episode().required_keywords if term.lower() not in keyword_items]
        if not summary_text:
            violations.append("missing_summary_line")
        if not keyword_items:
            violations.append("missing_keywords_line")
        if summary_text and _word_count(summary_text) > self._episode().max_summary_words:
            violations.append("summary_too_long")
        if keyword_items and len(keyword_items) != 2:
            violations.append("keyword_count_mismatch")
        if missing_summary_terms:
            violations.append("missing_summary_terms")
        if missing_keywords:
            violations.append("missing_keywords")
        partial_score = self.score(output)
        return {
            "done": self.done(output),
            "partial_score": partial_score,
            "progress_label": _progress_label(done=self.done(output), candidate=_strip_output(output), partial_score=partial_score),
            "constraint_violations": violations,
            "missing_summary_terms": missing_summary_terms,
            "missing_keywords": missing_keywords,
        }

    def stop_checker(self, output: str) -> bool:
        parsed = self._parse_output(output)
        if parsed["summary"] and parsed["keywords"]:
            return True
        return _word_count(_strip_output(output)) >= (self._episode().max_summary_words + 10)

    def worker_runtime_kwargs(self) -> dict[str, Any]:
        episode = self._episode()
        return {
            "task_id": self.task_id,
            "goal_hint": self.goal_hint,
            "constraints": self.constraints,
            "max_generated_tokens": max(20, episode.max_summary_words * 3),
            "stop_checker": self.stop_checker,
            "task_feedback_fn": self.task_feedback,
        }

    def _parse_output(self, output: str) -> dict[str, Any]:
        raw = _strip_output(output)
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        summary_text = ""
        keywords_text = ""
        for line in lines:
            lower = line.lower()
            if lower.startswith("summary:") and not summary_text:
                summary_text = line.split(":", 1)[1].strip()
            elif lower.startswith("keywords:") and not keywords_text:
                keywords_text = line.split(":", 1)[1].strip()

        if not summary_text or not keywords_text:
            lowered = raw.lower()
            summary_index = lowered.find("summary:")
            keywords_index = lowered.find("keywords:")
            if summary_index >= 0 and not summary_text:
                end = keywords_index if keywords_index > summary_index else len(raw)
                summary_text = raw[summary_index + len("summary:") : end].strip()
            if keywords_index >= 0 and not keywords_text:
                keywords_text = raw[keywords_index + len("keywords:") :].strip()

        keyword_items = [item.strip().lower() for item in keywords_text.split(",") if item.strip()]
        return {"summary": _compact_whitespace(summary_text), "keywords": keyword_items}

    def _episode(self) -> StructuredSummaryEpisode:
        if self.current_episode is None:
            raise RuntimeError("reset(seed=...) must be called before scoring or feedback")
        return self.current_episode
