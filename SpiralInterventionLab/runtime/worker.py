from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .adapter import ModelAdapter
from .codecs import TextCodec, resolve_text_codec
from .compiler import StepContext, compile_command
from .edit_budget import LOOP_RESCUE_EDIT_BUDGET_POOL, MAIN_EDIT_BUDGET_POOL
from .effects import build_edit_effect, summarize_effects
from .schema import SurfaceInfo
from .trace_recorder import StepAlignedTrace, StepAlignedTraceRecorder


_CONTROLLER_REFLECTION_MODES = {"off", "structured"}
_DECODER_CONTROL_MODE_SPECS = {
    "off": {
        "track": "baseline",
        "objective": "No worker-side decoder shaping; internal edits remain the primary control path.",
    },
    "loop_aware": {
        "track": "auxiliary",
        "objective": "Apply task-agnostic anti-loop penalties to reduce short local attractors.",
    },
    "loop_aware_prune": {
        "track": "auxiliary",
        "objective": "Break local attractors, then demote the current overconfident top token to widen search.",
    },
    "loop_aware_constraint": {
        "track": "auxiliary",
        "objective": "Break local attractors, then softly bias toward explicitly missing constraint tokens without forcing output.",
    },
    "loop_aware_entity_recall": {
        "track": "auxiliary",
        "objective": "Break local attractors, then softly continue or start explicit missing entity tokens from task feedback.",
    },
    "logit_bias_entity_soft": {
        "track": "auxiliary",
        "objective": "Apply a bounded soft logit bias toward tokens from explicit missing entities without forcing a prefix or full answer.",
    },
}
_DECODER_CONTROL_MODES = set(_DECODER_CONTROL_MODE_SPECS)
_CONTROLLER_MEMORY_STRING_FIELDS = (
    "hypothesis",
    "micro_rationale",
    "expected_effect",
    "observed_outcome",
    "why_failed_or_helped",
    "next_change",
    "stop_condition",
)
_CONTROLLER_MEMORY_ALLOWED_OUTCOMES = {"unknown", "helpful", "harmful", "neutral", "mixed"}
_CONTROLLER_MEMORY_ALLOWED_NEXT_ACTIONS = {
    "wait",
    "noop",
    "apply",
    "rollback",
    "request_observer_check",
    "stop",
}
_CONTROLLER_TOOL_NAMES = {"tokenize_terms", "constraint_scorer", "dry_run_decode"}
_SOFT_CONSTRAINT_FEEDBACK_KEYS = ("missing_required_terms", "missing_keywords", "missing_summary_terms")
_ENTITY_RECALL_FEEDBACK_KEYS = ("entity_recall_terms",) + _SOFT_CONSTRAINT_FEEDBACK_KEYS


@dataclass
class _TokenSegment:
    kind: str
    token_ids: list[int]


@dataclass(frozen=True)
class _TargetTokenSequence:
    term: str
    token_ids: tuple[int, ...]
    variant: str


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.numel() != right.numel():
        return None
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    value = torch.dot(left, right) / ((left.norm() * right.norm()) + 1e-8)
    return float(value.item())


def _normalize_controller_reflection_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _CONTROLLER_REFLECTION_MODES:
        raise ValueError(
            f"Unsupported controller reflection mode '{value}'; expected one of {sorted(_CONTROLLER_REFLECTION_MODES)}"
        )
    return normalized


def _normalize_decoder_control_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _DECODER_CONTROL_MODES:
        raise ValueError(
            f"Unsupported decoder control mode '{value}'; expected one of {sorted(_DECODER_CONTROL_MODES)}"
        )
    return normalized


def _decoder_control_spec(mode: str) -> Mapping[str, str]:
    normalized = _normalize_decoder_control_mode(mode)
    return _DECODER_CONTROL_MODE_SPECS[normalized]


def _clean_controller_memory_text(value: Any, *, limit: int = 160) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip()
    if not text:
        return None
    return text[:limit]


def _normalize_controller_memory_label(value: Any, *, limit: int = 48) -> str | None:
    text = _clean_controller_memory_text(value, limit=limit)
    if text is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return normalized[:limit] or None


def _normalize_controller_memory_entry(
    value: Mapping[str, Any] | None,
    *,
    decision: str | None = None,
    recorded_step: int | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    entry: dict[str, Any] = {}
    for key in _CONTROLLER_MEMORY_STRING_FIELDS:
        text = _clean_controller_memory_text(value.get(key))
        if text is not None:
            if key == "observed_outcome":
                normalized_outcome = text.lower().replace(" ", "_")
                if normalized_outcome not in _CONTROLLER_MEMORY_ALLOWED_OUTCOMES:
                    normalized_outcome = "mixed"
                entry[key] = normalized_outcome
            else:
                entry[key] = text
    next_trigger = _normalize_controller_memory_label(value.get("next_trigger"), limit=60)
    if next_trigger is not None:
        entry["next_trigger"] = next_trigger
    next_action = _normalize_controller_memory_label(value.get("next_action"), limit=32)
    if next_action in _CONTROLLER_MEMORY_ALLOWED_NEXT_ACTIONS:
        entry["next_action"] = next_action
    confidence = value.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        entry["confidence"] = max(0.0, min(1.0, float(confidence)))
    if decision is not None:
        entry["decision"] = str(decision)
    if recorded_step is not None:
        entry["recorded_step"] = int(recorded_step)
    return entry or None


def _normalize_observer_check_request(value: Any) -> dict[str, Any] | None:
    if value is True:
        return {"kind": "semantic_progress"}
    if not isinstance(value, Mapping):
        return None
    kind = " ".join(str(value.get("kind", "semantic_progress")).split()).strip().lower().replace("-", "_")
    if not kind:
        kind = "semantic_progress"
    if kind != "semantic_progress":
        return None
    normalized: dict[str, Any] = {"kind": kind}
    reason = _clean_controller_memory_text(value.get("reason"), limit=120)
    if reason is not None:
        normalized["reason"] = reason
    trigger = _clean_controller_memory_text(value.get("trigger"), limit=40)
    if trigger is not None:
        normalized["trigger"] = trigger.lower().replace(" ", "_")
    return normalized


def _normalize_controller_tool_name(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split()).strip().lower().replace("-", "_")
    return text or None


def _normalize_controller_tool_request(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    tool_name = _normalize_controller_tool_name(value.get("tool") or value.get("name") or value.get("kind"))
    if tool_name not in _CONTROLLER_TOOL_NAMES:
        return None
    normalized: dict[str, Any] = {"tool": tool_name}
    reason = _clean_controller_memory_text(value.get("reason"), limit=120)
    if reason is not None:
        normalized["reason"] = reason
    if tool_name == "tokenize_terms":
        raw_terms = value.get("terms") or value.get("features") or ()
        if not isinstance(raw_terms, SequenceABC) or isinstance(raw_terms, (str, bytes, bytearray)):
            return None
        terms: list[str] = []
        for raw_term in raw_terms:
            term = _clean_controller_memory_text(raw_term, limit=80)
            if term is None or term in terms:
                continue
            terms.append(term)
        if not terms:
            return None
        normalized["terms"] = terms[:8]
        return normalized
    if tool_name == "constraint_scorer":
        candidate = _clean_controller_memory_text(value.get("candidate"), limit=240)
        if candidate is not None:
            normalized["candidate"] = candidate
        source_text = _clean_controller_memory_text(value.get("source"), limit=240)
        if source_text is not None:
            normalized["source"] = source_text
        constraints = value.get("constraints")
        if isinstance(constraints, Mapping):
            normalized["constraints"] = dict(constraints)
        return normalized
    candidate_edit = value.get("candidate_edit") or value.get("edit")
    if not isinstance(candidate_edit, Mapping):
        return None
    normalized["candidate_edit"] = dict(candidate_edit)
    max_new_tokens = value.get("max_new_tokens", 4)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        max_new_tokens = 4
    normalized["max_new_tokens"] = max(1, min(6, int(max_new_tokens)))
    top_k = value.get("top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        top_k = 5
    normalized["top_k"] = max(1, min(8, int(top_k)))
    return normalized


def _tool_contains_term(text: str, term: str) -> bool:
    haystack = " ".join(str(text).split()).strip()
    needle = " ".join(str(term).split()).strip()
    if not haystack or not needle:
        return False
    escaped = re.escape(needle)
    if any(char.isalnum() for char in needle):
        return re.search(rf"(?i)(?<!\w){escaped}(?!\w)", haystack) is not None
    return needle.lower() in haystack.lower()


def _tool_term_span_progress(text: str, term: str) -> float:
    haystack = " ".join(str(text).lower().split()).strip()
    needle = " ".join(str(term).lower().split()).strip()
    if not haystack or not needle:
        return 0.0
    if needle in haystack:
        return 1.0
    best = 0
    for width in range(len(needle), 0, -1):
        if needle[:width] in haystack:
            best = width
            break
    return round(best / max(1, len(needle)), 6)


class HookedTransformerWorkerRuntime:
    def __init__(
        self,
        *,
        runtime_state: Any,
        adapter: ModelAdapter,
        surface_catalog: Sequence[SurfaceInfo | Mapping[str, Any]],
        codec: TextCodec | None = None,
        model: Any | None = None,
        run_id: str = "run_v0",
        episode_id: str = "episode_v0",
        worker_id: str = "os_0",
        task_id: str = "task_v0",
        task_view_mode: str = "redacted",
        goal_hint: str | None = None,
        constraints: Sequence[str] | None = None,
        max_generated_tokens: int = 32,
        min_generated_tokens: int = 0,
        max_edits_per_step: int = 1,
        max_edits_per_run: int = 4,
        max_total_alpha: float = 0.5,
        max_total_edit_cost: float | None = None,
        max_loop_rescue_edits_per_run: int = 0,
        max_loop_rescue_alpha: float = 0.0,
        max_loop_rescue_edit_cost: float | None = None,
        max_active_patch_slots: int = 1,
        generated_tail_chars: int = 80,
        recent_token_count: int = 6,
        stop_token_ids: Sequence[int] | None = None,
        allowed_token_ids: Sequence[int] | None = None,
        stop_checker: Callable[[str], bool] | None = None,
        trace_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        task_feedback_fn: Callable[[str], Mapping[str, Any] | None] | None = None,
        observer_check_fn: Callable[..., Mapping[str, Any] | None] | None = None,
        trace_recorder: StepAlignedTraceRecorder | None = None,
        controller_reflection_mode: str = "off",
        controller_memory_window: int = 3,
        decoder_control_mode: str = "off",
        max_observer_checks_per_run: int = 4,
        observer_check_window: int = 4,
        max_tool_calls_per_run: int = 6,
        tool_result_window: int = 6,
    ) -> None:
        self.runtime_state = runtime_state
        self.adapter = adapter
        self.model = model if model is not None else getattr(runtime_state, "model", None)
        self.codec = resolve_text_codec(self.model, codec)

        self.run_id = run_id
        self.episode_id = episode_id
        self.worker_id = worker_id
        self.task_id = task_id
        self.task_view_mode = task_view_mode
        self.goal_hint = goal_hint
        self.constraints = tuple(constraints or ())
        self.max_generated_tokens = max_generated_tokens
        self.min_generated_tokens = max(0, int(min_generated_tokens))
        self.max_edits_per_step = max_edits_per_step
        self.max_edits_per_run = max_edits_per_run
        self.max_total_alpha = max_total_alpha
        self.max_total_edit_cost = float(max_total_edit_cost if max_total_edit_cost is not None else max_total_alpha)
        self.max_loop_rescue_edits_per_run = max(0, int(max_loop_rescue_edits_per_run))
        self.max_loop_rescue_alpha = max(0.0, float(max_loop_rescue_alpha))
        self.max_loop_rescue_edit_cost = float(
            max_loop_rescue_edit_cost if max_loop_rescue_edit_cost is not None else self.max_loop_rescue_alpha
        )
        self.max_active_patch_slots = max_active_patch_slots
        self.generated_tail_chars = generated_tail_chars
        self.recent_token_count = recent_token_count
        self.stop_token_ids = set(int(token_id) for token_id in (stop_token_ids or ()))
        self.allowed_token_ids = tuple(sorted({int(token_id) for token_id in (allowed_token_ids or ())}))
        self.stop_checker = stop_checker
        self.trace_metadata = {str(trace_id): dict(meta) for trace_id, meta in (trace_metadata or {}).items()}
        self.task_feedback_fn = task_feedback_fn
        self.observer_check_fn = observer_check_fn
        self.controller_reflection_mode = _normalize_controller_reflection_mode(controller_reflection_mode)
        self.controller_memory_window = max(1, int(controller_memory_window))
        self.decoder_control_mode = _normalize_decoder_control_mode(decoder_control_mode)
        self.max_observer_checks_per_run = max(0, int(max_observer_checks_per_run))
        self.observer_check_window = max(1, int(observer_check_window))
        self.max_tool_calls_per_run = max(0, int(max_tool_calls_per_run))
        self.tool_result_window = max(1, int(tool_result_window))

        self.surface_catalog = tuple(
            surface if isinstance(surface, SurfaceInfo) else SurfaceInfo.from_dict(surface) for surface in surface_catalog
        )
        self._surface_catalog_raw = [self._surface_to_dict(surface) for surface in self.surface_catalog]
        self.trace_recorder = trace_recorder or StepAlignedTraceRecorder(
            surface_catalog=self.surface_catalog,
            adapter=self.adapter,
        )

        self.prompt = ""
        self._segments: list[_TokenSegment] = []
        self._steps = 0
        self._last_metrics: dict[str, float] = {
            "entropy": 0.0,
            "top1_margin": 0.0,
            "repetition_score": 0.0,
        }
        self._last_task_feedback: dict[str, Any] = {"done": False, "progress_label": "progressing"}
        self._last_status = "thinking"
        self._previous_probe_vectors: dict[str, torch.Tensor] = {}
        self._recent_effects: list[dict[str, Any]] = []
        self._latest_completed_effects: list[dict[str, Any]] = []
        self._recent_effect_summary: dict[str, Any] = summarize_effects(())
        self._controller_memory: list[dict[str, Any]] = []
        self._observer_checks: list[dict[str, Any]] = []
        self._latest_observer_check: dict[str, Any] | None = None
        self._pending_observer_check_events: list[dict[str, Any]] = []
        self._tool_results: list[dict[str, Any]] = []
        self._latest_tool_results: list[dict[str, Any]] = []
        self._pending_tool_events: list[dict[str, Any]] = []
        self._last_observer_candidate_hash: str | None = None
        self._feature_prototype_cache: dict[str, torch.Tensor] = {}
        self._kv_projection_cache: dict[tuple[int, str, int, int, int], torch.Tensor | None] = {}
        self._kv_canary_eval_active = False
        self._last_simulate_decode_error: str | None = None
        self._last_decoder_control: dict[str, Any] = {
            "mode": self.decoder_control_mode,
            "track": _decoder_control_spec(self.decoder_control_mode)["track"],
            "active": False,
            "loop_cycle_length": 0,
        }
        self._pending_effects: list[dict[str, Any]] = []
        self._seen_registration_ids: set[str] = set()
        self._spent_budget_alpha: dict[str, float] = {}
        self._spent_budget_cost: dict[str, float] = {}
        self._spent_loop_rescue_alpha: dict[str, float] = {}
        self._spent_loop_rescue_cost: dict[str, float] = {}
        self._last_packet: dict[str, Any] | None = None
        self._no_progress_steps = 0

    def reset(self, prompt: str) -> None:
        self._clear_episode_state()
        self.prompt = prompt
        prompt_tokens = self.codec.encode(prompt).detach().reshape(-1).to(dtype=torch.long).tolist()
        if not prompt_tokens:
            raise ValueError("prompt must encode to at least one token")
        self._segments = [_TokenSegment(kind="prompt", token_ids=prompt_tokens)]

    def step(self) -> None:
        previous_metrics = dict(self._last_metrics)
        previous_feedback = dict(self._last_task_feedback)
        previous_status = str(self._last_status)
        previous_no_progress_steps = int(self._no_progress_steps)
        tokens = self._current_token_tensor()
        logits, _cache = self.runtime_state.run_with_cache(tokens, return_type="logits")
        if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
            raise ValueError(f"runtime_state.run_with_cache must return logits shaped [batch, pos, vocab], got {type(logits)!r}")
        next_logits = self._apply_token_constraints(logits[0, -1].detach())
        next_logits, decoder_control = self._apply_decoder_control(next_logits)
        next_token = int(torch.argmax(next_logits).item())
        self._append_output_token(next_token)
        self._steps += 1
        self._last_decoder_control = decoder_control
        self._last_metrics = self._compute_metrics(next_logits)
        self._update_status()
        self._last_task_feedback = self._compute_task_feedback()
        self._maybe_run_observer_check(
            previous_feedback=previous_feedback,
            previous_metrics=previous_metrics,
            previous_status=previous_status,
            previous_no_progress_steps=previous_no_progress_steps,
        )
        self._record_trace_step(emitted_token_id=next_token)
        self._last_packet = None

    def done(self) -> bool:
        output_tokens = self._output_token_ids()
        generated_tokens = len(output_tokens)
        if self.max_generated_tokens > 0 and generated_tokens >= self.max_generated_tokens:
            return True
        if generated_tokens < self.min_generated_tokens:
            return False
        if output_tokens and output_tokens[-1] in self.stop_token_ids:
            return True
        if self.stop_checker is not None and self.stop_checker(self.final_text()):
            return True
        return False

    def append_prompt_hint(self, hint: str) -> bool:
        hint_tokens = self.codec.encode(hint).detach().reshape(-1).to(dtype=torch.long).tolist()
        if not hint_tokens:
            return False
        self._segments.append(_TokenSegment(kind="hint", token_ids=hint_tokens))
        self._last_packet = None
        return True

    def build_controller_packet(self) -> dict[str, Any]:
        if hasattr(self.runtime_state, "set_trace_alignment"):
            self.runtime_state.set_trace_alignment(self._steps)
        active_edits = self._collect_active_edits()
        promoted_cache_surfaces = self._promoted_cache_surfaces()
        control_phase_hint = self._control_phase_hint()
        strategy_hints = self._strategy_hints(
            control_phase_hint=control_phase_hint,
            promoted_cache_surfaces=promoted_cache_surfaces,
        )
        latest_observer_check = self._observer_check_with_cache_surface_ids(
            self._latest_observer_check,
            promoted_cache_surfaces=promoted_cache_surfaces,
        )
        packet = {
            "version": "0.1",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "worker_id": self.worker_id,
            "step": self._steps,
            "horizon": {
                "generated_tokens": len(self._output_token_ids()),
                "max_generated_tokens": self.max_generated_tokens,
                "done": self.done(),
            },
            "task_view": self._task_view(),
            "worker_view": self._worker_view(),
            "telemetry": dict(self._last_metrics)
            | {
                "repeat_flag": self._repeat_flag(),
                "no_progress_steps": self._no_progress_steps,
                "loop_cycle_length": int(self._loop_cycle_length() or 0),
                "observer_check_count": len(self._observer_checks),
                "observer_check_budget_left": max(0, self.max_observer_checks_per_run - len(self._observer_checks)),
                "observer_check_last_trigger": None
                if self._latest_observer_check is None
                else str(self._latest_observer_check.get("trigger", "")),
                "tool_call_count": len(self._tool_results),
                "tool_call_budget_left": max(0, self.max_tool_calls_per_run - len(self._tool_results)),
                "decoder_control_mode": self.decoder_control_mode,
                "decoder_control_track": str(self._last_decoder_control.get("track", "baseline")),
                "decoder_rescue_active": bool(self._last_decoder_control.get("active", False)),
                "decoder_candidate_prune_active": bool(self._last_decoder_control.get("candidate_prune_active", False)),
                "decoder_constraint_target_count": int(self._last_decoder_control.get("constraint_target_count", 0) or 0),
                "decoder_entity_prior_active": bool(self._last_decoder_control.get("entity_prior_active", False)),
                "decoder_entity_target_count": int(self._last_decoder_control.get("entity_target_count", 0) or 0),
                "decoder_entity_prefix_depth": int(self._last_decoder_control.get("entity_prefix_depth", 0) or 0),
                "decoder_logit_bias_active": bool(self._last_decoder_control.get("logit_bias_active", False)),
                "decoder_logit_bias_term_count": int(self._last_decoder_control.get("logit_bias_term_count", 0) or 0),
                "decoder_logit_bias_token_count": int(self._last_decoder_control.get("logit_bias_token_count", 0) or 0),
                "decoder_logit_bias_focus_term_count": int(
                    self._last_decoder_control.get("logit_bias_focus_term_count", 0) or 0
                ),
            },
            "surface_catalog": self._packet_surface_catalog(promoted_cache_surfaces=promoted_cache_surfaces),
            "probe_frames": self._build_probe_frames(),
            "trace_bank": self._build_trace_bank(),
            "active_edits": active_edits,
            "recent_effects": [dict(effect) for effect in self._recent_effects],
            "recent_effect_summary": dict(self._recent_effect_summary),
            "budget": self._budget_state(active_edits),
            "task_feedback": dict(self._last_task_feedback),
            "tool_catalog": self._build_tool_catalog(),
            "control_phase_hint": control_phase_hint,
            "strategy_hints": strategy_hints,
        }
        if latest_observer_check is not None:
            packet["latest_observer_check"] = latest_observer_check
        if self._observer_checks:
            packet["recent_observer_checks"] = [
                self._observer_check_with_cache_surface_ids(check, promoted_cache_surfaces=promoted_cache_surfaces)
                for check in self._observer_checks
            ]
        if self._latest_tool_results:
            packet["latest_tool_results"] = [dict(result) for result in self._latest_tool_results]
        if self._tool_results:
            packet["recent_tool_results"] = [dict(result) for result in self._tool_results]
        if self.controller_reflection_mode != "off":
            packet["controller_memory"] = [dict(entry) for entry in self._controller_memory]
        self._last_packet = packet
        return packet

    def record_controller_memory(self, entry: Mapping[str, Any], *, decision: str | None = None) -> dict[str, Any] | None:
        if self.controller_reflection_mode == "off":
            return None
        normalized = _normalize_controller_memory_entry(entry, decision=decision, recorded_step=self._steps)
        if normalized is None:
            return None
        self._controller_memory.append(normalized)
        self._controller_memory = self._controller_memory[-self.controller_memory_window :]
        self._last_packet = None
        return dict(normalized)

    def pop_observer_check_events(self) -> list[dict[str, Any]]:
        events = [dict(event) for event in self._pending_observer_check_events]
        self._pending_observer_check_events = []
        return events

    def pop_tool_events(self) -> list[dict[str, Any]]:
        events = [dict(event) for event in self._pending_tool_events]
        self._pending_tool_events = []
        return events

    def latent_feature_scan(
        self,
        *,
        feature_groups: Mapping[str, Any],
        max_features_per_group: int = 4,
        max_surface_hits: int = 2,
    ) -> dict[str, Any] | None:
        if not isinstance(feature_groups, Mapping):
            return None
        surface_vectors = self._feature_scan_surface_vectors()
        if not surface_vectors:
            return None

        groups: list[dict[str, Any]] = []
        top_feature_hits: list[dict[str, Any]] = []
        term_progress = self._feedback_term_progress_by_term()

        for raw_group_name, raw_spec in feature_groups.items():
            group_name = " ".join(str(raw_group_name).split()).strip().lower().replace(" ", "_")
            if not group_name:
                continue

            polarity = "promote"
            feature_kind = "term"
            raw_terms: Any = raw_spec
            if isinstance(raw_spec, Mapping):
                raw_terms = raw_spec.get("terms", ())
                polarity = " ".join(str(raw_spec.get("polarity", "promote")).split()).strip().lower().replace(" ", "_")
                feature_kind = " ".join(str(raw_spec.get("feature_kind", "term")).split()).strip().lower().replace(" ", "_")
            if not isinstance(raw_terms, SequenceABC) or isinstance(raw_terms, (str, bytes, bytearray)):
                continue

            feature_rows: list[dict[str, Any]] = []
            seen_terms: set[str] = set()
            for raw_term in raw_terms:
                term = " ".join(str(raw_term).split()).strip()
                if not term or term in seen_terms:
                    continue
                seen_terms.add(term)
                prototype = self._feature_prototype_vector(term)
                if prototype is None:
                    continue

                surface_hits: list[dict[str, Any]] = []
                for surface_id, surface_vector in surface_vectors.items():
                    alignment = _cosine_similarity(surface_vector, prototype)
                    if alignment is None:
                        continue
                    surface_hits.append(
                        {
                            "surface_id": str(surface_id),
                            "alignment": round(max(0.0, float(alignment)), 6),
                        }
                    )
                if not surface_hits:
                    continue
                surface_hits.sort(key=lambda item: (-float(item["alignment"]), str(item["surface_id"])))
                best_hit = surface_hits[0]
                feature_rows.append(
                    {
                        "feature": term,
                        "feature_kind": feature_kind,
                        "polarity": polarity,
                        "alignment": best_hit["alignment"],
                        "surface_id": best_hit["surface_id"],
                        "surface_hits": surface_hits[: max(1, int(max_surface_hits))],
                        "coverage_progress": round(float(term_progress.get(term, 0.0) or 0.0), 6),
                    }
                )

            if not feature_rows:
                continue
            feature_rows.sort(
                key=lambda item: (
                    -float(item["alignment"]),
                    -float(item.get("coverage_progress", 0.0) or 0.0),
                    str(item["feature"]).lower(),
                )
            )
            top_rows = feature_rows[: max(1, int(max_features_per_group))]
            mean_alignment = sum(float(item["alignment"]) for item in feature_rows) / len(feature_rows)
            groups.append(
                {
                    "group": group_name,
                    "polarity": polarity,
                    "feature_kind": feature_kind,
                    "feature_count": len(feature_rows),
                    "mean_alignment": round(mean_alignment, 6),
                    "top_features": top_rows,
                }
            )
            for item in top_rows:
                top_feature_hits.append(
                    {
                        "group": group_name,
                        "feature": item["feature"],
                        "feature_kind": item["feature_kind"],
                        "polarity": item["polarity"],
                        "alignment": item["alignment"],
                        "surface_id": item["surface_id"],
                        "coverage_progress": item["coverage_progress"],
                    }
                )

        if not groups:
            return None

        top_feature_hits.sort(
            key=lambda item: (
                -float(item["alignment"]),
                -float(item.get("coverage_progress", 0.0) or 0.0),
                str(item["feature"]).lower(),
            )
        )
        mean_alignment = sum(float(group["mean_alignment"]) for group in groups) / len(groups)
        max_alignment = max(float(item["alignment"]) for item in top_feature_hits)
        return {
            "prototype_mode": "token_embedding_mean",
            "surface_count": len(surface_vectors),
            "group_count": len(groups),
            "mean_alignment": round(mean_alignment, 6),
            "max_alignment": round(max_alignment, 6),
            "groups": groups,
            "top_feature_hits": top_feature_hits[:6],
        }

    def kv_feature_scan(
        self,
        *,
        feature_groups: Mapping[str, Any],
        max_features_per_group: int = 4,
        max_surface_hits: int = 2,
    ) -> dict[str, Any] | None:
        if not isinstance(feature_groups, Mapping):
            return None
        cache_vectors = self._kv_scan_surface_vectors()
        if not cache_vectors:
            return None

        def _group_priority(item: Mapping[str, Any]) -> int:
            group = str(item.get("group", "") or "")
            if group == "required_terms":
                return 0
            if group == "missing_keywords":
                return 1
            if group == "missing_summary_terms":
                return 2
            return 3

        def _polarity_priority(item: Mapping[str, Any]) -> int:
            return 0 if str(item.get("polarity", "promote") or "promote") == "promote" else 1

        def _site_priority(item: Mapping[str, Any]) -> int:
            return 0 if str(item.get("site", "")) == "v_cache" else 1

        groups: list[dict[str, Any]] = []
        top_feature_hits: list[dict[str, Any]] = []
        term_progress = self._feedback_term_progress_by_term()
        position_records = self._token_position_records()
        cache_tensor_cache: dict[tuple[int, str], torch.Tensor | None] = {}

        for raw_group_name, raw_spec in feature_groups.items():
            group_name = " ".join(str(raw_group_name).split()).strip().lower().replace(" ", "_")
            if not group_name:
                continue

            polarity = "promote"
            feature_kind = "term"
            raw_terms: Any = raw_spec
            if isinstance(raw_spec, Mapping):
                raw_terms = raw_spec.get("terms", ())
                polarity = " ".join(str(raw_spec.get("polarity", "promote")).split()).strip().lower().replace(" ", "_")
                feature_kind = " ".join(str(raw_spec.get("feature_kind", "term")).split()).strip().lower().replace(" ", "_")
            if not isinstance(raw_terms, SequenceABC) or isinstance(raw_terms, (str, bytes, bytearray)):
                continue

            feature_rows: list[dict[str, Any]] = []
            seen_terms: set[str] = set()
            for raw_term in raw_terms:
                term = " ".join(str(raw_term).split()).strip()
                if not term or term in seen_terms:
                    continue
                seen_terms.add(term)
                prototype = self._feature_prototype_vector(term)
                if prototype is None:
                    continue

                surface_hits: list[dict[str, Any]] = []
                for cache_spec in cache_vectors:
                    cache_key = (int(cache_spec["layer"]), str(cache_spec["site"]))
                    cache_tensor = cache_tensor_cache.get(cache_key)
                    if cache_key not in cache_tensor_cache:
                        cache_tensor = self._cache_tensor_for_scan(layer=int(cache_spec["layer"]), site=str(cache_spec["site"]))
                        cache_tensor_cache[cache_key] = cache_tensor
                    if cache_tensor is None:
                        continue
                    source_positions = self._kv_source_positions_for_feature(
                        prototype,
                        cache_tensor=cache_tensor,
                        layer=int(cache_spec["layer"]),
                        site=str(cache_spec["site"]),
                        head=int(cache_spec["head"]),
                        width=int(cache_spec["width"]),
                        head_count=int(cache_spec["head_count"]),
                        position_records=position_records,
                        max_positions=max_surface_hits + 1,
                    )
                    if not source_positions:
                        continue
                    best_source = source_positions[0]
                    hit = {
                        "site": cache_spec["site"],
                        "layer": cache_spec["layer"],
                        "head": cache_spec["head"],
                        "token_mode": cache_spec["token_mode"],
                        "alignment": round(float(best_source["alignment"]), 6),
                        "argmax_pos": int(best_source["position"]),
                        "argmax_relative_index": int(best_source["relative_index"]),
                        "argmax_piece": str(best_source["piece"]),
                        "argmax_segment_kind": str(best_source["segment_kind"]),
                        "source_positions": [
                            {
                                "position": int(item["position"]),
                                "relative_index": int(item["relative_index"]),
                                "segment_kind": str(item["segment_kind"]),
                                "piece": str(item["piece"]),
                                "alignment": round(float(item["alignment"]), 6),
                            }
                            for item in source_positions[: max(1, int(max_surface_hits))]
                        ],
                    }
                    if cache_spec.get("surface_id"):
                        hit["surface_id"] = cache_spec["surface_id"]
                    surface_hits.append(hit)
                if not surface_hits:
                    continue
                surface_hits.sort(
                    key=lambda item: (
                        -float(item["alignment"]),
                        str(item["site"]),
                        int(item["layer"]),
                        int(item["head"]),
                    )
                )
                best_hit = surface_hits[0]
                feature_rows.append(
                    {
                        "feature": term,
                        "feature_kind": feature_kind,
                        "polarity": polarity,
                        "alignment": best_hit["alignment"],
                        "site": best_hit["site"],
                        "layer": best_hit["layer"],
                        "head": best_hit["head"],
                        "token_mode": best_hit["token_mode"],
                        "surface_id": best_hit.get("surface_id"),
                        "argmax_pos": best_hit.get("argmax_pos"),
                        "argmax_relative_index": best_hit.get("argmax_relative_index"),
                        "argmax_piece": best_hit.get("argmax_piece"),
                        "argmax_segment_kind": best_hit.get("argmax_segment_kind"),
                        "source_positions": list(best_hit.get("source_positions", [])),
                        "surface_hits": surface_hits[: max(1, int(max_surface_hits))],
                        "coverage_progress": round(float(term_progress.get(term, 0.0) or 0.0), 6),
                    }
                )

            if not feature_rows:
                continue
            feature_rows.sort(
                key=lambda item: (
                    -float(item["alignment"]),
                    -float(item.get("coverage_progress", 0.0) or 0.0),
                    str(item["feature"]).lower(),
                )
            )
            top_rows = feature_rows[: max(1, int(max_features_per_group))]
            mean_alignment = sum(float(item["alignment"]) for item in feature_rows) / len(feature_rows)
            groups.append(
                {
                    "group": group_name,
                    "polarity": polarity,
                    "feature_kind": feature_kind,
                    "feature_count": len(feature_rows),
                    "mean_alignment": round(mean_alignment, 6),
                    "top_features": top_rows,
                }
            )
            for item in top_rows:
                top_feature_hits.append(
                    {
                        "group": group_name,
                        "feature": item["feature"],
                        "feature_kind": item["feature_kind"],
                        "polarity": item["polarity"],
                        "site": item["site"],
                        "layer": item["layer"],
                        "head": item["head"],
                        "token_mode": item["token_mode"],
                        "alignment": item["alignment"],
                        "surface_id": item.get("surface_id"),
                        "argmax_pos": item.get("argmax_pos"),
                        "argmax_relative_index": item.get("argmax_relative_index"),
                        "argmax_piece": item.get("argmax_piece"),
                        "argmax_segment_kind": item.get("argmax_segment_kind"),
                        "source_positions": list(item.get("source_positions", [])),
                        "coverage_progress": item["coverage_progress"],
                    }
                )

        if not groups:
            return None

        top_feature_hits.sort(
            key=lambda item: (
                _group_priority(item),
                _polarity_priority(item),
                _site_priority(item),
                -float(item["alignment"]),
                -float(item.get("coverage_progress", 0.0) or 0.0),
                int(item["layer"]),
                int(item["head"]),
                str(item.get("feature", "")).lower(),
            )
        )
        mean_alignment = sum(float(group["mean_alignment"]) for group in groups) / len(groups)
        max_alignment = max(float(item["alignment"]) for item in top_feature_hits)
        return {
            "projection_mode": "attn_weight_head_projection",
            "surface_count": len(cache_vectors),
            "group_count": len(groups),
            "mean_alignment": round(mean_alignment, 6),
            "max_alignment": round(max_alignment, 6),
            "groups": groups,
            "top_feature_hits": top_feature_hits[:6],
        }

    def request_observer_check(
        self,
        request: Mapping[str, Any] | None = None,
        *,
        source: str = "controller",
    ) -> dict[str, Any] | None:
        normalized_request = _normalize_observer_check_request(request if request is not None else {"kind": "semantic_progress"})
        if normalized_request is None or self.observer_check_fn is None or self.max_observer_checks_per_run <= 0:
            return None
        if len(self._observer_checks) >= self.max_observer_checks_per_run:
            return None
        candidate_text = self.final_text()
        if not candidate_text.strip():
            return None
        candidate_hash = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
        trigger = str(normalized_request.get("trigger", f"{source}_request"))
        result = self._invoke_observer_check(candidate_text, trigger=trigger)
        normalized = self._normalize_observer_check_result(
            result,
            trigger=trigger,
            candidate_hash=candidate_hash,
            previous_result=self._latest_observer_check,
        )
        if normalized is None:
            return None
        normalized["requested_by"] = str(source)
        normalized["request_kind"] = str(normalized_request.get("kind", "semantic_progress"))
        if normalized_request.get("reason") is not None:
            normalized["request_reason"] = str(normalized_request["reason"])
        self._latest_observer_check = normalized
        self._observer_checks.append(normalized)
        self._observer_checks = self._observer_checks[-self.observer_check_window :]
        self._pending_observer_check_events.append(dict(normalized))
        self._last_observer_candidate_hash = candidate_hash
        self._last_packet = None
        return dict(normalized)

    def request_controller_tools(
        self,
        requests: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None,
        *,
        source: str = "controller",
    ) -> list[dict[str, Any]]:
        if self.max_tool_calls_per_run <= 0:
            return []
        raw_items: Sequence[Any]
        if isinstance(requests, Mapping):
            raw_items = [requests]
        elif isinstance(requests, SequenceABC) and not isinstance(requests, (str, bytes, bytearray)):
            raw_items = requests
        else:
            return []
        budget_left = max(0, self.max_tool_calls_per_run - len(self._tool_results))
        if budget_left <= 0:
            return []

        results: list[dict[str, Any]] = []
        for raw_request in raw_items[:budget_left]:
            request = _normalize_controller_tool_request(raw_request)
            if request is None:
                continue
            result = self._execute_controller_tool_request(request, source=source)
            if result is None:
                continue
            results.append(result)

        if not results:
            return []
        self._latest_tool_results = [dict(result) for result in results]
        self._tool_results.extend(self._latest_tool_results)
        self._tool_results = self._tool_results[-self.tool_result_window :]
        self._pending_tool_events.extend(dict(result) for result in self._latest_tool_results)
        self._last_packet = None
        return [dict(result) for result in self._latest_tool_results]

    def observe_recent_effects(self) -> None:
        current_metrics = self._effect_metrics()
        completed = [
            build_edit_effect(
                edit_id=pending["edit_id"],
                surface_id=pending["surface_id"],
                observed_window_steps=1,
                before=pending["before"],
                after=current_metrics,
                hypothesis=pending.get("hypothesis"),
                expected_effect=pending.get("expected_effect"),
                controller_confidence=pending.get("controller_confidence"),
                op=pending.get("op"),
                step_size=pending.get("step_size"),
                edit_cost=pending.get("edit_cost"),
            )
            for pending in self._pending_effects
        ]
        if completed:
            self._recent_effects.extend(completed)
            self._recent_effects = self._recent_effects[-8:]
        self._latest_completed_effects = completed
        self._recent_effect_summary = summarize_effects(self._recent_effects)
        self._pending_effects = []

        for active in self._collect_active_edits():
            edit_id = str(active["edit_id"])
            if edit_id in self._seen_registration_ids:
                continue
            self._seen_registration_ids.add(edit_id)
            budget_key = str(active.get("budget_key", edit_id.split(":", 1)[0]))
            budget_pool = str(active.get("budget_pool", MAIN_EDIT_BUDGET_POOL) or MAIN_EDIT_BUDGET_POOL)
            if budget_pool == LOOP_RESCUE_EDIT_BUDGET_POOL:
                self._spent_loop_rescue_alpha.setdefault(budget_key, float(active["alpha"]))
                self._spent_loop_rescue_cost.setdefault(budget_key, float(active.get("edit_cost", 0.0) or 0.0))
            else:
                self._spent_budget_alpha.setdefault(budget_key, float(active["alpha"]))
                self._spent_budget_cost.setdefault(budget_key, float(active.get("edit_cost", 0.0) or 0.0))
            self._pending_effects.append(
                {
                    "edit_id": edit_id,
                    "surface_id": str(active["surface_id"]),
                    "before": current_metrics,
                    "hypothesis": active.get("hypothesis"),
                    "expected_effect": active.get("expected_effect"),
                    "controller_confidence": active.get("controller_confidence"),
                    "op": active.get("op"),
                    "step_size": active.get("step_size"),
                    "edit_cost": active.get("edit_cost"),
                }
            )

    def tick_ttl(self) -> None:
        if hasattr(self.runtime_state, "tick_ttl"):
            self.runtime_state.tick_ttl()

    def cleanup_expired(self) -> None:
        if hasattr(self.runtime_state, "cleanup_expired"):
            self.runtime_state.cleanup_expired()

    def final_text(self) -> str:
        return self.codec.decode(self._output_token_ids())

    def current_tokens(self) -> torch.Tensor:
        return self._current_token_tensor().detach().clone()

    def export_step_trace(
        self,
        trace_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepAlignedTrace | None:
        if self.trace_recorder is None or self.trace_recorder.step_count == 0:
            return None
        merged_metadata = dict(self.trace_metadata.get(trace_id, {}))
        merged_metadata.update(metadata or {})
        return self.trace_recorder.snapshot(trace_id, metadata=merged_metadata)

    def snapshot_trace(
        self,
        trace_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> StepAlignedTrace | Mapping[str, torch.Tensor] | None:
        trace = self.export_step_trace(trace_id, metadata=metadata)
        if trace is not None:
            if hasattr(self.runtime_state, "put_step_trace"):
                self.runtime_state.put_step_trace(trace_id, trace)
            return trace
        last_cache = getattr(self.runtime_state, "last_cache", None)
        if last_cache is not None and hasattr(self.runtime_state, "put_trace_cache"):
            frozen = {str(name): tensor.detach().clone() for name, tensor in last_cache.items()}
            self.runtime_state.put_trace_cache(trace_id, frozen)
            return frozen
        return None

    def _clear_episode_state(self) -> None:
        if hasattr(self.runtime_state, "clear_edits"):
            self.runtime_state.clear_edits()
        else:
            for edit_id in list(getattr(self.runtime_state, "hooks", {})):
                self.runtime_state.remove_edit(edit_id)
            for edit_id in list(getattr(self.runtime_state, "overlays", {})):
                self.runtime_state.remove_edit(edit_id)
        self._segments = []
        self._steps = 0
        self._last_metrics = {"entropy": 0.0, "top1_margin": 0.0, "repetition_score": 0.0}
        self._last_task_feedback = {"done": False, "progress_label": "progressing"}
        self._last_status = "thinking"
        self._previous_probe_vectors = {}
        self._recent_effects = []
        self._latest_completed_effects = []
        self._recent_effect_summary = summarize_effects(())
        self._controller_memory = []
        self._observer_checks = []
        self._latest_observer_check = None
        self._pending_observer_check_events = []
        self._tool_results = []
        self._latest_tool_results = []
        self._pending_tool_events = []
        self._last_observer_candidate_hash = None
        self._kv_canary_eval_active = False
        self._last_simulate_decode_error = None
        self._last_decoder_control = {
            "mode": self.decoder_control_mode,
            "track": _decoder_control_spec(self.decoder_control_mode)["track"],
            "active": False,
            "loop_cycle_length": 0,
        }
        self._pending_effects = []
        self._seen_registration_ids = set()
        self._spent_budget_alpha = {}
        self._spent_budget_cost = {}
        self._spent_loop_rescue_alpha = {}
        self._spent_loop_rescue_cost = {}
        self._last_packet = None
        self._no_progress_steps = 0
        if self.trace_recorder is not None:
            self.trace_recorder.reset()

    def _current_token_tensor(self) -> torch.Tensor:
        tokens: list[int] = []
        for segment in self._segments:
            tokens.extend(segment.token_ids)
        tensor_kwargs: dict[str, Any] = {"dtype": torch.long}
        device = self._model_device()
        if device is not None:
            tensor_kwargs["device"] = device
        return torch.tensor([tokens], **tensor_kwargs)

    def _model_device(self) -> torch.device | None:
        model = self.model if self.model is not None else getattr(self.runtime_state, "model", None)
        if model is None:
            return None
        cfg = getattr(model, "cfg", None)
        cfg_device = getattr(cfg, "device", None)
        if cfg_device is not None:
            try:
                return torch.device(cfg_device)
            except Exception:
                pass
        try:
            return next(model.parameters()).device
        except Exception:
            return None

    def _append_output_token(self, token_id: int) -> None:
        if self._segments and self._segments[-1].kind == "output":
            self._segments[-1].token_ids.append(int(token_id))
            return
        self._segments.append(_TokenSegment(kind="output", token_ids=[int(token_id)]))

    def _apply_token_constraints(self, next_logits: torch.Tensor) -> torch.Tensor:
        if not self.allowed_token_ids:
            return next_logits
        allowed = [token_id for token_id in self.allowed_token_ids if 0 <= token_id < next_logits.shape[-1]]
        if not allowed:
            return next_logits
        masked = torch.full_like(next_logits, float("-inf"))
        index = torch.tensor(allowed, dtype=torch.long, device=next_logits.device)
        masked[index] = next_logits[index]
        return masked

    def _apply_decoder_control(self, next_logits: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        cycle_length = self._loop_cycle_length()
        spec = _decoder_control_spec(self.decoder_control_mode)
        state = {
            "mode": self.decoder_control_mode,
            "track": spec["track"],
            "objective": spec["objective"],
            "active": False,
            "loop_cycle_length": int(cycle_length or 0),
            "candidate_prune_active": False,
            "constraint_target_count": 0,
            "entity_prior_active": False,
            "entity_target_count": 0,
            "entity_prefix_depth": 0,
            "logit_bias_active": False,
            "logit_bias_term_count": 0,
            "logit_bias_token_count": 0,
            "logit_bias_focus_term_count": 0,
        }
        if self.decoder_control_mode == "off":
            return next_logits, state

        tokens = self._output_token_ids()
        rescue_active = cycle_length is not None or self._repeat_flag() or self._no_progress_steps > 0
        bias_ready = bool(self._feedback_terms(_ENTITY_RECALL_FEEDBACK_KEYS))
        if not tokens:
            return next_logits, state
        if self.decoder_control_mode != "logit_bias_entity_soft" and not rescue_active:
            return next_logits, state
        if self.decoder_control_mode == "logit_bias_entity_soft" and not (rescue_active or bias_ready):
            return next_logits, state

        adjusted = next_logits.clone()
        if rescue_active:
            adjusted, loop_state = self._apply_loop_aware_penalties(adjusted, cycle_length=cycle_length)
            state.update(loop_state)
            state["active"] = bool(loop_state.get("active", False))

        if self.decoder_control_mode == "loop_aware_prune":
            adjusted, prune_state = self._apply_candidate_prune(adjusted)
            state.update(prune_state)
            state["active"] = state["active"] or bool(prune_state.get("candidate_prune_active", False))
            return adjusted, state

        if self.decoder_control_mode == "loop_aware_constraint":
            adjusted, constraint_state = self._apply_soft_constraint_bias(adjusted)
            state.update(constraint_state)
            state["active"] = state["active"] or bool(constraint_state.get("constraint_bias_active", False))
            return adjusted, state

        if self.decoder_control_mode == "loop_aware_entity_recall":
            adjusted, entity_state = self._apply_entity_recall_prior(adjusted)
            state.update(entity_state)
            state["active"] = state["active"] or bool(entity_state.get("entity_prior_active", False))
            return adjusted, state

        if self.decoder_control_mode == "logit_bias_entity_soft":
            adjusted, bias_state = self._apply_soft_entity_logit_bias(adjusted)
            state.update(bias_state)
            state["active"] = state["active"] or bool(bias_state.get("logit_bias_active", False))
            return adjusted, state

        return adjusted, state

    def _apply_loop_aware_penalties(
        self,
        adjusted: torch.Tensor,
        *,
        cycle_length: int | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        penalties: dict[int, float] = {}
        state: dict[str, Any] = {"active": False}
        tokens = self._output_token_ids()
        for offset, token_id in enumerate(reversed(tokens[-6:]), start=1):
            penalties[int(token_id)] = penalties.get(int(token_id), 0.0) + max(0.0, 1.8 - (0.2 * (offset - 1)))

        blocked_token_ids: list[int] = []
        if cycle_length is not None:
            blocked_token_ids = [int(token_id) for token_id in dict.fromkeys(tokens[-cycle_length:])]
            cycle_penalty = 8.0 if cycle_length == 1 else 6.0
            for token_id in blocked_token_ids:
                penalties[token_id] = penalties.get(token_id, 0.0) + cycle_penalty
        elif tokens:
            blocked_token_ids = [int(tokens[-1])]
            penalties[blocked_token_ids[0]] = penalties.get(blocked_token_ids[0], 0.0) + 8.0

        for token_id, penalty in penalties.items():
            if 0 <= token_id < adjusted.shape[-1]:
                adjusted[token_id] = adjusted[token_id] - penalty

        valid_blocked = [token_id for token_id in blocked_token_ids if 0 <= token_id < adjusted.shape[-1]]
        if valid_blocked:
            blocked_index = torch.tensor(valid_blocked, dtype=torch.long, device=adjusted.device)
            alternative_mask = torch.isfinite(adjusted)
            alternative_mask[blocked_index] = False
            if bool(alternative_mask.any()) and int(torch.argmax(adjusted).item()) in valid_blocked:
                adjusted[blocked_index] = float("-inf")

        state["active"] = True
        state["blocked_token_ids"] = valid_blocked
        return adjusted, state

    def _apply_candidate_prune(self, adjusted: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        finite_indices = torch.nonzero(torch.isfinite(adjusted), as_tuple=False).reshape(-1)
        if finite_indices.numel() < 2:
            return adjusted, {"candidate_prune_active": False}
        top_values, top_indices = torch.topk(adjusted, k=2, dim=-1)
        gap = float((top_values[0] - top_values[1]).item())
        if gap <= 0.0:
            return adjusted, {"candidate_prune_active": False}
        prune_penalty = gap + 0.05
        adjusted[int(top_indices[0].item())] = adjusted[int(top_indices[0].item())] - prune_penalty
        return adjusted, {
            "candidate_prune_active": True,
            "pruned_token_id": int(top_indices[0].item()),
            "promoted_alternative_token_id": int(top_indices[1].item()),
            "candidate_prune_gap": gap,
        }

    def _apply_soft_constraint_bias(self, adjusted: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        target_token_ids, target_terms = self._soft_constraint_target_token_ids(vocab_size=int(adjusted.shape[-1]))
        if not target_token_ids:
            return adjusted, {"constraint_bias_active": False, "constraint_target_count": 0}

        bias = min(1.4, 0.6 + (0.2 * max(1, self._no_progress_steps)))
        applied_ids: list[int] = []
        for token_id in target_token_ids:
            if 0 <= token_id < adjusted.shape[-1] and torch.isfinite(adjusted[token_id]):
                adjusted[token_id] = adjusted[token_id] + bias
                applied_ids.append(int(token_id))
        if not applied_ids:
            return adjusted, {"constraint_bias_active": False, "constraint_target_count": 0}
        return adjusted, {
            "constraint_bias_active": True,
            "constraint_bias": float(bias),
            "constraint_target_count": len(applied_ids),
            "constraint_target_terms": list(target_terms),
            "constraint_target_token_ids": applied_ids,
        }

    def _apply_entity_recall_prior(self, adjusted: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        sequences, target_terms = self._entity_recall_target_sequences(vocab_size=int(adjusted.shape[-1]))
        if not sequences:
            return adjusted, {"entity_prior_active": False, "entity_target_count": 0, "entity_prefix_depth": 0}

        output_tokens = self._output_token_ids()
        bias_by_token: dict[int, float] = {}
        max_prefix_depth = 0
        matched_terms: list[str] = []
        continued_terms: list[str] = []
        start_bias = min(1.8, 0.8 + (0.2 * max(1, self._no_progress_steps)))
        continuation_bias = min(2.8, start_bias + 0.9)

        for sequence in sequences:
            prefix_depth = self._target_prefix_depth(output_tokens, sequence.token_ids)
            next_index = prefix_depth if prefix_depth < len(sequence.token_ids) else None
            if next_index is None:
                continue
            next_token_id = int(sequence.token_ids[next_index])
            if not (0 <= next_token_id < adjusted.shape[-1]) or not torch.isfinite(adjusted[next_token_id]):
                continue
            bias = start_bias if prefix_depth == 0 else min(3.0, continuation_bias + (0.25 * (prefix_depth - 1)))
            bias_by_token[next_token_id] = max(bias_by_token.get(next_token_id, 0.0), float(bias))
            max_prefix_depth = max(max_prefix_depth, prefix_depth)
            if sequence.term not in matched_terms:
                matched_terms.append(sequence.term)
            if prefix_depth > 0 and sequence.term not in continued_terms:
                continued_terms.append(sequence.term)

        if not bias_by_token:
            return adjusted, {"entity_prior_active": False, "entity_target_count": 0, "entity_prefix_depth": 0}

        for token_id, bias in bias_by_token.items():
            adjusted[token_id] = adjusted[token_id] + bias

        return adjusted, {
            "entity_prior_active": True,
            "entity_target_count": len(sequences),
            "entity_target_terms": list(target_terms),
            "entity_target_token_ids": sorted(bias_by_token),
            "entity_prefix_depth": int(max_prefix_depth),
            "entity_continuation_active": bool(continued_terms),
            "entity_continued_terms": continued_terms[:6],
            "entity_matched_terms": matched_terms[:6],
            "entity_start_bias": float(start_bias),
            "entity_continuation_bias": float(continuation_bias),
        }

    def _apply_soft_entity_logit_bias(self, adjusted: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        sequences, target_terms = self._entity_recall_target_sequences(vocab_size=int(adjusted.shape[-1]))
        if not sequences:
            return adjusted, {
                "logit_bias_active": False,
                "logit_bias_term_count": 0,
                "logit_bias_token_count": 0,
                "logit_bias_focus_term_count": 0,
            }

        output_tokens = self._output_token_ids()
        term_progress = self._feedback_term_progress_by_term()
        focus_terms, easy_terms, hard_terms, focus_mode = self._entity_bias_focus_terms(
            target_terms,
            term_progress,
            max_terms=2,
        )
        if not focus_terms:
            focus_terms = list(target_terms[:2])
        easy_term_set = set(easy_terms)
        token_biases: dict[int, float] = {}
        base_bias = min(1.45, 0.6 + (0.15 * max(1, self._no_progress_steps)) + (0.15 * float(self._repeat_flag())))
        max_bias = 0.0
        space_prefixed_terms: list[str] = []
        for sequence in sequences:
            if sequence.term not in focus_terms:
                continue
            prefix_depth = self._target_prefix_depth(output_tokens, sequence.token_ids)
            progress = max(0.0, min(1.0, float(term_progress.get(sequence.term, 0.0))))
            if progress <= 0.01:
                term_scale = 1.45
            elif progress <= 0.34:
                term_scale = 1.1
            else:
                term_scale = 0.75
            if sequence.term in easy_term_set:
                term_scale *= 1.12
            elif easy_term_set:
                term_scale *= 0.88
            has_space_prefixed_variant = any(
                candidate.term == sequence.term and str(candidate.variant).startswith(" ")
                for candidate in sequences
            )
            if has_space_prefixed_variant and str(sequence.variant).startswith(" "):
                variant_scale = 1.3
                if sequence.term not in space_prefixed_terms:
                    space_prefixed_terms.append(sequence.term)
            elif has_space_prefixed_variant:
                variant_scale = 0.72
            else:
                variant_scale = 1.0
            for index, token_id in enumerate(sequence.token_ids):
                if not (0 <= token_id < adjusted.shape[-1]) or not torch.isfinite(adjusted[token_id]):
                    continue
                if index == 0:
                    position_scale = 1.2
                elif index == 1:
                    position_scale = 0.85
                else:
                    position_scale = 0.65
                if prefix_depth > 0 and index == min(prefix_depth, len(sequence.token_ids) - 1):
                    position_scale = max(position_scale, 1.05)
                bias = min(2.25, float(base_bias * term_scale * variant_scale * position_scale))
                token_biases[token_id] = max(token_biases.get(token_id, 0.0), bias)
                max_bias = max(max_bias, bias)

        if not token_biases:
            return adjusted, {
                "logit_bias_active": False,
                "logit_bias_term_count": 0,
                "logit_bias_token_count": 0,
                "logit_bias_focus_term_count": 0,
            }

        for token_id, bias in token_biases.items():
            adjusted[token_id] = adjusted[token_id] + bias

        return adjusted, {
            "logit_bias_active": True,
            "logit_bias_weight": float(base_bias),
            "logit_bias_max_weight": float(max_bias),
            "logit_bias_term_count": len(target_terms),
            "logit_bias_token_count": len(token_biases),
            "logit_bias_terms": list(target_terms),
            "logit_bias_focus_term_count": len(focus_terms),
            "logit_bias_focus_terms": list(focus_terms),
            "logit_bias_focus_mode": focus_mode,
            "logit_bias_easy_terms": list(easy_terms),
            "logit_bias_hard_terms": list(hard_terms),
            "logit_bias_prefer_space_terms": space_prefixed_terms[:6],
            "logit_bias_token_ids": sorted(token_biases),
        }

    def _soft_constraint_target_token_ids(self, *, vocab_size: int) -> tuple[list[int], list[str]]:
        if bool(self._last_task_feedback.get("done", False)):
            return [], []
        terms = self._feedback_terms(_SOFT_CONSTRAINT_FEEDBACK_KEYS)
        if not terms:
            return [], []

        token_ids: list[int] = []
        seen_token_ids: set[int] = set()
        for term in terms:
            for variant in self._constraint_token_variants(term):
                try:
                    encoded = self.codec.encode(variant).detach().reshape(-1).to(dtype=torch.long).tolist()
                except Exception:
                    continue
                if not encoded:
                    continue
                token_id = int(encoded[0])
                if token_id in seen_token_ids or not (0 <= token_id < vocab_size):
                    continue
                try:
                    token_text = self.codec.decode([token_id])
                except Exception:
                    token_text = str(variant)
                if not str(token_text).strip():
                    continue
                seen_token_ids.add(token_id)
                token_ids.append(token_id)
        return token_ids, terms[:6]

    def _entity_recall_target_sequences(self, *, vocab_size: int) -> tuple[list[_TargetTokenSequence], list[str]]:
        if bool(self._last_task_feedback.get("done", False)):
            return [], []
        terms = self._feedback_terms(_ENTITY_RECALL_FEEDBACK_KEYS)
        if not terms:
            return [], []
        return self._target_token_sequences(terms, vocab_size=vocab_size), terms[:6]

    def _feedback_terms(self, keys: Sequence[str]) -> list[str]:
        terms: list[str] = []
        seen_terms: set[str] = set()
        for key in keys:
            value = self._last_task_feedback.get(key)
            if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes, bytearray)):
                continue
            for item in value:
                term = " ".join(str(item).split()).strip()
                if not term or term in seen_terms:
                    continue
                seen_terms.add(term)
                terms.append(term)
        return terms

    def _feedback_term_progress_by_term(self) -> dict[str, float]:
        progress_by_term: dict[str, float] = {}
        for key in (
            "entity_recall_progress_by_term",
            "required_term_span_progress_by_term",
            "summary_term_span_progress_by_term",
        ):
            value = self._last_task_feedback.get(key)
            if not isinstance(value, Mapping):
                continue
            for raw_term, raw_progress in value.items():
                term = " ".join(str(raw_term).split()).strip()
                if not term:
                    continue
                try:
                    progress = max(0.0, min(1.0, float(raw_progress)))
                except Exception:
                    continue
                progress_by_term[term] = progress
        return progress_by_term

    def _focus_terms_by_progress(self, terms: Sequence[str], progress_by_term: Mapping[str, float], *, max_terms: int) -> list[str]:
        if max_terms <= 0:
            return []
        indexed_terms = list(enumerate(terms))
        indexed_terms.sort(key=lambda item: (float(progress_by_term.get(item[1], 0.0) or 0.0), item[0]))
        return [str(term) for _index, term in indexed_terms[: min(max_terms, len(indexed_terms))]]

    def _entity_bias_focus_terms(
        self,
        terms: Sequence[str],
        progress_by_term: Mapping[str, float],
        *,
        max_terms: int,
    ) -> tuple[list[str], list[str], list[str], str]:
        if max_terms <= 0:
            return [], [], [], "off"
        allowed_terms = {str(term) for term in terms}
        latest_tokenize = self._latest_tokenize_terms_result()
        easy_terms = self._intersect_terms(latest_tokenize.get("soft_logit_bias_ok_terms"), allowed_terms)
        hard_terms = self._intersect_terms(latest_tokenize.get("needs_sequence_support_terms"), allowed_terms)
        if easy_terms:
            focus_terms = self._focus_terms_by_progress(easy_terms, progress_by_term, max_terms=max_terms)
            if len(focus_terms) < max_terms:
                ranked_terms = self._focus_terms_by_progress(terms, progress_by_term, max_terms=len(terms))
                for term in ranked_terms:
                    if term in focus_terms:
                        continue
                    focus_terms.append(term)
                    if len(focus_terms) >= max_terms:
                        break
            return focus_terms[:max_terms], easy_terms, hard_terms, "easy_terms_first"
        focus_terms = self._focus_terms_by_progress(terms, progress_by_term, max_terms=max_terms)
        return focus_terms[:max_terms], easy_terms, hard_terms, "progress_only"

    def _loop_severity_hint(self) -> str:
        cycle_length = self._loop_cycle_length()
        output_len = len(self._output_token_ids())
        if self._last_status == "looping":
            return "high"
        if self._repeat_flag() and (self._no_progress_steps >= 2 or (cycle_length is not None and output_len >= 4)):
            return "high"
        if self._repeat_flag() or cycle_length is not None or self._no_progress_steps > 0:
            return "low"
        return "none"

    def _effect_has_coverage_progress(self, effect: Mapping[str, Any]) -> bool:
        delta = effect.get("delta", {})
        if not isinstance(delta, Mapping):
            return False
        return (
            float(delta.get("required_term_recall", 0.0) or 0.0) > 1e-6
            or float(delta.get("required_term_span_progress", 0.0) or 0.0) > 0.01
            or float(delta.get("partial_score", 0.0) or 0.0) > 1e-6
            or float(delta.get("semantic_progress_score", 0.0) or 0.0) > 0.03
        )

    def _effect_looks_like_recall_edit(self, effect: Mapping[str, Any]) -> bool:
        text = " ".join(
            str(effect.get(key, "") or "")
            for key in ("hypothesis", "edit_id", "expected_effect")
        ).lower()
        if not text:
            return False
        term_markers = ("required", "term", "entity", "coverage", "recall", "insertion", "shot")
        loop_markers = ("loop", "rescue", "repeat")
        return any(marker in text for marker in term_markers) and not any(marker in text for marker in loop_markers)

    def _effect_looks_like_loop_break(self, effect: Mapping[str, Any]) -> bool:
        text = " ".join(
            str(effect.get(key, "") or "")
            for key in ("hypothesis", "edit_id", "expected_effect")
        ).lower()
        if not text:
            return False
        return any(marker in text for marker in ("loop", "rescue", "repeat", "stall"))

    def _recent_loop_break_attempt_count(self, *, window: int = 8) -> int:
        count = 0
        for effect in list(self._recent_effects)[-window:]:
            if isinstance(effect, Mapping) and self._effect_looks_like_loop_break(effect):
                count += 1
        return count

    def _recent_stabilizing_only_count(self, *, window: int = 8) -> int:
        count = 0
        for effect in list(self._recent_effects)[-window:]:
            if not isinstance(effect, Mapping):
                continue
            if not self._effect_looks_like_loop_break(effect):
                continue
            if str(effect.get("signal_profile", "flat")) == "stabilizing_only":
                count += 1
        return count

    def _recent_recall_failure_count(self, *, window: int = 8) -> int:
        count = 0
        for effect in list(self._recent_effects)[-window:]:
            if not isinstance(effect, Mapping):
                continue
            if not self._effect_looks_like_recall_edit(effect):
                continue
            if self._effect_has_coverage_progress(effect):
                continue
            verdict = str(effect.get("verdict", "unknown"))
            signal_profile = str(effect.get("signal_profile", "flat"))
            if verdict == "harmful" or signal_profile in {"stabilizing_only", "flat"}:
                count += 1
        return count

    def _recent_tool_result_count(self, tool_names: Sequence[str], *, window: int = 6) -> int:
        allowed = {str(name) for name in tool_names}
        count = 0
        for item in list(self._tool_results)[-window:]:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("tool", "")) in allowed:
                count += 1
        return count

    def _shot_mode_ready(self) -> bool:
        missing_terms = self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        if not missing_terms:
            return False
        if self._loop_severity_hint() == "high":
            return False
        recent_coverage_progress = any(
            isinstance(effect, Mapping) and self._effect_has_coverage_progress(effect)
            for effect in list(self._recent_effects)[-6:]
        )
        if recent_coverage_progress:
            return False
        stabilizing_only_count = self._recent_stabilizing_only_count()
        recall_failure_count = self._recent_recall_failure_count()
        loop_break_attempt_count = self._recent_loop_break_attempt_count()
        return stabilizing_only_count >= 1 or recall_failure_count >= 1 or loop_break_attempt_count >= 2

    def _shot_candidate_edits(self, *, avoid_surfaces: Sequence[str] = ()) -> list[dict[str, Any]]:
        avoid = {str(surface_id) for surface_id in avoid_surfaces}
        candidates: list[tuple[int, int, dict[str, Any]]] = []
        for surface in self.surface_catalog:
            target = getattr(surface, "target", None)
            if getattr(target, "kind", None) != "activation" or getattr(target, "site", None) != "resid_pre":
                continue
            surface_id = str(surface.surface_id)
            if surface_id in avoid:
                continue
            token = getattr(target, "token", None)
            mode = getattr(token, "mode", None)
            token_priority = 99
            role = "shot_generic"
            alpha = 0.04
            if mode == "index" and getattr(token, "value", None) == -2:
                token_priority = 0
                role = "shot_prev_anchor"
                alpha = 0.04
            elif mode == "last":
                token_priority = 1
                role = "shot_last_anchor"
                alpha = 0.05 if getattr(target, "layer", 0) >= 4 else 0.04
            else:
                continue
            layer = int(getattr(target, "layer", 0) or 0)
            candidates.append(
                (
                    token_priority,
                    layer,
                    {
                        "surface_id": surface_id,
                        "kind": "resid_add",
                        "alpha": float(alpha),
                        "ttl_steps": 1,
                        "step_size": float(alpha),
                        "role": role,
                    },
                )
            )
        candidates.sort(key=lambda item: (item[0], item[1], str(item[2]["surface_id"])))
        return [candidate for _priority, _layer, candidate in candidates[:3]]

    def _shot_probe_needed(self, *, shot_mode_ready: bool) -> bool:
        if not shot_mode_ready:
            return False
        if max(0, self.max_tool_calls_per_run - len(self._tool_results)) <= 0:
            return False
        return self._recent_tool_result_count(("constraint_scorer", "dry_run_decode"), window=4) == 0

    def _control_phase_hint(self) -> str:
        if self._loop_severity_hint() == "high":
            return "loop_break"
        missing_terms = self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        if missing_terms:
            if self._shot_mode_ready():
                return "shot_mode"
            return "entity_insertion"
        if self._loop_severity_hint() == "low":
            return "loop_break"
        return "monitor"

    def _strategy_hints(
        self,
        *,
        control_phase_hint: str,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        latest_tokenize = self._latest_tokenize_terms_result()
        missing_terms = set(
            self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        )
        easy_terms = self._intersect_terms(latest_tokenize.get("soft_logit_bias_ok_terms"), missing_terms)
        hard_terms = self._intersect_terms(latest_tokenize.get("needs_sequence_support_terms"), missing_terms)
        watch_terms = self._intersect_terms(latest_tokenize.get("span_progress_watch_terms"), missing_terms)
        prefer_auxiliary_entity_bias = bool(easy_terms) and self.decoder_control_mode == "logit_bias_entity_soft"
        shot_mode_ready = self._shot_mode_ready()
        loop_break_attempt_count = self._recent_loop_break_attempt_count()
        stabilizing_only_count = self._recent_stabilizing_only_count()
        avoid_surfaces = ["s_resid_pre_l4_last"] if self._l4_term_nudge_cooldown_active() else []
        shot_candidate_edits = self._shot_candidate_edits(avoid_surfaces=avoid_surfaces)
        kv_candidate_edits = self._kv_candidate_edits(
            promoted_cache_surfaces=promoted_cache_surfaces,
            canary_enabled=control_phase_hint == "shot_mode" and not self._kv_canary_eval_active,
        )
        shot_probe_needed = self._shot_probe_needed(shot_mode_ready=shot_mode_ready)
        kv_canary_checked = sum(1 for item in kv_candidate_edits if bool(item.get("canary_checked")))
        kv_canary_positive_count = sum(1 for item in kv_candidate_edits if bool(item.get("canary_pass")))
        kv_canary_rejected_count = sum(
            1 for item in kv_candidate_edits if bool(item.get("canary_checked")) and not bool(item.get("canary_pass"))
        )
        hints = {
            "control_phase_hint": control_phase_hint,
            "loop_severity": self._loop_severity_hint(),
            "prefer_space_prefixed_logit_bias": bool(hard_terms or easy_terms),
            "prefer_auxiliary_entity_bias": prefer_auxiliary_entity_bias,
            "direct_entity_edit_gate": (
                "shot_mode_first"
                if control_phase_hint == "shot_mode"
                else ("auxiliary_first" if prefer_auxiliary_entity_bias else "direct_edit_ok")
            ),
            "easy_entity_terms": easy_terms,
            "hard_entity_terms": hard_terms,
            "watch_span_progress_terms": watch_terms,
            "shot_mode_ready": shot_mode_ready,
            "loop_break_attempt_count": loop_break_attempt_count,
            "stabilizing_only_count": stabilizing_only_count,
            "shot_probe_needed": shot_probe_needed,
            "kv_probe_needed": bool(shot_probe_needed and kv_candidate_edits and kv_canary_checked == 0),
            "shot_candidate_edits": shot_candidate_edits,
            "kv_candidate_edits": kv_candidate_edits,
            "l4_term_nudge_cooldown": self._l4_term_nudge_cooldown_active(),
        }
        if kv_canary_checked > 0:
            hints["kv_canary_checked"] = kv_canary_checked
            hints["kv_canary_positive_count"] = kv_canary_positive_count
            hints["kv_canary_rejected_count"] = kv_canary_rejected_count
        if hints["l4_term_nudge_cooldown"]:
            hints["avoid_recall_surfaces"] = list(avoid_surfaces)
        if control_phase_hint == "loop_break":
            hints["phase_policy"] = "Prefer loop relief, patience, or observer/tool checks before entity insertion edits."
        elif control_phase_hint == "shot_mode":
            hints["phase_policy"] = (
                "De-loop already helped but coverage is still zero; prefer constraint_scorer, dry_run_decode, or a dedicated shot edit before another loop-break."
            )
            hints["prefer_shot_tools"] = ["constraint_scorer", "dry_run_decode"]
            if shot_candidate_edits:
                hints["preferred_shot_surface_id"] = str(shot_candidate_edits[0]["surface_id"])
            if kv_candidate_edits:
                passing_kv_candidates = [item for item in kv_candidate_edits if bool(item.get("canary_pass"))]
                if passing_kv_candidates:
                    hints["preferred_kv_surface_id"] = str(passing_kv_candidates[0]["surface_id"])
                    hints["phase_policy"] = (
                        "De-loop already helped but coverage is still zero; prefer constraint_scorer plus the canary-cleared shot or kv candidate before another loop-break."
                    )
                else:
                    hints["phase_policy"] = (
                        "De-loop already helped but coverage is still zero; if kv canaries fail, prefer constraint_scorer, residual shots, or noop over blind kv apply."
                    )
        elif control_phase_hint == "entity_insertion":
            if prefer_auxiliary_entity_bias:
                hints["phase_policy"] = (
                    "Favor auxiliary easy-term bias, dry-run checks, or observer checks before a new direct latent recall edit."
                )
            else:
                hints["phase_policy"] = "Favor entity insertion and dry-run checks; avoid new loop-rescue edits unless looping returns."
        else:
            hints["phase_policy"] = "Prefer noop or monitoring unless fresh evidence justifies a small edit."
        return hints

    def _latest_tokenize_terms_result(self) -> Mapping[str, Any]:
        for item in reversed(self._tool_results):
            if isinstance(item, Mapping) and str(item.get("tool", "")) == "tokenize_terms":
                return item
        return {}

    def _intersect_terms(self, raw_terms: Any, allowed_terms: set[str]) -> list[str]:
        if not isinstance(raw_terms, SequenceABC) or isinstance(raw_terms, (str, bytes, bytearray)):
            return []
        terms: list[str] = []
        for raw_term in raw_terms:
            term = " ".join(str(raw_term).split()).strip()
            if not term or term not in allowed_terms or term in terms:
                continue
            terms.append(term)
        return terms[:6]

    def _latest_kv_feature_scan(self) -> Mapping[str, Any]:
        if not isinstance(self._latest_observer_check, Mapping):
            return {}
        value = self._latest_observer_check.get("kv_feature_scan")
        if not isinstance(value, Mapping):
            return {}
        return value

    def _kv_feature_hits(self, value: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        scan = value if isinstance(value, Mapping) else self._latest_kv_feature_scan()
        hits: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str, int, int, str]] = set()

        def _push(raw_hit: Mapping[str, Any], *, group: str | None = None, polarity: str | None = None) -> None:
            site = str(raw_hit.get("site", "") or "")
            token_mode = str(raw_hit.get("token_mode", "last") or "last")
            layer = raw_hit.get("layer")
            head = raw_hit.get("head")
            if site not in {"k_cache", "v_cache"}:
                return
            if token_mode != "last":
                return
            if isinstance(layer, bool) or not isinstance(layer, int):
                return
            if isinstance(head, bool) or not isinstance(head, int):
                return
            alignment = raw_hit.get("alignment")
            normalized = dict(raw_hit)
            normalized["site"] = site
            normalized["layer"] = int(layer)
            normalized["head"] = int(head)
            normalized["token_mode"] = token_mode
            if group is not None and normalized.get("group") in (None, ""):
                normalized["group"] = str(group)
            if polarity is not None and normalized.get("polarity") in (None, ""):
                normalized["polarity"] = str(polarity)
            hit_group = str(normalized.get("group", "") or "")
            hit_polarity = str(normalized.get("polarity", "") or "")
            feature = str(normalized.get("feature", "") or "")
            dedupe_key = (hit_group, hit_polarity, site, int(layer), int(head), token_mode)
            if dedupe_key in seen_keys:
                return
            seen_keys.add(dedupe_key)
            if isinstance(alignment, (int, float)) and not isinstance(alignment, bool):
                normalized["alignment"] = round(float(alignment), 6)
            hits.append(normalized)

        top_hits = scan.get("top_feature_hits")
        if isinstance(top_hits, SequenceABC) and not isinstance(top_hits, (str, bytes, bytearray)):
            for item in top_hits:
                if isinstance(item, Mapping):
                    _push(item)

        groups = scan.get("groups")
        if isinstance(groups, SequenceABC) and not isinstance(groups, (str, bytes, bytearray)):
            for group_item in groups:
                if not isinstance(group_item, Mapping):
                    continue
                group = str(group_item.get("group", "") or "")
                polarity = str(group_item.get("polarity", "") or "")
                top_features = group_item.get("top_features")
                if not isinstance(top_features, SequenceABC) or isinstance(top_features, (str, bytes, bytearray)):
                    continue
                for item in top_features:
                    if isinstance(item, Mapping):
                        _push(item, group=group, polarity=polarity)

        def _group_priority(item: Mapping[str, Any]) -> int:
            group = str(item.get("group", "") or "")
            if group == "required_terms":
                return 0
            if group == "missing_keywords":
                return 1
            if group == "missing_summary_terms":
                return 2
            return 3

        def _site_priority(item: Mapping[str, Any]) -> int:
            return 0 if str(item.get("site", "")) == "v_cache" else 1

        hits.sort(
            key=lambda item: (
                _group_priority(item),
                0 if str(item.get("polarity", "promote") or "promote") == "promote" else 1,
                _site_priority(item),
                -float(item.get("alignment", 0.0) or 0.0),
                int(item.get("layer", 0) or 0),
                int(item.get("head", 0) or 0),
            )
        )
        return hits

    def _promoted_cache_surface_id(self, *, site: str, layer: int, head: int, token_mode: str) -> str:
        return f"s_{str(site)}_l{int(layer)}_h{int(head)}_{str(token_mode)}_promoted"

    def _cache_surface_records(
        self,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for surface in self.surface_catalog:
            target = getattr(surface, "target", None)
            if getattr(target, "kind", None) != "cache":
                continue
            token = getattr(target, "token", None)
            if token is None or getattr(target, "head", None) is None:
                continue
            records.append(
                {
                    "surface_id": str(surface.surface_id),
                    "site": str(target.site),
                    "layer": int(target.layer),
                    "head": int(target.head),
                    "token_mode": str(token.mode),
                    "max_alpha": float(surface.caps.max_alpha),
                    "step_size": None if surface.caps.step_size is None else float(surface.caps.step_size),
                    "norm_clip": None if surface.caps.norm_clip is None else float(surface.caps.norm_clip),
                }
            )
        for surface in promoted_cache_surfaces:
            if not isinstance(surface, Mapping):
                continue
            target = surface.get("target")
            caps = surface.get("caps")
            if not isinstance(target, Mapping) or not isinstance(caps, Mapping):
                continue
            token = target.get("token")
            if not isinstance(token, Mapping):
                continue
            site = str(target.get("site", "") or "")
            layer = target.get("layer")
            head = target.get("head")
            if site not in {"k_cache", "v_cache"}:
                continue
            if isinstance(layer, bool) or not isinstance(layer, int):
                continue
            if isinstance(head, bool) or not isinstance(head, int):
                continue
            records.append(
                {
                    "surface_id": str(surface.get("surface_id", "") or ""),
                    "site": site,
                    "layer": int(layer),
                    "head": int(head),
                    "token_mode": str(token.get("mode", "last") or "last"),
                    "max_alpha": float(caps.get("max_alpha", 0.06) or 0.06),
                    "step_size": None if caps.get("step_size") is None else float(caps.get("step_size")),
                    "norm_clip": None if caps.get("norm_clip") is None else float(caps.get("norm_clip")),
                }
            )
        return records

    def _cache_surface_lookup(
        self,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> dict[tuple[str, int, int, str], dict[str, Any]]:
        lookup: dict[tuple[str, int, int, str], dict[str, Any]] = {}
        for record in self._cache_surface_records(promoted_cache_surfaces=promoted_cache_surfaces):
            key = (
                str(record.get("site", "") or ""),
                int(record.get("layer", 0) or 0),
                int(record.get("head", 0) or 0),
                str(record.get("token_mode", "last") or "last"),
            )
            surface_id = str(record.get("surface_id", "") or "")
            if not surface_id or key in lookup:
                continue
            lookup[key] = record
        return lookup

    def _promoted_cache_surfaces(self) -> list[dict[str, Any]]:
        existing_lookup = self._cache_surface_lookup()
        promoted: list[dict[str, Any]] = []
        seen_keys = set(existing_lookup)
        for hit in self._kv_feature_hits():
            if str(hit.get("polarity", "promote") or "promote") != "promote":
                continue
            if str(hit.get("group", "") or "").startswith("forbidden"):
                continue
            site = str(hit.get("site", "") or "")
            layer = hit.get("layer")
            head = hit.get("head")
            token_mode = str(hit.get("token_mode", "last") or "last")
            if site not in {"k_cache", "v_cache"} or token_mode != "last":
                continue
            if isinstance(layer, bool) or not isinstance(layer, int):
                continue
            if isinstance(head, bool) or not isinstance(head, int):
                continue
            key = (site, int(layer), int(head), token_mode)
            if key in seen_keys:
                continue
            max_alpha = 0.06 if site == "k_cache" else 0.08
            step_size = 0.03 if site == "k_cache" else 0.04
            promoted.append(
                {
                    "surface_id": self._promoted_cache_surface_id(
                        site=site,
                        layer=int(layer),
                        head=int(head),
                        token_mode=token_mode,
                    ),
                    "target": {
                        "kind": "cache",
                        "worker": self.worker_id,
                        "site": site,
                        "layer": int(layer),
                        "head": int(head),
                        "token": {"mode": token_mode},
                    },
                    "allow_ops": ["kv_mix"],
                    "caps": {
                        "max_alpha": float(max_alpha),
                        "max_ttl_steps": 1,
                        "norm_clip": 1.0,
                        "step_size": float(step_size),
                        "revertible_only": True,
                    },
                }
            )
            seen_keys.add(key)
            if len(promoted) >= 2:
                break
        return promoted

    def _observer_check_with_cache_surface_ids(
        self,
        value: Mapping[str, Any] | None,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        lookup = self._cache_surface_lookup(promoted_cache_surfaces=promoted_cache_surfaces)
        normalized = dict(value)
        kv_scan = value.get("kv_feature_scan")
        if not isinstance(kv_scan, Mapping):
            return normalized
        kv_scan_copy = dict(kv_scan)

        def _inject_surface_id(raw_hit: Mapping[str, Any]) -> dict[str, Any]:
            item = dict(raw_hit)
            site = str(item.get("site", "") or "")
            token_mode = str(item.get("token_mode", "last") or "last")
            layer = item.get("layer")
            head = item.get("head")
            if site in {"k_cache", "v_cache"} and isinstance(layer, int) and isinstance(head, int):
                record = lookup.get((site, int(layer), int(head), token_mode))
                if record is not None and record.get("surface_id"):
                    item["surface_id"] = str(record["surface_id"])
            return item

        top_hits = kv_scan.get("top_feature_hits")
        if isinstance(top_hits, SequenceABC) and not isinstance(top_hits, (str, bytes, bytearray)):
            kv_scan_copy["top_feature_hits"] = [
                _inject_surface_id(item) if isinstance(item, Mapping) else item for item in top_hits
            ]

        groups = kv_scan.get("groups")
        if isinstance(groups, SequenceABC) and not isinstance(groups, (str, bytes, bytearray)):
            groups_copy: list[Any] = []
            for group in groups:
                if not isinstance(group, Mapping):
                    groups_copy.append(group)
                    continue
                group_copy = dict(group)
                top_features = group.get("top_features")
                if isinstance(top_features, SequenceABC) and not isinstance(top_features, (str, bytes, bytearray)):
                    group_copy["top_features"] = [
                        _inject_surface_id(item) if isinstance(item, Mapping) else item for item in top_features
                    ]
                groups_copy.append(group_copy)
            kv_scan_copy["groups"] = groups_copy

        normalized["kv_feature_scan"] = kv_scan_copy
        return normalized

    def _packet_surface_catalog(
        self,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        catalog = [dict(surface) for surface in self._surface_catalog_raw]
        seen_surface_ids = {str(surface.get("surface_id", "") or "") for surface in catalog}
        for surface in promoted_cache_surfaces:
            if not isinstance(surface, Mapping):
                continue
            surface_id = str(surface.get("surface_id", "") or "")
            if not surface_id or surface_id in seen_surface_ids:
                continue
            catalog.append(dict(surface))
            seen_surface_ids.add(surface_id)
        return catalog

    def _kv_candidate_edits(
        self,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
        canary_enabled: bool = False,
    ) -> list[dict[str, Any]]:
        missing_terms = set(
            self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        )
        if not missing_terms:
            return []
        surface_lookup = self._cache_surface_lookup(promoted_cache_surfaces=promoted_cache_surfaces)
        candidates: list[tuple[int, float, int, int, dict[str, Any]]] = []
        seen_surface_ids: set[str] = set()
        for hit in self._kv_feature_hits():
            if str(hit.get("polarity", "promote") or "promote") != "promote":
                continue
            if str(hit.get("group", "") or "").startswith("forbidden"):
                continue
            feature = str(hit.get("feature", "") or "")
            if feature and feature not in missing_terms:
                continue
            site = str(hit.get("site", "") or "")
            token_mode = str(hit.get("token_mode", "last") or "last")
            layer = hit.get("layer")
            head = hit.get("head")
            if site not in {"k_cache", "v_cache"} or token_mode != "last":
                continue
            if isinstance(layer, bool) or not isinstance(layer, int):
                continue
            if isinstance(head, bool) or not isinstance(head, int):
                continue
            record = surface_lookup.get((site, int(layer), int(head), token_mode))
            if record is None:
                continue
            surface_id = str(record.get("surface_id", "") or "")
            if not surface_id or surface_id in seen_surface_ids:
                continue
            seen_surface_ids.add(surface_id)
            source_position = hit.get("argmax_pos")
            if isinstance(source_position, bool) or not isinstance(source_position, int):
                source_positions = hit.get("source_positions")
                if isinstance(source_positions, SequenceABC) and not isinstance(source_positions, (str, bytes, bytearray)):
                    for item in source_positions:
                        if isinstance(item, Mapping):
                            candidate_pos = item.get("position")
                            if isinstance(candidate_pos, int) and not isinstance(candidate_pos, bool):
                                source_position = int(candidate_pos)
                                break
            if isinstance(source_position, bool) or not isinstance(source_position, int):
                continue
            max_alpha = float(record.get("max_alpha", 0.06) or 0.06)
            alpha = min(max_alpha, 0.04 if site == "v_cache" else 0.03)
            step_cap = record.get("step_size")
            step_size = float(alpha if step_cap is None else min(float(step_cap), alpha))
            norm_clip = record.get("norm_clip")
            if norm_clip is None:
                norm_clip = 1.0
            which = "v" if site == "v_cache" else "k"
            source_expr = {
                "ref": {
                    "scope": "runtime",
                    "worker": self.worker_id,
                    "tensor": site,
                    "layer": int(layer),
                    "head": int(head),
                    "token": {"mode": "index", "value": int(source_position)},
                }
            }
            candidate = {
                "surface_id": surface_id,
                "kind": "kv_mix",
                "role": f"kv_shot_{which}_source_anchor",
                "site": site,
                "layer": int(layer),
                "head": int(head),
                "token_mode": token_mode,
                "focus_feature": feature,
                "alignment": round(float(hit.get("alignment", 0.0) or 0.0), 6),
                "source_position": int(source_position),
                "source_relative_index": (
                    int(hit.get("argmax_relative_index"))
                    if isinstance(hit.get("argmax_relative_index"), int) and not isinstance(hit.get("argmax_relative_index"), bool)
                    else None
                ),
                "source_piece": None if hit.get("argmax_piece") in (None, "") else str(hit.get("argmax_piece")),
                "source_segment_kind": (
                    None if hit.get("argmax_segment_kind") in (None, "") else str(hit.get("argmax_segment_kind"))
                ),
                "source": {
                    "dtype": "cache_pair",
                    which: source_expr,
                },
                "op": {"kind": "kv_mix", "alpha": float(alpha), "which": which},
                "budget": {
                    "ttl_steps": 1,
                    "norm_clip": float(norm_clip),
                    "step_size": float(step_size),
                    "revertible": True,
                },
                "meta": {
                    "hypothesis": f"kv_{which}_recall_probe",
                    "expected_effect": "small_cache_recall_support",
                },
            }
            site_priority = 0 if site == "v_cache" else 1
            candidates.append((site_priority, -float(candidate["alignment"]), int(layer), int(head), candidate))
            if len(candidates) >= 4:
                continue
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], str(item[4]["surface_id"])))
        selected = [candidate for _site_priority, _alignment, _layer, _head, candidate in candidates[:2]]
        if canary_enabled:
            return self._annotate_kv_candidates_with_canary(selected)
        return selected

    def _kv_source_positions_for_feature(
        self,
        prototype: torch.Tensor,
        *,
        cache_tensor: torch.Tensor,
        layer: int,
        site: str,
        head: int,
        width: int,
        head_count: int,
        position_records: Sequence[Mapping[str, Any]],
        max_positions: int,
    ) -> list[dict[str, Any]]:
        if not isinstance(cache_tensor, torch.Tensor) or cache_tensor.ndim != 4:
            return []
        if cache_tensor.shape[0] <= 0 or cache_tensor.shape[1] <= 0:
            return []
        if head < 0 or cache_tensor.shape[2] <= head:
            return []
        seq_len = int(cache_tensor.shape[1])
        max_positions = max(1, int(max_positions))
        rows: list[dict[str, Any]] = []
        projection_cache: dict[int, torch.Tensor | None] = {}

        def _segment_priority(kind: str) -> int:
            if kind == "hint":
                return 0
            if kind == "prompt":
                return 1
            if kind == "output":
                return 2
            return 3

        for position in range(seq_len):
            projected = projection_cache.get(position)
            if position not in projection_cache:
                projected = self._project_feature_into_kv_head(
                    prototype,
                    layer=layer,
                    site=site,
                    head=head,
                    token_index=position,
                    width=width,
                    head_count=head_count,
                )
                projection_cache[position] = projected
            if projected is None:
                continue
            cache_vector = cache_tensor[0, position, head, :].detach().reshape(-1).cpu().float()
            if cache_vector.numel() != int(width):
                continue
            alignment = _cosine_similarity(projected, cache_vector)
            if alignment is None:
                continue
            record = position_records[position] if position < len(position_records) else {}
            segment_kind = str(record.get("segment_kind", "unknown") or "unknown")
            piece = str(record.get("piece", "") or "")
            relative_index = record.get("relative_index")
            if isinstance(relative_index, bool) or not isinstance(relative_index, int):
                relative_index = int(position) - seq_len
            rows.append(
                {
                    "position": int(position),
                    "relative_index": int(relative_index),
                    "segment_kind": segment_kind,
                    "piece": piece,
                    "alignment": float(alignment),
                    "_segment_priority": _segment_priority(segment_kind),
                }
            )
        rows.sort(
            key=lambda item: (
                int(item["_segment_priority"]),
                -float(item["alignment"]),
                abs(int(item["relative_index"])),
                int(item["position"]),
            )
        )
        return [
            {
                "position": int(item["position"]),
                "relative_index": int(item["relative_index"]),
                "segment_kind": str(item["segment_kind"]),
                "piece": str(item["piece"]),
                "alignment": float(item["alignment"]),
            }
            for item in rows[:max_positions]
        ]

    def _annotate_kv_candidates_with_canary(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        terms = self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        if not terms:
            return [dict(candidate) for candidate in candidates]

        baseline = self._simulate_decode(max_new_tokens=1, top_k=5)
        if baseline is None:
            return [dict(candidate) for candidate in candidates]

        annotated: list[dict[str, Any]] = []
        baseline_logits = baseline["first_logits"].detach().cpu().float()
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        vocab_size = int(baseline_logits.shape[-1])
        for raw_candidate in candidates:
            candidate = dict(raw_candidate)
            focus_terms = []
            feature = str(candidate.get("focus_feature", "") or "")
            if feature:
                focus_terms.append(feature)
            for term in terms:
                if term not in focus_terms:
                    focus_terms.append(term)
            target_sequences = self._target_token_sequences(focus_terms, vocab_size=vocab_size)
            command = self._build_dry_run_command(candidate)
            if not target_sequences or command is None:
                candidate["canary_checked"] = False
                candidate["canary_pass"] = False
                candidate["canary_reason"] = "missing_target_sequence"
                annotated.append(candidate)
                continue
            try:
                self._kv_canary_eval_active = True
                edited = self._simulate_decode(max_new_tokens=1, top_k=5, command=command)
            finally:
                self._kv_canary_eval_active = False
            if edited is None:
                candidate["canary_checked"] = False
                candidate["canary_pass"] = False
                candidate["canary_reason"] = "dry_run_failed"
                if self._last_simulate_decode_error:
                    candidate["canary_error"] = str(self._last_simulate_decode_error)
                annotated.append(candidate)
                continue

            edited_logits = edited["first_logits"].detach().cpu().float()
            edited_probs = torch.softmax(edited_logits, dim=-1)
            best_focus: dict[str, Any] | None = None
            seen_token_ids: set[int] = set()
            for sequence in target_sequences:
                token_id = int(sequence.token_ids[0])
                if token_id in seen_token_ids:
                    continue
                seen_token_ids.add(token_id)
                row = {
                    "term": str(sequence.term),
                    "variant": str(sequence.variant),
                    "token_id": token_id,
                    "piece": self.codec.decode([token_id]),
                    "baseline_logit": float(baseline_logits[token_id].item()),
                    "edited_logit": float(edited_logits[token_id].item()),
                    "logit_delta": float(edited_logits[token_id].item() - baseline_logits[token_id].item()),
                    "baseline_prob": float(baseline_probs[token_id].item()),
                    "edited_prob": float(edited_probs[token_id].item()),
                    "prob_delta": float(edited_probs[token_id].item() - baseline_probs[token_id].item()),
                }
                if best_focus is None or (
                    float(row["prob_delta"]),
                    float(row["logit_delta"]),
                ) > (
                    float(best_focus["prob_delta"]),
                    float(best_focus["logit_delta"]),
                ):
                    best_focus = row

            candidate["canary_checked"] = True
            candidate["canary_repeat_flag_delta"] = int(bool(edited["repeat_flag"])) - int(bool(baseline["repeat_flag"]))
            candidate["canary_entropy_delta"] = round(float(edited["entropy"]) - float(baseline["entropy"]), 6)
            edited_scoring = edited.get("scoring", {})
            baseline_scoring = baseline.get("scoring", {})
            candidate["canary_required_term_recall_delta"] = round(
                float(edited_scoring.get("required_term_recall") or 0.0)
                - float(baseline_scoring.get("required_term_recall") or 0.0),
                6,
            )
            candidate["canary_semantic_progress_delta"] = round(
                float(edited_scoring.get("semantic_progress_score") or 0.0)
                - float(baseline_scoring.get("semantic_progress_score") or 0.0),
                6,
            )
            if best_focus is not None:
                candidate["canary_focus_term"] = str(best_focus["term"])
                candidate["canary_focus_piece"] = str(best_focus["piece"])
                candidate["canary_focus_token_id"] = int(best_focus["token_id"])
                candidate["canary_focus_logit_delta"] = round(float(best_focus["logit_delta"]), 6)
                candidate["canary_focus_prob_delta"] = round(float(best_focus["prob_delta"]), 6)
            else:
                candidate["canary_focus_logit_delta"] = 0.0
                candidate["canary_focus_prob_delta"] = 0.0

            focus_logit_delta = float(candidate.get("canary_focus_logit_delta", 0.0) or 0.0)
            focus_prob_delta = float(candidate.get("canary_focus_prob_delta", 0.0) or 0.0)
            recall_delta = float(candidate.get("canary_required_term_recall_delta", 0.0) or 0.0)
            semantic_delta = float(candidate.get("canary_semantic_progress_delta", 0.0) or 0.0)
            repeat_delta = int(candidate.get("canary_repeat_flag_delta", 0) or 0)
            canary_pass = (
                (recall_delta > 0.0)
                or ((focus_logit_delta >= 0.001 or focus_prob_delta >= 0.0001) and repeat_delta <= 0 and semantic_delta >= -0.02)
            )
            candidate["canary_pass"] = bool(canary_pass)
            candidate["canary_reason"] = "focus_token_improved" if canary_pass else "focus_token_flat"
            annotated.append(candidate)
        return annotated

    def _l4_term_nudge_cooldown_active(self) -> bool:
        for effect in reversed(self._recent_effects):
            if not self._effect_looks_like_l4_term_nudge(effect):
                continue
            verdict = str(effect.get("verdict", "unknown"))
            delta = effect.get("delta", {})
            if not isinstance(delta, Mapping):
                return verdict in {"harmful", "neutral"}
            flat = (
                float(delta.get("required_term_recall", 0.0) or 0.0) <= 0.0
                and float(delta.get("required_term_span_progress", 0.0) or 0.0) <= 0.0
                and float(delta.get("partial_score", 0.0) or 0.0) <= 0.0
                and float(delta.get("semantic_progress_score", 0.0) or 0.0) <= 0.0
            )
            return verdict == "harmful" or flat or verdict == "neutral"
        return False

    def _effect_looks_like_l4_term_nudge(self, effect: Mapping[str, Any]) -> bool:
        if str(effect.get("surface_id", "")) != "s_resid_pre_l4_last":
            return False
        if str(effect.get("op", "")) != "resid_add":
            return False
        return self._effect_looks_like_recall_edit(effect)

    def _target_token_sequences(self, terms: Sequence[str], *, vocab_size: int) -> list[_TargetTokenSequence]:
        sequences: list[_TargetTokenSequence] = []
        seen_sequences: set[tuple[int, ...]] = set()
        for term in terms:
            for variant in self._constraint_token_variants(term):
                try:
                    encoded = self.codec.encode(variant).detach().reshape(-1).to(dtype=torch.long).tolist()
                except Exception:
                    continue
                if not encoded:
                    continue
                token_ids = tuple(int(token_id) for token_id in encoded)
                if token_ids in seen_sequences or any(token_id < 0 or token_id >= vocab_size for token_id in token_ids):
                    continue
                seen_sequences.add(token_ids)
                sequences.append(_TargetTokenSequence(term=str(term), token_ids=token_ids, variant=str(variant)))
        return sequences

    def _target_prefix_depth(self, output_tokens: Sequence[int], target_token_ids: Sequence[int]) -> int:
        if not output_tokens or not target_token_ids:
            return 0
        max_depth = min(len(output_tokens), max(0, len(target_token_ids) - 1))
        for depth in range(max_depth, 0, -1):
            if list(output_tokens[-depth:]) == list(target_token_ids[:depth]):
                return depth
        return 0

    def _constraint_token_variants(self, term: str) -> tuple[str, ...]:
        stripped = " ".join(str(term).split()).strip()
        variants: list[str] = []
        for candidate in (
            stripped,
            stripped.lower(),
            stripped.capitalize(),
            f" {stripped}",
            f" {stripped.lower()}",
            f" {stripped.capitalize()}",
        ):
            if candidate not in variants:
                variants.append(candidate)
        return tuple(variants)

    def _output_token_ids(self) -> list[int]:
        output_tokens: list[int] = []
        for segment in self._segments:
            if segment.kind == "output":
                output_tokens.extend(segment.token_ids)
        return output_tokens

    def _token_position_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        position = 0
        for segment in self._segments:
            for token_id in segment.token_ids:
                records.append(
                    {
                        "position": int(position),
                        "segment_kind": str(segment.kind),
                        "token_id": int(token_id),
                        "piece": str(self.codec.decode([int(token_id)])),
                    }
                )
                position += 1
        seq_len = len(records)
        for record in records:
            record["relative_index"] = int(record["position"]) - seq_len
        return records

    def _compute_metrics(self, next_logits: torch.Tensor) -> dict[str, float]:
        probs = torch.softmax(next_logits.float(), dim=-1)
        entropy = float((-(probs * probs.clamp_min(1e-8).log()).sum()).item())
        top_probs = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        margin = float((top_probs[0] - top_probs[1]).item()) if top_probs.shape[0] >= 2 else float(top_probs[0].item())
        return {
            "entropy": entropy,
            "top1_margin": margin,
            "repetition_score": self._repetition_score(),
        }

    def _repetition_score(self) -> float:
        tokens = self._output_token_ids()
        if len(tokens) < 2:
            return 0.0
        window = tokens[-4:]
        repeated = max(window.count(token_id) for token_id in set(window))
        return float((repeated - 1) / max(1, len(window) - 1))

    def _repeat_flag(self) -> bool:
        tokens = self._output_token_ids()
        if len(tokens) >= 3 and len(set(tokens[-3:])) == 1:
            return True
        if self._loop_cycle_length() is not None:
            return True
        if len(tokens) >= 4 and self._repetition_score() >= 0.66:
            return True
        return False

    def _loop_cycle_length(self, *, max_period: int = 4) -> int | None:
        tokens = self._output_token_ids()
        max_candidate = min(max_period, len(tokens) // 2)
        for period in range(1, max_candidate + 1):
            if tokens[-period:] == tokens[-2 * period : -period]:
                return period
        return None

    def _update_status(self) -> None:
        if self._repeat_flag():
            self._no_progress_steps += 1
            self._last_status = "looping"
            return
        self._no_progress_steps = 0
        self._last_status = "acting" if self._output_token_ids() else "thinking"

    def _task_view(self) -> dict[str, Any]:
        prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        view = {
            "mode": self.task_view_mode,
            "task_id": self.task_id,
            "prompt_hash": f"sha256:{prompt_hash}",
            "goal_hint": self.goal_hint,
            "constraints": list(self.constraints),
        }
        if self.task_view_mode == "full":
            view["prompt"] = self.prompt
        return view

    def _worker_view(self) -> dict[str, Any]:
        output_text = self.final_text()
        recent_tokens = self._output_token_ids()[-self.recent_token_count :]
        return {
            "generated_tail": output_text[-self.generated_tail_chars :],
            "recent_tokens": [self.codec.decode([token_id]) for token_id in recent_tokens],
            "status": self._last_status,
        }

    def _build_tool_catalog(self) -> list[dict[str, Any]]:
        catalog = [
            {
                "tool": "tokenize_terms",
                "available": True,
                "cost_hint": "cheap",
                "objective": "Inspect how local terms map onto worker tokenizer pieces and token ids.",
            },
            {
                "tool": "constraint_scorer",
                "available": bool(self.task_feedback_fn is not None),
                "cost_hint": "medium",
                "objective": "Score a candidate against task constraints and expose missing local coverage.",
            },
            {
                "tool": "dry_run_decode",
                "available": bool(self.surface_catalog),
                "cost_hint": "expensive",
                "objective": "Preview a small reversible edit for a few tokens before spending a real apply.",
            },
        ]
        for item in catalog:
            item["budget_left"] = max(0, self.max_tool_calls_per_run - len(self._tool_results))
        return catalog

    def _execute_controller_tool_request(self, request: Mapping[str, Any], *, source: str) -> dict[str, Any] | None:
        tool_name = str(request.get("tool", "") or "")
        base = {
            "tool": tool_name,
            "requested_by": str(source),
            "recorded_step": int(self._steps),
        }
        reason = request.get("reason")
        if reason is not None:
            base["request_reason"] = str(reason)
        if tool_name == "tokenize_terms":
            payload = self._tokenize_terms_tool_result(request)
        elif tool_name == "constraint_scorer":
            payload = self._constraint_scorer_tool_result(request)
        elif tool_name == "dry_run_decode":
            payload = self._dry_run_decode_tool_result(request)
        else:
            return None
        if payload is None:
            return None
        return base | payload

    def _tokenize_terms_tool_result(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        raw_terms = request.get("terms")
        if not isinstance(raw_terms, SequenceABC) or isinstance(raw_terms, (str, bytes, bytearray)):
            return None
        rows: list[dict[str, Any]] = []
        single_token_terms: list[str] = []
        multi_piece_terms: list[str] = []
        soft_logit_bias_ok_terms: list[str] = []
        needs_sequence_support_terms: list[str] = []
        span_progress_watch_terms: list[str] = []
        for raw_term in raw_terms:
            term = _clean_controller_memory_text(raw_term, limit=80)
            if term is None:
                continue
            segmentations: list[dict[str, Any]] = []
            seen_tokenizations: set[tuple[int, ...]] = set()
            for variant in self._constraint_token_variants(term):
                try:
                    token_ids = tuple(int(token_id) for token_id in self.codec.encode(variant).detach().reshape(-1).tolist())
                except Exception:
                    continue
                if not token_ids or token_ids in seen_tokenizations:
                    continue
                seen_tokenizations.add(token_ids)
                pieces = [self.codec.decode([token_id]) for token_id in token_ids]
                cursor = 0
                boundaries = []
                for piece in pieces:
                    start = cursor
                    cursor += len(piece)
                    boundaries.append({"piece": piece, "start": start, "end": cursor})
                segmentations.append(
                    {
                        "variant": variant,
                        "token_ids": list(token_ids),
                        "pieces": pieces,
                        "piece_boundaries": boundaries,
                        "biasable_units": [
                            {"token_id": int(token_id), "piece": piece, "position": idx}
                            for idx, (token_id, piece) in enumerate(zip(token_ids, pieces, strict=False))
                        ],
                    }
                )
            if not segmentations:
                continue
            primary = segmentations[0]
            piece_count = len(primary["token_ids"])
            if piece_count <= 1:
                control_profile = "single_token_bias_ok"
                soft_logit_bias_ok_terms.append(term)
                single_token_terms.append(term)
            elif piece_count == 2:
                control_profile = "sequence_bias_plus_patience"
                needs_sequence_support_terms.append(term)
                span_progress_watch_terms.append(term)
                multi_piece_terms.append(term)
            else:
                control_profile = "edit_plus_sequence_support"
                needs_sequence_support_terms.append(term)
                span_progress_watch_terms.append(term)
                multi_piece_terms.append(term)
            rows.append(
                {
                    "term": term,
                    "token_ids": list(primary["token_ids"]),
                    "pieces": list(primary["pieces"]),
                    "piece_count": piece_count,
                    "is_single_token": piece_count == 1,
                    "control_profile": control_profile,
                    "piece_boundaries": list(primary["piece_boundaries"]),
                    "alternative_segmentations": [
                        {
                            "variant": item["variant"],
                            "token_ids": list(item["token_ids"]),
                            "pieces": list(item["pieces"]),
                        }
                        for item in segmentations[1:4]
                    ],
                    "biasable_units": list(primary["biasable_units"]),
                }
            )
        if not rows:
            return None
        return {
            "status": "ok",
            "term_count": len(rows),
            "terms": rows,
            "single_token_terms": single_token_terms[:8],
            "multi_piece_terms": multi_piece_terms[:8],
            "soft_logit_bias_ok_terms": soft_logit_bias_ok_terms[:8],
            "needs_sequence_support_terms": needs_sequence_support_terms[:8],
            "span_progress_watch_terms": span_progress_watch_terms[:8],
        }

    def _constraint_scorer_tool_result(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        candidate_text = _clean_controller_memory_text(request.get("candidate"), limit=320) or self.final_text()
        if not candidate_text.strip():
            return None
        scoring = self._score_candidate_text(
            candidate_text,
            trigger="tool_constraint_scorer",
            constraints=request.get("constraints"),
        )
        result = {
            "status": "ok",
            "candidate_preview": candidate_text[:160],
            "required_term_recall": scoring.get("required_term_recall"),
            "forbidden_clean": scoring.get("forbidden_clean"),
            "brevity_score": scoring.get("brevity_score"),
            "per_term_coverage": scoring.get("per_term_coverage"),
            "explanation_tags": scoring.get("explanation_tags", []),
        }
        if scoring.get("semantic_progress_score") is not None:
            result["semantic_progress_score"] = scoring.get("semantic_progress_score")
        if scoring.get("progress_label") is not None:
            result["progress_label"] = scoring.get("progress_label")
        if scoring.get("constraint_violations") is not None:
            result["constraint_violations"] = scoring.get("constraint_violations")
        return result

    def _generic_constraint_feedback(self, candidate_text: str, constraints: Mapping[str, Any] | None) -> dict[str, Any]:
        if not isinstance(constraints, Mapping):
            return {}
        required_terms = constraints.get("required_terms") or ()
        forbidden_terms = constraints.get("forbidden_terms") or ()
        max_words = constraints.get("max_words")
        if not isinstance(required_terms, SequenceABC) or isinstance(required_terms, (str, bytes, bytearray)):
            required_terms = ()
        if not isinstance(forbidden_terms, SequenceABC) or isinstance(forbidden_terms, (str, bytes, bytearray)):
            forbidden_terms = ()
        required_terms = tuple(" ".join(str(term).split()).strip() for term in required_terms if str(term).strip())
        forbidden_terms = tuple(" ".join(str(term).split()).strip() for term in forbidden_terms if str(term).strip())
        present_required = [term for term in required_terms if _tool_contains_term(candidate_text, term)]
        missing_required = [term for term in required_terms if term not in present_required]
        forbidden_present = [term for term in forbidden_terms if _tool_contains_term(candidate_text, term)]
        span_progress = {term: _tool_term_span_progress(candidate_text, term) for term in required_terms}
        word_count = len([part for part in candidate_text.split() if part])
        brevity_score = None
        budget_ok = None
        if isinstance(max_words, int) and max_words > 0:
            budget_ok = word_count <= max_words
            brevity_score = 1.0 if budget_ok else round(max_words / max(1, word_count), 6)
        required_term_recall = round(len(present_required) / max(1, len(required_terms)), 6) if required_terms else 1.0
        forbidden_clean = round(
            sum(1 for term in forbidden_terms if term not in forbidden_present) / max(1, len(forbidden_terms)),
            6,
        ) if forbidden_terms else 1.0
        done = bool(required_term_recall >= 1.0 and forbidden_clean >= 1.0 and (budget_ok is not False))
        violations: list[str] = []
        if missing_required:
            violations.append("missing_required_terms")
        if forbidden_present:
            violations.append("forbidden_terms_present")
        if budget_ok is False:
            violations.append("over_word_budget")
        return {
            "done": done,
            "partial_score": round(
                (0.55 * required_term_recall)
                + (0.25 * forbidden_clean)
                + (0.20 * (1.0 if brevity_score is None else float(brevity_score))),
                6,
            ),
            "progress_label": "progressing" if present_required else "stalled",
            "required_term_recall": required_term_recall,
            "required_term_span_progress": round(
                sum(float(value) for value in span_progress.values()) / max(1, len(span_progress)),
                6,
            ) if span_progress else 0.0,
            "required_term_span_progress_by_term": span_progress,
            "missing_required_terms": missing_required,
            "forbidden_term_clean": forbidden_clean,
            "forbidden_terms_present": forbidden_present,
            "word_budget_score": brevity_score,
            "budget_ok": budget_ok,
            "constraint_violations": violations,
        }

    def _score_candidate_text(
        self,
        candidate_text: str,
        *,
        trigger: str,
        constraints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        feedback: dict[str, Any] = {}
        if self.task_feedback_fn is not None:
            try:
                feedback = dict(self.task_feedback_fn(candidate_text) or {})
            except Exception:
                feedback = {}
        if not feedback:
            feedback = self._generic_constraint_feedback(candidate_text, constraints)

        observer_result = self._invoke_observer_check(candidate_text, trigger=trigger, task_feedback=feedback)
        semantic_progress_score = None
        if isinstance(observer_result, Mapping):
            score = observer_result.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                semantic_progress_score = round(float(score), 6)

        required_term_recall = feedback.get("required_term_recall")
        forbidden_clean = feedback.get("forbidden_term_clean")
        brevity_score = feedback.get("word_budget_score")
        if brevity_score is None and feedback.get("budget_ok") is not None:
            brevity_score = 1.0 if bool(feedback.get("budget_ok")) else 0.0
        span_by_term = feedback.get("required_term_span_progress_by_term")
        per_term_coverage: dict[str, Any] = {}
        if isinstance(span_by_term, Mapping):
            missing_terms = set()
            raw_missing = feedback.get("missing_required_terms")
            if isinstance(raw_missing, SequenceABC) and not isinstance(raw_missing, (str, bytes, bytearray)):
                missing_terms = {str(term) for term in raw_missing}
            for raw_term, raw_progress in span_by_term.items():
                term = str(raw_term)
                progress = round(float(raw_progress or 0.0), 6)
                per_term_coverage[term] = {
                    "present": term not in missing_terms,
                    "span_progress": progress,
                }

        explanation_tags: list[str] = []
        progress_label = feedback.get("progress_label")
        if progress_label:
            explanation_tags.append(str(progress_label))
        for violation in feedback.get("constraint_violations", []) if isinstance(feedback.get("constraint_violations"), SequenceABC) else []:
            if violation:
                explanation_tags.append(str(violation))
        if required_term_recall is not None and float(required_term_recall or 0.0) <= 0.0:
            explanation_tags.append("required_recall_zero")
        if forbidden_clean is not None and float(forbidden_clean or 0.0) >= 1.0:
            explanation_tags.append("forbidden_clean")
        if brevity_score is not None and float(brevity_score or 0.0) >= 1.0:
            explanation_tags.append("brevity_ok")
        if semantic_progress_score is not None:
            explanation_tags.append("semantic_checked")

        return {
            "semantic_progress_score": semantic_progress_score,
            "required_term_recall": None if required_term_recall is None else round(float(required_term_recall), 6),
            "forbidden_clean": None if forbidden_clean is None else round(float(forbidden_clean), 6),
            "brevity_score": None if brevity_score is None else round(float(brevity_score), 6),
            "per_term_coverage": per_term_coverage,
            "explanation_tags": explanation_tags[:6],
            "progress_label": progress_label,
            "constraint_violations": list(feedback.get("constraint_violations", []) or ()),
        }

    def _dry_run_decode_tool_result(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        candidate_edit = request.get("candidate_edit")
        if not isinstance(candidate_edit, Mapping):
            return None
        max_new_tokens = int(request.get("max_new_tokens", 4) or 4)
        top_k = int(request.get("top_k", 5) or 5)
        baseline = self._simulate_decode(max_new_tokens=max_new_tokens, top_k=top_k)
        if baseline is None:
            return None
        command = self._build_dry_run_command(candidate_edit)
        if command is None:
            return {
                "status": "error",
                "error": "unsupported_candidate_edit",
                "candidate_edit": dict(candidate_edit),
            }
        edited = self._simulate_decode(max_new_tokens=max_new_tokens, top_k=top_k, command=command)
        if edited is None:
            return {
                "status": "error",
                "error": "dry_run_failed",
                "candidate_edit": dict(candidate_edit),
            }
        baseline_score = baseline.get("scoring", {})
        edited_score = edited.get("scoring", {})
        return {
            "status": "ok",
            "candidate_edit": dict(candidate_edit),
            "max_new_tokens": max_new_tokens,
            "sampled_continuations": [
                {"variant": "baseline", "text": baseline["continuation"]},
                {"variant": "candidate", "text": edited["continuation"]},
            ],
            "topk_token_diff": self._topk_token_diff(baseline["first_logits"], edited["first_logits"], top_k=top_k),
            "entropy_delta": round(float(edited["entropy"]) - float(baseline["entropy"]), 6),
            "repeat_flag_delta": int(bool(edited["repeat_flag"])) - int(bool(baseline["repeat_flag"])),
            "required_term_recall_delta": round(
                float(edited_score.get("required_term_recall") or 0.0) - float(baseline_score.get("required_term_recall") or 0.0),
                6,
            ),
            "semantic_progress_delta": round(
                float(edited_score.get("semantic_progress_score") or 0.0)
                - float(baseline_score.get("semantic_progress_score") or 0.0),
                6,
            ),
        }

    def _build_dry_run_command(self, candidate_edit: Mapping[str, Any]) -> dict[str, Any] | None:
        if "source" in candidate_edit and "op" in candidate_edit and "budget" in candidate_edit:
            edit = dict(candidate_edit)
            edit.setdefault("id", f"dry_run_{self._steps}")
            if "target" not in edit and candidate_edit.get("surface_id") is not None:
                edit["target"] = {"surface_id": candidate_edit.get("surface_id")}
            return {"version": "0.1", "decision": "apply", "edits": [edit]}

        surface_id = candidate_edit.get("surface_id")
        if surface_id is None and isinstance(candidate_edit.get("target"), Mapping):
            surface_id = candidate_edit["target"].get("surface_id")
        if not isinstance(surface_id, str) or not surface_id:
            return None
        op_kind = _normalize_controller_tool_name(candidate_edit.get("kind") or candidate_edit.get("op", {}).get("kind"))
        if op_kind != "resid_add":
            return None
        surface = next((surface for surface in self.surface_catalog if str(surface.surface_id) == surface_id), None)
        if surface is None or getattr(surface.target, "kind", None) != "activation":
            return None
        layer = getattr(surface.target, "layer", None)
        if not isinstance(layer, int):
            return None
        alpha = candidate_edit.get("alpha", 0.04)
        ttl_steps = candidate_edit.get("ttl_steps", 1)
        step_size = candidate_edit.get("step_size", alpha)
        try:
            alpha = float(alpha)
            step_size = float(step_size)
            ttl_steps = int(ttl_steps)
        except Exception:
            return None
        if alpha <= 0.0 or step_size <= 0.0 or ttl_steps <= 0:
            return None
        norm_clip = getattr(getattr(surface, "caps", None), "norm_clip", None) or 1.0
        edit = {
            "id": f"dry_run_{self._steps}_{surface_id}",
            "target": {"surface_id": surface_id},
            "source": {
                "dtype": "vector",
                "expr": {
                    "fn": "clip_norm",
                    "max_norm": float(norm_clip),
                    "arg": {
                        "fn": "scale",
                        "by": step_size,
                        "arg": {
                            "fn": "normalize",
                            "arg": {
                                "ref": {
                                    "scope": "runtime",
                                    "worker": self.worker_id,
                                    "tensor": "hidden",
                                    "layer": layer,
                                    "token": {"mode": "last"},
                                }
                            },
                        },
                    },
                },
            },
            "op": {"kind": "resid_add", "alpha": alpha},
            "budget": {
                "ttl_steps": ttl_steps,
                "norm_clip": float(norm_clip),
                "step_size": step_size,
                "revertible": True,
            },
        }
        return {"version": "0.1", "decision": "apply", "edits": [edit]}

    def _simulate_decode(
        self,
        *,
        max_new_tokens: int,
        top_k: int,
        command: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if max_new_tokens <= 0:
            return None
        self._last_simulate_decode_error = None
        saved_segments = [_TokenSegment(kind=segment.kind, token_ids=list(segment.token_ids)) for segment in self._segments]
        saved_last_packet = self._last_packet
        saved_last_tokens = None if getattr(self.runtime_state, "last_tokens", None) is None else self.runtime_state.last_tokens.detach().clone()
        saved_last_logits = None if getattr(self.runtime_state, "last_logits", None) is None else self.runtime_state.last_logits.detach().clone()
        saved_last_cache = None
        if getattr(self.runtime_state, "last_cache", None) is not None:
            saved_last_cache = {str(name): tensor.detach().clone() for name, tensor in self.runtime_state.last_cache.items()}

        temporary_edits = []
        try:
            if command is not None:
                if getattr(self.runtime_state, "last_cache", None) is None:
                    primer_tokens = self._current_token_tensor()
                    self.runtime_state.run_with_cache(primer_tokens, return_type="logits")
                packet = self.build_controller_packet()
                ctx = StepContext(packet=packet, runtime_state=self.runtime_state, traces={}, stats={}, adapter=self.adapter, active_edits={})
                compiled = compile_command(command, packet, ctx)
                for item in compiled:
                    item.apply(ctx)
                    temporary_edits.append((item, ctx))

            baseline_output_len = len(self._output_token_ids())
            first_logits: torch.Tensor | None = None
            first_entropy = 0.0
            for _ in range(max_new_tokens):
                tokens = self._current_token_tensor()
                logits, _cache = self.runtime_state.run_with_cache(tokens, return_type="logits")
                next_logits = self._apply_token_constraints(logits[0, -1].detach())
                next_logits, _decoder_state = self._apply_decoder_control(next_logits)
                if first_logits is None:
                    first_logits = next_logits.detach().cpu().float()
                    first_entropy = float(self._compute_metrics(next_logits)["entropy"])
                next_token = int(torch.argmax(next_logits).item())
                self._append_output_token(next_token)
                continuation_ids = self._output_token_ids()[baseline_output_len:]
                if continuation_ids and continuation_ids[-1] in self.stop_token_ids:
                    break
                if self.stop_checker is not None and self.stop_checker(self.final_text()):
                    break
            continuation_ids = self._output_token_ids()[baseline_output_len:]
            if first_logits is None:
                return None
            scoring = self._score_candidate_text(self.final_text(), trigger="tool_dry_run_candidate")
            return {
                "continuation": self.codec.decode(continuation_ids),
                "first_logits": first_logits,
                "entropy": first_entropy,
                "repeat_flag": self._repeat_flag(),
                "scoring": scoring,
            }
        except Exception as exc:
            self._last_simulate_decode_error = f"{type(exc).__name__}: {exc}"
            return None
        finally:
            for compiled, ctx in reversed(temporary_edits):
                try:
                    compiled.rollback(ctx)
                except Exception:
                    pass
            self._segments = [_TokenSegment(kind=segment.kind, token_ids=list(segment.token_ids)) for segment in saved_segments]
            self._last_packet = saved_last_packet
            self.runtime_state.last_tokens = saved_last_tokens
            self.runtime_state.last_logits = saved_last_logits
            self.runtime_state.last_cache = saved_last_cache

    def _topk_token_diff(self, baseline_logits: torch.Tensor, edited_logits: torch.Tensor, *, top_k: int) -> list[dict[str, Any]]:
        baseline = baseline_logits.detach().float()
        edited = edited_logits.detach().float()
        baseline_probs = torch.softmax(baseline, dim=-1)
        edited_probs = torch.softmax(edited, dim=-1)
        baseline_ids = torch.topk(baseline, k=min(top_k, baseline.shape[-1]), dim=-1).indices.tolist()
        edited_ids = torch.topk(edited, k=min(top_k, edited.shape[-1]), dim=-1).indices.tolist()
        token_ids = list(dict.fromkeys([int(token_id) for token_id in baseline_ids + edited_ids]))
        rows: list[dict[str, Any]] = []
        for token_id in token_ids:
            rows.append(
                {
                    "token_id": int(token_id),
                    "piece": self.codec.decode([int(token_id)]),
                    "baseline_logit": round(float(baseline[token_id].item()), 6),
                    "edited_logit": round(float(edited[token_id].item()), 6),
                    "logit_delta": round(float(edited[token_id].item() - baseline[token_id].item()), 6),
                    "baseline_prob": round(float(baseline_probs[token_id].item()), 6),
                    "edited_prob": round(float(edited_probs[token_id].item()), 6),
                    "prob_delta": round(float(edited_probs[token_id].item() - baseline_probs[token_id].item()), 6),
                }
            )
        rows.sort(key=lambda item: (-abs(float(item["prob_delta"])), str(item["piece"])))
        return rows[:top_k]

    def _task_feedback(self) -> dict[str, Any]:
        return dict(self._last_task_feedback)

    def _maybe_run_observer_check(
        self,
        *,
        previous_feedback: Mapping[str, Any],
        previous_metrics: Mapping[str, Any],
        previous_status: str,
        previous_no_progress_steps: int,
    ) -> None:
        if self.observer_check_fn is None or self.max_observer_checks_per_run <= 0:
            return
        if len(self._observer_checks) >= self.max_observer_checks_per_run:
            return
        candidate_text = self.final_text()
        candidate_hash = hashlib.sha256(candidate_text.encode("utf-8")).hexdigest()
        if candidate_hash == self._last_observer_candidate_hash:
            return
        trigger = self._observer_check_trigger(
            previous_feedback=previous_feedback,
            previous_metrics=previous_metrics,
            previous_status=previous_status,
            previous_no_progress_steps=previous_no_progress_steps,
        )
        if trigger is None:
            return
        result = self._invoke_observer_check(candidate_text, trigger=trigger)
        normalized = self._normalize_observer_check_result(
            result,
            trigger=trigger,
            candidate_hash=candidate_hash,
            previous_result=self._latest_observer_check,
        )
        if normalized is None:
            return
        self._latest_observer_check = normalized
        self._observer_checks.append(normalized)
        self._observer_checks = self._observer_checks[-self.observer_check_window :]
        self._pending_observer_check_events.append(dict(normalized))
        self._last_observer_candidate_hash = candidate_hash

    def _observer_check_trigger(
        self,
        *,
        previous_feedback: Mapping[str, Any],
        previous_metrics: Mapping[str, Any],
        previous_status: str,
        previous_no_progress_steps: int,
    ) -> str | None:
        current_feedback = self._last_task_feedback
        for key in (
            "required_term_recall",
            "required_term_span_progress",
            "summary_term_span_progress",
            "keyword_recall",
            "partial_score",
        ):
            current_value = float(current_feedback.get(key, 0.0) or 0.0)
            previous_value = float(previous_feedback.get(key, 0.0) or 0.0)
            if current_value - previous_value > 0.01:
                return "coverage_progress" if "term" in key or "keyword" in key else "partial_progress"
        if previous_status == "looping" and self._last_status != "looping":
            return "loop_relief"
        if previous_no_progress_steps > 0 and self._no_progress_steps == 0 and self._last_status != "looping":
            return "stall_relief"
        previous_repetition = float(previous_metrics.get("repetition_score", 0.0) or 0.0)
        current_repetition = float(self._last_metrics.get("repetition_score", 0.0) or 0.0)
        if previous_repetition - current_repetition > 0.2 and self._last_status != "looping":
            return "trajectory_shift"
        return None

    def _invoke_observer_check(
        self,
        candidate_text: str,
        *,
        trigger: str,
        task_feedback: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        if self.observer_check_fn is None:
            return None
        feedback = dict(self._last_task_feedback if task_feedback is None else task_feedback)
        try:
            result = self.observer_check_fn(
                candidate_text,
                task_feedback=feedback,
                trigger=trigger,
                worker_runtime=self,
            )
        except TypeError:
            try:
                result = self.observer_check_fn(
                    candidate_text,
                    task_feedback=feedback,
                    trigger=trigger,
                )
            except TypeError:
                try:
                    result = self.observer_check_fn(candidate_text, feedback, trigger)
                except TypeError:
                    result = self.observer_check_fn(candidate_text)
        if not isinstance(result, Mapping):
            return None
        return result

    def _normalize_observer_check_result(
        self,
        value: Mapping[str, Any] | None,
        *,
        trigger: str,
        candidate_hash: str,
        previous_result: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        score = value.get("score")
        raw_score = value.get("raw_score")
        coverage_weight = value.get("coverage_weight")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            return None
        normalized = {
            "check_type": str(value.get("check_type", "semantic_progress")),
            "trigger": str(value.get("trigger", trigger)),
            "score": round(float(score), 6),
            "candidate_hash": str(candidate_hash),
            "recorded_step": int(self._steps),
        }
        if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
            normalized["raw_score"] = round(float(raw_score), 6)
        if isinstance(coverage_weight, (int, float)) and not isinstance(coverage_weight, bool):
            normalized["coverage_weight"] = round(float(coverage_weight), 6)
        for key in ("coverage_signal", "mode", "model_name", "reference_kind"):
            if value.get(key) is not None:
                normalized[key] = value.get(key)
        latent_feature_scan = value.get("latent_feature_scan")
        if isinstance(latent_feature_scan, Mapping):
            normalized["latent_feature_scan"] = dict(latent_feature_scan)
        kv_feature_scan = value.get("kv_feature_scan")
        if isinstance(kv_feature_scan, Mapping):
            normalized["kv_feature_scan"] = dict(kv_feature_scan)
        previous_score = None
        if isinstance(previous_result, Mapping):
            last_value = previous_result.get("score")
            if isinstance(last_value, (int, float)) and not isinstance(last_value, bool):
                previous_score = float(last_value)
        if previous_score is not None:
            delta = round(float(normalized["score"]) - previous_score, 6)
            normalized["delta_vs_last_check"] = delta
            if delta > 0.03:
                normalized["verdict"] = "improved"
            elif delta < -0.03:
                normalized["verdict"] = "regressed"
            else:
                normalized["verdict"] = "flat"
        else:
            normalized["verdict"] = "baseline"
        return normalized

    def _compute_task_feedback(self) -> dict[str, Any]:
        feedback = {"done": self.done(), "progress_label": "stalled" if self._last_status == "looping" else "progressing"}
        if self.task_feedback_fn is None:
            return feedback
        custom = self.task_feedback_fn(self.final_text()) or {}
        return feedback | dict(custom)

    def _effect_metrics(self) -> dict[str, float]:
        metrics = dict(self._last_metrics)
        metrics["repeat_flag"] = float(self._repeat_flag())
        metrics["no_progress_steps"] = float(self._no_progress_steps)
        partial_score = self._last_task_feedback.get("partial_score")
        if partial_score is not None:
            metrics["partial_score"] = float(partial_score)
        for key in ("required_term_recall", "forbidden_term_clean", "word_budget_score"):
            value = self._last_task_feedback.get(key)
            if value is not None:
                metrics[key] = float(value)
        required_term_span_progress = self._last_task_feedback.get("required_term_span_progress")
        if required_term_span_progress is not None:
            metrics["required_term_span_progress"] = float(required_term_span_progress)
        semantic_progress_score = None if self._latest_observer_check is None else self._latest_observer_check.get("score")
        if semantic_progress_score is not None:
            metrics["semantic_progress_score"] = float(semantic_progress_score)
        budget_ok = self._last_task_feedback.get("budget_ok")
        if budget_ok is not None:
            metrics["budget_ok"] = float(bool(budget_ok))
        metrics["progress_score"] = self._progress_score(self._last_task_feedback)
        metrics["task_violation_count"] = float(self._task_violation_count(self._last_task_feedback))
        metrics["done"] = float(bool(self._last_task_feedback.get("done", False)))
        return metrics

    def _progress_score(self, feedback: Mapping[str, Any]) -> float:
        if bool(feedback.get("done", False)):
            return 1.0
        semantic_score = None if self._latest_observer_check is None else self._latest_observer_check.get("score")
        semantic_adjustment = 0.0
        if semantic_score is not None:
            semantic_value = float(semantic_score)
            if semantic_value >= 0.75:
                semantic_adjustment = 0.15
            elif semantic_value <= 0.35:
                semantic_adjustment = -0.15
        label = str(feedback.get("progress_label", "") or "").strip().lower()
        if label == "progressing":
            return 0.25 + semantic_adjustment
        if label == "regressing":
            return -0.5 + semantic_adjustment
        return semantic_adjustment

    def _task_violation_count(self, feedback: Mapping[str, Any]) -> int:
        total = 0
        violations = feedback.get("constraint_violations")
        if isinstance(violations, SequenceABC) and not isinstance(violations, (str, bytes, bytearray)):
            total += sum(1 for item in violations if item)
        for key in ("missing_required_terms", "forbidden_terms_present", "missing_keywords", "missing_summary_terms"):
            detail = feedback.get(key)
            if isinstance(detail, SequenceABC) and not isinstance(detail, (str, bytes, bytearray)):
                total += sum(1 for item in detail if item)
        return total

    def _feature_scan_surface_vectors(self) -> dict[str, torch.Tensor]:
        if getattr(self.runtime_state, "last_cache", None) is None:
            return {}
        if hasattr(self.runtime_state, "set_trace_alignment"):
            self.runtime_state.set_trace_alignment(self._steps)
        ctx = StepContext(packet={"step": self._steps}, runtime_state=self.runtime_state, traces={}, stats={}, adapter=self.adapter, active_edits={})
        vectors: dict[str, torch.Tensor] = {}
        for surface in self.surface_catalog:
            bound = self.adapter.bind_surface(surface)
            if bound.kind != "activation":
                continue
            vector = self._surface_probe_vector(surface, bound, ctx)
            if vector is None:
                continue
            vectors[str(surface.surface_id)] = vector.detach().reshape(-1).cpu().float()
        return vectors

    def _kv_scan_surface_vectors(self) -> list[dict[str, Any]]:
        if getattr(self.runtime_state, "last_cache", None) is None:
            return []
        vectors: list[dict[str, Any]] = []
        for layer in self._feature_scan_layers():
            for site in ("k_cache", "v_cache"):
                cache_tensor = self._cache_tensor_for_scan(layer=layer, site=site)
                if cache_tensor is None or cache_tensor.ndim != 4 or cache_tensor.shape[1] <= 0:
                    continue
                batch0 = cache_tensor[0]
                token_index = batch0.shape[0] - 1
                head_count = int(batch0.shape[1])
                width = int(batch0.shape[2])
                for head in range(head_count):
                    vector = batch0[token_index, head, :].detach().reshape(-1).cpu().float()
                    if vector.numel() != width:
                        continue
                    vectors.append(
                        {
                            "site": site,
                            "layer": int(layer),
                            "head": int(head),
                            "token_mode": "last",
                            "token_index": int(token_index),
                            "head_count": head_count,
                            "width": width,
                            "vector": vector,
                            "surface_id": self._cache_surface_id(site=site, layer=int(layer), head=int(head), token_mode="last"),
                        }
                    )
        return vectors

    def _feature_embedding_matrix(self) -> torch.Tensor | None:
        model = self.model if self.model is not None else getattr(self.runtime_state, "model", None)
        if model is None:
            return None
        for candidate in (
            getattr(model, "W_E", None),
            getattr(getattr(model, "embed", None), "W_E", None),
            getattr(getattr(model, "embed", None), "weight", None),
        ):
            if isinstance(candidate, torch.nn.Parameter):
                return candidate.detach().float()
            if isinstance(candidate, torch.Tensor):
                return candidate.detach().float()
        return None

    def _feature_prototype_vector(self, term: str) -> torch.Tensor | None:
        normalized_term = " ".join(str(term).split()).strip()
        if not normalized_term:
            return None
        cached = self._feature_prototype_cache.get(normalized_term)
        if cached is not None:
            return cached
        embedding_matrix = self._feature_embedding_matrix()
        if embedding_matrix is None:
            return None

        variant_vectors: list[torch.Tensor] = []
        for variant in self._constraint_token_variants(normalized_term):
            try:
                encoded = self.codec.encode(variant).detach().reshape(-1).to(dtype=torch.long).tolist()
            except Exception:
                continue
            if not encoded:
                continue
            if any(token_id < 0 or token_id >= embedding_matrix.shape[0] for token_id in encoded):
                continue
            token_index = torch.tensor(encoded, dtype=torch.long, device=embedding_matrix.device)
            vector = embedding_matrix.index_select(0, token_index).mean(dim=0).reshape(-1).float()
            norm = float(vector.norm().item())
            if norm <= 0.0:
                continue
            variant_vectors.append(vector / norm)
        if not variant_vectors:
            return None

        prototype = torch.stack(variant_vectors, dim=0).mean(dim=0)
        norm = float(prototype.norm().item())
        if norm <= 0.0:
            return None
        prototype = (prototype / norm).detach().cpu().float()
        self._feature_prototype_cache[normalized_term] = prototype
        return prototype

    def _feature_scan_layers(self) -> tuple[int, ...]:
        layers = sorted(
            {
                int(getattr(surface.target, "layer", 0))
                for surface in self.surface_catalog
                if getattr(surface.target, "kind", None) in {"activation", "cache"}
            }
        )
        if layers:
            return tuple(layers)
        cache = getattr(self.runtime_state, "last_cache", None) or {}
        discovered: set[int] = set()
        for hook_name in cache:
            match = re.search(r"blocks\.(\d+)\.attn\.hook_[kv]$", str(hook_name))
            if match:
                discovered.add(int(match.group(1)))
        return tuple(sorted(discovered))

    def _cache_hook_name(self, *, layer: int, site: str) -> str:
        hook_name_for_ref = getattr(self.adapter, "_hook_name_for_ref", None)
        if callable(hook_name_for_ref):
            try:
                return str(hook_name_for_ref(site, layer))
            except Exception:
                pass
        templates = getattr(self.adapter, "HOOK_SITE_TEMPLATES", None)
        if isinstance(templates, Mapping) and site in templates:
            return str(templates[site]).format(layer=layer)
        suffix = "k" if site == "k_cache" else "v"
        return f"blocks.{int(layer)}.attn.hook_{suffix}"

    def _cache_tensor_for_scan(self, *, layer: int, site: str) -> torch.Tensor | None:
        cache = getattr(self.runtime_state, "last_cache", None) or {}
        hook_name = self._cache_hook_name(layer=int(layer), site=str(site))
        tensor = cache.get(hook_name)
        if isinstance(tensor, torch.Tensor):
            return tensor.detach()
        return None

    def _cache_surface_id(self, *, site: str, layer: int, head: int, token_mode: str) -> str | None:
        for surface in self.surface_catalog:
            target = surface.target
            if getattr(target, "kind", None) != "cache":
                continue
            if str(getattr(target, "site", "")) != str(site):
                continue
            if int(getattr(target, "layer", -1)) != int(layer):
                continue
            if int(getattr(target, "head", -1) if getattr(target, "head", None) is not None else -1) != int(head):
                continue
            token = getattr(target, "token", None)
            if token is None or str(getattr(token, "mode", "")) != str(token_mode):
                continue
            return str(surface.surface_id)
        return None

    def _project_feature_into_kv_head(
        self,
        prototype: torch.Tensor,
        *,
        layer: int,
        site: str,
        head: int,
        token_index: int,
        width: int,
        head_count: int,
    ) -> torch.Tensor | None:
        projection = self._kv_projection_tensor(
            layer=layer,
            site=site,
            head_count=head_count,
            d_model=int(prototype.numel()),
            width=width,
        )
        if projection is None or projection.shape[0] <= head:
            return None
        proto = prototype.detach().cpu().float().reshape(-1)
        projected = torch.matmul(proto, projection[int(head)].cpu().float())
        if projected.numel() != width:
            return None
        if site == "k_cache":
            projected = self._apply_k_rotary_projection(projected, layer=layer, token_index=token_index, head=head)
            if projected is None:
                return None
        norm = float(projected.norm().item())
        if norm <= 0.0:
            return None
        return (projected / norm).detach().cpu().float()

    def _kv_projection_tensor(
        self,
        *,
        layer: int,
        site: str,
        head_count: int,
        d_model: int,
        width: int,
    ) -> torch.Tensor | None:
        cache_key = (int(layer), str(site), int(head_count), int(d_model), int(width))
        if cache_key in self._kv_projection_cache:
            return self._kv_projection_cache[cache_key]
        model = self.model if self.model is not None else getattr(self.runtime_state, "model", None)
        block = None if model is None else getattr(getattr(model, "blocks", None), "__getitem__", None)
        if block is None:
            self._kv_projection_cache[cache_key] = None
            return None
        try:
            attn = model.blocks[int(layer)].attn
        except Exception:
            self._kv_projection_cache[cache_key] = None
            return None
        raw = getattr(attn, "W_K" if site == "k_cache" else "W_V", None)
        tensor = self._reshape_kv_projection_tensor(raw, head_count=head_count, d_model=d_model, width=width)
        self._kv_projection_cache[cache_key] = tensor
        return tensor

    def _reshape_kv_projection_tensor(
        self,
        raw: Any,
        *,
        head_count: int,
        d_model: int,
        width: int,
    ) -> torch.Tensor | None:
        if isinstance(raw, torch.nn.Parameter):
            tensor = raw.detach().float()
        elif isinstance(raw, torch.Tensor):
            tensor = raw.detach().float()
        else:
            return None
        canonical: torch.Tensor | None = None
        if tensor.ndim == 3:
            if tensor.shape[1:] == (d_model, width):
                canonical = tensor
            elif tensor.shape[0] == d_model and tensor.shape[2] == width:
                canonical = tensor.permute(1, 0, 2).contiguous()
            elif tensor.shape[0] == d_model and tensor.shape[1] == width:
                canonical = tensor.permute(2, 0, 1).contiguous()
            elif tensor.shape[1] == width and tensor.shape[2] == d_model:
                canonical = tensor.permute(0, 2, 1).contiguous()
        elif tensor.ndim == 2:
            if tensor.shape[0] == d_model and tensor.shape[1] % width == 0:
                canonical = tensor.reshape(d_model, tensor.shape[1] // width, width).permute(1, 0, 2).contiguous()
            elif tensor.shape[1] == d_model and tensor.shape[0] % width == 0:
                canonical = tensor.reshape(tensor.shape[0] // width, width, d_model).permute(0, 2, 1).contiguous()
        if canonical is None:
            return None
        return self._align_kv_projection_heads(canonical, head_count=head_count, d_model=d_model, width=width)

    def _align_kv_projection_heads(
        self,
        tensor: torch.Tensor,
        *,
        head_count: int,
        d_model: int,
        width: int,
    ) -> torch.Tensor | None:
        if tensor.ndim != 3 or tensor.shape[1:] != (d_model, width):
            return None
        if tensor.shape[0] == head_count:
            return tensor.contiguous()

        raw_head_count = int(tensor.shape[0])
        if raw_head_count > head_count and raw_head_count % head_count == 0:
            # GQA models can expose W_K/W_V on expanded query heads while the live cache
            # keeps only KV heads. Collapse each contiguous query-head group back to one KV head.
            group_size = raw_head_count // head_count
            return tensor.reshape(head_count, group_size, d_model, width).mean(dim=1).contiguous()

        if raw_head_count < head_count and head_count % raw_head_count == 0:
            repeat_factor = head_count // raw_head_count
            return tensor.repeat_interleave(repeat_factor, dim=0).contiguous()

        return None

    def _apply_k_rotary_projection(
        self,
        projected: torch.Tensor,
        *,
        layer: int,
        token_index: int,
        head: int,
    ) -> torch.Tensor | None:
        model = self.model if self.model is not None else getattr(self.runtime_state, "model", None)
        try:
            attn = model.blocks[int(layer)].attn if model is not None else None
        except Exception:
            attn = None
        if attn is None:
            return projected
        apply_rotary = getattr(attn, "apply_rotary", None)
        if not callable(apply_rotary):
            return projected
        try:
            rotated = apply_rotary(
                projected.detach().float().reshape(1, 1, 1, -1),
                past_kv_pos_offset=max(0, int(token_index)),
            )
        except Exception:
            return None
        if not isinstance(rotated, torch.Tensor):
            return None
        return rotated[0, 0, 0].detach().cpu().float()

    def _build_trace_bank(self) -> list[dict[str, Any]]:
        trace_caches = getattr(self.runtime_state, "trace_caches", {})
        trace_sequences = getattr(self.runtime_state, "trace_sequences", {})
        trace_bank: list[dict[str, Any]] = []
        trace_ids = sorted(set(trace_caches) | set(trace_sequences))
        for trace_id in trace_ids:
            meta = dict(getattr(trace_sequences.get(trace_id), "metadata", {}) or {})
            meta.update(self.trace_metadata.get(trace_id, {}))
            tags = list(meta.get("tags", ()))
            if trace_id in trace_sequences and "step_aligned" not in tags:
                tags.append("step_aligned")
            trace_bank.append(
                {
                    "trace_id": trace_id,
                    "origin": meta.get("origin", trace_id),
                    "compatible": bool(meta.get("compatible", True)),
                    "similarity_hint": meta.get("similarity_hint"),
                    "tags": tags,
                }
            )
        return trace_bank

    def _build_probe_frames(self) -> list[dict[str, Any]]:
        if getattr(self.runtime_state, "last_cache", None) is None:
            return []

        if hasattr(self.runtime_state, "set_trace_alignment"):
            self.runtime_state.set_trace_alignment(self._steps)
        ctx = StepContext(packet={"step": self._steps}, runtime_state=self.runtime_state, traces={}, stats={}, adapter=self.adapter, active_edits={})
        frames: list[dict[str, Any]] = []
        trace_ids = {trace["trace_id"] for trace in self._build_trace_bank()}

        for surface in self.surface_catalog:
            bound = self.adapter.bind_surface(surface)
            vector = self._surface_probe_vector(surface, bound, ctx)
            if vector is None:
                frames.append({"surface_id": surface.surface_id, "stats": {"norm": 0.0, "delta_prev": 0.0}})
                continue

            prev = self._previous_probe_vectors.get(surface.surface_id)
            delta_prev = 0.0 if prev is None or prev.numel() != vector.numel() else float((vector - prev).norm().item())
            stats: dict[str, Any] = {
                "norm": float(vector.norm().item()),
                "delta_prev": delta_prev,
            }

            for trace_id, field_name in (
                ("best_success", "cosine_to_best_success"),
                ("last_success", "cosine_to_last_success"),
                ("paired_baseline", "cosine_to_paired_baseline"),
            ):
                if trace_id not in trace_ids:
                    continue
                trace_vector = self._surface_probe_vector(surface, bound, ctx, scope="trace", trace_id=trace_id)
                if trace_vector is None:
                    continue
                cosine = _cosine_similarity(vector, trace_vector)
                if cosine is not None:
                    stats[field_name] = cosine

            if bound.kind == "cache":
                stats["cache_drift"] = delta_prev

            self._previous_probe_vectors[surface.surface_id] = vector.detach().clone()
            frames.append({"surface_id": surface.surface_id, "stats": stats})

        return frames

    def _surface_probe_vector(
        self,
        surface: SurfaceInfo,
        bound: Any,
        ctx: StepContext,
        *,
        scope: str = "runtime",
        trace_id: str | None = None,
    ) -> torch.Tensor | None:
        target = surface.target
        if bound.kind == "weight":
            module_ref = bound.module_ref
            if isinstance(module_ref, torch.nn.Parameter):
                return module_ref.detach().reshape(-1).float()
            if hasattr(module_ref, "weight") and isinstance(module_ref.weight, torch.nn.Parameter):
                return module_ref.weight.detach().reshape(-1).float()
            return None

        ref = {
            "scope": scope,
            "worker": self.worker_id,
            "tensor": "hidden" if target.kind == "activation" and target.site == "resid_post" else target.site,
            "layer": target.layer,
            "token": {
                "mode": target.token.mode,
                **({"value": target.token.value} if target.token.value is not None else {}),
                **({"start": target.token.start, "end": target.token.end, "pool": target.token.pool} if target.token.start is not None else {}),
            },
        }
        if trace_id is not None:
            ref["trace_id"] = trace_id
        if getattr(target, "head", None) is not None:
            ref["head"] = target.head
        try:
            return self.adapter.read_ref(ref, ctx).detach().reshape(-1).float()
        except Exception:
            return None

    def _collect_active_edits(self) -> list[dict[str, Any]]:
        active: list[dict[str, Any]] = []
        for edit_id, registration in getattr(self.runtime_state, "hooks", {}).items():
            active.append(self._active_edit_record(edit_id, registration, default_op="resid_add"))
        for edit_id, registration in getattr(self.runtime_state, "overlays", {}).items():
            active.append(self._active_edit_record(edit_id, registration, default_op="rank1_patch"))
        return active

    def _active_edit_record(self, edit_id: str, registration: Any, *, default_op: str) -> dict[str, Any]:
        metadata = self._registration_metadata(registration)
        ttl_left = self._registration_field(registration, "ttl_steps", default=0)
        revertible = bool(self._registration_field(registration, "revertible", default=True))
        alpha = float(metadata.get("alpha", 0.0))
        return {
            "edit_id": str(edit_id),
            "surface_id": str(metadata.get("surface_id", "unknown")),
            "op": str(metadata.get("op", default_op)),
            "alpha": alpha,
            "ttl_left": int(ttl_left),
            "revertible": revertible,
            "step_size": None if metadata.get("step_size") is None else float(metadata["step_size"]),
            "edit_cost": None if metadata.get("edit_cost") is None else float(metadata["edit_cost"]),
            "hypothesis": metadata.get("hypothesis"),
            "expected_effect": metadata.get("expected_effect"),
            "controller_confidence": None
            if metadata.get("controller_confidence") is None
            else float(metadata["controller_confidence"]),
            "budget_key": str(metadata.get("budget_key", str(edit_id).split(":", 1)[0])),
            "budget_pool": str(metadata.get("budget_pool", MAIN_EDIT_BUDGET_POOL) or MAIN_EDIT_BUDGET_POOL),
        }

    def _registration_field(self, registration: Any, field_name: str, *, default: Any) -> Any:
        if hasattr(registration, field_name):
            return getattr(registration, field_name)
        if isinstance(registration, Mapping):
            return registration.get(field_name, default)
        return default

    def _registration_metadata(self, registration: Any) -> Mapping[str, Any]:
        if hasattr(registration, "metadata"):
            return getattr(registration, "metadata") or {}
        if isinstance(registration, Mapping):
            return registration.get("metadata", {}) or {}
        return {}

    def _budget_state(self, active_edits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        active_patch_slots = sum(1 for edit in active_edits if edit.get("op") == "rank1_patch")
        spent_alpha = sum(self._spent_budget_alpha.values())
        spent_edit_cost = sum(self._spent_budget_cost.values())
        spent_loop_rescue_alpha = sum(self._spent_loop_rescue_alpha.values())
        spent_loop_rescue_edit_cost = sum(self._spent_loop_rescue_cost.values())
        return {
            "edits_left_this_step": self.max_edits_per_step,
            "edits_left_this_run": max(0, self.max_edits_per_run - len(self._spent_budget_alpha)),
            "alpha_left_total": max(0.0, self.max_total_alpha - spent_alpha),
            "edit_cost_left_total": max(0.0, self.max_total_edit_cost - spent_edit_cost),
            "loop_rescue_edits_left_this_run": max(
                0,
                self.max_loop_rescue_edits_per_run - len(self._spent_loop_rescue_alpha),
            ),
            "loop_rescue_alpha_left_total": max(0.0, self.max_loop_rescue_alpha - spent_loop_rescue_alpha),
            "loop_rescue_edit_cost_left_total": max(
                0.0,
                self.max_loop_rescue_edit_cost - spent_loop_rescue_edit_cost,
            ),
            "active_patch_slots_left": max(0, self.max_active_patch_slots - active_patch_slots),
            "rollbackable_ids": [str(edit["edit_id"]) for edit in active_edits if bool(edit.get("revertible", True))],
        }

    def latest_effect_trace(self) -> dict[str, Any]:
        return {
            "completed_effects": [dict(effect) for effect in self._latest_completed_effects],
            "summary": dict(self._recent_effect_summary),
        }

    def _record_trace_step(self, *, emitted_token_id: int | None) -> None:
        if self.trace_recorder is None:
            return
        self.trace_recorder.record_step(
            runtime_state=self.runtime_state,
            step=self._steps,
            generated_tokens=len(self._output_token_ids()),
            emitted_token_id=emitted_token_id,
            output_text=self.final_text(),
            telemetry=self._last_metrics,
            active_edits=self._collect_active_edits(),
        )

    def _surface_to_dict(self, surface: SurfaceInfo) -> dict[str, Any]:
        target = surface.target
        payload: dict[str, Any]
        if target.kind == "activation":
            payload = {
                "kind": target.kind,
                "worker": target.worker,
                "site": target.site,
                "layer": target.layer,
                "token": self._token_to_dict(target.token),
            }
        elif target.kind == "cache":
            payload = {
                "kind": target.kind,
                "worker": target.worker,
                "site": target.site,
                "layer": target.layer,
                "token": self._token_to_dict(target.token),
            }
            if target.head is not None:
                payload["head"] = target.head
        else:
            payload = {
                "kind": target.kind,
                "worker": target.worker,
                "module": target.module,
                "layer": target.layer,
            }
        return {
            "surface_id": surface.surface_id,
            "target": payload,
            "allow_ops": list(surface.allow_ops),
            "caps": {
                "max_alpha": surface.caps.max_alpha,
                "max_ttl_steps": surface.caps.max_ttl_steps,
                "norm_clip": surface.caps.norm_clip,
                "step_size": surface.caps.step_size,
                "rank_cap": surface.caps.rank_cap,
                "revertible_only": surface.caps.revertible_only,
            },
        }

    def _token_to_dict(self, token: Any) -> dict[str, Any]:
        data = {"mode": token.mode}
        if token.value is not None:
            data["value"] = token.value
        if token.start is not None:
            data["start"] = token.start
            data["end"] = token.end
            data["pool"] = token.pool
        return data
