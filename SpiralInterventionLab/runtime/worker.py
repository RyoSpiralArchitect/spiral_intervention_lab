from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, replace
from typing import Any, Callable, Collection, Mapping, Sequence

import torch

from .adapter import ModelAdapter
from .codecs import TextCodec, resolve_text_codec
from .compiler import StepContext, compile_command
from .edit_budget import LOOP_RESCUE_EDIT_BUDGET_POOL, MAIN_EDIT_BUDGET_POOL
from .effects import build_edit_effect, summarize_effects
from .policy import GlobalBudget, HarnessPolicy
from .schema import SurfaceInfo
from .sidecar import (
    ReadoutSidecarAnalyzer,
    ReadoutSidecarCapture,
    ReadoutSidecarSiteCapture,
    normalize_readout_sidecar_hints,
)
from .trace_recorder import StepAlignedTrace, StepAlignedTraceRecorder


_CONTROLLER_REFLECTION_MODES = {"off", "structured"}
_READOUT_ANALYZER_RERANK_MODES = {"off", "shadow", "apply"}
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
    "blocked_by",
    "next_change",
    "next_evidence_needed",
    "diagnostic_request",
    "stop_condition",
    "focus_term",
)
_CONTROLLER_MEMORY_LABEL_FIELDS = {
    "noop_reason": 48,
    "apply_block_reason": 64,
    "surface_family_key": 96,
}
_CONTROLLER_MEMORY_ALLOWED_OUTCOMES = {"unknown", "helpful", "harmful", "neutral", "mixed"}
_CONTROLLER_MEMORY_ALLOWED_NEXT_ACTIONS = {
    "wait",
    "noop",
    "apply",
    "rollback",
    "request_observer_check",
    "request_operator_diagnostic",
    "request_attention_head_ablation",
    "request_sae_feature_scan",
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


def _normalize_readout_analyzer_rerank_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in _READOUT_ANALYZER_RERANK_MODES:
        raise ValueError(
            f"Unsupported readout analyzer rerank mode '{value}'; expected one of {sorted(_READOUT_ANALYZER_RERANK_MODES)}"
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
    for key, limit in _CONTROLLER_MEMORY_LABEL_FIELDS.items():
        label = _normalize_controller_memory_label(value.get(key), limit=limit)
        if label is not None:
            entry[key] = label
    confidence = value.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        entry["confidence"] = max(0.0, min(1.0, float(confidence)))
    for key in ("transfer_confidence", "same_family_escalation_risk"):
        raw_value = value.get(key)
        if isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
            entry[key] = max(0.0, min(1.0, float(raw_value)))
    finish_budget_reserved = value.get("finish_budget_reserved")
    if isinstance(finish_budget_reserved, bool):
        entry["finish_budget_reserved"] = finish_budget_reserved
    elif isinstance(finish_budget_reserved, (int, float)) and not isinstance(finish_budget_reserved, bool):
        entry["finish_budget_reserved"] = max(0, min(8, int(round(float(finish_budget_reserved)))))
    raw_bullets = value.get("evidence_bullets")
    if isinstance(raw_bullets, SequenceABC) and not isinstance(raw_bullets, (str, bytes, bytearray)):
        bullets: list[str] = []
        for item in raw_bullets:
            text = _clean_controller_memory_text(item, limit=96)
            if text is None or text in bullets:
                continue
            bullets.append(text)
            if len(bullets) >= 4:
                break
        if bullets:
            entry["evidence_bullets"] = bullets
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
        readout_sidecar_analyzer: ReadoutSidecarAnalyzer | None = None,
        readout_analyzer_rerank_mode: str = "apply",
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
        self.readout_sidecar_analyzer = readout_sidecar_analyzer
        self.readout_analyzer_rerank_mode = _normalize_readout_analyzer_rerank_mode(readout_analyzer_rerank_mode)

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
        self._latest_readout_sidecar_capture: ReadoutSidecarCapture | None = None
        self._latest_readout_sidecar_capture_summary: dict[str, Any] | None = None
        self._latest_readout_sidecar_hints: dict[str, Any] = {}
        self._operator_certification_table: dict[str, dict[str, Any]] = {}
        self._operator_bridge_plan_table: dict[str, dict[str, Any]] = {}

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
        answer_readout_canary = self._current_answer_readout_canary(
            promoted_cache_surfaces=promoted_cache_surfaces,
        )
        control_phase_hint = self._control_phase_hint(answer_readout_canary=answer_readout_canary)
        readout_sidecar_capture = self._build_readout_sidecar_capture(
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
        )
        readout_sidecar_hints = self._analyze_readout_sidecar_capture(readout_sidecar_capture)
        if self._latest_readout_sidecar_hints:
            readout_sidecar_hints = dict(self._latest_readout_sidecar_hints)
        strategy_hints = self._strategy_hints(
            control_phase_hint=control_phase_hint,
            promoted_cache_surfaces=promoted_cache_surfaces,
            answer_readout_canary=answer_readout_canary,
            readout_sidecar_hints=readout_sidecar_hints,
        )
        latest_observer_check = self._observer_check_with_cache_surface_ids(
            self._latest_observer_check,
            promoted_cache_surfaces=promoted_cache_surfaces,
        )
        worker_view = self._worker_view()
        if answer_readout_canary is not None:
            worker_view["answer_readout_canary"] = answer_readout_canary
            strategy_hints["answer_readout_summary"] = {
                "semantic_focus_term": answer_readout_canary.get("semantic_focus_term"),
                "semantic_focus_source": answer_readout_canary.get("semantic_focus_source"),
                "reachable_focus_term": answer_readout_canary.get("reachable_focus_term"),
                "reachable_focus_piece": answer_readout_canary.get("reachable_focus_piece"),
                "reachable_focus_rank": answer_readout_canary.get("reachable_focus_rank"),
                "target_mass": answer_readout_canary.get("target_mass"),
                "target_top20_hits": answer_readout_canary.get("target_top20_hits"),
                "attractor_family_mass": answer_readout_canary.get("attractor_family_mass"),
                "attractor_family_top_overlap": answer_readout_canary.get("attractor_family_top_overlap"),
                "attractor_family_overlap_tokens": list(answer_readout_canary.get("attractor_family_overlap_tokens", [])[:5]),
                "top_tokens": list(answer_readout_canary.get("top_tokens", [])[:5]),
            }
        if self._latest_readout_sidecar_capture_summary is not None:
            worker_view["readout_sidecar_capture_summary"] = dict(self._latest_readout_sidecar_capture_summary)
            strategy_hints["readout_sidecar_capture_summary"] = dict(self._latest_readout_sidecar_capture_summary)
            worker_view["readout_analyzer_capture_summary"] = dict(self._latest_readout_sidecar_capture_summary)
            strategy_hints["readout_analyzer_capture_summary"] = dict(self._latest_readout_sidecar_capture_summary)
        if readout_sidecar_hints:
            worker_view["readout_sidecar_hints"] = dict(readout_sidecar_hints)
            strategy_hints["readout_sidecar_hints"] = dict(readout_sidecar_hints)
            worker_view["readout_analyzer_hints"] = dict(readout_sidecar_hints)
            strategy_hints["readout_analyzer_hints"] = dict(readout_sidecar_hints)
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
            "worker_view": worker_view,
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
        missing_required_terms = packet["task_feedback"].get("missing_required_terms")
        if not isinstance(missing_required_terms, SequenceABC) or isinstance(
            missing_required_terms, (str, bytes, bytearray)
        ):
            missing_required_terms = ()
        edits_left_this_run = int(packet["budget"].get("edits_left_this_run", 0) or 0)
        if control_phase_hint in {"shot_mode", "readout_escape"}:
            if len(tuple(missing_required_terms)) == 1 and edits_left_this_run <= 1:
                strategy_hints["finish_budget_reserve_suggested"] = True
                strategy_hints["finish_budget_reserve_reason"] = "single_missing_term_with_low_budget"
                strategy_hints.setdefault("suggested_noop_reason", "hold_finish_budget")
            if strategy_hints.get("selected_bundle_key") not in (None, "") and int(
                strategy_hints.get("target_top20_hits", 0) or 0
            ) == 0:
                strategy_hints["same_family_alpha_escalation_requires_gain"] = True
                strategy_hints["same_family_alpha_escalation_reason"] = "no_first_token_hit_yet"
                strategy_hints.setdefault("suggested_noop_reason", "needs_canary_gain")
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
                        feature=term,
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
                surface_family_key=pending.get("surface_family_key"),
                operator_recipe_id=pending.get("operator_recipe_id"),
                operator_recipe_seed_key=pending.get("operator_recipe_seed_key"),
                bundle_key=pending.get("bundle_key"),
                objective_bundle_key=pending.get("objective_bundle_key"),
                step_actuator_bundle_key=pending.get("step_actuator_bundle_key"),
                apply_kind=pending.get("apply_kind"),
                production_apply_allowed=pending.get("production_apply_allowed"),
                production_policy_would_apply=pending.get("production_policy_would_apply"),
                certified_for_apply=pending.get("certified_for_apply"),
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
                    "surface_family_key": active.get("surface_family_key"),
                    "operator_recipe_id": active.get("operator_recipe_id"),
                    "operator_recipe_seed_key": active.get("operator_recipe_seed_key"),
                    "bundle_key": active.get("bundle_key"),
                    "objective_bundle_key": active.get("objective_bundle_key"),
                    "step_actuator_bundle_key": active.get("step_actuator_bundle_key"),
                    "apply_kind": active.get("apply_kind"),
                    "production_apply_allowed": active.get("production_apply_allowed"),
                    "production_policy_would_apply": active.get("production_policy_would_apply"),
                    "certified_for_apply": active.get("certified_for_apply"),
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
        self._latest_readout_sidecar_capture = None
        self._latest_readout_sidecar_capture_summary = None
        self._latest_readout_sidecar_hints = {}
        self._operator_bridge_plan_table = {}
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

    def _canonical_effect_family_piece(self, value: Any) -> str:
        raw = "".join(ch for ch in str(value).lower().strip() if ch.isalnum())
        return raw[:16]

    def _effect_family_signature_from_probe_item(self, item: Mapping[str, Any]) -> tuple[str, ...]:
        signature: list[str] = []
        sampled = item.get("sampled_continuations")
        if isinstance(sampled, SequenceABC) and not isinstance(sampled, (str, bytes, bytearray)):
            for entry in sampled:
                if not isinstance(entry, Mapping):
                    continue
                if str(entry.get("variant", "") or "") != "candidate":
                    continue
                anchor = self._canonical_effect_family_piece(entry.get("text", ""))
                if anchor:
                    signature.append(f"cont:{anchor}")
                    break
        topk = item.get("topk_token_diff")
        if isinstance(topk, SequenceABC) and not isinstance(topk, (str, bytes, bytearray)):
            rows: list[tuple[float, str]] = []
            for token_row in topk:
                if not isinstance(token_row, Mapping):
                    continue
                piece = self._canonical_effect_family_piece(token_row.get("piece", ""))
                if not piece:
                    continue
                prob_delta = float(token_row.get("prob_delta", 0.0) or 0.0)
                logit_delta = float(token_row.get("logit_delta", 0.0) or 0.0)
                magnitude = max(abs(prob_delta), abs(logit_delta))
                if magnitude <= 0.0:
                    continue
                direction = "up" if (prob_delta > 0.0 or (prob_delta == 0.0 and logit_delta > 0.0)) else "down"
                rows.append((magnitude, f"{direction}:{piece}"))
            rows.sort(key=lambda item: (-float(item[0]), item[1]))
            for _magnitude, label in rows[:3]:
                if label not in signature:
                    signature.append(label)
        return tuple(signature[:4])

    def _probe_family_kind(self, candidate_edit: Mapping[str, Any]) -> str:
        family_kind = self._candidate_family_kind(candidate_edit)
        return family_kind if family_kind and family_kind != "unknown" else str(candidate_edit.get("kind", "") or "unknown")

    def _probe_phase_profile(self, candidate_edit: Mapping[str, Any]) -> str:
        phase_objective = str(candidate_edit.get("phase_objective", "") or "")
        if phase_objective == "readout_escape":
            return "readout_escape"
        if phase_objective in {"shot_mode", "entity_insertion", "composition"}:
            return "composition"
        if phase_objective == "loop_break":
            return "trajectory"
        family_kind = self._probe_family_kind(candidate_edit)
        if family_kind in {"kv_v", "kv_k", "kv_mix"}:
            return "readout_escape"
        if family_kind in {"shot_bridge", "resid_add"}:
            return "composition"
        return "balanced"

    def _classify_probe_result(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        candidate_edit = item.get("candidate_edit")
        if not isinstance(candidate_edit, Mapping):
            return None
        surface_id = str(candidate_edit.get("surface_id", "") or "")
        if not surface_id:
            return None
        probe_family = self._probe_family_kind(candidate_edit)
        if not probe_family:
            return None
        phase_profile = self._probe_phase_profile(candidate_edit)
        recall_delta = float(item.get("required_term_recall_delta", 0.0) or 0.0)
        span_delta = float(item.get("required_term_span_progress_delta", 0.0) or 0.0)
        semantic_delta = float(item.get("semantic_progress_delta", 0.0) or 0.0)
        repeat_delta = int(item.get("repeat_flag_delta", 0) or 0)
        target_mass_delta = float(item.get("target_mass_delta", 0.0) or 0.0)
        target_mass_edited = float(item.get("target_mass_edited", 0.0) or 0.0)
        target_top20_hits_edited = int(item.get("target_top20_hits_edited", 0) or 0)
        target_top20_hit_delta = int(item.get("target_top20_hit_delta", 0) or 0)
        focus_rank_delta = int(item.get("focus_rank_delta", 0) or 0)
        focus_rank_edited = int(item.get("focus_rank_edited", 0) or 0)
        rank_focus_delta = int(item.get("rank_focus_delta", 0) or 0)
        rank_focus_rank_edited = int(item.get("rank_focus_rank_edited", 0) or 0)
        focus_logit_delta = float(item.get("focus_logit_delta", 0.0) or 0.0)
        focus_prob_delta = float(item.get("focus_prob_delta", 0.0) or 0.0)
        candidate_meta = candidate_edit.get("meta") if isinstance(candidate_edit.get("meta"), Mapping) else {}
        retry_stage = (
            str(candidate_edit.get("retry_stage", "") or "")
            or str(candidate_meta.get("retry_stage", "") or "")
        )
        prefix_depth_delta = int(
            item.get("prefix_depth_delta", item.get("canary_prefix_depth_delta", 0)) or 0
        )
        topk = item.get("topk_token_diff")
        max_logit_delta = 0.0
        max_prob_delta = 0.0
        if isinstance(topk, SequenceABC) and not isinstance(topk, (str, bytes, bytearray)):
            for token_row in topk:
                if not isinstance(token_row, Mapping):
                    continue
                max_logit_delta = max(max_logit_delta, float(token_row.get("logit_delta", 0.0) or 0.0))
                max_prob_delta = max(max_prob_delta, float(token_row.get("prob_delta", 0.0) or 0.0))
        semantic_focus_term = str(candidate_edit.get("focus_feature", "") or "")
        reachable_focus_term = str(
            item.get("rank_focus_term")
            or item.get("focus_term")
            or semantic_focus_term
            or ""
        )
        reachable_focus_rank = 10**9
        if item.get("rank_focus_term") not in (None, "") and rank_focus_rank_edited > 0:
            reachable_focus_rank = int(rank_focus_rank_edited)
        elif item.get("focus_term") not in (None, "") and focus_rank_edited > 0:
            reachable_focus_rank = int(focus_rank_edited)
        effect_family_signature = self._effect_family_signature_from_probe_item(item)
        effect_family_key = "|".join(effect_family_signature) if effect_family_signature else f"surface:{surface_id}"
        candidate_key = self._candidate_sidecar_key(candidate_edit)

        readout_score = 0.0
        if target_top20_hit_delta > 0:
            readout_score += 1.25 + (0.15 * float(target_top20_hit_delta))
        if target_top20_hits_edited > 0:
            readout_score += min(0.75, 0.25 * float(target_top20_hits_edited))
        readout_score += min(0.9, 900.0 * max(0.0, target_mass_delta))
        readout_score += min(0.9, 700.0 * max(0.0, target_mass_edited))
        readout_score += min(0.75, 0.04 * float(max(0, focus_rank_delta, rank_focus_delta)))
        if reachable_focus_rank != 10**9 and reachable_focus_rank > 0:
            readout_score += min(0.85, max(0.0, (512.0 - min(float(reachable_focus_rank), 512.0)) / 512.0))
        readout_score += min(0.6, 600.0 * max(0.0, max_logit_delta))
        readout_score += min(0.6, 6000.0 * max(0.0, max_prob_delta))
        if prefix_depth_delta > 0:
            readout_score += min(0.75, 0.35 * float(prefix_depth_delta))

        constraint_score = 0.0
        if recall_delta > 0.0:
            constraint_score += min(1.5, 4.0 * recall_delta)
        if span_delta > 0.0:
            constraint_score += min(1.2, 3.0 * span_delta)
        if semantic_delta > 0.0:
            constraint_score += min(0.8, 6.0 * semantic_delta)

        trajectory_score = 0.0
        if repeat_delta < 0:
            trajectory_score += min(1.0, 0.75 * abs(float(repeat_delta)))
        if prefix_depth_delta > 0:
            trajectory_score += min(0.5, 0.2 * float(prefix_depth_delta))
        if semantic_delta > 0.0:
            trajectory_score += min(0.4, 3.0 * semantic_delta)
        if target_top20_hit_delta > 0 or focus_rank_delta > 0 or rank_focus_delta > 0:
            trajectory_score += min(0.4, 0.01 * float(max(target_top20_hit_delta, focus_rank_delta, rank_focus_delta)))

        positive_axes: list[str] = []
        if (
            target_top20_hit_delta > 0
            or target_mass_delta > 0.00001
            or target_mass_edited >= 0.0003
            or focus_rank_delta >= 4
            or rank_focus_delta >= 4
            or max_logit_delta >= 0.0005
            or max_prob_delta >= 0.00005
            or prefix_depth_delta > 0
            or readout_score >= 0.35
        ):
            positive_axes.append("readout")
        if (
            recall_delta > 0.0
            or span_delta > 0.0
            or semantic_delta > 0.01
            or constraint_score >= 0.4
        ):
            positive_axes.append("constraint")
        if repeat_delta < 0 or prefix_depth_delta > 0 or trajectory_score >= 0.35:
            positive_axes.append("trajectory")

        actionable_axes: list[str] = []
        if (
            target_top20_hits_edited > 0
            or target_mass_edited >= 0.001
            or reachable_focus_rank <= 150
            or prefix_depth_delta > 0
        ):
            actionable_axes.append("readout")
        if recall_delta > 0.0 or span_delta > 0.0 or semantic_delta >= 0.03:
            actionable_axes.append("constraint")
        if repeat_delta < 0 and (target_top20_hit_delta > 0 or target_mass_delta > 0.0 or semantic_delta > 0.0):
            actionable_axes.append("trajectory")

        weight_map = {
            "readout_escape": {"readout": 0.6, "constraint": 0.2, "trajectory": 0.2},
            "composition": {"readout": 0.2, "constraint": 0.6, "trajectory": 0.2},
            "trajectory": {"readout": 0.15, "constraint": 0.15, "trajectory": 0.7},
            "balanced": {"readout": 0.34, "constraint": 0.43, "trajectory": 0.23},
        }
        weights = weight_map.get(phase_profile, weight_map["balanced"])
        composite_score = (
            (weights["readout"] * readout_score)
            + (weights["constraint"] * constraint_score)
            + (weights["trajectory"] * trajectory_score)
        )

        harmful_signal = bool(repeat_delta > 0 or semantic_delta < -0.01)
        dead_actuator_signal = bool(
            target_top20_hit_delta <= 0
            and target_mass_delta <= 0.00001
            and abs(focus_logit_delta) < 0.001
            and abs(focus_prob_delta) < 0.00005
            and focus_rank_delta < 4
            and rank_focus_delta < 4
            and recall_delta <= 0.0
            and span_delta <= 0.0
            and semantic_delta <= 0.01
            and prefix_depth_delta <= 0
        )

        if harmful_signal:
            label = "harmful"
            score = -2.0 - (0.75 * max(0, repeat_delta)) - (12.0 * max(0.0, -semantic_delta))
        elif dead_actuator_signal:
            label = "dead_actuator"
            score = -1.5
        else:
            if phase_profile == "readout_escape":
                actionable = bool("readout" in actionable_axes or (readout_score >= 1.6 and "readout" in positive_axes))
                strong_positive = bool(
                    readout_score >= 1.15
                    or ("readout" in positive_axes and "trajectory" in positive_axes)
                    or constraint_score >= 0.6
                )
            elif phase_profile == "composition":
                actionable = bool(
                    "constraint" in actionable_axes or (constraint_score >= 1.0 and "constraint" in positive_axes)
                )
                strong_positive = bool(
                    constraint_score >= 0.75
                    or ("constraint" in positive_axes and ("readout" in positive_axes or "trajectory" in positive_axes))
                )
            elif phase_profile == "trajectory":
                actionable = bool("trajectory" in actionable_axes or trajectory_score >= 0.9)
                strong_positive = bool(trajectory_score >= 0.5 or "trajectory" in positive_axes)
            else:
                actionable = bool(actionable_axes)
                strong_positive = bool(composite_score >= 0.85 or len(positive_axes) >= 2)

            if actionable:
                label = "actionable_positive"
                score = 1.75 + composite_score
            elif strong_positive:
                label = "positive"
                score = 1.0 + composite_score
            elif positive_axes:
                label = "weak_positive_subthreshold"
                score = 0.45 + composite_score
            else:
                label = "flat"
                score = -1.0 + composite_score

        return {
            "surface_id": surface_id,
            "candidate_key": candidate_key,
            "probe_family": probe_family,
            "probe_phase_profile": phase_profile,
            "label": label,
            "score": round(float(score), 6),
            "readout_score": round(float(readout_score), 6),
            "constraint_score": round(float(constraint_score), 6),
            "trajectory_score": round(float(trajectory_score), 6),
            "composite_score": round(float(composite_score), 6),
            "positive_axes": list(positive_axes),
            "actionable_axes": list(actionable_axes),
            "required_term_recall_delta": round(recall_delta, 6),
            "required_term_span_progress_delta": round(span_delta, 6),
            "semantic_progress_delta": round(semantic_delta, 6),
            "repeat_flag_delta": int(repeat_delta),
            "target_mass_delta": round(target_mass_delta, 6),
            "target_mass_edited": round(target_mass_edited, 6),
            "target_top20_hit_delta": int(target_top20_hit_delta),
            "target_top20_hits_edited": int(target_top20_hits_edited),
            "focus_logit_delta": round(focus_logit_delta, 6),
            "focus_prob_delta": round(focus_prob_delta, 6),
            "focus_rank_delta": int(focus_rank_delta),
            "focus_rank_edited": int(focus_rank_edited),
            "rank_focus_delta": int(rank_focus_delta),
            "rank_focus_rank_edited": int(rank_focus_rank_edited),
            "prefix_depth_delta": int(prefix_depth_delta),
            "max_logit_delta": round(max_logit_delta, 6),
            "max_prob_delta": round(max_prob_delta, 6),
            "semantic_focus_term": semantic_focus_term or None,
            "reachable_focus_term": reachable_focus_term or None,
            "reachable_focus_rank": None if reachable_focus_rank == 10**9 else int(reachable_focus_rank),
            "effect_family_key": effect_family_key,
            "effect_family_signature": list(effect_family_signature),
            "dead_actuator": bool(label == "dead_actuator"),
            "subthreshold_only": bool(label == "weak_positive_subthreshold"),
            "was_retry": bool(retry_stage),
        }

    def _classify_kv_probe_result(self, item: Mapping[str, Any]) -> dict[str, Any] | None:
        summary = self._classify_probe_result(item)
        if summary is None:
            return None
        if str(summary.get("probe_family", "") or "") not in {"kv_v", "kv_k", "kv_mix"}:
            return None
        return summary

    def _recent_probe_history(
        self,
        *,
        window: int = 8,
        probe_families: Collection[str] | None = None,
        phase_profiles: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        allowed_families = {str(item) for item in probe_families} if probe_families is not None else None
        allowed_profiles = {str(item) for item in phase_profiles} if phase_profiles is not None else None
        for item in reversed(list(self._tool_results)[-window:]):
            if not isinstance(item, Mapping):
                continue
            if str(item.get("tool", "") or "") != "dry_run_decode":
                continue
            summary = self._classify_probe_result(item)
            if summary is None:
                continue
            if allowed_families is not None and str(summary.get("probe_family", "") or "") not in allowed_families:
                continue
            if allowed_profiles is not None and str(summary.get("probe_phase_profile", "") or "") not in allowed_profiles:
                continue
            history.append(summary)
        return history

    def _recent_kv_probe_history(self, *, window: int = 8) -> list[dict[str, Any]]:
        return self._recent_probe_history(window=window, probe_families={"kv_v", "kv_k", "kv_mix"})

    def _recent_probe_outcomes_by_candidate_key(
        self,
        *,
        window: int = 8,
        probe_families: Collection[str] | None = None,
        phase_profiles: Collection[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        outcomes: dict[str, dict[str, Any]] = {}
        for summary in self._recent_probe_history(
            window=window,
            probe_families=probe_families,
            phase_profiles=phase_profiles,
        ):
            candidate_key = str(summary.get("candidate_key", "") or "")
            if not candidate_key or candidate_key in outcomes:
                continue
            outcomes[candidate_key] = dict(summary)
        return outcomes

    def _recent_kv_probe_outcomes(self, *, window: int = 8) -> dict[str, dict[str, Any]]:
        outcomes: dict[str, dict[str, Any]] = {}
        for summary in self._recent_probe_history(window=window, probe_families={"kv_v", "kv_k", "kv_mix"}):
            surface_id = str(summary.get("surface_id", "") or "")
            if not surface_id or surface_id in outcomes:
                continue
            outcomes[surface_id] = dict(summary)
        return outcomes

    def _encode_text_token_ids(self, text: str) -> list[int]:
        try:
            encoded = self.codec.encode(text)
        except Exception:
            return []
        if isinstance(encoded, torch.Tensor):
            return encoded.detach().reshape(-1).to(dtype=torch.long).tolist()
        if isinstance(encoded, SequenceABC) and not isinstance(encoded, (str, bytes, bytearray)):
            token_ids: list[int] = []
            for item in encoded:
                if isinstance(item, bool) or not isinstance(item, int):
                    return []
                token_ids.append(int(item))
            return token_ids
        return []

    def _provenance_weight(self, provenance_class: str) -> float:
        return {
            "source_body": 1.0,
            "answer_prefix": 0.2,
            "constraint_header": -0.3,
            "misc_prompt": -0.5,
        }.get(str(provenance_class), -0.75)

    def _family_weight(self, family_kind: str) -> float:
        return {
            "kv_pair_ready": 1.3,
            "kv_v": 1.0,
            "kv_k": 0.9,
            "shot_bridge": 0.45,
            "resid_add": 0.2,
        }.get(str(family_kind), 0.0)

    def _span_weight(self, span_kind: str) -> float:
        return {
            "exact_prompt_span_mean": 0.4,
            "exact_prompt_piece": 0.35,
            "source_position_single": 0.1,
        }.get(str(span_kind), 0.0)

    def _candidate_family_kind(self, candidate: Mapping[str, Any]) -> str:
        kind = str(candidate.get("kind", "") or "")
        site = str(candidate.get("site", "") or "")
        role = str(candidate.get("role", "") or "")
        if kind == "kv_mix":
            if site == "v_cache":
                return "kv_v"
            if site == "k_cache":
                return "kv_k"
        if kind == "resid_add":
            if role.startswith("shot_source_bridge"):
                return "shot_bridge"
            return "resid_add"
        return kind or "unknown"

    def _candidate_span_id(self, candidate: Mapping[str, Any]) -> str:
        source_span = candidate.get("source_span")
        if isinstance(source_span, Mapping):
            try:
                start = int(source_span.get("start", 0) or 0)
                end = int(source_span.get("end", start + 1) or (start + 1))
                return f"{start}:{end}"
            except Exception:
                pass
        source_position = candidate.get("source_position")
        if isinstance(source_position, int) and not isinstance(source_position, bool):
            return f"{int(source_position)}:{int(source_position) + 1}"
        return "unknown"

    def _bundle_focus_term(
        self,
        bundle_key: str,
        members: Sequence[Mapping[str, Any]],
    ) -> str:
        for item in members:
            feature = str(item.get("focus_feature", "") or item.get("focus_term", "") or "")
            if feature:
                return feature
        parts = str(bundle_key or "").split(":")
        if len(parts) >= 2:
            return str(parts[1] or "")
        return ""

    def _operator_recipe_id(self, item: Mapping[str, Any]) -> str:
        phase_objective = str(
            item.get("phase_objective")
            or item.get("probe_phase_profile")
            or item.get("control_phase_hint")
            or "unknown"
        )
        family = str(item.get("bundle_family", "") or "")
        if not family:
            family = self._candidate_family_kind(item)
        provenance_class = str(item.get("provenance_class", "") or "unknown")
        localization = str(
            item.get("recipe_localization")
            or item.get("span_kind")
            or "unknown"
        )
        pooling = str(item.get("recipe_pooling", "") or "unknown")
        contrast_mode = str(item.get("contrast_mode", "") or "none")
        contrast_scale = float(item.get("recipe_contrast_scale", 1.0) or 1.0)
        contrast_tag = (
            contrast_mode
            if contrast_mode == "none" or abs(float(contrast_scale) - 1.0) <= 1e-8
            else f"{contrast_mode}@{round(float(contrast_scale), 3):.3f}"
        )
        if family == "kv_pair_source_anchor":
            k_alpha = float(item.get("recipe_k_alpha", item.get("k_alpha", 0.0)) or 0.0)
            v_alpha = float(item.get("recipe_v_alpha", item.get("v_alpha", 0.0)) or 0.0)
            alpha_key = f"k{round(k_alpha, 4):.4f}|v{round(v_alpha, 4):.4f}"
        else:
            alpha = float(item.get("recipe_alpha", item.get("alpha", item.get("op", {}).get("alpha", 0.0))) or 0.0)
            alpha_key = f"a{round(alpha, 4):.4f}"
        return f"{phase_objective}|{family}|{provenance_class}|{localization}|{pooling}|{contrast_tag}|{alpha_key}"

    def _operator_recipe_seed_key(
        self,
        *,
        mode: str,
        localization: str,
        pooling: str,
    ) -> str:
        return f"{str(mode or 'unknown')}|{str(localization or 'unknown')}|{str(pooling or 'unknown')}"

    def _operator_family_key(self, item: Mapping[str, Any]) -> str:
        phase_objective = str(
            item.get("phase_objective")
            or item.get("probe_phase_profile")
            or item.get("control_phase_hint")
            or "unknown"
        )
        family = str(item.get("bundle_family", "") or "")
        if not family:
            family = self._candidate_family_kind(item)
        provenance_class = str(item.get("provenance_class", "") or "unknown")
        span_kind = str(item.get("span_kind", "") or "unknown")
        return f"{phase_objective}|{family}|{provenance_class}|{span_kind}"

    def _term_readout_lift_score(self, metrics: Mapping[str, Any]) -> float:
        target_mass_delta = float(metrics.get("target_mass_delta", 0.0) or 0.0)
        target_top20_hit_delta = int(metrics.get("target_top20_hit_delta", 0) or 0)
        focus_rank_delta = int(metrics.get("focus_rank_delta", 0) or 0)
        rank_focus_delta = int(metrics.get("rank_focus_delta", 0) or 0)
        return round(
            max(0.0, 1000.0 * target_mass_delta)
            + (0.35 * max(0, target_top20_hit_delta))
            + (0.01 * max(0, focus_rank_delta, rank_focus_delta)),
            6,
        )

    def _term_readout_deltas(
        self,
        baseline_logits: torch.Tensor,
        edited_logits: torch.Tensor,
        *,
        focus_terms: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        term_metrics: dict[str, dict[str, Any]] = {}
        seen_terms: set[str] = set()
        for raw_term in focus_terms:
            term = str(raw_term or "")
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            metrics = self._first_token_target_readout_metrics(
                baseline_logits,
                edited_logits,
                focus_terms=[term],
            )
            if not metrics:
                continue
            metrics = dict(metrics)
            metrics["lift_score"] = self._term_readout_lift_score(metrics)
            term_metrics[term] = metrics
        return term_metrics

    def _classify_actual_delta_result(self, item: Mapping[str, Any]) -> str:
        continuation_baseline = str(item.get("continuation_baseline", "") or "")
        continuation_candidate = str(item.get("continuation_candidate", "") or "")
        continuation_changed = continuation_baseline != continuation_candidate
        recall_delta = float(item.get("required_term_recall_delta", 0.0) or 0.0)
        span_delta = float(item.get("required_term_span_progress_delta", 0.0) or 0.0)
        semantic_delta = float(item.get("semantic_progress_delta", 0.0) or 0.0)
        repeat_delta = int(item.get("repeat_flag_delta", 0) or 0)
        entropy_delta = float(item.get("entropy_delta", 0.0) or 0.0)
        top1_margin_delta = float(item.get("top1_margin_delta", 0.0) or 0.0)
        repetition_score_delta = float(item.get("repetition_score_delta", 0.0) or 0.0)
        target_mass_delta = float(item.get("target_mass_delta", 0.0) or 0.0)
        target_top20_hit_delta = int(item.get("target_top20_hit_delta", 0) or 0)
        focus_rank_delta = int(item.get("focus_rank_delta", 0) or 0)
        rank_focus_delta = int(item.get("rank_focus_delta", 0) or 0)
        prefix_depth_delta = int(item.get("prefix_depth_delta", item.get("canary_prefix_depth_delta", 0)) or 0)
        if (
            recall_delta <= 0.0
            and span_delta <= 0.0
            and semantic_delta <= 0.01
            and repeat_delta > 0
            and entropy_delta < -0.02
            and top1_margin_delta > 0.005
            and repetition_score_delta > 0.05
        ):
            return "collapse_sharpener"
        if repeat_delta > 0 or semantic_delta < -0.03 or recall_delta < 0.0 or span_delta < 0.0:
            return "harmful"
        if (
            recall_delta > 0.0
            or span_delta > 0.0
            or target_top20_hit_delta > 0
            or target_mass_delta > 0.00002
            or focus_rank_delta >= 6
            or rank_focus_delta >= 6
            or prefix_depth_delta > 0
        ):
            return "target_lift"
        if (
            continuation_changed
            and recall_delta <= 0.0
            and span_delta <= 0.0
            and semantic_delta <= 0.01
            and (target_mass_delta < -0.0001 or target_top20_hit_delta < 0)
        ):
            return "collapse_isomorphic"
        if (
            not continuation_changed
            and abs(target_mass_delta) <= 0.00001
            and target_top20_hit_delta == 0
            and focus_rank_delta < 4
            and rank_focus_delta < 4
            and abs(semantic_delta) <= 0.01
            and recall_delta <= 0.0
            and span_delta <= 0.0
        ):
            return "dead_actuator"
        return "neutral"

    def _candidate_fingerprint(
        self,
        candidate_edits: Sequence[Mapping[str, Any]],
        *,
        intended_bundle_key: str | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        if not candidate_edits:
            return {}
        primary = dict(candidate_edits[0])
        op = primary.get("op") if isinstance(primary.get("op"), Mapping) else {}
        budget = primary.get("budget") if isinstance(primary.get("budget"), Mapping) else {}
        source_span = primary.get("source_span") if isinstance(primary.get("source_span"), Mapping) else None
        source_start = int(primary.get("source_position", 0) or 0)
        if source_span is not None:
            source_start = int(source_span.get("start", source_start) or source_start)
        recipe_name = None
        if label:
            recipe_name = str(label).split(":", 1)[0] or None
        return {
            "bundle_key": str(primary.get("bundle_key", "") or "") or None,
            "objective_bundle_key": str(intended_bundle_key or "") or None,
            "actuator_bundle_key": str(primary.get("bundle_key", "") or "") or None,
            "term": str(primary.get("focus_term", "") or primary.get("focus_feature", "") or "") or None,
            "recipe_name": recipe_name,
            "site": str(primary.get("site", "") or "") or None,
            "layer": None if primary.get("layer") is None else int(primary.get("layer", 0) or 0),
            "head": None if primary.get("head") is None else int(primary.get("head", 0) or 0),
            "source_token_index": int(source_start),
            "source_piece": primary.get("source_piece"),
            "source_segment_kind": primary.get("source_segment_kind"),
            "op_kind": str(primary.get("kind", "") or op.get("kind", "") or "") or None,
            "which": str(op.get("which", "") or "") or None,
            "alpha": None if op.get("alpha") is None else float(op.get("alpha", 0.0) or 0.0),
            "ttl_steps": None if budget.get("ttl_steps") is None else int(budget.get("ttl_steps", 0) or 0),
            "norm_clip": None if budget.get("norm_clip") is None else float(budget.get("norm_clip", 0.0) or 0.0),
            "edit_count": len(candidate_edits),
        }

    def _eval_context_fingerprint(
        self,
        *,
        max_new_tokens: int,
        top_k: int,
        max_edits_per_step: int,
        focus_terms: Sequence[str],
    ) -> dict[str, Any]:
        prompt_hash = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        return {
            "prompt_hash": f"sha256:{prompt_hash}",
            "tokenizer": type(self.codec).__name__,
            "decode_step": int(self._steps),
            "answer_prefix": self.final_text(),
            "max_new_tokens": int(max_new_tokens),
            "top_k": int(top_k),
            "active_patch_count": len(self._collect_active_edits()),
            "max_edits_per_step": int(max_edits_per_step),
            "score_terms": [str(term) for term in focus_terms if str(term)],
        }

    def _replay_policy(self, *, max_edits_per_step_override: int | None = None) -> HarnessPolicy | None:
        if max_edits_per_step_override is None:
            return None
        base_policy = HarnessPolicy.default_v0()
        global_budget = replace(
            base_policy.global_budget,
            max_edits_per_step=max(1, int(max_edits_per_step_override)),
        )
        return replace(base_policy, global_budget=global_budget)

    def _summarize_operator_certifications(
        self,
        evaluations: Sequence[Mapping[str, Any]],
        *,
        key_field: str = "operator_family_key",
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for item in evaluations:
            if not isinstance(item, Mapping):
                continue
            certification_key = str(item.get(key_field, "") or "")
            if not certification_key:
                continue
            grouped.setdefault(certification_key, []).append(item)
        summaries: list[dict[str, Any]] = []
        for certification_key, items in grouped.items():
            class_counts: dict[str, int] = {}
            best_target_mass_delta = 0.0
            best_target_top20_hit_delta = 0
            best_readout_probe_score = 0.0
            family_labels: set[str] = set()
            for item in items:
                cls = str(
                    item.get("actual_delta_class")
                    or ("error" if str(item.get("status", "") or "") == "error" else "neutral")
                )
                class_counts[cls] = int(class_counts.get(cls, 0) or 0) + 1
                best_target_mass_delta = max(best_target_mass_delta, float(item.get("target_mass_delta", 0.0) or 0.0))
                best_target_top20_hit_delta = max(
                    best_target_top20_hit_delta,
                    int(item.get("target_top20_hit_delta", 0) or 0),
                )
                probe_summary = item.get("probe_summary") if isinstance(item.get("probe_summary"), Mapping) else {}
                best_readout_probe_score = max(
                    best_readout_probe_score,
                    float(probe_summary.get("readout_score", 0.0) or 0.0),
                )
                if item.get("label") not in (None, ""):
                    family_labels.add(str(item.get("label")))
            target_lift_count = int(class_counts.get("target_lift", 0) or 0)
            harmful_count = int(class_counts.get("harmful", 0) or 0)
            collapse_sharpener_count = int(class_counts.get("collapse_sharpener", 0) or 0)
            collapse_count = int(class_counts.get("collapse_isomorphic", 0) or 0)
            dead_count = int(class_counts.get("dead_actuator", 0) or 0)
            neutral_count = int(class_counts.get("neutral", 0) or 0)
            error_count = int(class_counts.get("error", 0) or 0)
            if harmful_count > 0 or collapse_sharpener_count > 0:
                status = "veto"
                certified_for_apply = False
                reason = "collapse_sharpener_seen" if collapse_sharpener_count > 0 else "harmful_actual_delta_seen"
            elif target_lift_count > 0:
                status = "apply_eligible"
                certified_for_apply = True
                reason = "target_lift_observed"
            elif error_count > 0:
                status = "shadow_only"
                certified_for_apply = False
                reason = "replay_error"
            else:
                status = "shadow_only"
                certified_for_apply = False
                reason = "no_target_lift_yet"
            family_prior_score = (
                (1.0 * target_lift_count)
                + (0.1 * neutral_count)
                - (0.1 * error_count)
                - (0.25 * dead_count)
                - (0.5 * collapse_count)
                - (0.75 * collapse_sharpener_count)
                - (1.0 * harmful_count)
            ) / max(len(items), 1)
            summaries.append(
                {
                    key_field: certification_key,
                    "certification_key": certification_key,
                    "certified_for_apply": bool(certified_for_apply),
                    "certification_status": status,
                    "certification_reason": reason,
                    "family_prior_score": round(float(family_prior_score), 6),
                    "evaluation_count": len(items),
                    "actual_delta_class_counts": dict(sorted(class_counts.items())),
                    "best_target_mass_delta": round(float(best_target_mass_delta), 6),
                    "best_target_top20_hit_delta": int(best_target_top20_hit_delta),
                    "best_readout_probe_score": round(float(best_readout_probe_score), 6),
                    "labels": sorted(family_labels),
                }
            )
        summaries.sort(
            key=lambda item: (
                0 if bool(item.get("certified_for_apply", False)) else 1,
                -float(item.get("family_prior_score", 0.0) or 0.0),
                str(item.get("certification_key", "") or ""),
            )
        )
        return summaries

    def _summarize_operator_recipe_bundle_ownership(
        self,
        evaluations: Sequence[Mapping[str, Any]],
        *,
        bundle_term_by_key: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for item in evaluations:
            if not isinstance(item, Mapping):
                continue
            recipe_id = str(item.get("operator_recipe_id", "") or "")
            recipe_seed_key = str(item.get("operator_recipe_seed_key", "") or recipe_id)
            intended_bundle_key = str(item.get("intended_bundle_key", "") or "")
            if not recipe_id or not intended_bundle_key:
                continue
            grouped.setdefault((recipe_id, intended_bundle_key), []).append(item)

        summaries: list[dict[str, Any]] = []
        tau_self = 0.03
        tau_positive = 0.005
        tau_align = 0.01
        tau_bridge = 0.03
        tau_dead = 0.01
        for (recipe_id, intended_bundle_key), items in grouped.items():
            intended_term = str(bundle_term_by_key.get(intended_bundle_key, "") or "")
            best_summary: dict[str, Any] | None = None
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                term_deltas = item.get("term_readout_deltas") if isinstance(item.get("term_readout_deltas"), Mapping) else {}
                bundle_scores: dict[str, float] = {}
                for bundle_key, bundle_term in bundle_term_by_key.items():
                    metrics = term_deltas.get(str(bundle_term)) if isinstance(term_deltas, Mapping) else None
                    score = 0.0
                    if isinstance(metrics, Mapping):
                        score = float(metrics.get("lift_score", 0.0) or 0.0)
                    bundle_scores[str(bundle_key)] = float(score)
                realized_lift_bundle_key = max(
                    bundle_scores.items(),
                    key=lambda entry: (float(entry[1]), str(entry[0])),
                )[0] if bundle_scores else intended_bundle_key
                self_delta = float(bundle_scores.get(intended_bundle_key, 0.0) or 0.0)
                cross_candidates = [
                    float(score)
                    for bundle_key, score in bundle_scores.items()
                    if str(bundle_key) != intended_bundle_key
                ]
                cross_delta = max(cross_candidates, default=0.0)
                alignment_margin = float(self_delta - cross_delta)
                best_eval_actual_delta_class = str(item.get("actual_delta_class", "") or "")
                if best_eval_actual_delta_class == "collapse_sharpener":
                    actuator_class = "collapse_sharpener"
                elif self_delta > tau_self and alignment_margin > tau_align:
                    actuator_class = "self_actuator"
                elif self_delta > tau_positive and alignment_margin < -tau_align and str(realized_lift_bundle_key) != intended_bundle_key:
                    actuator_class = "cross_bound"
                elif cross_delta > tau_bridge and str(realized_lift_bundle_key) != intended_bundle_key:
                    actuator_class = "bridge_actuator"
                elif max(self_delta, cross_delta) <= tau_dead:
                    actuator_class = "dead_actuator"
                else:
                    actuator_class = "noisy_or_harmful"
                candidate_summary = {
                    "operator_recipe_id": recipe_id,
                    "operator_recipe_seed_key": recipe_seed_key,
                    "contrast_mode": str(item.get("contrast_mode", "") or "none"),
                    "intended_bundle_key": intended_bundle_key,
                    "intended_term": intended_term or None,
                    "realized_lift_bundle_key": None if not bundle_scores else str(realized_lift_bundle_key),
                    "realized_lift_term": None
                    if not bundle_scores
                    else str(bundle_term_by_key.get(str(realized_lift_bundle_key), "") or ""),
                    "self_delta": round(float(self_delta), 6),
                    "cross_delta": round(float(cross_delta), 6),
                    "alignment_margin": round(float(alignment_margin), 6),
                    "actuator_class": actuator_class,
                    "bridge_plan_bundle_key": str(realized_lift_bundle_key)
                    if actuator_class in {"bridge_actuator", "cross_bound"} and str(realized_lift_bundle_key) != intended_bundle_key
                    else None,
                    "best_eval_label": item.get("label"),
                    "best_eval_status": item.get("status"),
                    "best_eval_actual_delta_class": best_eval_actual_delta_class,
                    "best_eval_entropy_delta": round(float(item.get("entropy_delta", 0.0) or 0.0), 6),
                    "best_eval_top1_margin_delta": round(float(item.get("top1_margin_delta", 0.0) or 0.0), 6),
                    "best_eval_repeat_flag_delta": int(item.get("repeat_flag_delta", 0) or 0),
                    "best_eval_repetition_score_delta": round(float(item.get("repetition_score_delta", 0.0) or 0.0), 6),
                    "best_eval_required_term_recall_delta": round(float(item.get("required_term_recall_delta", 0.0) or 0.0), 6),
                    "best_eval_required_term_span_progress_delta": round(float(item.get("required_term_span_progress_delta", 0.0) or 0.0), 6),
                    "best_eval_target_mass_delta": round(float(item.get("target_mass_delta", 0.0) or 0.0), 6),
                    "best_eval_target_top20_hit_delta": int(item.get("target_top20_hit_delta", 0) or 0),
                    "best_eval_candidate_fingerprint": dict(item.get("candidate_fingerprint", {}))
                    if isinstance(item.get("candidate_fingerprint"), Mapping)
                    else {},
                    "best_eval_context_fingerprint": dict(item.get("eval_context_fingerprint", {}))
                    if isinstance(item.get("eval_context_fingerprint"), Mapping)
                    else {},
                    "bundle_lift_scores": {
                        str(bundle_key): round(float(score), 6)
                        for bundle_key, score in sorted(bundle_scores.items(), key=lambda entry: str(entry[0]))
                    },
                }
                if best_summary is None or (
                    float(candidate_summary["alignment_margin"]),
                    float(candidate_summary["self_delta"]),
                    -float(candidate_summary["cross_delta"]),
                    str(candidate_summary["best_eval_label"] or ""),
                ) > (
                    float(best_summary["alignment_margin"]),
                    float(best_summary["self_delta"]),
                    -float(best_summary["cross_delta"]),
                    str(best_summary["best_eval_label"] or ""),
                ):
                    best_summary = candidate_summary
            if best_summary is not None:
                summaries.append(best_summary)

        summaries.sort(
            key=lambda item: (
                str(item.get("operator_recipe_id", "") or ""),
                str(item.get("intended_bundle_key", "") or ""),
            )
        )
        return summaries

    def _infer_recipe_stealer_bundles(
        self,
        ownership: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, str], str]:
        stealer_map: dict[tuple[str, str], str] = {}
        fallback_by_bundle: dict[str, tuple[float, str]] = {}
        for item in ownership:
            if not isinstance(item, Mapping):
                continue
            recipe_id = str(item.get("operator_recipe_seed_key", "") or item.get("operator_recipe_id", "") or "")
            intended_bundle_key = str(item.get("intended_bundle_key", "") or "")
            realized_bundle_key = str(item.get("realized_lift_bundle_key", "") or "")
            actuator_class = str(item.get("actuator_class", "") or "")
            cross_delta = float(item.get("cross_delta", 0.0) or 0.0)
            if not recipe_id or not intended_bundle_key or not realized_bundle_key:
                continue
            if actuator_class not in {"bridge_actuator", "self_actuator", "cross_bound", "noisy_or_harmful"}:
                continue
            if realized_bundle_key == intended_bundle_key:
                continue
            stealer_map[(recipe_id, intended_bundle_key)] = realized_bundle_key
            previous = fallback_by_bundle.get(intended_bundle_key)
            if previous is None or float(cross_delta) > float(previous[0]):
                fallback_by_bundle[intended_bundle_key] = (float(cross_delta), realized_bundle_key)
        for intended_bundle_key, (_cross_delta, realized_bundle_key) in fallback_by_bundle.items():
            stealer_map[("*", intended_bundle_key)] = realized_bundle_key
        return stealer_map

    def _summarize_bridge_plan_recommendations(
        self,
        ownership: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        bundle_terms: dict[str, str] = {}
        for item in ownership:
            if not isinstance(item, Mapping):
                continue
            intended_bundle_key = str(item.get("intended_bundle_key", "") or "")
            if intended_bundle_key:
                intended_term = str(item.get("intended_term", "") or "")
                if intended_term:
                    bundle_terms.setdefault(intended_bundle_key, intended_term)
            realized_bundle_key = str(item.get("realized_lift_bundle_key", "") or "")
            if realized_bundle_key:
                realized_term = str(item.get("realized_lift_term", "") or "")
                if realized_term:
                    bundle_terms.setdefault(realized_bundle_key, realized_term)

        grouped: dict[str, dict[str, Any]] = {}
        for item in ownership:
            if not isinstance(item, Mapping):
                continue
            intended_bundle_key = str(item.get("intended_bundle_key", "") or "")
            actuator_class = str(item.get("actuator_class", "") or "")
            if intended_bundle_key:
                bucket = grouped.setdefault(intended_bundle_key, {"has_self": False, "candidates": []})
                if actuator_class == "self_actuator":
                    bucket["has_self"] = True
            realized_bundle_key = str(item.get("realized_lift_bundle_key", "") or "")
            if (
                actuator_class not in {"bridge_actuator", "cross_bound"}
                or not intended_bundle_key
                or not realized_bundle_key
                or realized_bundle_key == intended_bundle_key
            ):
                continue
            objective_bucket = grouped.setdefault(realized_bundle_key, {"has_self": False, "candidates": []})
            objective_lift_delta = float(
                (
                    item.get("bundle_lift_scores", {}).get(realized_bundle_key, 0.0)
                    if isinstance(item.get("bundle_lift_scores"), Mapping)
                    else 0.0
                )
                or 0.0
            )
            candidate = dict(item)
            candidate["bridge_objective_bundle_key"] = realized_bundle_key
            candidate["bridge_objective_term"] = bundle_terms.get(realized_bundle_key, str(item.get("realized_lift_term", "") or ""))
            candidate["bridge_objective_delta"] = objective_lift_delta
            candidate["bridge_alignment_margin"] = round(
                float(objective_lift_delta - float(item.get("self_delta", 0.0) or 0.0)),
                6,
            )
            objective_bucket["candidates"].append(candidate)

        recommendations: list[dict[str, Any]] = []
        for objective_bundle_key, bucket in grouped.items():
            if bool(bucket.get("has_self", False)):
                continue
            candidates = [item for item in bucket.get("candidates", []) if isinstance(item, Mapping)]
            if not candidates:
                continue
            best = max(
                candidates,
                key=lambda item: (
                    1 if str(item.get("actuator_class", "") or "") == "cross_bound" else 0,
                    float(item.get("bridge_objective_delta", 0.0) or 0.0),
                    float(item.get("bridge_alignment_margin", -10.0) or -10.0),
                    float(item.get("self_delta", 0.0) or 0.0),
                    -float(item.get("cross_delta", 0.0) or 0.0),
                    str(item.get("operator_recipe_id", "") or ""),
                ),
            )
            actuator_bundle_key = str(best.get("intended_bundle_key", "") or "")
            recommendation = {
                "objective_bundle_key": objective_bundle_key,
                "objective_term": str(best.get("bridge_objective_term", "") or bundle_terms.get(objective_bundle_key, "")),
                "actuator_bundle_key": actuator_bundle_key,
                "actuator_term": str(best.get("intended_term", "") or bundle_terms.get(actuator_bundle_key, "")),
                "operator_recipe_id": str(best.get("operator_recipe_id", "") or ""),
                "operator_recipe_seed_key": str(best.get("operator_recipe_seed_key", "") or ""),
                "actuator_class": str(best.get("actuator_class", "") or ""),
                "objective_lift_delta": round(float(best.get("bridge_objective_delta", 0.0) or 0.0), 6),
                "self_delta": round(float(best.get("self_delta", 0.0) or 0.0), 6),
                "cross_delta": round(float(best.get("cross_delta", 0.0) or 0.0), 6),
                "alignment_margin": round(float(best.get("alignment_margin", 0.0) or 0.0), 6),
                "bridge_alignment_margin": round(float(best.get("bridge_alignment_margin", 0.0) or 0.0), 6),
                "bridge_required": True,
                "bridge_plan_reason": (
                    "no_certified_self_actuator_target_lift_owned_by_other_bundle"
                    if str(best.get("actuator_class", "") or "") == "cross_bound"
                    else "no_certified_self_actuator_bridge_bundle_available"
                ),
            }
            recommendations.append(recommendation)

        recommendations.sort(
            key=lambda item: (
                str(item.get("objective_bundle_key", "") or ""),
                -float(item.get("objective_lift_delta", 0.0) or 0.0),
                -float(item.get("bridge_alignment_margin", 0.0) or 0.0),
                -float(item.get("self_delta", 0.0) or 0.0),
            )
        )
        return recommendations

    def _attach_operator_certifications(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        attached: list[dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            candidate = dict(item)
            family_key = str(candidate.get("operator_family_key", "") or self._operator_family_key(candidate))
            candidate["operator_family_key"] = family_key
            certification = self._operator_certification_table.get(family_key)
            if certification is not None:
                candidate["operator_certification"] = dict(certification)
                candidate["operator_certified_for_apply"] = bool(certification.get("certified_for_apply", False))
                candidate["operator_certification_status"] = str(
                    certification.get("certification_status", "shadow_only") or "shadow_only"
                )
            attached.append(candidate)
        return attached

    def latest_readout_sidecar_capture(self) -> ReadoutSidecarCapture | None:
        return self._latest_readout_sidecar_capture

    def _bridge_plan_report_for_bundle(
        self,
        bundle_key: str,
        *,
        objective_term: str | None = None,
    ) -> dict[str, Any] | None:
        normalized = str(bundle_key or "")
        if normalized:
            report = self._operator_bridge_plan_table.get(normalized)
            if isinstance(report, Mapping):
                return dict(report)
        normalized_term = str(objective_term or "").strip().lower()
        if not normalized_term:
            return None
        matching = [
            dict(value)
            for value in self._operator_bridge_plan_table.values()
            if isinstance(value, Mapping) and str(value.get("objective_term", "") or "").strip().lower() == normalized_term
        ]
        if len(matching) == 1:
            return matching[0]
        return None

    def _candidate_sidecar_key(self, candidate: Mapping[str, Any]) -> str:
        bundle_key = str(candidate.get("bundle_key", "") or "")
        if bundle_key:
            return bundle_key
        focus_term = str(candidate.get("focus_feature", "") or "")
        provenance_class = str(candidate.get("provenance_class", "misc_prompt") or "misc_prompt")
        span_id = self._candidate_span_id(candidate)
        family_kind = self._candidate_family_kind(candidate)
        span_kind = str(candidate.get("span_kind", "") or "")
        return f"{focus_term}|{provenance_class}|{span_id}|{family_kind}|{span_kind}"

    def _candidate_sidecar_vetoed(self, candidate: Mapping[str, Any]) -> bool:
        hints = self._latest_readout_sidecar_hints if isinstance(self._latest_readout_sidecar_hints, Mapping) else {}
        key_vetoes = hints.get("candidate_key_vetoes")
        if isinstance(key_vetoes, SequenceABC) and not isinstance(key_vetoes, (str, bytes, bytearray)):
            if self._candidate_sidecar_key(candidate) in {str(item) for item in key_vetoes}:
                return True
        family_vetoes = hints.get("candidate_family_vetoes")
        if isinstance(family_vetoes, SequenceABC) and not isinstance(family_vetoes, (str, bytes, bytearray)):
            family_names = {str(item) for item in family_vetoes}
            candidate_family = str(candidate.get("candidate_family", "") or "")
            if candidate_family and candidate_family in family_names:
                return True
            if self._candidate_family_kind(candidate) in family_names:
                return True
        return False

    def _candidate_sidecar_bonus(self, candidate: Mapping[str, Any]) -> float:
        hints = self._latest_readout_sidecar_hints if isinstance(self._latest_readout_sidecar_hints, Mapping) else {}
        bonus = 0.0
        if self._candidate_sidecar_vetoed(candidate):
            return -1.2
        focus_term = str(candidate.get("focus_feature", "") or "")
        term_strengths = hints.get("term_anchor_strength_by_term")
        if isinstance(term_strengths, Mapping) and focus_term:
            try:
                term_strength = float(term_strengths.get(focus_term, 0.0) or 0.0)
            except Exception:
                term_strength = 0.0
            bonus += max(-0.25, min(0.25, 0.2 * term_strength))
        support_terms = hints.get("candidate_support_terms")
        if isinstance(support_terms, Mapping) and focus_term:
            try:
                term_support = float(support_terms.get(focus_term, 0.0) or 0.0)
            except Exception:
                term_support = 0.0
            bonus += max(-0.2, min(0.2, 0.15 * term_support))
        support_scores = hints.get("candidate_support_scores")
        if isinstance(support_scores, Mapping):
            score_value = support_scores.get(self._candidate_sidecar_key(candidate))
            if score_value is None and candidate.get("bundle_key") is not None:
                score_value = support_scores.get(str(candidate.get("bundle_key", "") or ""))
            if score_value is not None:
                try:
                    support_score = float(score_value)
                except Exception:
                    support_score = 0.0
                bonus += max(-0.45, min(0.45, 0.3 * support_score))
        return bonus

    def _candidate_builder_score(
        self,
        candidate: Mapping[str, Any],
        *,
        include_sidecar: bool = True,
    ) -> float:
        provenance_class = str(candidate.get("provenance_class", "misc_prompt") or "misc_prompt")
        family_kind = self._candidate_family_kind(candidate)
        span_kind = str(candidate.get("span_kind", "") or "")
        alignment = float(candidate.get("alignment", 0.0) or 0.0)
        recent_probe = candidate.get("recent_probe") if isinstance(candidate.get("recent_probe"), Mapping) else {}
        probe_bonus = 0.0
        label = str(recent_probe.get("label", "") or "")
        if label == "actionable_positive":
            probe_bonus = 0.4
        elif label == "weak_positive_subthreshold":
            probe_bonus = 0.15
        elif label == "dead_actuator":
            probe_bonus = -0.6
        phase_objective = str(candidate.get("phase_objective", "") or "")
        target_term_priority = candidate.get("target_term_priority")
        term_priority_bonus = 0.0
        if phase_objective == "readout_escape" and isinstance(target_term_priority, int) and not isinstance(target_term_priority, bool):
            bounded_priority = max(0, min(int(target_term_priority), 3))
            term_priority_bonus = 0.18 * float(3 - bounded_priority)
        score = (
            (4.0 * self._provenance_weight(provenance_class))
            + self._family_weight(family_kind)
            + self._span_weight(span_kind)
            + probe_bonus
            + term_priority_bonus
            + (0.5 * alignment)
        )
        if include_sidecar:
            score += self._candidate_sidecar_bonus(candidate)
        return score

    def _candidate_sidecar_support_score(self, candidate: Mapping[str, Any]) -> float:
        hints = self._latest_readout_sidecar_hints if isinstance(self._latest_readout_sidecar_hints, Mapping) else {}
        support_scores = hints.get("candidate_support_scores")
        if not isinstance(support_scores, Mapping):
            return 0.0
        keys = [self._candidate_sidecar_key(candidate)]
        bundle_key = str(candidate.get("bundle_key", "") or "")
        if bundle_key:
            keys.insert(0, bundle_key)
        for key in keys:
            try:
                value = float(support_scores.get(key, 0.0) or 0.0)
            except Exception:
                value = 0.0
            if value != 0.0:
                return value
        return 0.0

    def _bundle_sidecar_support_score(self, bundle_key: str) -> float:
        if not bundle_key:
            return 0.0
        hints = self._latest_readout_sidecar_hints if isinstance(self._latest_readout_sidecar_hints, Mapping) else {}
        bundle_scores = hints.get("bundle_support_scores")
        if isinstance(bundle_scores, Mapping):
            try:
                bundle_value = float(bundle_scores.get(bundle_key, 0.0) or 0.0)
            except Exception:
                bundle_value = 0.0
            if bundle_value != 0.0:
                return bundle_value
        support_scores = hints.get("candidate_support_scores")
        if isinstance(support_scores, Mapping):
            try:
                support_value = float(support_scores.get(bundle_key, 0.0) or 0.0)
            except Exception:
                support_value = 0.0
            if support_value != 0.0:
                return support_value
        return 0.0

    def _bundle_group_key(self, candidate: Mapping[str, Any]) -> str:
        bundle_key = str(candidate.get("bundle_key", "") or "")
        if bundle_key:
            return bundle_key
        return self._candidate_sidecar_key(candidate)

    def _bundle_groups(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        groups: dict[str, dict[str, Any]] = {}
        ordered_keys: list[str] = []
        for raw_candidate in candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = dict(raw_candidate)
            group_key = self._bundle_group_key(candidate)
            if group_key not in groups:
                ordered_keys.append(group_key)
                groups[group_key] = {
                    "group_key": group_key,
                    "bundle_key": str(candidate.get("bundle_key", "") or "") or None,
                    "bundle_ready": bool(candidate.get("bundle_ready", False)),
                    "focus_term": str(candidate.get("focus_feature", "") or ""),
                    "provenance_class": str(candidate.get("provenance_class", "misc_prompt") or "misc_prompt"),
                    "span_kind": str(candidate.get("span_kind", "") or ""),
                    "members": [],
                }
            groups[group_key]["members"].append(candidate)
        return [groups[key] for key in ordered_keys]

    def _bundle_base_score(self, members: Sequence[Mapping[str, Any]]) -> float:
        member_scores = sorted(
            (
                self._candidate_builder_score(item, include_sidecar=False)
                for item in members
                if isinstance(item, Mapping)
            ),
            reverse=True,
        )
        if not member_scores:
            return -10.0
        top = float(member_scores[0])
        second = float(member_scores[1]) if len(member_scores) > 1 else 0.0
        return top + (0.4 * second)

    def _bundle_sidecar_support(self, bundle: Mapping[str, Any]) -> float:
        members = bundle.get("members")
        if not isinstance(members, SequenceABC) or isinstance(members, (str, bytes, bytearray)):
            return 0.0
        support_values: list[float] = []
        bundle_key = str(bundle.get("bundle_key", "") or "")
        if bundle_key:
            bundle_score = self._bundle_sidecar_support_score(bundle_key)
            if bundle_score != 0.0:
                support_values.append(bundle_score)
        for member in members:
            if not isinstance(member, Mapping):
                continue
            support_value = self._candidate_sidecar_support_score(member)
            if support_value != 0.0:
                support_values.append(float(support_value))
        if not support_values:
            return 0.0
        support_values = sorted(set(round(float(value), 6) for value in support_values), reverse=True)
        top = float(support_values[0])
        second = float(support_values[1]) if len(support_values) > 1 else 0.0
        return top + (0.4 * max(0.0, second))

    def _bundle_sidecar_support_margin(self, bundle: Mapping[str, Any]) -> float:
        focus_term = str(bundle.get("focus_term", "") or "")
        if not focus_term:
            return 0.0
        hints = self._latest_readout_sidecar_hints if isinstance(self._latest_readout_sidecar_hints, Mapping) else {}
        support_terms = hints.get("candidate_support_terms")
        if not isinstance(support_terms, Mapping):
            return 0.0
        try:
            focus_score = float(support_terms.get(focus_term, 0.0) or 0.0)
        except Exception:
            focus_score = 0.0
        other_scores: list[float] = []
        for raw_term, raw_value in support_terms.items():
            if str(raw_term) == focus_term:
                continue
            try:
                other_scores.append(float(raw_value or 0.0))
            except Exception:
                continue
        other_max = max(other_scores, default=0.0)
        return focus_score - other_max

    def _bundle_sidecar_evidence(self, bundle: Mapping[str, Any]) -> dict[str, Any]:
        hints = self._latest_readout_sidecar_hints if isinstance(self._latest_readout_sidecar_hints, Mapping) else {}
        bundle_key = str(bundle.get("bundle_key", "") or bundle.get("group_key", "") or "")
        evidence_vectors = hints.get("bundle_evidence_vectors")
        raw_vector = evidence_vectors.get(bundle_key) if isinstance(evidence_vectors, Mapping) else None
        vector = raw_vector if isinstance(raw_vector, Mapping) else {}
        focus_term = str(bundle.get("focus_term", "") or "")
        term_strengths = hints.get("term_anchor_strength_by_term")
        fallback_anchor = 0.0
        if isinstance(term_strengths, Mapping) and focus_term:
            try:
                fallback_anchor = float(term_strengths.get(focus_term, 0.0) or 0.0)
            except Exception:
                fallback_anchor = 0.0
        provenance_class = str(vector.get("provenance_class", bundle.get("provenance_class", "misc_prompt")) or "misc_prompt")
        span_kind = str(vector.get("span_kind", "") or "")
        if provenance_class == "source_body":
            fallback_tier = 1.0
        elif provenance_class == "answer_prefix":
            fallback_tier = 0.4
        elif provenance_class == "constraint_header":
            fallback_tier = 0.2
        else:
            fallback_tier = 0.0
        if span_kind.startswith("exact_prompt_span"):
            fallback_span_precision = 1.0
        elif span_kind.startswith("exact_prompt"):
            fallback_span_precision = 0.82
        else:
            fallback_span_precision = 0.45 if provenance_class == "source_body" else 0.2
        semantic_support = self._bundle_sidecar_support(bundle)
        try:
            semantic_support = float(vector.get("semantic_residual_support", semantic_support) or semantic_support)
        except Exception:
            semantic_support = float(semantic_support)
        try:
            anchor_strength = float(vector.get("anchor_strength", fallback_anchor) or fallback_anchor)
        except Exception:
            anchor_strength = float(fallback_anchor)
        try:
            span_precision = float(vector.get("span_precision", fallback_span_precision) or fallback_span_precision)
        except Exception:
            span_precision = float(fallback_span_precision)
        try:
            provenance_tier_hint = float(vector.get("provenance_tier_hint", fallback_tier) or fallback_tier)
        except Exception:
            provenance_tier_hint = float(fallback_tier)
        source_body_exact = bool(
            vector.get(
                "source_body_exact",
                provenance_class == "source_body" and span_kind.startswith("exact_prompt"),
            )
        )
        return {
            "semantic_residual_support": round(float(semantic_support), 6),
            "anchor_strength": round(float(anchor_strength), 6),
            "span_precision": round(float(span_precision), 6),
            "provenance_tier_hint": round(float(provenance_tier_hint), 6),
            "source_body_exact": bool(source_body_exact),
            "provenance_class": provenance_class,
            "span_kind": span_kind,
        }

    def _bundle_first_piece_reachability(
        self,
        bundle: Mapping[str, Any],
        *,
        answer_readout_canary: Mapping[str, Any] | None,
    ) -> float:
        if not isinstance(answer_readout_canary, Mapping):
            return 0.0
        focus_term = str(bundle.get("focus_term", "") or "")
        reachable_term = str(answer_readout_canary.get("reachable_focus_term", "") or "")
        reachable_piece = self._canonical_effect_family_piece(answer_readout_canary.get("reachable_focus_piece", ""))
        try:
            reachable_rank = int(answer_readout_canary.get("reachable_focus_rank", 10**9) or 10**9)
        except Exception:
            reachable_rank = 10**9
        try:
            target_mass = float(answer_readout_canary.get("target_mass", 0.0) or 0.0)
        except Exception:
            target_mass = 0.0
        target_top20_hits = int(answer_readout_canary.get("target_top20_hits", 0) or 0)
        rank_score = max(0.0, min(1.0, (1024.0 - min(float(reachable_rank), 1024.0)) / 1024.0))
        mass_score = max(0.0, min(1.0, target_mass / 0.001))
        hit_score = 1.0 if target_top20_hits > 0 else 0.0
        piece_match = False
        members = bundle.get("members")
        if isinstance(members, SequenceABC) and not isinstance(members, (str, bytes, bytearray)) and reachable_piece:
            for member in members:
                if not isinstance(member, Mapping):
                    continue
                source_piece = self._canonical_effect_family_piece(member.get("source_piece", ""))
                if source_piece and source_piece == reachable_piece:
                    piece_match = True
                    break
        score = 0.0
        if focus_term and focus_term == reachable_term:
            score += (0.55 * rank_score) + (0.25 * mass_score) + (0.2 * hit_score)
        if piece_match:
            score += 0.15
        return round(min(1.25, score), 6)

    def _bundle_probe_bonus(self, bundle: Mapping[str, Any]) -> float:
        members = bundle.get("members")
        if not isinstance(members, SequenceABC) or isinstance(members, (str, bytes, bytearray)):
            return 0.0
        best = 0.0
        for member in members:
            if not isinstance(member, Mapping):
                continue
            recent_probe = member.get("recent_probe") if isinstance(member.get("recent_probe"), Mapping) else {}
            label = str(recent_probe.get("label", "") or "")
            if label == "actionable_positive":
                best = max(best, 1.0)
            elif label == "weak_positive_subthreshold":
                best = max(best, 0.6)
            elif label == "positive":
                best = max(best, 0.8)
        return best

    def _bundle_probe_axis_scores(self, bundle: Mapping[str, Any]) -> dict[str, float]:
        members = bundle.get("members")
        if not isinstance(members, SequenceABC) or isinstance(members, (str, bytes, bytearray)):
            return {
                "readout_score": 0.0,
                "constraint_score": 0.0,
                "trajectory_score": 0.0,
            }
        readout_score = 0.0
        constraint_score = 0.0
        trajectory_score = 0.0
        for member in members:
            if not isinstance(member, Mapping):
                continue
            recent_probe = member.get("recent_probe") if isinstance(member.get("recent_probe"), Mapping) else {}
            probe_summary = recent_probe.get("probe_summary") if isinstance(recent_probe.get("probe_summary"), Mapping) else {}
            try:
                readout_score = max(
                    readout_score,
                    float(
                        recent_probe.get(
                            "readout_score",
                            probe_summary.get("readout_score", 0.0),
                        )
                        or 0.0
                    ),
                )
            except Exception:
                pass
            try:
                constraint_score = max(
                    constraint_score,
                    float(
                        recent_probe.get(
                            "constraint_score",
                            probe_summary.get("constraint_score", 0.0),
                        )
                        or 0.0
                    ),
                )
            except Exception:
                pass
            try:
                trajectory_score = max(
                    trajectory_score,
                    float(
                        recent_probe.get(
                            "trajectory_score",
                            probe_summary.get("trajectory_score", 0.0),
                        )
                        or 0.0
                    ),
                )
            except Exception:
                pass
        return {
            "readout_score": round(float(readout_score), 6),
            "constraint_score": round(float(constraint_score), 6),
            "trajectory_score": round(float(trajectory_score), 6),
        }

    def _bundle_harmful_family_penalty(self, bundle: Mapping[str, Any]) -> float:
        members = bundle.get("members")
        if not isinstance(members, SequenceABC) or isinstance(members, (str, bytes, bytearray)):
            return 0.0
        penalty = 0.0
        recent_vetoes = self._recent_effect_veto_index()
        bundle_key = str(bundle.get("bundle_key", "") or "")
        for member in members:
            if not isinstance(member, Mapping):
                continue
            recent_probe = member.get("recent_probe") if isinstance(member.get("recent_probe"), Mapping) else {}
            label = str(recent_probe.get("label", "") or "")
            if label == "harmful":
                penalty = max(penalty, 1.0)
            elif label == "dead_actuator":
                penalty = max(penalty, 0.7)
            if bool(member.get("effect_family_collapsed_member", False)):
                penalty = max(penalty, 0.35)
            family_key = str(member.get("candidate_family", "") or "")
            recipe_id = str(member.get("operator_recipe_id", "") or "")
            member_bundle_key = str(member.get("bundle_key", "") or bundle_key)
            if (
                family_key in recent_vetoes["collapse_families"]
                or recipe_id in recent_vetoes["collapse_recipe_ids"]
                or member_bundle_key in recent_vetoes["collapse_bundle_keys"]
            ):
                penalty = max(penalty, 1.2)
            elif (
                family_key in recent_vetoes["harmful_families"]
                or recipe_id in recent_vetoes["harmful_recipe_ids"]
                or member_bundle_key in recent_vetoes["harmful_bundle_keys"]
            ):
                penalty = max(penalty, 1.0)
        return penalty

    def _recent_effect_veto_index(self) -> dict[str, set[str]]:
        summary = self._recent_effect_summary if isinstance(self._recent_effect_summary, Mapping) else {}
        return {
            "harmful_families": {
                str(item)
                for item in summary.get("recent_harmful_surface_family_keys", ())
                if str(item)
            },
            "collapse_families": {
                str(item)
                for item in summary.get("recent_collapse_sharpener_surface_family_keys", ())
                if str(item)
            },
            "harmful_recipe_ids": {
                str(item)
                for item in summary.get("recent_harmful_operator_recipe_ids", ())
                if str(item)
            },
            "collapse_recipe_ids": {
                str(item)
                for item in summary.get("recent_collapse_sharpener_operator_recipe_ids", ())
                if str(item)
            },
            "harmful_bundle_keys": {
                str(item)
                for item in summary.get("recent_harmful_bundle_keys", ())
                if str(item)
            },
            "collapse_bundle_keys": {
                str(item)
                for item in summary.get("recent_collapse_sharpener_bundle_keys", ())
                if str(item)
            },
        }

    def _bundle_duplicate_family_penalty(
        self,
        bundle: Mapping[str, Any],
        *,
        term_counts: Mapping[str, int],
    ) -> float:
        focus_term = str(bundle.get("focus_term", "") or "")
        members = bundle.get("members")
        if not isinstance(members, SequenceABC) or isinstance(members, (str, bytes, bytearray)):
            return 0.0
        penalty = 0.0
        if int(term_counts.get(focus_term, 0) or 0) > 1 and not bool(bundle.get("bundle_ready", False)):
            penalty += 0.4
        family_kinds = {
            self._candidate_family_kind(member)
            for member in members
            if isinstance(member, Mapping)
        }
        if len(family_kinds) == 1 and "shot_bridge" in family_kinds:
            penalty += 0.25
        elif len(family_kinds) == 1 and ("kv_v" in family_kinds or "kv_k" in family_kinds):
            penalty += 0.15
        return penalty

    def _bundle_support_confident(
        self,
        *,
        sidecar_support: float,
        sidecar_margin: float,
    ) -> bool:
        return bool(
            sidecar_margin >= 0.08
            and (sidecar_support >= 0.45 or (sidecar_support >= 0.3 and sidecar_margin >= 0.12))
        )

    def _bundle_evidence_agreement(
        self,
        bundle: Mapping[str, Any],
        *,
        first_piece_reachability: float,
        probe_bonus: float,
    ) -> bool:
        provenance_class = str(bundle.get("provenance_class", "misc_prompt") or "misc_prompt")
        bundle_ready = bool(bundle.get("bundle_ready", False))
        if provenance_class != "source_body":
            return False
        return bool(bundle_ready or first_piece_reachability >= 0.08 or probe_bonus >= 0.6)

    def _bundle_is_actionable_candidate(
        self,
        bundle: Mapping[str, Any],
        *,
        first_piece_reachability: float,
        probe_bonus: float,
        harmful_penalty: float,
    ) -> bool:
        if harmful_penalty >= 0.7:
            return False
        if probe_bonus >= 1.0:
            return True
        if bool(bundle.get("bundle_ready", False)) and first_piece_reachability >= 0.12:
            return True
        if self._bundle_provenance_tier(bundle) >= 3 and first_piece_reachability >= 0.2:
            return True
        return False

    def _bundle_provenance_tier(self, bundle: Mapping[str, Any]) -> int:
        provenance_class = str(bundle.get("provenance_class", "misc_prompt") or "misc_prompt")
        members = bundle.get("members")
        if not isinstance(members, SequenceABC) or isinstance(members, (str, bytes, bytearray)):
            return 0
        family_kinds = {
            self._candidate_family_kind(member)
            for member in members
            if isinstance(member, Mapping)
        }
        span_kinds = {
            str(member.get("span_kind", "") or "")
            for member in members
            if isinstance(member, Mapping)
        }
        exact_prompt = any(kind.startswith("exact_prompt") for kind in span_kinds)
        if provenance_class == "source_body":
            if exact_prompt and any(kind in {"kv_v", "kv_k"} for kind in family_kinds):
                return 3
            if exact_prompt or any(kind == "shot_bridge" for kind in family_kinds):
                return 2
            return 1
        if provenance_class in {"constraint_header", "answer_prefix"}:
            return 1
        return 0

    def _bundle_rerank_vetoes(
        self,
        bundle: Mapping[str, Any],
        *,
        harmful_penalty: float,
        duplicate_penalty: float,
    ) -> list[str]:
        vetoes: list[str] = []
        recent_vetoes = self._recent_effect_veto_index()
        bundle_key = str(bundle.get("bundle_key", "") or "")
        members = bundle.get("members")
        if isinstance(members, SequenceABC) and not isinstance(members, (str, bytes, bytearray)):
            labels = {
                str(((member.get("recent_probe") if isinstance(member.get("recent_probe"), Mapping) else {}) or {}).get("label", "") or "")
                for member in members
                if isinstance(member, Mapping)
            }
            if "harmful" in labels:
                vetoes.append("recent_harmful_family")
            if "dead_actuator" in labels:
                vetoes.append("dead_actuator_family")
            if any(bool(member.get("effect_family_collapsed_member", False)) for member in members if isinstance(member, Mapping)):
                vetoes.append("family_collapse")
            for member in members:
                if not isinstance(member, Mapping):
                    continue
                family_key = str(member.get("candidate_family", "") or "")
                recipe_id = str(member.get("operator_recipe_id", "") or "")
                member_bundle_key = str(member.get("bundle_key", "") or bundle_key)
                if (
                    family_key in recent_vetoes["harmful_families"]
                    or recipe_id in recent_vetoes["harmful_recipe_ids"]
                    or member_bundle_key in recent_vetoes["harmful_bundle_keys"]
                ):
                    vetoes.append("recent_harmful_family")
                if (
                    family_key in recent_vetoes["collapse_families"]
                    or recipe_id in recent_vetoes["collapse_recipe_ids"]
                    or member_bundle_key in recent_vetoes["collapse_bundle_keys"]
                ):
                    vetoes.append("collapse_sharpener_veto")
        if duplicate_penalty >= 0.4:
            vetoes.append("duplicate_overflow")
        if harmful_penalty >= 0.7 and "recent_harmful_family" not in vetoes and "dead_actuator_family" not in vetoes:
            vetoes.append("harmful_family_block")
        if harmful_penalty >= 1.0 and "collapse_sharpener_veto" not in vetoes:
            vetoes.append("collapse_sharpener_veto")
        return vetoes

    def _bundle_member_sort_key(self, candidate: Mapping[str, Any]) -> tuple[float, float, int, str]:
        family_kind = self._candidate_family_kind(candidate)
        family_priority = {
            "kv_v": 0,
            "kv_k": 1,
            "shot_bridge": 2,
            "resid_add": 3,
        }.get(family_kind, 4)
        return (
            -self._candidate_builder_score(candidate, include_sidecar=False),
            family_priority,
            str(candidate.get("surface_id", "") or ""),
        )

    def _selector_phase_weights(self, control_phase_hint: str) -> dict[str, float]:
        if control_phase_hint == "readout_escape":
            return {
                "sidecar_support": 0.35,
                "sidecar_margin": 0.15,
                "first_piece": 0.15,
                "probe_bonus": 0.10,
                "duplicate_penalty": 0.15,
                "harmful_penalty": 0.20,
                "top_k": 5.0,
                "base_gap_threshold": 0.65,
                "rerank_gap_threshold": 0.05,
            }
        if control_phase_hint in {"shot_mode", "composition", "entity_insertion"}:
            return {
                "sidecar_support": 0.08,
                "sidecar_margin": 0.04,
                "first_piece": 0.03,
                "probe_bonus": 0.08,
                "duplicate_penalty": 0.10,
                "harmful_penalty": 0.16,
                "top_k": 4.0,
                "base_gap_threshold": 0.20,
                "rerank_gap_threshold": 0.18,
            }
        if control_phase_hint == "loop_break":
            return {
                "sidecar_support": 0.02,
                "sidecar_margin": 0.0,
                "first_piece": 0.0,
                "probe_bonus": 0.04,
                "duplicate_penalty": 0.06,
                "harmful_penalty": 0.22,
                "top_k": 3.0,
                "base_gap_threshold": 0.12,
                "rerank_gap_threshold": 0.20,
            }
        return {
            "sidecar_support": 0.02,
            "sidecar_margin": 0.0,
            "first_piece": 0.0,
            "probe_bonus": 0.04,
            "duplicate_penalty": 0.06,
            "harmful_penalty": 0.12,
            "top_k": 3.0,
            "base_gap_threshold": 0.08,
            "rerank_gap_threshold": 0.20,
        }

    def _post_bundle_rerank(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        answer_readout_canary: Mapping[str, Any] | None,
        debug: dict[str, Any],
        control_phase_hint: str,
    ) -> list[dict[str, Any]]:
        bundle_groups = self._bundle_groups(candidates)
        phase_weights = self._selector_phase_weights(control_phase_hint)
        if len(bundle_groups) <= 1:
            if bundle_groups:
                debug["selected_bundle_key"] = bundle_groups[0].get("bundle_key") or bundle_groups[0].get("group_key")
            debug["bundle_score_debug"] = [
                {
                    "bundle_key": str(group.get("bundle_key") or group.get("group_key") or ""),
                    "focus_term": str(group.get("focus_term", "") or ""),
                    "bundle_ready": bool(group.get("bundle_ready", False)),
                    "base_bundle_score": round(self._bundle_base_score(group.get("members", [])), 6),
                    "rerank_score": round(self._bundle_base_score(group.get("members", [])), 6),
                    "selected": True,
                }
                for group in bundle_groups
            ]
            debug["bundle_reorder_applied"] = False
            debug["bundle_reorder_reason"] = "single_bundle"
            debug["bundle_selector_phase"] = str(control_phase_hint)
            debug["bundle_rerank_gate_open"] = False
            debug["bundle_rerank_gate_reasons"] = ["single_bundle"]
            return [
                member
                for group in bundle_groups
                for member in sorted(
                    [dict(item) for item in group.get("members", []) if isinstance(item, Mapping)],
                    key=self._bundle_member_sort_key,
                )
            ]

        term_counts: dict[str, int] = {}
        for group in bundle_groups:
            focus_term = str(group.get("focus_term", "") or "")
            if focus_term:
                term_counts[focus_term] = int(term_counts.get(focus_term, 0) or 0) + 1

        scored_groups: list[dict[str, Any]] = []
        for group in bundle_groups:
            members = [dict(item) for item in group.get("members", []) if isinstance(item, Mapping)]
            base_score = self._bundle_base_score(members)
            sidecar_support = self._bundle_sidecar_support(group)
            sidecar_margin = self._bundle_sidecar_support_margin(group)
            sidecar_evidence = self._bundle_sidecar_evidence(group)
            first_piece_reachability = self._bundle_first_piece_reachability(
                group,
                answer_readout_canary=answer_readout_canary,
            )
            probe_bonus = self._bundle_probe_bonus(group)
            probe_axis_scores = self._bundle_probe_axis_scores(group)
            duplicate_penalty = self._bundle_duplicate_family_penalty(group, term_counts=term_counts)
            harmful_penalty = self._bundle_harmful_family_penalty(group)
            support_confident = self._bundle_support_confident(
                sidecar_support=sidecar_support,
                sidecar_margin=sidecar_margin,
            )
            evidence_agreement = self._bundle_evidence_agreement(
                group,
                first_piece_reachability=first_piece_reachability,
                probe_bonus=probe_bonus,
            )
            actionable_candidate = self._bundle_is_actionable_candidate(
                group,
                first_piece_reachability=first_piece_reachability,
                probe_bonus=probe_bonus,
                harmful_penalty=harmful_penalty,
            )
            rerank_score = (
                base_score
                + (0.08 * float(sidecar_evidence.get("semantic_residual_support", 0.0) or 0.0))
                + (0.04 * float(sidecar_evidence.get("anchor_strength", 0.0) or 0.0))
                + (phase_weights["first_piece"] * first_piece_reachability)
                + (0.08 * float(probe_axis_scores.get("readout_score", 0.0) or 0.0))
                + (0.04 * float(probe_axis_scores.get("constraint_score", 0.0) or 0.0))
                + (phase_weights["probe_bonus"] * probe_bonus)
                - (phase_weights["duplicate_penalty"] * duplicate_penalty)
                - (phase_weights["harmful_penalty"] * harmful_penalty)
            )
            scored_groups.append(
                {
                    **dict(group),
                    "members": members,
                    "operator_family_key": self._operator_family_key(group),
                    "base_bundle_score": float(base_score),
                    "sidecar_bundle_support": float(sidecar_support),
                    "sidecar_support_margin": float(sidecar_margin),
                    "sidecar_evidence": dict(sidecar_evidence),
                    "first_piece_reachability": float(first_piece_reachability),
                    "probe_bonus": float(probe_bonus),
                    "readout_probe_score": float(probe_axis_scores.get("readout_score", 0.0) or 0.0),
                    "constraint_probe_score": float(probe_axis_scores.get("constraint_score", 0.0) or 0.0),
                    "trajectory_probe_score": float(probe_axis_scores.get("trajectory_score", 0.0) or 0.0),
                    "duplicate_family_penalty": float(duplicate_penalty),
                    "harmful_family_penalty": float(harmful_penalty),
                    "bundle_support_confident": bool(support_confident),
                    "bundle_evidence_agreement": bool(evidence_agreement),
                    "bundle_is_actionable_candidate": bool(actionable_candidate),
                    "bundle_provenance_tier": int(self._bundle_provenance_tier(group)),
                    "rerank_vetoes": self._bundle_rerank_vetoes(
                        group,
                        harmful_penalty=harmful_penalty,
                        duplicate_penalty=duplicate_penalty,
                    ),
                    "rerank_score": float(rerank_score),
                }
            )

        base_sorted = sorted(
            scored_groups,
            key=lambda item: (
                -float(item["base_bundle_score"]),
                0 if bool(item.get("bundle_ready", False)) else 1,
                str(item.get("bundle_key") or item.get("group_key") or ""),
            ),
        )
        top_k = min(max(1, int(phase_weights["top_k"])), len(base_sorted))
        rerank_candidates = list(base_sorted[:top_k])
        best_first_piece = max((float(item.get("first_piece_reachability", 0.0) or 0.0) for item in rerank_candidates), default=0.0)
        for item in rerank_candidates:
            reachability_ratio = (
                1.0
                if best_first_piece <= 1e-6
                else float(item.get("first_piece_reachability", 0.0) or 0.0) / max(best_first_piece, 1e-6)
            )
            item["reachability_ratio"] = round(float(min(1.25, max(0.0, reachability_ratio))), 6)
            eligibility_reasons: list[str] = []
            if control_phase_hint != "readout_escape":
                eligibility_reasons.append("not_readout_escape")
            if not bool(item.get("bundle_ready", False)):
                eligibility_reasons.append("bundle_not_ready")
            if int(item.get("bundle_provenance_tier", 0) or 0) < 2:
                eligibility_reasons.append("provenance_tier_low")
            if item.get("rerank_vetoes"):
                eligibility_reasons.append("has_vetoes")
            item["eligibility_reasons"] = list(eligibility_reasons)
            item["eligible_for_rerank"] = not eligibility_reasons
            evidence_votes = 0
            if bool(item.get("bundle_support_confident", False)):
                evidence_votes += 1
            if int(item.get("bundle_provenance_tier", 0) or 0) >= 2:
                evidence_votes += 1
            if float(item.get("probe_bonus", 0.0) or 0.0) >= 0.6 and float(item.get("harmful_family_penalty", 0.0) or 0.0) < 0.7:
                evidence_votes += 1
            if float(item.get("reachability_ratio", 0.0) or 0.0) >= 0.75:
                evidence_votes += 1
            item["evidence_votes"] = int(evidence_votes)

        rerank_mode = self.readout_analyzer_rerank_mode
        reorder_applied = False
        reorder_reason = "base_order_kept"
        gate_open = False
        gate_reasons: list[str] = []
        rerank_vetoes: list[str] = []
        selection_source = "base"
        rerank_gate_mode = "none"
        base_winner = rerank_candidates[0] if rerank_candidates else None
        challenger = None
        hard_margin = 0.0
        soft_margin = 0.0
        pairwise_delta = 0.0
        base_gap = 10.0
        base_gap_norm = 10.0
        pairwise_margin_breakdown: dict[str, float] = {}
        common_mode_evidence: dict[str, Any] = {}
        support_common_vs_discriminative: dict[str, float] = {}
        controller_pairwise_reason_text = ""
        if base_winner is not None:
            eligible_challengers = [item for item in rerank_candidates[1:] if bool(item.get("eligible_for_rerank", False))]
            pairwise_rows: list[tuple[dict[str, Any], dict[str, float], dict[str, Any], dict[str, float]]] = []
            if eligible_challengers:
                base_evidence = (
                    dict(base_winner.get("sidecar_evidence", {}))
                    if isinstance(base_winner.get("sidecar_evidence"), Mapping)
                    else {}
                )
                base_support = float(base_winner.get("sidecar_bundle_support", 0.0) or 0.0)
                base_readout_probe = float(base_winner.get("readout_probe_score", 0.0) or 0.0)
                base_constraint_probe = float(base_winner.get("constraint_probe_score", 0.0) or 0.0)
                base_trajectory_probe = float(base_winner.get("trajectory_probe_score", 0.0) or 0.0)
                base_span_precision = float(base_evidence.get("span_precision", 0.0) or 0.0)
                base_semantic_support = float(base_evidence.get("semantic_residual_support", base_support) or base_support)
                base_anchor_strength = float(base_evidence.get("anchor_strength", 0.0) or 0.0)
                base_reachability_ratio = float(base_winner.get("reachability_ratio", 0.0) or 0.0)
                base_first_piece = float(base_winner.get("first_piece_reachability", 0.0) or 0.0)
                base_actionable = 1.0 if bool(base_winner.get("bundle_is_actionable_candidate", False)) else 0.0

                def _clip_unit(value: float) -> float:
                    return max(-1.0, min(1.0, float(value)))

                for item in eligible_challengers:
                    item_evidence = (
                        dict(item.get("sidecar_evidence", {}))
                        if isinstance(item.get("sidecar_evidence"), Mapping)
                        else {}
                    )
                    challenger_support = float(item.get("sidecar_bundle_support", 0.0) or 0.0)
                    challenger_readout_probe = float(item.get("readout_probe_score", 0.0) or 0.0)
                    challenger_constraint_probe = float(item.get("constraint_probe_score", 0.0) or 0.0)
                    challenger_trajectory_probe = float(item.get("trajectory_probe_score", 0.0) or 0.0)
                    challenger_span_precision = float(item_evidence.get("span_precision", 0.0) or 0.0)
                    challenger_semantic_support = float(
                        item_evidence.get("semantic_residual_support", challenger_support) or challenger_support
                    )
                    challenger_anchor_strength = float(item_evidence.get("anchor_strength", 0.0) or 0.0)
                    reachability_adv = max(0.0, float(item.get("reachability_ratio", 0.0) or 0.0) - base_reachability_ratio)
                    first_piece_adv = max(
                        0.0,
                        float(item.get("first_piece_reachability", 0.0) or 0.0) - base_first_piece,
                    )
                    actionable_adv = max(
                        0.0,
                        (1.0 if bool(item.get("bundle_is_actionable_candidate", False)) else 0.0) - base_actionable,
                    )
                    probe_readout_adv = _clip_unit(challenger_readout_probe - base_readout_probe)
                    probe_constraint_adv = _clip_unit(challenger_constraint_probe - base_constraint_probe)
                    probe_trajectory_adv = _clip_unit(challenger_trajectory_probe - base_trajectory_probe)
                    span_precision_adv = max(0.0, challenger_span_precision - base_span_precision)
                    harmful_penalty_adv = max(
                        0.0,
                        float(item.get("harmful_family_penalty", 0.0) or 0.0)
                        - float(base_winner.get("harmful_family_penalty", 0.0) or 0.0),
                    )
                    duplicate_penalty_adv = max(
                        0.0,
                        float(item.get("duplicate_family_penalty", 0.0) or 0.0)
                        - float(base_winner.get("duplicate_family_penalty", 0.0) or 0.0),
                    )
                    semantic_residual_adv = _clip_unit(challenger_semantic_support - base_semantic_support)
                    anchor_strength_adv = _clip_unit(challenger_anchor_strength - base_anchor_strength)
                    hard_candidate = (
                        (0.55 * reachability_adv)
                        + (0.35 * first_piece_adv)
                        + (0.40 * actionable_adv)
                        + (0.30 * probe_readout_adv)
                        + (0.10 * probe_constraint_adv)
                        + (0.05 * probe_trajectory_adv)
                        + (0.15 * span_precision_adv)
                        - (0.40 * harmful_penalty_adv)
                        - (0.20 * duplicate_penalty_adv)
                    )
                    soft_candidate = (0.20 * semantic_residual_adv) + (0.10 * anchor_strength_adv)
                    common_candidate = {
                        "source_body_exact_both": bool(
                            base_evidence.get("source_body_exact", False) and item_evidence.get("source_body_exact", False)
                        ),
                        "bundle_ready_both": bool(base_winner.get("bundle_ready", False) and item.get("bundle_ready", False)),
                        "evidence_agreement_both": bool(
                            base_winner.get("bundle_evidence_agreement", False)
                            and item.get("bundle_evidence_agreement", False)
                        ),
                        "same_provenance_tier": bool(
                            int(base_winner.get("bundle_provenance_tier", 0) or 0)
                            == int(item.get("bundle_provenance_tier", 0) or 0)
                        ),
                    }
                    breakdown = {
                        "reachability_adv": round(float(reachability_adv), 6),
                        "first_piece_adv": round(float(first_piece_adv), 6),
                        "actionable_adv": round(float(actionable_adv), 6),
                        "probe_readout_adv": round(float(probe_readout_adv), 6),
                        "probe_constraint_adv": round(float(probe_constraint_adv), 6),
                        "probe_trajectory_adv": round(float(probe_trajectory_adv), 6),
                        "span_precision_adv": round(float(span_precision_adv), 6),
                        "harmful_penalty": round(float(harmful_penalty_adv), 6),
                        "duplicate_penalty": round(float(duplicate_penalty_adv), 6),
                        "soft_semantic_adv": round(float(semantic_residual_adv), 6),
                        "soft_anchor_adv": round(float(anchor_strength_adv), 6),
                        "hard_margin": round(float(hard_candidate), 6),
                        "soft_margin": round(float(soft_candidate), 6),
                    }
                    support_split = {
                        "common": round(float(min(base_support, challenger_support)), 6),
                        "discriminative": round(float(max(abs(semantic_residual_adv), abs(anchor_strength_adv))), 6),
                    }
                    pairwise_rows.append((item, breakdown, common_candidate, support_split))
            if pairwise_rows:
                challenger, pairwise_margin_breakdown, common_mode_evidence, support_common_vs_discriminative = max(
                    pairwise_rows,
                    key=lambda row: (
                        float(row[1].get("hard_margin", 0.0) or 0.0),
                        float(row[1].get("soft_margin", 0.0) or 0.0),
                        float(row[0].get("reachability_ratio", 0.0) or 0.0),
                        float(row[0].get("base_bundle_score", -10.0) or -10.0),
                        -float(row[0].get("harmful_family_penalty", 0.0) or 0.0),
                        str(row[0].get("bundle_key") or row[0].get("group_key") or ""),
                    ),
                )
            if rerank_mode == "off":
                gate_reasons.append("mode_off")
            elif control_phase_hint != "readout_escape":
                gate_reasons.append("not_readout_escape")
            elif challenger is None:
                gate_reasons.append("no_eligible_challenger")
            else:
                rerank_vetoes = list(challenger.get("rerank_vetoes", []) or [])
                gate_reasons.extend(rerank_vetoes)
                base_gap = float(base_winner.get("base_bundle_score", 0.0) or 0.0) - float(challenger.get("base_bundle_score", 0.0) or 0.0)
                base_gap_norm = base_gap / max(abs(float(base_winner.get("base_bundle_score", 0.0) or 0.0)), 1.0)
                hard_margin = float(pairwise_margin_breakdown.get("hard_margin", 0.0) or 0.0)
                soft_margin = float(pairwise_margin_breakdown.get("soft_margin", 0.0) or 0.0)
                pairwise_delta = hard_margin + (soft_margin if hard_margin >= 0.25 else 0.0)
                if base_gap_norm >= 0.08:
                    gate_reasons.append("base_gap_large")
                min_reachability = max(0.6, float(base_winner.get("reachability_ratio", 0.0) or 0.0) - 0.25)
                if float(challenger.get("reachability_ratio", 0.0) or 0.0) < min_reachability:
                    gate_reasons.append("reachability_too_low")
                if int(challenger.get("evidence_votes", 0) or 0) < 2:
                    gate_reasons.append("insufficient_evidence_votes")
                if hard_margin <= 0.18:
                    gate_reasons.append("hard_margin_insufficient")
                if (
                    float(base_winner.get("probe_bonus", 0.0) or 0.0) >= 1.0
                    and float(challenger.get("probe_bonus", 0.0) or 0.0) < 1.0
                ):
                    gate_reasons.append("base_actionable_winner")
                shadow_gate_open = not gate_reasons
                if shadow_gate_open and rerank_mode == "shadow":
                    gate_open = True
                    reorder_applied = False
                    reorder_reason = f"{control_phase_hint}_pairwise_shadow"
                    selection_source = "base_shadow"
                    rerank_gate_mode = "shadow"
                elif shadow_gate_open and rerank_mode == "apply":
                    apply_ready = bool(
                        hard_margin >= 1.1
                        or float(pairwise_margin_breakdown.get("probe_readout_adv", 0.0) or 0.0) > 0.15
                        or float(challenger.get("probe_bonus", 0.0) or 0.0) >= 1.0
                        or (hard_margin >= 0.45 and soft_margin > 0.08)
                    )
                    if not apply_ready:
                        gate_reasons.append("apply_evidence_insufficient")
                    else:
                        gate_open = True
                        reorder_applied = True
                        reorder_reason = f"{control_phase_hint}_pairwise_promote"
                        selection_source = "sidecar_tiebreak"
                        rerank_gate_mode = (
                            "promote_body" if int(challenger.get("bundle_provenance_tier", 0) or 0) >= 3 else "tiebreak"
                        )
                elif "base_actionable_winner" in gate_reasons:
                    selection_source = "probe_lock"
                reason_parts: list[str] = []
                if float(pairwise_margin_breakdown.get("reachability_adv", 0.0) or 0.0) > 0.0:
                    reason_parts.append("challenger had better reachability")
                if float(pairwise_margin_breakdown.get("actionable_adv", 0.0) or 0.0) > 0.0:
                    reason_parts.append("challenger was actionable while base was not")
                if float(pairwise_margin_breakdown.get("first_piece_adv", 0.0) or 0.0) > 0.0:
                    reason_parts.append("challenger had stronger first-piece readout")
                if not rerank_vetoes:
                    reason_parts.append("no veto applied")
                controller_pairwise_reason_text = "; ".join(reason_parts[:3]) if reason_parts else "base winner remained preferred"

        debug["base_winner_bundle_key"] = (
            str(base_winner.get("bundle_key") or base_winner.get("group_key") or "") if base_winner is not None else None
        )
        debug["challenger_bundle_key"] = (
            str(challenger.get("bundle_key") or challenger.get("group_key") or "") if challenger is not None else None
        )
        debug["bundle_base_gap"] = round(float(base_gap), 6)
        debug["base_gap_norm"] = round(float(base_gap_norm), 6)
        debug["bundle_rerank_gap"] = round(float(pairwise_delta), 6)
        debug["hard_margin"] = round(float(hard_margin), 6)
        debug["soft_margin"] = round(float(soft_margin), 6)
        debug["pairwise_delta"] = round(float(pairwise_delta), 6)
        debug["pairwise_margin_breakdown"] = dict(pairwise_margin_breakdown)
        debug["common_mode_evidence"] = dict(common_mode_evidence)
        debug["support_common_vs_discriminative"] = dict(support_common_vs_discriminative)
        debug["controller_pairwise_reason_text"] = controller_pairwise_reason_text
        debug["rerank_vetoes"] = list(rerank_vetoes)
        debug["selection_source"] = str(selection_source)
        debug["rerank_gate_mode"] = str(rerank_gate_mode)
        debug["gate_report_base_winner_bundle_key"] = debug["base_winner_bundle_key"]
        debug["gate_report_challenger_bundle_key"] = debug["challenger_bundle_key"]
        debug["gate_report_vetoes"] = list(rerank_vetoes)
        debug["gate_report_selection_source"] = str(selection_source)
        debug["gate_report_mode"] = str(rerank_gate_mode)
        debug["gate_report_pairwise_reason_text"] = controller_pairwise_reason_text

        if reorder_applied and challenger is not None and base_winner is not None:
            final_frontier = [challenger, base_winner] + [
                group
                for group in rerank_candidates[1:]
                if str(group.get("bundle_key") or group.get("group_key") or "")
                != str(challenger.get("bundle_key") or challenger.get("group_key") or "")
            ]
        else:
            final_frontier = list(rerank_candidates)
        final_groups = list(final_frontier) + list(base_sorted[top_k:])

        selected_key = str(final_groups[0].get("bundle_key") or final_groups[0].get("group_key") or "")
        debug["selected_bundle_key"] = selected_key
        debug["gate_report_frontier_bundle_key"] = selected_key
        debug["bundle_reorder_applied"] = bool(reorder_applied)
        debug["bundle_reorder_reason"] = str(reorder_reason)
        debug["bundle_selector_phase"] = str(control_phase_hint)
        debug["bundle_rerank_mode"] = str(rerank_mode)
        debug["bundle_rerank_gate_open"] = bool(gate_open)
        debug["bundle_rerank_gate_reasons"] = list(gate_reasons)
        debug["gate_report_open"] = bool(gate_open)
        debug["gate_report_reasons"] = list(gate_reasons)
        debug["gate_report_rejected_signals"] = list(dict.fromkeys([*gate_reasons, *rerank_vetoes]))
        debug["bundle_score_debug"] = [
            {
                "bundle_key": str(group.get("bundle_key") or group.get("group_key") or ""),
                "focus_term": str(group.get("focus_term", "") or ""),
                "operator_family_key": str(group.get("operator_family_key", "") or ""),
                "bundle_ready": bool(group.get("bundle_ready", False)),
                "member_count": len(group.get("members", [])),
                "base_bundle_score": round(float(group["base_bundle_score"]), 6),
                "sidecar_bundle_support": round(float(group["sidecar_bundle_support"]), 6),
                "sidecar_support_margin": round(float(group["sidecar_support_margin"]), 6),
                "sidecar_evidence": dict(group.get("sidecar_evidence", {})),
                "first_piece_reachability": round(float(group["first_piece_reachability"]), 6),
                "probe_bonus": round(float(group["probe_bonus"]), 6),
                "readout_probe_score": round(float(group.get("readout_probe_score", 0.0) or 0.0), 6),
                "constraint_probe_score": round(float(group.get("constraint_probe_score", 0.0) or 0.0), 6),
                "trajectory_probe_score": round(float(group.get("trajectory_probe_score", 0.0) or 0.0), 6),
                "duplicate_family_penalty": round(float(group["duplicate_family_penalty"]), 6),
                "harmful_family_penalty": round(float(group["harmful_family_penalty"]), 6),
                "bundle_support_confident": bool(group["bundle_support_confident"]),
                "bundle_evidence_agreement": bool(group["bundle_evidence_agreement"]),
                "bundle_is_actionable_candidate": bool(group["bundle_is_actionable_candidate"]),
                "bundle_provenance_tier": int(group.get("bundle_provenance_tier", 0) or 0),
                "rerank_vetoes": list(group.get("rerank_vetoes", []) or []),
                "eligibility_reasons": list(group.get("eligibility_reasons", []) or []),
                "eligible_for_rerank": bool(group.get("eligible_for_rerank", False)),
                "reachability_ratio": round(float(group.get("reachability_ratio", 0.0) or 0.0), 6),
                "evidence_votes": int(group.get("evidence_votes", 0) or 0),
                "pairwise_delta_vs_base": round(
                    float(pairwise_delta)
                    if (
                        challenger is not None
                        and str(group.get("bundle_key") or group.get("group_key") or "")
                        == str(challenger.get("bundle_key") or challenger.get("group_key") or "")
                    )
                    else 0.0,
                    6,
                ),
                "pairwise_margin_breakdown_vs_base": (
                    dict(pairwise_margin_breakdown)
                    if (
                        challenger is not None
                        and str(group.get("bundle_key") or group.get("group_key") or "")
                        == str(challenger.get("bundle_key") or challenger.get("group_key") or "")
                    )
                    else {}
                ),
                "common_mode_evidence_vs_base": (
                    dict(common_mode_evidence)
                    if (
                        challenger is not None
                        and str(group.get("bundle_key") or group.get("group_key") or "")
                        == str(challenger.get("bundle_key") or challenger.get("group_key") or "")
                    )
                    else {}
                ),
                "support_common_vs_discriminative": (
                    dict(support_common_vs_discriminative)
                    if (
                        challenger is not None
                        and str(group.get("bundle_key") or group.get("group_key") or "")
                        == str(challenger.get("bundle_key") or challenger.get("group_key") or "")
                    )
                    else {}
                ),
                "operator_certification": (
                    dict(self._operator_certification_table.get(str(group.get("operator_family_key", "") or ""), {}))
                    if str(group.get("operator_family_key", "") or "") in self._operator_certification_table
                    else {}
                ),
                "rerank_score": round(float(group["rerank_score"]), 6),
                "selected": bool(str(group.get("bundle_key") or group.get("group_key") or "") == selected_key),
            }
            for group in final_groups[:top_k]
        ]
        ordered_candidates: list[dict[str, Any]] = []
        for group in final_groups:
            members = [
                dict(item)
                for item in group.get("members", [])
                if isinstance(item, Mapping)
            ]
            members.sort(key=self._bundle_member_sort_key)
            ordered_candidates.extend(members)
        return ordered_candidates

    def _prompt_term_spans(self, term: str, *, max_spans: int = 2) -> list[dict[str, Any]]:
        normalized_term = " ".join(str(term).split()).strip()
        if not normalized_term:
            return []
        records = self._token_position_records()
        if not records:
            return []
        variant_token_ids: list[tuple[str, list[int]]] = []
        seen_variants: set[tuple[int, ...]] = set()
        for raw_text in (f" {normalized_term}", normalized_term):
            token_ids = self._encode_text_token_ids(raw_text)
            token_key = tuple(int(token_id) for token_id in token_ids)
            if not token_key or token_key in seen_variants:
                continue
            seen_variants.add(token_key)
            variant_token_ids.append((raw_text, list(token_key)))

        matches: list[dict[str, Any]] = []
        seen_spans: set[tuple[int, int]] = set()
        for _raw_text, token_ids in variant_token_ids:
            width = len(token_ids)
            for start in range(0, len(records) - width + 1):
                window = records[start : start + width]
                if not window:
                    continue
                if any(str(item.get("segment_kind", "")) not in {"prompt", "hint"} for item in window):
                    continue
                if [int(item.get("token_id", -1)) for item in window] != token_ids:
                    continue
                span_key = (int(window[0]["position"]), int(window[-1]["position"]) + 1)
                if span_key in seen_spans:
                    continue
                seen_spans.add(span_key)
                provenance_counts: dict[str, int] = {}
                for item in window:
                    provenance_class = str(item.get("provenance_class", "misc_prompt") or "misc_prompt")
                    provenance_counts[provenance_class] = int(provenance_counts.get(provenance_class, 0) or 0) + 1
                provenance_class = max(
                    provenance_counts,
                    key=lambda key: (self._provenance_weight(key), provenance_counts.get(key, 0), key),
                )
                matches.append(
                    {
                        "start": int(window[0]["position"]),
                        "end": int(window[-1]["position"]) + 1,
                        "length": len(window),
                        "segment_kind": str(window[0].get("segment_kind", "prompt") or "prompt"),
                        "provenance_class": provenance_class,
                        "pieces": [str(item.get("piece", "")) for item in window],
                        "text": "".join(str(item.get("piece", "")) for item in window),
                        "span_kind": "exact_prompt_span",
                    }
                )
        collapsed_matches: list[dict[str, Any]] = []
        for item in matches:
            replaced = False
            normalized_text = " ".join(str(item.get("text", "")).split()).strip().lower()
            for index, existing in enumerate(collapsed_matches):
                existing_text = " ".join(str(existing.get("text", "")).split()).strip().lower()
                overlaps = not (int(item["end"]) <= int(existing["start"]) or int(existing["end"]) <= int(item["start"]))
                if normalized_text and normalized_text == existing_text and overlaps:
                    if int(item["length"]) > int(existing["length"]):
                        collapsed_matches[index] = item
                    replaced = True
                    break
            if not replaced:
                collapsed_matches.append(item)
        matches = collapsed_matches
        matches.sort(
            key=lambda item: (
                -self._provenance_weight(str(item.get("provenance_class", "misc_prompt") or "misc_prompt")),
                0 if str(item["segment_kind"]) == "prompt" else 1,
                -int(item["end"]),
                -int(item["length"]),
                int(item["start"]),
            )
        )
        return matches[: max(1, int(max_spans))]

    def _bundle_inputs_for_candidates(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for raw_candidate in candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            family_kind = self._candidate_family_kind(raw_candidate)
            if family_kind not in {"kv_v", "kv_k"}:
                continue
            focus_term = str(raw_candidate.get("focus_feature", "") or "")
            if not focus_term:
                continue
            provenance_class = str(raw_candidate.get("provenance_class", "misc_prompt") or "misc_prompt")
            span_id = self._candidate_span_id(raw_candidate)
            group_key = (focus_term, provenance_class, span_id)
            group = grouped.setdefault(
                group_key,
                {
                    "term": focus_term,
                    "provenance_class": provenance_class,
                    "span_id": span_id,
                    "span_kind": str(raw_candidate.get("span_kind", "") or ""),
                    "source_span": dict(raw_candidate["source_span"]) if isinstance(raw_candidate.get("source_span"), Mapping) else None,
                    "families": {},
                },
            )
            group["families"][family_kind] = {
                "surface_id": str(raw_candidate.get("surface_id", "") or ""),
                "candidate_family": str(raw_candidate.get("candidate_family", "") or ""),
            }
        bundles: list[dict[str, Any]] = []
        for group in grouped.values():
            families = group.get("families", {})
            if not isinstance(families, Mapping) or "kv_v" not in families or "kv_k" not in families:
                continue
            bundle_key = (
                f"kv_pair:{group['term']}:{group['provenance_class']}:{group['span_id']}"
            )
            bundles.append(
                {
                    "bundle_key": bundle_key,
                    "term": group["term"],
                    "provenance_class": group["provenance_class"],
                    "span_kind": group["span_kind"],
                    "source_span": group.get("source_span"),
                    "v": dict(families["kv_v"]),
                    "k": dict(families["kv_k"]),
                }
            )
        bundles.sort(
            key=lambda item: (
                -self._provenance_weight(str(item.get("provenance_class", "misc_prompt") or "misc_prompt")),
                str(item.get("term", "")),
                str(item.get("bundle_key", "")),
            )
        )
        return bundles[:4]

    def _prompt_records_for_span(self, start: int, end: int) -> list[dict[str, Any]]:
        def _position(record: Mapping[str, Any]) -> int:
            value = record.get("position", -1)
            return int(-1 if value is None else value)

        return [
            dict(record)
            for record in self._token_position_records()
            if str(record.get("segment_kind", "") or "") == "prompt"
            and _position(record) >= int(start)
            and _position(record) < int(end)
        ]

    def _prompt_window_positions(self, start: int, end: int, *, padding: int = 0) -> list[int]:
        def _position(record: Mapping[str, Any]) -> int:
            value = record.get("position", -1)
            return int(-1 if value is None else value)

        prompt_positions = [
            _position(record)
            for record in self._token_position_records()
            if str(record.get("segment_kind", "") or "") == "prompt"
        ]
        prompt_positions = sorted(position for position in prompt_positions if position >= 0)
        if not prompt_positions:
            return []
        try:
            start_index = prompt_positions.index(int(start))
        except ValueError:
            return []
        end_position = int(end) - 1
        try:
            end_index = prompt_positions.index(end_position)
        except ValueError:
            end_index = start_index
        left = max(0, start_index - max(0, int(padding)))
        right = min(len(prompt_positions), end_index + 1 + max(0, int(padding)))
        return prompt_positions[left:right]

    def _content_piece_score(self, piece: str) -> tuple[float, int, str]:
        text = str(piece or "")
        normalized = "".join(ch for ch in text.strip().lower() if ch.isalnum())
        if not normalized:
            return (0.0, 0, text)
        starts_with_space = 1 if text.startswith(" ") else 0
        return (1.0 + (0.2 * starts_with_space) + min(0.6, len(normalized) / 8.0), len(normalized), normalized)

    def _cache_ref_expr(
        self,
        *,
        site: str,
        layer: int,
        head: int,
        token_selector: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "ref": {
                "scope": "runtime",
                "worker": self.worker_id,
                "tensor": str(site),
                "layer": int(layer),
                "head": int(head),
                "token": dict(token_selector),
            }
        }

    def _expr_weighted_mean(
        self,
        refs_with_weights: Sequence[tuple[dict[str, Any], float]],
    ) -> dict[str, Any]:
        filtered = [(dict(expr), float(weight)) for expr, weight in refs_with_weights if abs(float(weight)) > 1e-8]
        if not filtered:
            raise ValueError("weighted mean requires at least one non-zero term")
        total_weight = sum(float(weight) for _expr, weight in filtered)
        if len(filtered) == 1:
            expr, weight = filtered[0]
            if abs(weight - 1.0) <= 1e-8:
                return expr
            return {"fn": "scale", "by": float(weight), "arg": expr}
        args: list[dict[str, Any]] = []
        for expr, weight in filtered:
            if abs(weight - 1.0) <= 1e-8:
                args.append(expr)
            else:
                args.append({"fn": "scale", "by": float(weight), "arg": expr})
        max_args = int(getattr(getattr(self, "policy", None), "max_expr_args", 4) or 4)
        summed = self._expr_add_tree(args, max_args=max(2, max_args))
        if abs(total_weight - 1.0) <= 1e-8:
            return summed
        return {"fn": "scale", "by": float(1.0 / total_weight), "arg": summed}

    def _expr_add_tree(
        self,
        args: Sequence[Mapping[str, Any]],
        *,
        max_args: int = 4,
    ) -> dict[str, Any]:
        normalized_args = [dict(arg) for arg in args if isinstance(arg, Mapping)]
        if len(normalized_args) < 2:
            raise ValueError("add tree requires at least two expressions")
        if len(normalized_args) <= max_args:
            return {"fn": "add", "args": normalized_args}
        branch_count = min(max_args, len(normalized_args))
        chunk_size = max(2, math.ceil(len(normalized_args) / branch_count))
        next_level: list[dict[str, Any]] = []
        for index in range(0, len(normalized_args), chunk_size):
            chunk = normalized_args[index : index + chunk_size]
            if len(chunk) == 1:
                next_level.append(dict(chunk[0]))
            elif len(chunk) == 2:
                next_level.append({"fn": "add", "args": [dict(chunk[0]), dict(chunk[1])]})
            else:
                next_level.append(self._expr_add_tree(chunk, max_args=max_args))
        return self._expr_add_tree(next_level, max_args=max_args)

    def _candidate_recipe_source_expr(
        self,
        candidate: Mapping[str, Any],
        *,
        localization: str,
        pooling: str,
        contrast_mode: str = "none",
        competitor_candidate: Mapping[str, Any] | None = None,
        contrast_scale: float = 1.0,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        site = str(candidate.get("site", "") or "")
        layer = candidate.get("layer")
        head = candidate.get("head")
        source_span = candidate.get("source_span") if isinstance(candidate.get("source_span"), Mapping) else None
        if site not in {"k_cache", "v_cache"} or isinstance(layer, bool) or not isinstance(layer, int):
            return None, {}
        if isinstance(head, bool) or not isinstance(head, int):
            return None, {}
        if source_span is None:
            selector = candidate.get("source", {}).get("dtype") if isinstance(candidate.get("source"), Mapping) else None
            if selector is None:
                return None, {}
            token_selector = {"mode": "index", "value": int(candidate.get("source_position", 0) or 0)}
            expr = self._cache_ref_expr(site=site, layer=int(layer), head=int(head), token_selector=token_selector)
            metadata = {
                "recipe_localization": "source_position_single",
                "recipe_pooling": "single",
                "contrast_mode": str(contrast_mode or "none"),
                "source_positions": [int(candidate.get("source_position", 0) or 0)],
            }
            return expr, metadata

        start = int(source_span.get("start", 0) or 0)
        end = int(source_span.get("end", start + 1) or (start + 1))
        span_records = self._prompt_records_for_span(start, end)
        span_positions: list[int] = []
        for record in span_records:
            value = record.get("position", -1)
            position = int(-1 if value is None else value)
            if position >= 0:
                span_positions.append(position)
        if not span_positions:
            return None, {}

        recipe_localization = str(localization or "exact_prompt_span_mean")
        recipe_pooling = str(pooling or "mean")
        refs_with_weights: list[tuple[dict[str, Any], float]] = []
        metadata_positions: list[int] = []

        if recipe_localization == "exact_prompt_span_mean":
            expr = self._cache_ref_expr(
                site=site,
                layer=int(layer),
                head=int(head),
                token_selector={"mode": "span", "start": start, "end": end, "pool": "mean"},
            )
            metadata_positions = list(span_positions)
        elif recipe_localization == "exact_term_token":
            best_record = max(
                span_records,
                key=lambda record: self._content_piece_score(str(record.get("piece", ""))),
            )
            token_position = int(best_record.get("position", start) or start)
            expr = self._cache_ref_expr(
                site=site,
                layer=int(layer),
                head=int(head),
                token_selector={"mode": "index", "value": token_position},
            )
            metadata_positions = [token_position]
        elif recipe_localization == "exact_term_fused":
            span_expr, span_meta = self._candidate_recipe_source_expr(
                candidate,
                localization="exact_prompt_span_mean",
                pooling="mean",
                contrast_mode="none",
            )
            token_expr, token_meta = self._candidate_recipe_source_expr(
                candidate,
                localization="exact_term_token",
                pooling="single",
                contrast_mode="none",
            )
            if span_expr is None or token_expr is None:
                return None, {}
            metadata_positions = list(
                dict.fromkeys(
                    list(span_meta.get("source_positions", []))
                    + list(token_meta.get("source_positions", []))
                )
            )
            expr = {
                "fn": "normalize",
                "arg": {
                    "fn": "add",
                    "args": [
                        {"fn": "scale", "by": 0.7, "arg": span_expr},
                        {"fn": "scale", "by": 0.3, "arg": token_expr},
                    ],
                },
            }
            recipe_pooling = "fused"
        elif recipe_localization in {"exact_term_window_pm1_weighted", "exact_term_window_pm2_weighted"}:
            padding = 1 if recipe_localization.endswith("pm1_weighted") else 2
            window_positions = self._prompt_window_positions(start, end, padding=padding)
            if not window_positions:
                return None, {}
            metadata_positions = list(window_positions)
            for position in window_positions:
                if start <= position < end:
                    weight = 1.0
                else:
                    distance = min(abs(position - start), abs(position - (end - 1)))
                    weight = 0.5 if distance <= 1 else 0.25
                refs_with_weights.append(
                    (
                        self._cache_ref_expr(
                            site=site,
                            layer=int(layer),
                            head=int(head),
                            token_selector={"mode": "index", "value": int(position)},
                        ),
                        float(weight),
                    )
                )
            expr = self._expr_weighted_mean(refs_with_weights)
        elif recipe_localization == "exact_term_centered_pm1":
            window_positions = self._prompt_window_positions(start, end, padding=1)
            if not window_positions:
                return None, {}
            metadata_positions = list(window_positions)
            positive_terms: list[tuple[dict[str, Any], float]] = []
            neighbor_terms: list[tuple[dict[str, Any], float]] = []
            for position in window_positions:
                base_expr = self._cache_ref_expr(
                    site=site,
                    layer=int(layer),
                    head=int(head),
                    token_selector={"mode": "index", "value": int(position)},
                )
                if start <= position < end:
                    positive_terms.append((base_expr, 1.0))
                else:
                    neighbor_terms.append((base_expr, 1.0))
            if not positive_terms:
                return None, {}
            positive_expr = self._expr_weighted_mean(positive_terms)
            if neighbor_terms:
                neighbor_expr = self._expr_weighted_mean(neighbor_terms)
                expr = {"fn": "sub", "args": [positive_expr, neighbor_expr]}
            else:
                expr = positive_expr
            recipe_pooling = "centered_mean"
        elif recipe_localization == "head_tail_mean":
            positions = [int(span_positions[0])]
            if int(span_positions[-1]) != int(span_positions[0]):
                positions.append(int(span_positions[-1]))
            metadata_positions = list(positions)
            expr = self._expr_weighted_mean(
                [
                    (
                        self._cache_ref_expr(
                            site=site,
                            layer=int(layer),
                            head=int(head),
                            token_selector={"mode": "index", "value": int(position)},
                        ),
                        1.0,
                    )
                    for position in positions
                ]
            )
        elif recipe_localization == "content_token_only_mean":
            content_records = [
                record for record in span_records if self._content_piece_score(str(record.get("piece", "")))[0] > 0.0
            ] or span_records
            metadata_positions = [int(record.get("position", start) or start) for record in content_records]
            expr = self._expr_weighted_mean(
                [
                    (
                        self._cache_ref_expr(
                            site=site,
                            layer=int(layer),
                            head=int(head),
                            token_selector={"mode": "index", "value": int(position)},
                        ),
                        1.0,
                    )
                    for position in metadata_positions
                ]
            )
        else:
            return None, {}

        if contrast_mode in {"minus_base", "minus_stealer"}:
            if not isinstance(competitor_candidate, Mapping):
                return None, {}
            competitor_expr, competitor_meta = self._candidate_recipe_source_expr(
                competitor_candidate,
                localization=localization,
                pooling=pooling,
                contrast_mode="none",
            )
            if competitor_expr is None:
                return None, {}
            metadata_positions = list(dict.fromkeys(metadata_positions + list(competitor_meta.get("source_positions", []))))
            competitor_term = (
                competitor_expr
                if abs(float(contrast_scale) - 1.0) <= 1e-8
                else {"fn": "scale", "by": float(contrast_scale), "arg": competitor_expr}
            )
            expr = {"fn": "sub", "args": [expr, competitor_term]}
        elif contrast_mode == "orthogonal_stealer":
            if not isinstance(competitor_candidate, Mapping):
                return None, {}
            competitor_expr, competitor_meta = self._candidate_recipe_source_expr(
                competitor_candidate,
                localization=localization,
                pooling=pooling,
                contrast_mode="none",
            )
            if competitor_expr is None:
                return None, {}
            metadata_positions = list(dict.fromkeys(metadata_positions + list(competitor_meta.get("source_positions", []))))
            expr = {"fn": "project_orthogonal", "arg": expr, "basis": competitor_expr}
        metadata = {
            "recipe_localization": recipe_localization,
            "recipe_pooling": recipe_pooling,
            "contrast_mode": str(contrast_mode or "none"),
            "contrast_scale": float(contrast_scale),
            "source_positions": metadata_positions,
        }
        return expr, metadata

    def _materialize_recipe_candidate(
        self,
        candidate: Mapping[str, Any],
        *,
        localization: str,
        pooling: str,
        contrast_mode: str = "none",
        competitor_candidate: Mapping[str, Any] | None = None,
        contrast_scale: float = 1.0,
        alpha_override: float | None = None,
    ) -> dict[str, Any] | None:
        expr, metadata = self._candidate_recipe_source_expr(
            candidate,
            localization=localization,
            pooling=pooling,
            contrast_mode=contrast_mode,
            competitor_candidate=competitor_candidate,
            contrast_scale=contrast_scale,
        )
        if expr is None:
            return None
        materialized = dict(candidate)
        source = dict(materialized.get("source", {}))
        source["dtype"] = "cache_pair"
        which = "v" if str(materialized.get("site", "") or "") == "v_cache" else "k"
        source[which] = expr
        materialized["source"] = source
        op = dict(materialized.get("op", {}))
        if alpha_override is not None:
            op["alpha"] = float(alpha_override)
        materialized["op"] = op
        materialized["recipe_localization"] = str(metadata.get("recipe_localization", localization))
        materialized["recipe_pooling"] = str(metadata.get("recipe_pooling", pooling))
        materialized["contrast_mode"] = str(metadata.get("contrast_mode", contrast_mode))
        materialized["recipe_contrast_scale"] = float(metadata.get("contrast_scale", contrast_scale))
        materialized["recipe_source_positions"] = list(metadata.get("source_positions", []))
        materialized["recipe_alpha"] = float(op.get("alpha", 0.0) or 0.0)
        materialized["operator_family_key"] = self._operator_family_key(materialized)
        materialized["operator_recipe_id"] = self._operator_recipe_id(materialized)
        return materialized

    def _prune_escape_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        debug: dict[str, Any],
    ) -> list[dict[str, Any]]:
        before_counts: dict[str, int] = {}
        for raw_candidate in candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            provenance_class = str(raw_candidate.get("provenance_class", "misc_prompt") or "misc_prompt")
            before_counts[provenance_class] = int(before_counts.get(provenance_class, 0) or 0) + 1
        debug["candidate_provenance_counts_before_prune"] = dict(before_counts)

        ranked_candidates = sorted(
            (dict(candidate) for candidate in candidates if isinstance(candidate, Mapping)),
            key=lambda item: (
                -self._candidate_builder_score(item, include_sidecar=False),
                int(item.get("surface_id", "") == ""),
                str(item.get("surface_id", "")),
            ),
        )
        selected: list[dict[str, Any]] = []
        seen_exact: set[tuple[str, str, str, str, str]] = set()
        kept_by_term_family: dict[tuple[str, str], dict[str, Any]] = {}
        same_term_family_drops = 0
        dominance_prune_drops = 0
        sidecar_veto_drops = 0

        for candidate in ranked_candidates:
            if self._candidate_sidecar_vetoed(candidate):
                sidecar_veto_drops += 1
                continue
            focus_term = str(candidate.get("focus_feature", "") or "")
            family_kind = self._candidate_family_kind(candidate)
            provenance_class = str(candidate.get("provenance_class", "misc_prompt") or "misc_prompt")
            span_kind = str(candidate.get("span_kind", "") or "")
            span_id = self._candidate_span_id(candidate)
            exact_key = (focus_term, provenance_class, span_id, family_kind, span_kind)
            if exact_key in seen_exact:
                same_term_family_drops += 1
                continue
            seen_exact.add(exact_key)
            term_family_key = (focus_term, family_kind)
            if term_family_key in kept_by_term_family:
                kept = kept_by_term_family[term_family_key]
                kept_provenance = str(kept.get("provenance_class", "misc_prompt") or "misc_prompt")
                if self._provenance_weight(provenance_class) < self._provenance_weight(kept_provenance):
                    dominance_prune_drops += 1
                    continue
                same_term_family_drops += 1
                continue
            kept_by_term_family[term_family_key] = candidate
            selected.append(candidate)

        bundle_inputs = self._bundle_inputs_for_candidates(selected)
        bundle_keys_by_candidate_group: dict[tuple[str, str, str], str] = {}
        for bundle in bundle_inputs:
            bundle_key = str(bundle.get("bundle_key", "") or "")
            term = str(bundle.get("term", "") or "")
            provenance_class = str(bundle.get("provenance_class", "misc_prompt") or "misc_prompt")
            source_span = bundle.get("source_span") if isinstance(bundle.get("source_span"), Mapping) else None
            if source_span is not None:
                span_id = f"{int(source_span.get('start', 0) or 0)}:{int(source_span.get('end', 0) or 0)}"
            else:
                span_id = str(bundle.get("span_id", "") or "")
            if term and bundle_key:
                bundle_keys_by_candidate_group[(term, provenance_class, span_id)] = bundle_key
        for candidate in selected:
            focus_term = str(candidate.get("focus_feature", "") or "")
            provenance_class = str(candidate.get("provenance_class", "misc_prompt") or "misc_prompt")
            span_id = self._candidate_span_id(candidate)
            bundle_key = bundle_keys_by_candidate_group.get((focus_term, provenance_class, span_id))
            if bundle_key:
                candidate["bundle_key"] = bundle_key
                candidate["bundle_ready"] = True
                candidate["bundle_family"] = "kv_pair_source_anchor"

        after_counts: dict[str, int] = {}
        for candidate in selected:
            provenance_class = str(candidate.get("provenance_class", "misc_prompt") or "misc_prompt")
            after_counts[provenance_class] = int(after_counts.get(provenance_class, 0) or 0) + 1
        debug["candidate_provenance_counts_after_prune"] = dict(after_counts)
        debug["dominance_prune_drops"] = int(dominance_prune_drops)
        debug["same_term_family_drops"] = int(same_term_family_drops)
        debug["sidecar_veto_drops"] = int(sidecar_veto_drops)
        debug["bundle_inputs"] = list(bundle_inputs)
        return selected

    def _ordered_missing_terms_for_phase(
        self,
        *,
        control_phase_hint: str,
        answer_readout_canary: Mapping[str, Any] | None = None,
        readout_sidecar_hints: Mapping[str, Any] | None = None,
        max_terms: int = 3,
    ) -> list[str]:
        missing_terms = self._feedback_terms(
            ("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms")
        )
        if not missing_terms:
            return []
        progress_by_term = self._feedback_term_progress_by_term()
        latest_tokenize = self._latest_tokenize_terms_result()
        allowed_terms = {str(term) for term in missing_terms}
        semantic_focus = (
            str(answer_readout_canary.get("semantic_focus_term", "") or "")
            if isinstance(answer_readout_canary, Mapping)
            else ""
        ) or self._semantic_focus_summary().get("semantic_focus_term")
        reachable_focus = (
            str(answer_readout_canary.get("reachable_focus_term", "") or "")
            if isinstance(answer_readout_canary, Mapping)
            else ""
        )
        easy_terms = self._intersect_terms(latest_tokenize.get("soft_logit_bias_ok_terms"), allowed_terms)
        exact_span_terms = [
            term for term in missing_terms if self._prompt_term_spans(term, max_spans=1)
        ]
        ordered: list[str] = []

        def _append(term: str | None) -> None:
            if term is None:
                return
            normalized = " ".join(str(term).split()).strip()
            if not normalized or normalized not in allowed_terms or normalized in ordered:
                return
            ordered.append(normalized)

        if control_phase_hint == "readout_escape":
            _append(reachable_focus)
            for term in exact_span_terms:
                if term == reachable_focus:
                    continue
                _append(term)
            for term in self._focus_terms_by_progress(easy_terms, progress_by_term, max_terms=len(easy_terms)):
                _append(term)
            _append(semantic_focus)
        else:
            _append(semantic_focus)
            _append(reachable_focus)
            for term in self._focus_terms_by_progress(easy_terms, progress_by_term, max_terms=len(easy_terms)):
                _append(term)
            for term in exact_span_terms:
                _append(term)
            for term in self._focus_terms_by_progress(missing_terms, progress_by_term, max_terms=len(missing_terms)):
                _append(term)

        if control_phase_hint == "readout_escape":
            for term in self._focus_terms_by_progress(missing_terms, progress_by_term, max_terms=len(missing_terms)):
                _append(term)
        return ordered[: max(1, int(max_terms))]

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

    def _source_bridge_shot_candidate_edits(
        self,
        *,
        avoid_surfaces: Sequence[str] = (),
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
        control_phase_hint: str = "monitor",
        answer_readout_canary: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        avoid = {str(surface_id) for surface_id in avoid_surfaces}
        activation_surfaces: dict[tuple[int, str], Any] = {}
        target_terms = self._ordered_missing_terms_for_phase(
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
            max_terms=2 if control_phase_hint == "readout_escape" else 3,
        )
        target_term_priority = {str(term): index for index, term in enumerate(target_terms)}
        for surface in self.surface_catalog:
            target = getattr(surface, "target", None)
            if getattr(target, "kind", None) != "activation" or getattr(target, "site", None) != "resid_pre":
                continue
            surface_id = str(surface.surface_id)
            if surface_id in avoid:
                continue
            token = getattr(target, "token", None)
            mode = getattr(token, "mode", None)
            if mode == "index" and getattr(token, "value", None) == -2:
                activation_surfaces[(int(getattr(target, "layer", 0) or 0), "prev")] = surface
            elif mode == "last":
                activation_surfaces[(int(getattr(target, "layer", 0) or 0), "last")] = surface

        candidates: list[tuple[int, int, int, float, dict[str, Any]]] = []
        seen_keys: set[tuple[str, str, int, int, str]] = set()
        for hit in self._actionable_kv_hits(limit=4, promoted_cache_surfaces=promoted_cache_surfaces):
            layer = hit.get("layer")
            if isinstance(layer, bool) or not isinstance(layer, int):
                continue
            feature = str(hit.get("feature", "") or "")
            if target_terms and feature and feature not in target_term_priority:
                continue
            source_position = self._kv_hit_source_position(hit)
            if source_position is None:
                continue
            recent_probe = hit.get("recent_probe") if isinstance(hit.get("recent_probe"), Mapping) else {}
            if str(recent_probe.get("label", "") or "") == "dead_actuator":
                continue
            chosen_surface = activation_surfaces.get((int(layer), "prev")) or activation_surfaces.get((int(layer), "last"))
            if chosen_surface is None:
                continue
            surface_id = str(chosen_surface.surface_id)
            target = getattr(chosen_surface, "target", None)
            surface_caps = getattr(chosen_surface, "caps", None)
            mode = "prev" if getattr(getattr(target, "token", None), "mode", None) == "index" else "last"
            norm_clip = 1.0 if getattr(surface_caps, "norm_clip", None) is None else float(surface_caps.norm_clip)
            alpha_cap = float(getattr(surface_caps, "max_alpha", 0.08) or 0.08)
            step_cap = getattr(surface_caps, "step_size", None)
            prompt_spans = self._prompt_term_spans(feature, max_spans=1)
            source_variants: list[dict[str, Any]] = []
            if control_phase_hint == "readout_escape":
                for span in prompt_spans:
                    selector = (
                        {"mode": "span", "start": int(span["start"]), "end": int(span["end"]), "pool": "mean"}
                        if int(span["length"]) > 1
                        else {"mode": "index", "value": int(span["start"])}
                    )
                    source_variants.append(
                        {
                            "variant_priority": 0,
                            "role": "shot_source_bridge_span_mean" if int(span["length"]) > 1 else f"shot_source_bridge_{mode}",
                            "span_kind": "exact_prompt_span_mean" if int(span["length"]) > 1 else "exact_prompt_piece",
                            "provenance_class": str(span.get("provenance_class", "misc_prompt") or "misc_prompt"),
                            "source_position": int(span["start"]),
                            "source_span": {"start": int(span["start"]), "end": int(span["end"])},
                            "source_piece": str(span["text"]),
                            "source_segment_kind": str(span["segment_kind"]),
                            "token_selector": selector,
                            "candidate_family": f"resid_bridge:{feature}:exact_prompt_span",
                            "phase_objective": "readout_escape",
                        }
                    )
            source_variants.append(
                {
                    "variant_priority": 1,
                    "role": f"shot_source_bridge_{mode}",
                    "span_kind": "source_position_single",
                    "provenance_class": "misc_prompt",
                    "source_position": int(source_position),
                    "source_span": None,
                    "source_piece": None if hit.get("argmax_piece") in (None, "") else str(hit.get("argmax_piece")),
                    "source_segment_kind": None
                    if hit.get("argmax_segment_kind") in (None, "")
                    else str(hit.get("argmax_segment_kind")),
                    "token_selector": {"mode": "index", "value": int(source_position)},
                    "candidate_family": f"resid_bridge:{feature}:source_position",
                    "phase_objective": "readout_escape" if control_phase_hint == "readout_escape" else "entity_insertion",
                }
            )

            for variant in source_variants:
                dedupe_key = (
                    surface_id,
                    str(feature),
                    int(variant["source_position"]),
                    int((variant.get("source_span") or {}).get("end", variant["source_position"] + 1)),
                    str(variant["span_kind"]),
                )
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                base_alpha = 0.035 if str(variant["span_kind"]).startswith("exact_prompt_span") else (0.04 if mode == "prev" else 0.045)
                alpha = min(alpha_cap, base_alpha)
                step_size = float(alpha if step_cap is None else min(float(step_cap), alpha))
                candidate = {
                    "surface_id": surface_id,
                    "kind": "resid_add",
                    "role": str(variant["role"]),
                    "focus_feature": feature,
                    "candidate_family": str(variant["candidate_family"]),
                    "phase_objective": str(variant["phase_objective"]),
                    "span_kind": str(variant["span_kind"]),
                    "provenance_class": str(variant.get("provenance_class", "misc_prompt") or "misc_prompt"),
                    "read_source_resolved": True,
                    "write_target_resolved": True,
                    "source_position": int(variant["source_position"]),
                    "source_piece": variant["source_piece"],
                    "source_segment_kind": variant["source_segment_kind"],
                    "recent_probe": dict(recent_probe),
                    "target": {"surface_id": surface_id},
                    "source": {
                        "dtype": "vector",
                        "expr": {
                            "fn": "clip_norm",
                            "max_norm": float(norm_clip),
                            "arg": {
                                "fn": "scale",
                                "by": float(step_size),
                                "arg": {
                                    "fn": "normalize",
                                    "arg": {
                                        "ref": {
                                            "scope": "runtime",
                                            "worker": self.worker_id,
                                            "tensor": "hidden",
                                            "layer": int(layer),
                                            "token": dict(variant["token_selector"]),
                                        }
                                    },
                                },
                            },
                        },
                    },
                    "op": {"kind": "resid_add", "alpha": float(alpha)},
                    "budget": {
                        "ttl_steps": 1,
                        "norm_clip": float(norm_clip),
                        "step_size": float(step_size),
                        "revertible": True,
                    },
                    "meta": {
                        "hypothesis": "source_position_residual_bridge",
                        "expected_effect": "source_anchor_readout_bridge",
                    },
                }
                if isinstance(variant.get("source_span"), Mapping):
                    candidate["source_span"] = dict(variant["source_span"])
                term_priority = int(target_term_priority.get(feature, 99))
                token_priority = 0 if str(variant["span_kind"]).startswith("exact_prompt_span") else (1 if mode == "prev" else 2)
                candidates.append(
                    (
                        term_priority,
                        token_priority + int(variant["variant_priority"]),
                        -self._provenance_weight(str(variant.get("provenance_class", "misc_prompt") or "misc_prompt")),
                        int(layer),
                        -float(hit.get("alignment", 0.0) or 0.0),
                        candidate,
                    )
                )
        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], str(item[5]["surface_id"])))
        return [candidate for _term_priority, _variant_priority, _provenance, _layer, _alignment, candidate in candidates[:3]]

    def _shot_candidate_edits(
        self,
        *,
        avoid_surfaces: Sequence[str] = (),
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
        control_phase_hint: str = "monitor",
        answer_readout_canary: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        avoid = {str(surface_id) for surface_id in avoid_surfaces}
        selected: list[dict[str, Any]] = []
        seen_surface_ids: set[str] = set()
        recent_probe_outcomes = self._recent_probe_outcomes_by_candidate_key(
            window=8,
            probe_families={"shot_bridge", "resid_add"},
        )
        anchored_selected: list[dict[str, Any]] = []
        for candidate in self._source_bridge_shot_candidate_edits(
            avoid_surfaces=avoid_surfaces,
            promoted_cache_surfaces=promoted_cache_surfaces,
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
        ):
            candidate_key = self._candidate_sidecar_key(candidate)
            if candidate_key in recent_probe_outcomes:
                candidate["recent_probe"] = dict(recent_probe_outcomes[candidate_key])
            recent_probe = candidate.get("recent_probe") if isinstance(candidate.get("recent_probe"), Mapping) else {}
            if str(recent_probe.get("label", "") or "") == "dead_actuator":
                continue
            surface_id = str(candidate.get("surface_id", "") or "")
            if not surface_id or surface_id in seen_surface_ids:
                continue
            anchored_selected.append(candidate)
            seen_surface_ids.add(surface_id)
        selected.extend(anchored_selected)
        if control_phase_hint == "readout_escape" and anchored_selected:
            return selected[:3]
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
        for _priority, _layer, candidate in candidates:
            candidate_key = self._candidate_sidecar_key(candidate)
            if candidate_key in recent_probe_outcomes:
                candidate["recent_probe"] = dict(recent_probe_outcomes[candidate_key])
            recent_probe = candidate.get("recent_probe") if isinstance(candidate.get("recent_probe"), Mapping) else {}
            if control_phase_hint == "readout_escape" and str(recent_probe.get("label", "") or "") == "dead_actuator":
                continue
            surface_id = str(candidate.get("surface_id", "") or "")
            if not surface_id or surface_id in seen_surface_ids:
                continue
            selected.append(candidate)
            seen_surface_ids.add(surface_id)
            if len(selected) >= 3:
                break
        for candidate in selected:
            candidate_key = self._candidate_sidecar_key(candidate)
            if candidate_key in recent_probe_outcomes:
                candidate["recent_probe"] = dict(recent_probe_outcomes[candidate_key])
        return selected[:3]

    def _shot_probe_needed(self, *, shot_mode_ready: bool) -> bool:
        if not shot_mode_ready:
            return False
        if max(0, self.max_tool_calls_per_run - len(self._tool_results)) <= 0:
            return False
        return self._recent_tool_result_count(("constraint_scorer", "dry_run_decode"), window=4) == 0

    def _preprobe_readout_escape_state(
        self,
        answer_readout_canary: Mapping[str, Any] | None,
    ) -> tuple[bool, dict[str, Any]]:
        if not isinstance(answer_readout_canary, Mapping):
            return False, {
                "mass_below_threshold": False,
                "rank_bad": False,
                "top20_hits_zero": False,
                "attractor_mass_high": False,
                "overlap_high": False,
                "preprobe_collapse": False,
            }
        target_top20_hits = int(answer_readout_canary.get("target_top20_hits", 0) or 0)
        target_mass = float(answer_readout_canary.get("target_mass", 0.0) or 0.0)
        reachable_focus_rank = answer_readout_canary.get("reachable_focus_rank")
        reachable_focus_rank_value = (
            10**9
            if isinstance(reachable_focus_rank, bool) or not isinstance(reachable_focus_rank, int)
            else int(reachable_focus_rank)
        )
        attractor_family_mass = float(answer_readout_canary.get("attractor_family_mass", 0.0) or 0.0)
        attractor_family_top_overlap = int(answer_readout_canary.get("attractor_family_top_overlap", 0) or 0)
        flags = {
            "mass_below_threshold": bool(target_mass < 0.001),
            "rank_bad": bool(reachable_focus_rank_value > 512),
            "top20_hits_zero": bool(target_top20_hits == 0),
            "attractor_mass_high": bool(attractor_family_mass > 0.20),
            "overlap_high": bool(attractor_family_top_overlap >= 2),
        }
        preprobe_collapse = (
            flags["top20_hits_zero"]
            and flags["mass_below_threshold"]
            and flags["rank_bad"]
            and flags["attractor_mass_high"]
        )
        return preprobe_collapse, flags | {"preprobe_collapse": bool(preprobe_collapse)}

    def _control_phase_hint(
        self,
        *,
        answer_readout_canary: Mapping[str, Any] | None = None,
    ) -> str:
        missing_terms = self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        if missing_terms:
            preprobe_collapse, _preprobe_flags = self._preprobe_readout_escape_state(answer_readout_canary)
            if preprobe_collapse:
                return "readout_escape"
            if self._loop_severity_hint() == "high":
                return "loop_break"
            if self._shot_mode_ready():
                return "shot_mode"
            return "entity_insertion"
        if self._loop_severity_hint() == "high":
            return "loop_break"
        if self._loop_severity_hint() == "low":
            return "loop_break"
        return "monitor"

    def _strategy_hints(
        self,
        *,
        control_phase_hint: str,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
        answer_readout_canary: Mapping[str, Any] | None = None,
        readout_sidecar_hints: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        latest_tokenize = self._latest_tokenize_terms_result()
        missing_terms = set(
            self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        )
        preprobe_collapse, preprobe_block_reason = self._preprobe_readout_escape_state(answer_readout_canary)
        easy_terms = self._intersect_terms(latest_tokenize.get("soft_logit_bias_ok_terms"), missing_terms)
        hard_terms = self._intersect_terms(latest_tokenize.get("needs_sequence_support_terms"), missing_terms)
        watch_terms = self._intersect_terms(latest_tokenize.get("span_progress_watch_terms"), missing_terms)
        prefer_auxiliary_entity_bias = bool(easy_terms) and self.decoder_control_mode == "logit_bias_entity_soft"
        shot_mode_ready = self._shot_mode_ready() or preprobe_collapse or control_phase_hint == "readout_escape"
        loop_break_attempt_count = self._recent_loop_break_attempt_count()
        stabilizing_only_count = self._recent_stabilizing_only_count()
        avoid_surfaces = ["s_resid_pre_l4_last"] if self._l4_term_nudge_cooldown_active() else []
        shot_candidate_edits = self._shot_candidate_edits(
            avoid_surfaces=avoid_surfaces,
            promoted_cache_surfaces=promoted_cache_surfaces,
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
        )
        kv_candidate_edits, kv_candidate_builder = self._kv_candidate_edits(
            promoted_cache_surfaces=promoted_cache_surfaces,
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
            readout_sidecar_hints=readout_sidecar_hints,
            canary_enabled=control_phase_hint in {"shot_mode", "readout_escape"} and not self._kv_canary_eval_active,
        )
        kv_candidate_edits, kv_effect_families = self._annotate_effect_families(kv_candidate_edits)
        kv_candidate_edits = self._attach_operator_certifications(kv_candidate_edits)
        kv_retry_candidate_edits = self._kv_retry_candidate_edits(
            base_candidates=kv_candidate_edits,
            canary_enabled=control_phase_hint in {"shot_mode", "readout_escape"} and not self._kv_canary_eval_active,
        )
        kv_retry_candidate_edits, kv_retry_effect_families = self._annotate_effect_families(kv_retry_candidate_edits)
        kv_retry_candidate_edits = self._attach_operator_certifications(kv_retry_candidate_edits)
        shot_probe_needed = False if preprobe_collapse else self._shot_probe_needed(shot_mode_ready=shot_mode_ready)
        kv_canary_checked = sum(1 for item in kv_candidate_edits if bool(item.get("canary_checked")))
        kv_canary_positive_count = sum(1 for item in kv_candidate_edits if bool(item.get("canary_pass")))
        kv_canary_rejected_count = sum(
            1 for item in kv_candidate_edits if bool(item.get("canary_checked")) and not bool(item.get("canary_pass"))
        )
        kv_retry_canary_checked = sum(1 for item in kv_retry_candidate_edits if bool(item.get("canary_checked")))
        kv_retry_positive_count = sum(1 for item in kv_retry_candidate_edits if bool(item.get("canary_pass")))
        semantic_focus = self._semantic_focus_summary(promoted_cache_surfaces=promoted_cache_surfaces)
        readout_escape_needed, readout_escape_reason, readout_escape_block_reason = self._readout_escape_needed(
            answer_readout_canary=answer_readout_canary,
            kv_candidate_edits=kv_candidate_edits,
        )
        sidecar_focus_term = (
            str(
                readout_sidecar_hints.get("suggested_focus_term", readout_sidecar_hints.get("focus_term_override", ""))
                or ""
            )
            if isinstance(readout_sidecar_hints, Mapping)
            else ""
        )
        controller_focus_term = (
            answer_readout_canary.get("reachable_focus_term")
            if readout_escape_needed and isinstance(answer_readout_canary, Mapping)
            else semantic_focus.get("semantic_focus_term")
        )
        controller_focus_source = "reachable_focus" if readout_escape_needed else "semantic_focus"
        hints = {
            "control_phase_hint": control_phase_hint,
            "loop_severity": self._loop_severity_hint(),
            "prefer_space_prefixed_logit_bias": bool(hard_terms or easy_terms),
            "prefer_auxiliary_entity_bias": prefer_auxiliary_entity_bias,
            "direct_entity_edit_gate": (
                "shot_mode_first"
                if control_phase_hint in {"shot_mode", "readout_escape"}
                else ("auxiliary_first" if prefer_auxiliary_entity_bias else "direct_edit_ok")
            ),
            "easy_entity_terms": easy_terms,
            "hard_entity_terms": hard_terms,
            "watch_span_progress_terms": watch_terms,
            "shot_mode_ready": shot_mode_ready,
            "loop_break_attempt_count": loop_break_attempt_count,
            "stabilizing_only_count": stabilizing_only_count,
            "shot_probe_needed": shot_probe_needed,
            "kv_probe_needed": bool(shot_probe_needed and kv_candidate_edits and kv_canary_checked == 0 and not readout_escape_needed),
            "kv_retry_needed": bool(control_phase_hint == "shot_mode" and kv_retry_candidate_edits),
            "shot_candidate_edits": shot_candidate_edits,
            "kv_candidate_edits": kv_candidate_edits,
            "kv_retry_candidate_edits": kv_retry_candidate_edits,
            "l4_term_nudge_cooldown": self._l4_term_nudge_cooldown_active(),
        }
        if isinstance(readout_sidecar_hints, Mapping) and readout_sidecar_hints:
            hints["readout_sidecar_active"] = True
            hints["readout_sidecar_hints"] = dict(readout_sidecar_hints)
            hints["readout_sidecar_report"] = dict(readout_sidecar_hints)
            hints["readout_analyzer_active"] = True
            hints["readout_analyzer_hints"] = dict(readout_sidecar_hints)
            hints["readout_analyzer_report"] = dict(readout_sidecar_hints)
            suggested_focus_term = readout_sidecar_hints.get(
                "suggested_focus_term",
                readout_sidecar_hints.get("focus_term_override"),
            )
            if suggested_focus_term not in (None, ""):
                hints["readout_sidecar_suggested_focus_term"] = suggested_focus_term
                hints["readout_analyzer_suggested_focus_term"] = suggested_focus_term
                hints["readout_sidecar_focus_term_override"] = suggested_focus_term
                hints["readout_analyzer_focus_term_override"] = suggested_focus_term
            suggested_bundle_key = readout_sidecar_hints.get("suggested_bundle_key")
            if suggested_bundle_key not in (None, ""):
                hints["readout_sidecar_suggested_bundle_key"] = str(suggested_bundle_key)
                hints["readout_analyzer_suggested_bundle_key"] = str(suggested_bundle_key)
        if semantic_focus:
            hints.update(semantic_focus)
        if controller_focus_term:
            hints["controller_focus_term"] = str(controller_focus_term)
            hints["controller_focus_source"] = str(controller_focus_source)
        if isinstance(answer_readout_canary, Mapping):
            hints["reachable_focus_term"] = answer_readout_canary.get("reachable_focus_term")
            hints["reachable_focus_rank"] = answer_readout_canary.get("reachable_focus_rank")
            hints["target_mass"] = answer_readout_canary.get("target_mass")
            hints["target_top20_hits"] = answer_readout_canary.get("target_top20_hits")
            hints["attractor_family_mass"] = answer_readout_canary.get("attractor_family_mass")
            hints["attractor_family_top_overlap"] = answer_readout_canary.get("attractor_family_top_overlap")
            hints["attractor_family_overlap_tokens"] = list(answer_readout_canary.get("attractor_family_overlap_tokens", [])[:5])
        if self._operator_certification_table:
            hints["operator_certification_count"] = len(self._operator_certification_table)
            hints["operator_certification_families"] = [
                {
                    "operator_family_key": str(key),
                    "certification_status": str(value.get("certification_status", "shadow_only") or "shadow_only"),
                    "certified_for_apply": bool(value.get("certified_for_apply", False)),
                    "family_prior_score": value.get("family_prior_score"),
                }
                for key, value in list(self._operator_certification_table.items())[:6]
            ]
        if self._operator_bridge_plan_table:
            hints["operator_bridge_plan_count"] = len(self._operator_bridge_plan_table)
            hints["operator_bridge_plan_objectives"] = [
                {
                    "objective_bundle_key": str(key),
                    "actuator_bundle_key": str(value.get("actuator_bundle_key", "") or ""),
                    "actuator_class": str(value.get("actuator_class", "") or ""),
                }
                for key, value in list(self._operator_bridge_plan_table.items())[:6]
                if isinstance(value, Mapping)
            ]
        hints["kv_effect_family_count"] = len([key for key in kv_effect_families if str(key)])
        hints["kv_effect_family_collapsed"] = bool(kv_candidate_edits) and len([key for key in kv_effect_families if str(key)]) < len(kv_candidate_edits)
        if kv_retry_candidate_edits:
            hints["kv_retry_effect_family_count"] = len([key for key in kv_retry_effect_families if str(key)])
        readout_escape_block_reason = dict(readout_escape_block_reason)
        readout_escape_block_reason["shot_mode_not_ready"] = bool(not shot_mode_ready)
        readout_escape_block_reason["no_candidates"] = bool(not kv_candidate_edits)
        hints["readout_escape_block_reason"] = readout_escape_block_reason
        if readout_escape_needed:
            hints["readout_escape_needed"] = True
            if readout_escape_reason:
                hints["readout_escape_reason"] = str(readout_escape_reason)
        hints["kv_candidate_builder_called"] = bool(kv_candidate_builder.get("called"))
        hints["kv_candidate_builder_source"] = kv_candidate_builder.get("source")
        hints["kv_candidate_builder_stage"] = kv_candidate_builder.get("stage")
        hints["kv_candidate_builder_input_count"] = int(kv_candidate_builder.get("input_hit_count", 0) or 0)
        hints["kv_candidate_builder_output_count"] = int(kv_candidate_builder.get("output_count", 0) or 0)
        hints["kv_candidate_builder_prune_reasons"] = list(kv_candidate_builder.get("prune_reasons", [])[:8])
        if isinstance(kv_candidate_builder.get("prune_reason_counts"), Mapping):
            hints["kv_candidate_builder_prune_reason_counts"] = dict(kv_candidate_builder["prune_reason_counts"])
        if control_phase_hint == "readout_escape":
            hints["escape_builder_called"] = True
            hints["escape_builder_target_terms"] = list(kv_candidate_builder.get("target_terms", [])[:4])
            hints["escape_builder_candidates_before_prune"] = int(
                kv_candidate_builder.get("candidate_count_before_select", 0) or 0
            )
            hints["escape_builder_candidates_after_prune"] = int(kv_candidate_builder.get("output_count", 0) or 0)
            hints["escape_builder_hit_limit"] = int(kv_candidate_builder.get("hit_limit", 0) or 0)
            hints["escape_builder_prune_reasons"] = list(kv_candidate_builder.get("prune_reasons", [])[:8])
            if isinstance(kv_candidate_builder.get("candidate_provenance_counts_before_prune"), Mapping):
                hints["candidate_provenance_counts_before_prune"] = dict(
                    kv_candidate_builder.get("candidate_provenance_counts_before_prune", {})
                )
            if isinstance(kv_candidate_builder.get("candidate_provenance_counts_after_prune"), Mapping):
                hints["candidate_provenance_counts_after_prune"] = dict(
                    kv_candidate_builder.get("candidate_provenance_counts_after_prune", {})
                )
            hints["dominance_prune_drops"] = int(kv_candidate_builder.get("dominance_prune_drops", 0) or 0)
            hints["same_term_family_drops"] = int(kv_candidate_builder.get("same_term_family_drops", 0) or 0)
            hints["sidecar_veto_drops"] = int(kv_candidate_builder.get("sidecar_veto_drops", 0) or 0)
            if isinstance(kv_candidate_builder.get("candidate_families"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("candidate_families"), (str, bytes, bytearray)
            ):
                hints["escape_builder_candidate_families"] = [
                    str(item) for item in kv_candidate_builder.get("candidate_families", [])[:6] if str(item)
                ]
            if isinstance(kv_candidate_builder.get("bundle_inputs"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("bundle_inputs"), (str, bytes, bytearray)
            ):
                hints["bundle_inputs"] = [
                    dict(item) for item in kv_candidate_builder.get("bundle_inputs", [])[:4] if isinstance(item, Mapping)
                ]
            if kv_candidate_builder.get("selected_bundle_key") not in (None, ""):
                hints["selected_bundle_key"] = str(kv_candidate_builder.get("selected_bundle_key"))
            if kv_candidate_builder.get("gate_report_frontier_bundle_key") not in (None, ""):
                hints["gate_report_frontier_bundle_key"] = str(kv_candidate_builder.get("gate_report_frontier_bundle_key"))
            hints["bundle_reorder_applied"] = bool(kv_candidate_builder.get("bundle_reorder_applied", False))
            if kv_candidate_builder.get("bundle_reorder_reason") not in (None, ""):
                hints["bundle_reorder_reason"] = str(kv_candidate_builder.get("bundle_reorder_reason"))
            hints["bundle_rerank_gate_open"] = bool(kv_candidate_builder.get("bundle_rerank_gate_open", False))
            hints["gate_report_open"] = bool(kv_candidate_builder.get("gate_report_open", False))
            if isinstance(kv_candidate_builder.get("bundle_rerank_gate_reasons"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("bundle_rerank_gate_reasons"), (str, bytes, bytearray)
            ):
                hints["bundle_rerank_gate_reasons"] = [
                    str(item) for item in kv_candidate_builder.get("bundle_rerank_gate_reasons", [])[:8] if str(item)
                ]
            if isinstance(kv_candidate_builder.get("gate_report_reasons"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("gate_report_reasons"), (str, bytes, bytearray)
            ):
                hints["gate_report_reasons"] = [
                    str(item) for item in kv_candidate_builder.get("gate_report_reasons", [])[:8] if str(item)
                ]
            if isinstance(kv_candidate_builder.get("rerank_vetoes"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("rerank_vetoes"), (str, bytes, bytearray)
            ):
                hints["rerank_vetoes"] = [str(item) for item in kv_candidate_builder.get("rerank_vetoes", [])[:8] if str(item)]
            if isinstance(kv_candidate_builder.get("gate_report_vetoes"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("gate_report_vetoes"), (str, bytes, bytearray)
            ):
                hints["gate_report_vetoes"] = [
                    str(item) for item in kv_candidate_builder.get("gate_report_vetoes", [])[:8] if str(item)
                ]
            if isinstance(kv_candidate_builder.get("gate_report_rejected_signals"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("gate_report_rejected_signals"), (str, bytes, bytearray)
            ):
                hints["gate_report_rejected_signals"] = [
                    str(item) for item in kv_candidate_builder.get("gate_report_rejected_signals", [])[:8] if str(item)
                ]
            if kv_candidate_builder.get("bundle_selector_phase") not in (None, ""):
                hints["bundle_selector_phase"] = str(kv_candidate_builder.get("bundle_selector_phase"))
            if kv_candidate_builder.get("bundle_rerank_mode") not in (None, ""):
                hints["bundle_rerank_mode"] = str(kv_candidate_builder.get("bundle_rerank_mode"))
            if kv_candidate_builder.get("rerank_gate_mode") not in (None, ""):
                hints["rerank_gate_mode"] = str(kv_candidate_builder.get("rerank_gate_mode"))
            if kv_candidate_builder.get("selection_source") not in (None, ""):
                hints["selection_source"] = str(kv_candidate_builder.get("selection_source"))
            if kv_candidate_builder.get("gate_report_selection_source") not in (None, ""):
                hints["gate_report_selection_source"] = str(kv_candidate_builder.get("gate_report_selection_source"))
            if kv_candidate_builder.get("base_winner_bundle_key") not in (None, ""):
                hints["base_winner_bundle_key"] = kv_candidate_builder.get("base_winner_bundle_key")
            if kv_candidate_builder.get("gate_report_base_winner_bundle_key") not in (None, ""):
                hints["gate_report_base_winner_bundle_key"] = kv_candidate_builder.get("gate_report_base_winner_bundle_key")
            if kv_candidate_builder.get("challenger_bundle_key") not in (None, ""):
                hints["challenger_bundle_key"] = kv_candidate_builder.get("challenger_bundle_key")
            if kv_candidate_builder.get("gate_report_challenger_bundle_key") not in (None, ""):
                hints["gate_report_challenger_bundle_key"] = kv_candidate_builder.get("gate_report_challenger_bundle_key")
            if kv_candidate_builder.get("bundle_base_gap") not in (None, ""):
                hints["bundle_base_gap"] = kv_candidate_builder.get("bundle_base_gap")
            if kv_candidate_builder.get("base_gap_norm") not in (None, ""):
                hints["base_gap_norm"] = kv_candidate_builder.get("base_gap_norm")
            if kv_candidate_builder.get("bundle_rerank_gap") not in (None, ""):
                hints["bundle_rerank_gap"] = kv_candidate_builder.get("bundle_rerank_gap")
            if kv_candidate_builder.get("hard_margin") not in (None, ""):
                hints["hard_margin"] = kv_candidate_builder.get("hard_margin")
            if kv_candidate_builder.get("soft_margin") not in (None, ""):
                hints["soft_margin"] = kv_candidate_builder.get("soft_margin")
            if kv_candidate_builder.get("pairwise_delta") not in (None, ""):
                hints["pairwise_delta"] = kv_candidate_builder.get("pairwise_delta")
            if isinstance(kv_candidate_builder.get("pairwise_margin_breakdown"), Mapping):
                hints["pairwise_margin_breakdown"] = dict(kv_candidate_builder.get("pairwise_margin_breakdown", {}))
            if isinstance(kv_candidate_builder.get("common_mode_evidence"), Mapping):
                hints["common_mode_evidence"] = dict(kv_candidate_builder.get("common_mode_evidence", {}))
            if isinstance(kv_candidate_builder.get("support_common_vs_discriminative"), Mapping):
                hints["support_common_vs_discriminative"] = dict(
                    kv_candidate_builder.get("support_common_vs_discriminative", {})
                )
            if kv_candidate_builder.get("controller_pairwise_reason_text") not in (None, ""):
                hints["controller_pairwise_reason_text"] = str(kv_candidate_builder.get("controller_pairwise_reason_text"))
            if kv_candidate_builder.get("gate_report_pairwise_reason_text") not in (None, ""):
                hints["gate_report_pairwise_reason_text"] = str(kv_candidate_builder.get("gate_report_pairwise_reason_text"))
            bridge_plan_key = ""
            for candidate_key in (
                kv_candidate_builder.get("selected_bundle_key"),
                kv_candidate_builder.get("gate_report_frontier_bundle_key"),
                kv_candidate_builder.get("base_winner_bundle_key"),
                kv_candidate_builder.get("challenger_bundle_key"),
            ):
                if candidate_key in (None, ""):
                    continue
                candidate_key_text = str(candidate_key)
                if candidate_key_text in self._operator_bridge_plan_table:
                    bridge_plan_key = candidate_key_text
                    break
            selected_bundle_focus_term = ""
            bundle_score_debug = kv_candidate_builder.get("bundle_score_debug")
            if isinstance(bundle_score_debug, SequenceABC) and not isinstance(bundle_score_debug, (str, bytes, bytearray)):
                for item in bundle_score_debug:
                    if not isinstance(item, Mapping):
                        continue
                    bundle_key_text = str(item.get("bundle_key", "") or "")
                    if bridge_plan_key and bundle_key_text == bridge_plan_key:
                        selected_bundle_focus_term = str(item.get("focus_term", "") or "")
                        break
            bridge_plan_report = self._bridge_plan_report_for_bundle(
                bridge_plan_key,
                objective_term=selected_bundle_focus_term or controller_focus_term,
            )
            if bridge_plan_report is not None:
                hints["bridge_plan_available"] = True
                hints["bridge_plan_required"] = bool(bridge_plan_report.get("bridge_required", True))
                hints["bridge_plan_report"] = dict(bridge_plan_report)
                if bridge_plan_report.get("objective_bundle_key") not in (None, ""):
                    hints["bridge_plan_objective_bundle_key"] = str(bridge_plan_report.get("objective_bundle_key"))
                if bridge_plan_report.get("objective_term") not in (None, ""):
                    hints["bridge_plan_objective_term"] = str(bridge_plan_report.get("objective_term"))
                if bridge_plan_report.get("actuator_bundle_key") not in (None, ""):
                    hints["bridge_plan_actuator_bundle_key"] = str(bridge_plan_report.get("actuator_bundle_key"))
                if bridge_plan_report.get("actuator_term") not in (None, ""):
                    hints["bridge_plan_actuator_term"] = str(bridge_plan_report.get("actuator_term"))
                if bridge_plan_report.get("operator_recipe_id") not in (None, ""):
                    hints["bridge_plan_recipe_id"] = str(bridge_plan_report.get("operator_recipe_id"))
                if bridge_plan_report.get("actuator_class") not in (None, ""):
                    hints["bridge_plan_actuator_class"] = str(bridge_plan_report.get("actuator_class"))
                if bridge_plan_report.get("bridge_plan_reason") not in (None, ""):
                    hints["bridge_plan_reason"] = str(bridge_plan_report.get("bridge_plan_reason"))
            if isinstance(kv_candidate_builder.get("bundle_score_debug"), SequenceABC) and not isinstance(
                kv_candidate_builder.get("bundle_score_debug"), (str, bytes, bytearray)
            ):
                hints["bundle_score_debug"] = [
                    dict(item) for item in kv_candidate_builder.get("bundle_score_debug", [])[:5] if isinstance(item, Mapping)
                ]
        if kv_canary_checked > 0:
            hints["kv_canary_checked"] = kv_canary_checked
            hints["kv_canary_positive_count"] = kv_canary_positive_count
            hints["kv_canary_rejected_count"] = kv_canary_rejected_count
        if kv_retry_canary_checked > 0:
            hints["kv_retry_canary_checked"] = kv_retry_canary_checked
            hints["kv_retry_positive_count"] = kv_retry_positive_count
        dead_kv_surface_ids = [
            str(item.get("surface_id", "") or "")
            for item in kv_candidate_edits
            if str(((item.get("recent_probe") or {}).get("label", "") if isinstance(item.get("recent_probe"), Mapping) else "") or "")
            == "dead_actuator"
        ]
        if dead_kv_surface_ids:
            hints["dead_kv_surface_ids"] = dead_kv_surface_ids
        if hints["l4_term_nudge_cooldown"]:
            hints["avoid_recall_surfaces"] = list(avoid_surfaces)
        if control_phase_hint == "loop_break":
            hints["phase_policy"] = "Prefer loop relief, patience, or observer/tool checks before entity insertion edits."
        elif control_phase_hint == "readout_escape":
            hints["phase_policy"] = (
                "First-token readout is already collapsed into a junk attractor; prefer escape-oriented moves, patience, or noop before another recall probe."
            )
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
            if kv_retry_candidate_edits:
                passing_retry_candidates = [item for item in kv_retry_candidate_edits if bool(item.get("canary_pass"))]
                hints["preferred_kv_retry_surface_id"] = str(kv_retry_candidate_edits[0]["surface_id"])
                if passing_retry_candidates:
                    hints["preferred_kv_surface_id"] = str(passing_retry_candidates[0]["surface_id"])
                    hints["phase_policy"] = (
                        "A prior kv probe was weak-but-promising on first-token rank or target mass; prefer the bounded retry candidate before inventing a fresh blind kv apply."
                    )
                else:
                    hints["phase_policy"] = (
                        "A prior kv probe showed weak first-token readout progress; prefer the bounded retry candidate or noop over switching to a totally new blind kv surface."
                    )
            if readout_escape_needed:
                hints["phase_policy"] = (
                    "Recent kv probes only nudged first-token reachability below threshold while the same attractor family still dominates; stop repeating recall probes and prefer escape-oriented moves, patience, or noop."
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

    def _latest_kv_feature_scan(
        self,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        if not isinstance(self._latest_observer_check, Mapping):
            return {}
        normalized = self._observer_check_with_cache_surface_ids(
            self._latest_observer_check,
            promoted_cache_surfaces=promoted_cache_surfaces,
        )
        value = None if normalized is None else normalized.get("kv_feature_scan")
        if not isinstance(value, Mapping):
            return {}
        return value

    def _kv_feature_hits(
        self,
        value: Mapping[str, Any] | None = None,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        scan = (
            value
            if isinstance(value, Mapping)
            else self._latest_kv_feature_scan(promoted_cache_surfaces=promoted_cache_surfaces)
        )
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
                    if not isinstance(item, Mapping):
                        continue
                    _push(item, group=group, polarity=polarity)
                    surface_hits = item.get("surface_hits")
                    if not isinstance(surface_hits, SequenceABC) or isinstance(surface_hits, (str, bytes, bytearray)):
                        continue
                    for surface_hit in surface_hits:
                        if not isinstance(surface_hit, Mapping):
                            continue
                        expanded = dict(item)
                        expanded.update(surface_hit)
                        _push(expanded, group=group, polarity=polarity)

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

    def _kv_feature_site_preferences(
        self,
        *,
        missing_terms: set[str] | None = None,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, str]:
        term_sites: dict[str, dict[str, float]] = {}
        for hit in self._kv_feature_hits(promoted_cache_surfaces=promoted_cache_surfaces):
            if str(hit.get("polarity", "promote") or "promote") != "promote":
                continue
            if str(hit.get("group", "") or "").startswith("forbidden"):
                continue
            feature = str(hit.get("feature", "") or "")
            if not feature:
                continue
            if missing_terms is not None and feature not in missing_terms:
                continue
            site = str(hit.get("site", "") or "")
            if site not in {"k_cache", "v_cache"}:
                continue
            alignment = float(hit.get("alignment", 0.0) or 0.0)
            site_scores = term_sites.setdefault(feature, {})
            site_scores[site] = max(float(site_scores.get(site, float("-inf"))), alignment)

        preferences: dict[str, str] = {}
        for feature, site_scores in term_sites.items():
            k_score = site_scores.get("k_cache")
            v_score = site_scores.get("v_cache")
            if k_score is None and v_score is None:
                continue
            if k_score is None:
                preferences[feature] = "v_cache"
                continue
            if v_score is None:
                preferences[feature] = "k_cache"
                continue
            if float(k_score) >= float(v_score) + 0.01:
                preferences[feature] = "k_cache"
            elif float(v_score) >= float(k_score) + 0.01:
                preferences[feature] = "v_cache"
            else:
                preferences[feature] = "balanced"
        return preferences

    def _actionable_kv_hits(
        self,
        *,
        limit: int = 3,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        missing_terms = set(
            self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        )
        if not missing_terms:
            return []
        site_preferences = self._kv_feature_site_preferences(
            missing_terms=missing_terms,
            promoted_cache_surfaces=promoted_cache_surfaces,
        )
        recent_probe_outcomes = self._recent_kv_probe_outcomes()
        ranked_hits: list[dict[str, Any]] = []

        def _group_priority(item: Mapping[str, Any]) -> int:
            group = str(item.get("group", "") or "")
            if group == "required_terms":
                return 0
            if group == "missing_keywords":
                return 1
            if group == "missing_summary_terms":
                return 2
            return 3

        def _site_role(site: str) -> str:
            return "retrieval_query_probe" if site == "k_cache" else "content_value_probe"

        for raw_hit in self._kv_feature_hits(promoted_cache_surfaces=promoted_cache_surfaces):
            if str(raw_hit.get("polarity", "promote") or "promote") != "promote":
                continue
            if str(raw_hit.get("group", "") or "").startswith("forbidden"):
                continue
            feature = str(raw_hit.get("feature", "") or "")
            if feature and feature not in missing_terms:
                continue
            site = str(raw_hit.get("site", "") or "")
            if site not in {"k_cache", "v_cache"}:
                continue
            normalized = dict(raw_hit)
            preferred_site = site_preferences.get(feature, "balanced")
            normalized["site_preference"] = preferred_site
            normalized["site_role"] = _site_role(site)
            surface_id = str(normalized.get("surface_id", "") or "")
            if surface_id and surface_id in recent_probe_outcomes:
                normalized["recent_probe"] = dict(recent_probe_outcomes[surface_id])
            ranked_hits.append(normalized)

        ranked_hits.sort(
            key=lambda item: (
                _group_priority(item),
                0 if str(item.get("polarity", "promote") or "promote") == "promote" else 1,
                -float(((item.get("recent_probe") or {}).get("score", 0.0) if isinstance(item.get("recent_probe"), Mapping) else 0.0)),
                0
                if str(item.get("site_preference", "balanced")) in {"k_cache", "v_cache"}
                and str(item.get("site")) == str(item.get("site_preference"))
                else 1,
                float(item.get("coverage_progress", 0.0) or 0.0),
                -float(item.get("alignment", 0.0) or 0.0),
                0 if str(item.get("site", "")) == "k_cache" else 1,
                int(item.get("layer", 0) or 0),
                int(item.get("head", 0) or 0),
            )
        )

        selected: list[dict[str, Any]] = []
        used_keys: set[tuple[str, int, int, str]] = set()

        def _append_if_fresh(item: Mapping[str, Any]) -> bool:
            key = (
                str(item.get("site", "") or ""),
                int(item.get("layer", 0) or 0),
                int(item.get("head", 0) or 0),
                str(item.get("token_mode", "last") or "last"),
            )
            if key in used_keys:
                return False
            used_keys.add(key)
            selected.append(dict(item))
            return True

        if ranked_hits:
            _append_if_fresh(ranked_hits[0])
        selected_sites = {str(item.get("site", "") or "") for item in selected}
        for required_site in ("k_cache", "v_cache"):
            if len(selected) >= max(1, int(limit)):
                break
            if required_site in selected_sites:
                continue
            for item in ranked_hits:
                if str(item.get("site", "") or "") != required_site:
                    continue
                if _append_if_fresh(item):
                    selected_sites.add(required_site)
                    break

        for item in ranked_hits:
            if len(selected) >= max(1, int(limit)):
                break
            _append_if_fresh(item)
        return selected[: max(1, int(limit))]

    def _kv_hit_source_position(self, hit: Mapping[str, Any]) -> int | None:
        source_position = hit.get("argmax_pos")
        if isinstance(source_position, int) and not isinstance(source_position, bool):
            return int(source_position)
        source_positions = hit.get("source_positions")
        if isinstance(source_positions, SequenceABC) and not isinstance(source_positions, (str, bytes, bytearray)):
            for item in source_positions:
                if not isinstance(item, Mapping):
                    continue
                candidate_pos = item.get("position")
                if isinstance(candidate_pos, int) and not isinstance(candidate_pos, bool):
                    return int(candidate_pos)
        return None

    def _semantic_focus_summary(
        self,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        missing_terms = self._feedback_terms(
            ("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms")
        )
        if not missing_terms:
            return {}
        normalized = self._observer_check_with_cache_surface_ids(
            self._latest_observer_check,
            promoted_cache_surfaces=promoted_cache_surfaces,
        )
        if isinstance(normalized, Mapping):
            for scan_name in ("kv_feature_scan", "latent_feature_scan"):
                scan = normalized.get(scan_name)
                if not isinstance(scan, Mapping):
                    continue
                top_hits = scan.get("top_feature_hits")
                if not isinstance(top_hits, SequenceABC) or isinstance(top_hits, (str, bytes, bytearray)):
                    continue
                for item in top_hits:
                    if not isinstance(item, Mapping):
                        continue
                    feature = str(item.get("feature", "") or "")
                    if feature and feature in missing_terms:
                        return {
                            "semantic_focus_term": feature,
                            "semantic_focus_source": str(scan_name),
                            "semantic_focus_alignment": round(float(item.get("alignment", 0.0) or 0.0), 6),
                        }
        return {
            "semantic_focus_term": str(missing_terms[0]),
            "semantic_focus_source": "missing_terms_fallback",
        }

    def _annotate_effect_families(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        family_counts: dict[str, int] = {}
        annotated: list[dict[str, Any]] = []
        for raw_candidate in candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            recent_probe = raw_candidate.get("recent_probe") if isinstance(raw_candidate.get("recent_probe"), Mapping) else {}
            family_key = str(
                recent_probe.get("effect_family_key")
                or raw_candidate.get("effect_family_key")
                or raw_candidate.get("surface_id")
                or "unknown_family"
            )
            family_counts[family_key] = int(family_counts.get(family_key, 0) or 0) + 1
            candidate = dict(raw_candidate)
            candidate["effect_family_key"] = family_key
            signature = recent_probe.get("effect_family_signature")
            if isinstance(signature, SequenceABC) and not isinstance(signature, (str, bytes, bytearray)):
                candidate["effect_family_signature"] = [str(item) for item in signature[:4] if str(item)]
            annotated.append(candidate)
        for candidate in annotated:
            family_key = str(candidate.get("effect_family_key", "") or "unknown_family")
            candidate["effect_family_size"] = int(family_counts.get(family_key, 1) or 1)
            candidate["effect_family_collapsed_member"] = bool(int(candidate["effect_family_size"]) > 1)
        return annotated, family_counts

    def _readout_escape_needed(
        self,
        *,
        answer_readout_canary: Mapping[str, Any] | None,
        kv_candidate_edits: Sequence[Mapping[str, Any]],
    ) -> tuple[bool, str | None, dict[str, Any]]:
        preprobe_collapse, block_reason = self._preprobe_readout_escape_state(answer_readout_canary)
        if not isinstance(answer_readout_canary, Mapping):
            return False, None, dict(block_reason)
        if preprobe_collapse:
            block_reason["needs_probe_history"] = False
            block_reason["probe_history_subthreshold_count"] = 0
            block_reason["probe_history_actionable_count"] = 0
            block_reason["candidate_family_count"] = len(
                [
                    str(candidate.get("effect_family_key", "") or "")
                    for candidate in kv_candidate_edits
                    if isinstance(candidate, Mapping) and str(candidate.get("effect_family_key", "") or "")
                ]
            )
            block_reason["no_candidates"] = bool(not kv_candidate_edits)
            return True, "preprobe_first_token_collapse", dict(block_reason)
        history = self._recent_probe_history(window=6, phase_profiles={"readout_escape"})
        actionable_count = sum(1 for item in history if str(item.get("label", "") or "") in {"positive", "actionable_positive"})
        subthreshold_count = sum(1 for item in history if str(item.get("label", "") or "") == "weak_positive_subthreshold")
        block_reason["needs_probe_history"] = bool(actionable_count <= 0 and subthreshold_count < 2)
        block_reason["probe_history_subthreshold_count"] = int(subthreshold_count)
        block_reason["probe_history_actionable_count"] = int(actionable_count)
        if actionable_count > 0 or subthreshold_count < 2:
            return False, None, dict(block_reason)
        target_top20_hits = int(answer_readout_canary.get("target_top20_hits", 0) or 0)
        target_mass = float(answer_readout_canary.get("target_mass", 0.0) or 0.0)
        reachable_focus_rank = answer_readout_canary.get("reachable_focus_rank")
        reachable_focus_rank_value = (
            10**9
            if isinstance(reachable_focus_rank, bool) or not isinstance(reachable_focus_rank, int)
            else int(reachable_focus_rank)
        )
        attractor_family_mass = float(answer_readout_canary.get("attractor_family_mass", 0.0) or 0.0)
        attractor_family_top_overlap = int(answer_readout_canary.get("attractor_family_top_overlap", 0) or 0)
        if target_top20_hits > 0 or target_mass >= 0.001 or reachable_focus_rank_value <= 150:
            return False, None, dict(block_reason)
        if attractor_family_top_overlap < 2 and attractor_family_mass < max(0.0005, target_mass * 2.0):
            return False, None, dict(block_reason)
        family_keys = {
            str(candidate.get("effect_family_key", "") or "")
            for candidate in kv_candidate_edits
            if isinstance(candidate, Mapping)
        }
        block_reason["candidate_family_count"] = len([key for key in family_keys if key])
        block_reason["no_candidates"] = bool(not kv_candidate_edits)
        if len([key for key in family_keys if key]) <= 1:
            return True, "collapsed_decode_basin", dict(block_reason)
        return True, "subthreshold_readout_stall", dict(block_reason)

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
        missing_terms = self._feedback_terms(
            ("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms")
        )
        promoted_limit = 4 if len(missing_terms) >= 2 else 3
        for hit in self._actionable_kv_hits(limit=promoted_limit):
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
            if len(promoted) >= promoted_limit:
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

    def _kv_source_variants_for_hit(
        self,
        hit: Mapping[str, Any],
        *,
        control_phase_hint: str,
    ) -> list[dict[str, Any]]:
        feature = str(hit.get("feature", "") or "")
        source_position = self._kv_hit_source_position(hit)
        variants: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int, int]] = set()
        if control_phase_hint == "readout_escape":
            for span in self._prompt_term_spans(feature, max_spans=2):
                start = int(span["start"])
                end = int(span["end"])
                span_key = ("span", start, end)
                if span_key in seen_keys:
                    continue
                seen_keys.add(span_key)
                variants.append(
                    {
                        "variant_priority": 0,
                        "span_kind": "exact_prompt_span_mean" if int(span["length"]) > 1 else "exact_prompt_piece",
                        "provenance_class": str(span.get("provenance_class", "misc_prompt") or "misc_prompt"),
                        "token_selector": (
                            {"mode": "span", "start": start, "end": end, "pool": "mean"}
                            if int(span["length"]) > 1
                            else {"mode": "index", "value": start}
                        ),
                        "source_position": start,
                        "source_span": {"start": start, "end": end},
                        "source_piece": str(span["text"]),
                        "source_segment_kind": str(span["segment_kind"]),
                        "candidate_family": f"kv_anchor:{feature}:exact_prompt_span",
                        "phase_objective": "readout_escape",
                    }
                )
        if source_position is not None:
            single_key = ("index", int(source_position), int(source_position) + 1)
            if single_key not in seen_keys:
                seen_keys.add(single_key)
                variants.append(
                    {
                        "variant_priority": 1,
                        "span_kind": "source_position_single",
                        "provenance_class": "misc_prompt",
                        "token_selector": {"mode": "index", "value": int(source_position)},
                        "source_position": int(source_position),
                        "source_span": None,
                        "source_piece": None if hit.get("argmax_piece") in (None, "") else str(hit.get("argmax_piece")),
                        "source_segment_kind": (
                            None
                            if hit.get("argmax_segment_kind") in (None, "")
                            else str(hit.get("argmax_segment_kind"))
                        ),
                        "candidate_family": f"kv_anchor:{feature}:source_position",
                        "phase_objective": "readout_escape" if control_phase_hint == "readout_escape" else "shot_mode",
                    }
                )
        return variants

    def _kv_candidate_edits(
        self,
        *,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
        control_phase_hint: str = "monitor",
        answer_readout_canary: Mapping[str, Any] | None = None,
        readout_sidecar_hints: Mapping[str, Any] | None = None,
        canary_enabled: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        debug: dict[str, Any] = {
            "called": True,
            "source": "runtime",
            "stage": "kv_candidate_builder",
            "input_hit_count": 0,
            "hit_limit": 0,
            "candidate_count_before_select": 0,
            "output_count": 0,
            "prune_reasons": [],
            "prune_reason_counts": {},
            "target_terms": [],
            "candidate_families": [],
            "candidate_provenance_counts_before_prune": {},
            "candidate_provenance_counts_after_prune": {},
            "dominance_prune_drops": 0,
            "same_term_family_drops": 0,
            "sidecar_veto_drops": 0,
            "bundle_inputs": [],
            "bundle_score_debug": [],
            "selected_bundle_key": None,
            "bundle_reorder_applied": False,
            "bundle_reorder_reason": "not_applicable",
            "bundle_rerank_gate_open": False,
            "bundle_rerank_gate_reasons": [],
        }

        def _note_prune(reason: str) -> None:
            counts = debug["prune_reason_counts"]
            counts[str(reason)] = int(counts.get(str(reason), 0) or 0) + 1

        missing_term_list = self._ordered_missing_terms_for_phase(
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
            readout_sidecar_hints=readout_sidecar_hints,
            max_terms=2 if control_phase_hint == "readout_escape" else 3,
        )
        debug["target_terms"] = list(missing_term_list)
        missing_terms = set(
            self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        )
        if not missing_terms:
            _note_prune("no_missing_terms")
            debug["prune_reasons"] = sorted(debug["prune_reason_counts"])
            return [], debug
        surface_lookup = self._cache_surface_lookup(promoted_cache_surfaces=promoted_cache_surfaces)
        term_priority = {str(term): index for index, term in enumerate(missing_term_list)}
        candidates: list[tuple[int, int, int, int, float, int, int, dict[str, Any]]] = []
        seen_keys: set[tuple[str, str, int, int, str]] = set()
        hit_limit = 3
        if control_phase_hint == "readout_escape":
            target_term_count = max(1, len(missing_term_list) or 1)
            hit_limit = min(6, max(4, 2 * target_term_count))
        debug["hit_limit"] = int(hit_limit)
        actionable_hits = self._actionable_kv_hits(limit=hit_limit, promoted_cache_surfaces=promoted_cache_surfaces)
        recent_probe_by_candidate_key = self._recent_probe_outcomes_by_candidate_key(
            window=8,
            probe_families={"kv_v", "kv_k", "kv_mix"},
        )
        debug["input_hit_count"] = len(actionable_hits)
        if not actionable_hits:
            _note_prune("no_feature_hits")
        for hit in actionable_hits:
            if str(hit.get("polarity", "promote") or "promote") != "promote":
                _note_prune("non_promote_hit")
                continue
            if str(hit.get("group", "") or "").startswith("forbidden"):
                _note_prune("forbidden_group")
                continue
            feature = str(hit.get("feature", "") or "")
            if feature and feature not in missing_terms:
                _note_prune("feature_not_missing")
                continue
            if term_priority and feature and feature not in term_priority:
                _note_prune("feature_not_selected")
                continue
            site = str(hit.get("site", "") or "")
            token_mode = str(hit.get("token_mode", "last") or "last")
            layer = hit.get("layer")
            head = hit.get("head")
            if site not in {"k_cache", "v_cache"} or token_mode != "last":
                _note_prune("unsupported_site_or_token_mode")
                continue
            if isinstance(layer, bool) or not isinstance(layer, int):
                _note_prune("invalid_layer")
                continue
            if isinstance(head, bool) or not isinstance(head, int):
                _note_prune("invalid_head")
                continue
            record = surface_lookup.get((site, int(layer), int(head), token_mode))
            if record is None:
                _note_prune("surface_lookup_miss")
                continue
            surface_id = str(record.get("surface_id", "") or "")
            if not surface_id:
                _note_prune("surface_lookup_miss")
                continue
            recent_probe = hit.get("recent_probe") if isinstance(hit.get("recent_probe"), Mapping) else {}
            if str(recent_probe.get("label", "") or "") == "dead_actuator":
                _note_prune("dead_actuator")
                continue
            source_variants = self._kv_source_variants_for_hit(hit, control_phase_hint=control_phase_hint)
            if not source_variants:
                _note_prune("missing_source_position")
                continue
            max_alpha = float(record.get("max_alpha", 0.06) or 0.06)
            step_cap = record.get("step_size")
            norm_clip = record.get("norm_clip")
            if norm_clip is None:
                norm_clip = 1.0
            which = "v" if site == "v_cache" else "k"
            for variant in source_variants:
                source_span = variant.get("source_span") if isinstance(variant.get("source_span"), Mapping) else None
                source_position = int(variant.get("source_position", 0) or 0)
                source_end = source_position + 1 if source_span is None else int(source_span.get("end", source_position + 1))
                dedupe_key = (
                    surface_id,
                    str(feature),
                    source_position,
                    source_end,
                    str(variant.get("span_kind", "")),
                )
                if dedupe_key in seen_keys:
                    _note_prune("duplicate_surface_variant")
                    continue
                seen_keys.add(dedupe_key)
                base_alpha = 0.03 if site == "k_cache" else 0.04
                if str(variant.get("span_kind", "")).startswith("exact_prompt_span"):
                    base_alpha = 0.035 if site == "k_cache" else 0.045
                alpha = min(max_alpha, base_alpha)
                step_size = float(alpha if step_cap is None else min(float(step_cap), alpha))
                source_expr = {
                    "ref": {
                        "scope": "runtime",
                        "worker": self.worker_id,
                        "tensor": site,
                        "layer": int(layer),
                        "head": int(head),
                        "token": dict(variant["token_selector"]),
                    }
                }
                candidate = {
                    "surface_id": surface_id,
                    "kind": "kv_mix",
                    "role": f"kv_shot_{which}_{'span_anchor' if str(variant.get('span_kind', '')).startswith('exact_prompt_span') else 'source_anchor'}",
                    "site": site,
                    "layer": int(layer),
                    "head": int(head),
                    "token_mode": token_mode,
                    "focus_feature": feature,
                    "alignment": round(float(hit.get("alignment", 0.0) or 0.0), 6),
                    "candidate_family": str(variant.get("candidate_family", "")),
                    "phase_objective": str(variant.get("phase_objective", "shot_mode")),
                    "target_term_priority": int(term_priority.get(feature, 99)),
                    "span_kind": str(variant.get("span_kind", "source_position_single")),
                    "recipe_localization": str(variant.get("span_kind", "source_position_single")),
                    "recipe_pooling": "mean"
                    if str(variant.get("span_kind", "")).startswith("exact_prompt_span")
                    else "single",
                    "contrast_mode": "none",
                    "provenance_class": str(variant.get("provenance_class", "misc_prompt") or "misc_prompt"),
                    "read_source_resolved": True,
                    "write_target_resolved": True,
                    "source_position": int(source_position),
                    "source_relative_index": (
                        int(hit.get("argmax_relative_index"))
                        if isinstance(hit.get("argmax_relative_index"), int) and not isinstance(hit.get("argmax_relative_index"), bool)
                        else None
                    ),
                    "source_piece": variant.get("source_piece"),
                    "source_segment_kind": variant.get("source_segment_kind"),
                    "site_preference": str(hit.get("site_preference", "balanced") or "balanced"),
                    "site_role": str(hit.get("site_role", "") or ""),
                    "recent_probe": dict(recent_probe),
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
                if source_span is not None:
                    candidate["source_span"] = {"start": int(source_span["start"]), "end": int(source_span["end"])}
                candidate["operator_family_key"] = self._operator_family_key(candidate)
                candidate["operator_recipe_id"] = self._operator_recipe_id(candidate)
                candidate_key = self._candidate_sidecar_key(candidate)
                if candidate_key in recent_probe_by_candidate_key:
                    candidate["recent_probe"] = dict(recent_probe_by_candidate_key[candidate_key])
                preferred_site = str(hit.get("site_preference", "balanced") or "balanced")
                preferred_priority = 0 if preferred_site in {"k_cache", "v_cache"} and site == preferred_site else 1
                site_priority = 0 if site == "k_cache" else 1
                candidates.append(
                    (
                        int(term_priority.get(feature, 99)),
                        int(variant.get("variant_priority", 1)),
                        preferred_priority,
                        site_priority,
                        -float(candidate["alignment"]),
                        int(layer),
                        int(head),
                        candidate,
                    )
                )
        debug["candidate_count_before_select"] = len(candidates)
        debug["candidate_families"] = [
            str(candidate.get("candidate_family", "") or "")
            for _term_priority, _variant_priority, _preferred, _site_priority, _alignment, _layer, _head, candidate in candidates
            if str(candidate.get("candidate_family", "") or "")
        ]
        candidates.sort(
            key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5], item[6], str(item[7]["surface_id"]))
        )
        selected = [candidate for _t, _v, _p, _s, _a, _l, _h, candidate in candidates]
        if control_phase_hint == "readout_escape":
            selected = self._prune_escape_candidates(selected, debug=debug)
            selected = self._post_bundle_rerank(
                selected,
                answer_readout_canary=answer_readout_canary,
                debug=debug,
                control_phase_hint=control_phase_hint,
            )
            selected = selected[:4]
        else:
            selected = selected[:3]
        if not selected and debug["input_hit_count"] > 0 and not debug["prune_reason_counts"]:
            _note_prune("selection_empty")
        if canary_enabled:
            selected = self._annotate_kv_candidates_with_canary(selected)
        debug["output_count"] = len(selected)
        debug["prune_reasons"] = sorted(debug["prune_reason_counts"])
        return selected, debug

    def _kv_retry_candidate_edits(
        self,
        *,
        base_candidates: Sequence[Mapping[str, Any]],
        canary_enabled: bool = False,
    ) -> list[dict[str, Any]]:
        retry_candidates: list[tuple[float, int, dict[str, Any]]] = []
        seen_surface_ids: set[str] = set()
        for raw_candidate in base_candidates:
            if not isinstance(raw_candidate, Mapping):
                continue
            recent_probe = raw_candidate.get("recent_probe")
            if not isinstance(recent_probe, Mapping):
                continue
            if str(recent_probe.get("label", "") or "") != "weak_positive_subthreshold":
                continue
            if bool(recent_probe.get("was_retry", False)):
                continue
            if int(recent_probe.get("repeat_flag_delta", 0) or 0) > 0:
                continue
            if float(recent_probe.get("semantic_progress_delta", 0.0) or 0.0) < -0.01:
                continue
            rank_delta = max(
                int(recent_probe.get("focus_rank_delta", 0) or 0),
                int(recent_probe.get("rank_focus_delta", 0) or 0),
            )
            mass_delta = float(recent_probe.get("target_mass_delta", 0.0) or 0.0)
            top20_delta = int(recent_probe.get("target_top20_hit_delta", 0) or 0)
            logit_delta = float(recent_probe.get("max_logit_delta", 0.0) or 0.0)
            prob_delta = float(recent_probe.get("max_prob_delta", 0.0) or 0.0)
            if (
                top20_delta <= 0
                and mass_delta < 0.00001
                and rank_delta < 4
                and logit_delta < 0.0003
                and prob_delta < 0.00003
            ):
                continue
            surface_id = str(raw_candidate.get("surface_id", "") or "")
            if not surface_id or surface_id in seen_surface_ids:
                continue
            seen_surface_ids.add(surface_id)
            candidate = dict(raw_candidate)
            for key in tuple(candidate):
                if key.startswith("canary_"):
                    candidate.pop(key, None)
            op = dict(candidate.get("op", {})) if isinstance(candidate.get("op"), Mapping) else {}
            budget = dict(candidate.get("budget", {})) if isinstance(candidate.get("budget"), Mapping) else {}
            meta = dict(candidate.get("meta", {})) if isinstance(candidate.get("meta"), Mapping) else {}
            site = str(candidate.get("site", "") or "")
            current_alpha = float(op.get("alpha", 0.03 if site == "k_cache" else 0.04) or (0.03 if site == "k_cache" else 0.04))
            alpha_cap = 0.06 if site == "k_cache" else 0.08
            alpha_step = 0.01 if site == "k_cache" else 0.015
            retry_alpha = min(alpha_cap, max(current_alpha, current_alpha + alpha_step, current_alpha * 1.4))
            op["alpha"] = round(float(retry_alpha), 6)
            budget["step_size"] = round(float(retry_alpha), 6)
            meta["retry_stage"] = "weak_positive_retry"
            which = str(op.get("which", "v") or "v")
            meta["hypothesis"] = f"kv_{which}_weak_retry_probe"
            meta["expected_effect"] = "slightly_stronger_cache_recall_support"
            candidate["op"] = op
            candidate["budget"] = budget
            candidate["meta"] = meta
            candidate["retry_stage"] = "weak_positive_retry"
            candidate["retry_reason"] = "target_rank_or_mass_up"
            candidate["retry_source"] = dict(recent_probe)
            retry_candidates.append(
                (
                    -float(recent_probe.get("score", 0.0) or 0.0),
                    0 if site == "k_cache" else 1,
                    candidate,
                )
            )
        retry_candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("surface_id", ""))))
        selected = [candidate for _score, _site_priority, candidate in retry_candidates[:2]]
        if canary_enabled:
            return self._annotate_kv_candidates_with_canary(selected)
        return selected

    def _kv_source_positions_for_feature(
        self,
        prototype: torch.Tensor,
        *,
        feature: str,
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

        def _source_anchor_quality(piece: str, *, segment_kind: str, relative_index: int) -> float:
            normalized_piece = "".join(ch for ch in str(piece).lower().strip() if ch.isalnum())
            normalized_feature = "".join(ch for ch in str(feature).lower().strip() if ch.isalnum())
            quality = 0.0
            if segment_kind == "hint":
                quality += 0.4
            elif segment_kind == "prompt":
                quality += 0.3
            if normalized_piece:
                quality += 0.25
                if normalized_feature:
                    if normalized_piece in normalized_feature:
                        quality += 1.4 + min(0.6, len(normalized_piece) / max(1, len(normalized_feature)))
                    overlap = len(set(normalized_piece) & set(normalized_feature))
                    if overlap > 0:
                        quality += 0.3 + (0.5 * overlap / max(1, len(set(normalized_piece))))
                if str(piece).startswith(" ") and normalized_feature.startswith(normalized_piece):
                    quality += 0.3
            else:
                quality -= 1.2
            if site == "k_cache":
                if not normalized_piece:
                    quality -= 1.5
                if segment_kind == "output":
                    quality -= 0.4
                if relative_index >= -2:
                    quality -= 0.25
            return float(quality)

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
            anchor_quality = _source_anchor_quality(piece, segment_kind=segment_kind, relative_index=int(relative_index))
            rows.append(
                {
                    "position": int(position),
                    "relative_index": int(relative_index),
                    "segment_kind": segment_kind,
                    "piece": piece,
                    "alignment": float(alignment),
                    "anchor_quality": float(anchor_quality),
                    "_segment_priority": _segment_priority(segment_kind),
                }
            )
        rows.sort(
            key=lambda item: (
                int(item["_segment_priority"]),
                -float(item["anchor_quality"]),
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
                "anchor_quality": round(float(item["anchor_quality"]), 6),
            }
            for item in rows[:max_positions]
        ]

    def _annotate_kv_candidates_with_canary(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        terms = self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms"))
        if not terms:
            return [dict(candidate) for candidate in candidates]

        baseline = self._simulate_decode(max_new_tokens=2, top_k=5)
        if baseline is None:
            return [dict(candidate) for candidate in candidates]

        annotated: list[dict[str, Any]] = []
        baseline_logits = baseline["first_logits"].detach().cpu().float()
        baseline_probs = torch.softmax(baseline_logits, dim=-1)
        vocab_size = int(baseline_logits.shape[-1])

        def _token_rank(logits: torch.Tensor, token_id: int) -> int:
            token_value = float(logits[token_id].item())
            higher = torch.count_nonzero(logits > token_value).item()
            return int(higher) + 1

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
                edited = self._simulate_decode(max_new_tokens=2, top_k=5, command=command)
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
            best_space_focus: dict[str, Any] | None = None
            best_rank_focus: dict[str, Any] | None = None
            seen_token_ids: set[int] = set()
            for sequence in target_sequences:
                token_id = int(sequence.token_ids[0])
                if token_id in seen_token_ids:
                    continue
                seen_token_ids.add(token_id)
                baseline_rank = _token_rank(baseline_logits, token_id)
                edited_rank = _token_rank(edited_logits, token_id)
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
                    "baseline_rank": int(baseline_rank),
                    "edited_rank": int(edited_rank),
                    "rank_delta": int(baseline_rank - edited_rank),
                }
                if best_focus is None or (
                    float(row["prob_delta"]),
                    float(row["logit_delta"]),
                ) > (
                    float(best_focus["prob_delta"]),
                    float(best_focus["logit_delta"]),
                ):
                    best_focus = row
                if str(sequence.variant).startswith(" ") and (
                    best_space_focus is None
                    or (float(row["prob_delta"]), float(row["logit_delta"]))
                    > (float(best_space_focus["prob_delta"]), float(best_space_focus["logit_delta"]))
                ):
                    best_space_focus = row
                if best_rank_focus is None or (
                    int(row["rank_delta"]),
                    -int(row["edited_rank"]),
                    float(row["prob_delta"]),
                    float(row["logit_delta"]),
                    1 if str(sequence.variant).startswith(" ") else 0,
                ) > (
                    int(best_rank_focus["rank_delta"]),
                    -int(best_rank_focus["edited_rank"]),
                    float(best_rank_focus["prob_delta"]),
                    float(best_rank_focus["logit_delta"]),
                    1 if str(best_rank_focus["variant"]).startswith(" ") else 0,
                ):
                    best_rank_focus = row
            target_token_ids = sorted(seen_token_ids)
            target_mass_baseline = sum(float(baseline_probs[token_id].item()) for token_id in target_token_ids)
            target_mass_edited = sum(float(edited_probs[token_id].item()) for token_id in target_token_ids)
            top_k_width = min(20, vocab_size)
            baseline_top_ids = {
                int(token_id)
                for token_id in torch.topk(baseline_logits, k=top_k_width).indices.detach().cpu().tolist()
            }
            edited_top_ids = {
                int(token_id)
                for token_id in torch.topk(edited_logits, k=top_k_width).indices.detach().cpu().tolist()
            }
            baseline_target_topk_hits = sum(1 for token_id in target_token_ids if token_id in baseline_top_ids)
            edited_target_topk_hits = sum(1 for token_id in target_token_ids if token_id in edited_top_ids)
            best_prefix: dict[str, Any] | None = None
            for sequence in target_sequences:
                baseline_prefix = 0
                edited_prefix = 0
                for baseline_token, target_token in zip(baseline.get("continuation_token_ids", []), sequence.token_ids, strict=False):
                    if int(baseline_token) != int(target_token):
                        break
                    baseline_prefix += 1
                for edited_token, target_token in zip(edited.get("continuation_token_ids", []), sequence.token_ids, strict=False):
                    if int(edited_token) != int(target_token):
                        break
                    edited_prefix += 1
                prefix_row = {
                    "term": str(sequence.term),
                    "variant": str(sequence.variant),
                    "baseline_prefix_depth": int(baseline_prefix),
                    "edited_prefix_depth": int(edited_prefix),
                    "prefix_depth_delta": int(edited_prefix - baseline_prefix),
                }
                if best_prefix is None or (
                    int(prefix_row["prefix_depth_delta"]),
                    int(prefix_row["edited_prefix_depth"]),
                    1 if str(sequence.variant).startswith(" ") else 0,
                ) > (
                    int(best_prefix["prefix_depth_delta"]),
                    int(best_prefix["edited_prefix_depth"]),
                    1 if str(best_prefix["variant"]).startswith(" ") else 0,
                ):
                    best_prefix = prefix_row

            candidate["canary_checked"] = True
            candidate["canary_max_new_tokens"] = 2
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
            candidate["canary_required_term_span_progress_delta"] = round(
                float(edited_scoring.get("required_term_span_progress") or 0.0)
                - float(baseline_scoring.get("required_term_span_progress") or 0.0),
                6,
            )
            candidate["canary_target_mass_baseline"] = round(target_mass_baseline, 6)
            candidate["canary_target_mass_edited"] = round(target_mass_edited, 6)
            candidate["canary_target_mass_delta"] = round(target_mass_edited - target_mass_baseline, 6)
            candidate["canary_target_top20_hits_baseline"] = int(baseline_target_topk_hits)
            candidate["canary_target_top20_hits_edited"] = int(edited_target_topk_hits)
            candidate["canary_target_top20_hit_delta"] = int(edited_target_topk_hits - baseline_target_topk_hits)
            if best_focus is not None:
                candidate["canary_focus_term"] = str(best_focus["term"])
                candidate["canary_focus_piece"] = str(best_focus["piece"])
                candidate["canary_focus_token_id"] = int(best_focus["token_id"])
                candidate["canary_focus_logit_delta"] = round(float(best_focus["logit_delta"]), 6)
                candidate["canary_focus_prob_delta"] = round(float(best_focus["prob_delta"]), 6)
                candidate["canary_focus_rank_baseline"] = int(best_focus["baseline_rank"])
                candidate["canary_focus_rank_edited"] = int(best_focus["edited_rank"])
                candidate["canary_focus_rank_delta"] = int(best_focus["rank_delta"])
            else:
                candidate["canary_focus_logit_delta"] = 0.0
                candidate["canary_focus_prob_delta"] = 0.0
                candidate["canary_focus_rank_baseline"] = 0
                candidate["canary_focus_rank_edited"] = 0
                candidate["canary_focus_rank_delta"] = 0
            if best_space_focus is not None:
                candidate["canary_space_focus_term"] = str(best_space_focus["term"])
                candidate["canary_space_focus_piece"] = str(best_space_focus["piece"])
                candidate["canary_space_focus_logit_delta"] = round(float(best_space_focus["logit_delta"]), 6)
                candidate["canary_space_focus_prob_delta"] = round(float(best_space_focus["prob_delta"]), 6)
            else:
                candidate["canary_space_focus_logit_delta"] = 0.0
                candidate["canary_space_focus_prob_delta"] = 0.0
            if best_rank_focus is not None:
                candidate["canary_rank_focus_term"] = str(best_rank_focus["term"])
                candidate["canary_rank_focus_piece"] = str(best_rank_focus["piece"])
                candidate["canary_rank_focus_baseline"] = int(best_rank_focus["baseline_rank"])
                candidate["canary_rank_focus_edited"] = int(best_rank_focus["edited_rank"])
                candidate["canary_rank_focus_delta"] = int(best_rank_focus["rank_delta"])
            else:
                candidate["canary_rank_focus_delta"] = 0
            if best_prefix is not None:
                candidate["canary_prefix_term"] = str(best_prefix["term"])
                candidate["canary_prefix_variant"] = str(best_prefix["variant"])
                candidate["canary_prefix_depth_baseline"] = int(best_prefix["baseline_prefix_depth"])
                candidate["canary_prefix_depth_edited"] = int(best_prefix["edited_prefix_depth"])
                candidate["canary_prefix_depth_delta"] = int(best_prefix["prefix_depth_delta"])
            else:
                candidate["canary_prefix_depth_delta"] = 0

            focus_logit_delta = float(candidate.get("canary_focus_logit_delta", 0.0) or 0.0)
            focus_prob_delta = float(candidate.get("canary_focus_prob_delta", 0.0) or 0.0)
            space_focus_logit_delta = float(candidate.get("canary_space_focus_logit_delta", 0.0) or 0.0)
            space_focus_prob_delta = float(candidate.get("canary_space_focus_prob_delta", 0.0) or 0.0)
            prefix_delta = int(candidate.get("canary_prefix_depth_delta", 0) or 0)
            recall_delta = float(candidate.get("canary_required_term_recall_delta", 0.0) or 0.0)
            semantic_delta = float(candidate.get("canary_semantic_progress_delta", 0.0) or 0.0)
            repeat_delta = int(candidate.get("canary_repeat_flag_delta", 0) or 0)
            span_progress_delta = float(candidate.get("canary_required_term_span_progress_delta", 0.0) or 0.0)
            target_mass_delta = float(candidate.get("canary_target_mass_delta", 0.0) or 0.0)
            target_topk_hit_delta = int(candidate.get("canary_target_top20_hit_delta", 0) or 0)
            focus_rank_delta = int(candidate.get("canary_focus_rank_delta", 0) or 0)
            rank_focus_delta = int(candidate.get("canary_rank_focus_delta", 0) or 0)
            site = str(candidate.get("site", "") or "")
            focus_logit_threshold = 0.0005 if site == "k_cache" else 0.001
            focus_prob_threshold = 0.00005 if site == "k_cache" else 0.0001
            target_mass_threshold = 0.00001 if site == "k_cache" else 0.00002
            target_rank_threshold = 4 if site == "k_cache" else 6
            canary_pass = (
                (recall_delta > 0.0)
                or (prefix_delta > 0)
                or (
                    (
                        target_topk_hit_delta > 0
                        or target_mass_delta >= target_mass_threshold
                        or focus_rank_delta >= target_rank_threshold
                        or rank_focus_delta >= target_rank_threshold
                    )
                    and repeat_delta <= 0
                    and semantic_delta >= -0.02
                )
                or (
                    (
                        focus_logit_delta >= focus_logit_threshold
                        or focus_prob_delta >= focus_prob_threshold
                        or space_focus_logit_delta >= focus_logit_threshold
                        or space_focus_prob_delta >= focus_prob_threshold
                        or span_progress_delta > 0.0
                    )
                    and repeat_delta <= 0
                    and semantic_delta >= -0.02
                )
            )
            candidate["canary_pass"] = bool(canary_pass)
            if canary_pass:
                if recall_delta > 0.0:
                    candidate["canary_reason"] = "recall_improved"
                elif prefix_delta > 0:
                    candidate["canary_reason"] = "target_prefix_improved"
                elif target_topk_hit_delta > 0:
                    candidate["canary_reason"] = "target_top20_improved"
                elif focus_rank_delta >= target_rank_threshold or rank_focus_delta >= target_rank_threshold:
                    candidate["canary_reason"] = "target_rank_improved"
                elif target_mass_delta >= target_mass_threshold:
                    candidate["canary_reason"] = "target_mass_improved"
                elif space_focus_logit_delta >= focus_logit_threshold or space_focus_prob_delta >= focus_prob_threshold:
                    candidate["canary_reason"] = "space_prefixed_focus_improved"
                else:
                    candidate["canary_reason"] = "focus_token_improved"
            else:
                candidate["canary_reason"] = "focus_token_flat"
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

    def _token_rank_for_logits(self, logits: torch.Tensor, token_id: int) -> int:
        token_value = float(logits[token_id].item())
        higher = torch.count_nonzero(logits > token_value).item()
        return int(higher) + 1

    def _first_token_target_readout_metrics(
        self,
        baseline_logits: torch.Tensor,
        edited_logits: torch.Tensor,
        *,
        focus_terms: Sequence[str],
    ) -> dict[str, Any]:
        vocab_size = int(baseline_logits.shape[-1])
        target_sequences = self._target_token_sequences(focus_terms, vocab_size=vocab_size)
        if not target_sequences:
            return {}
        baseline_probs = torch.softmax(baseline_logits.detach().cpu().float(), dim=-1)
        edited_probs = torch.softmax(edited_logits.detach().cpu().float(), dim=-1)
        baseline_logits = baseline_logits.detach().cpu().float()
        edited_logits = edited_logits.detach().cpu().float()
        best_focus: dict[str, Any] | None = None
        best_rank_focus: dict[str, Any] | None = None
        seen_token_ids: set[int] = set()
        for sequence in target_sequences:
            token_id = int(sequence.token_ids[0])
            if token_id in seen_token_ids:
                continue
            seen_token_ids.add(token_id)
            baseline_rank = self._token_rank_for_logits(baseline_logits, token_id)
            edited_rank = self._token_rank_for_logits(edited_logits, token_id)
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
                "baseline_rank": int(baseline_rank),
                "edited_rank": int(edited_rank),
                "rank_delta": int(baseline_rank - edited_rank),
            }
            if best_focus is None or (
                float(row["prob_delta"]),
                float(row["logit_delta"]),
            ) > (
                float(best_focus["prob_delta"]),
                float(best_focus["logit_delta"]),
            ):
                best_focus = row
            if best_rank_focus is None or (
                int(row["rank_delta"]),
                -int(row["edited_rank"]),
                float(row["prob_delta"]),
                float(row["logit_delta"]),
                1 if str(sequence.variant).startswith(" ") else 0,
            ) > (
                int(best_rank_focus["rank_delta"]),
                -int(best_rank_focus["edited_rank"]),
                float(best_rank_focus["prob_delta"]),
                float(best_rank_focus["logit_delta"]),
                1 if str(best_rank_focus["variant"]).startswith(" ") else 0,
            ):
                best_rank_focus = row
        target_token_ids = sorted(seen_token_ids)
        target_mass_baseline = sum(float(baseline_probs[token_id].item()) for token_id in target_token_ids)
        target_mass_edited = sum(float(edited_probs[token_id].item()) for token_id in target_token_ids)
        top_k_width = min(20, vocab_size)
        baseline_top_ids = {
            int(token_id)
            for token_id in torch.topk(baseline_logits, k=top_k_width).indices.detach().cpu().tolist()
        }
        edited_top_ids = {
            int(token_id)
            for token_id in torch.topk(edited_logits, k=top_k_width).indices.detach().cpu().tolist()
        }
        metrics: dict[str, Any] = {
            "target_mass_baseline": round(target_mass_baseline, 6),
            "target_mass_edited": round(target_mass_edited, 6),
            "target_mass_delta": round(target_mass_edited - target_mass_baseline, 6),
            "target_top20_hits_baseline": sum(1 for token_id in target_token_ids if token_id in baseline_top_ids),
            "target_top20_hits_edited": sum(1 for token_id in target_token_ids if token_id in edited_top_ids),
        }
        metrics["target_top20_hit_delta"] = int(metrics["target_top20_hits_edited"] - metrics["target_top20_hits_baseline"])
        if best_focus is not None:
            metrics["focus_term"] = str(best_focus["term"])
            metrics["focus_piece"] = str(best_focus["piece"])
            metrics["focus_logit_delta"] = round(float(best_focus["logit_delta"]), 6)
            metrics["focus_prob_delta"] = round(float(best_focus["prob_delta"]), 6)
            metrics["focus_rank_delta"] = int(best_focus["rank_delta"])
            metrics["focus_rank_baseline"] = int(best_focus["baseline_rank"])
            metrics["focus_rank_edited"] = int(best_focus["edited_rank"])
        if best_rank_focus is not None:
            metrics["rank_focus_term"] = str(best_rank_focus["term"])
            metrics["rank_focus_piece"] = str(best_rank_focus["piece"])
            metrics["rank_focus_delta"] = int(best_rank_focus["rank_delta"])
            metrics["rank_focus_rank_baseline"] = int(best_rank_focus["baseline_rank"])
            metrics["rank_focus_rank_edited"] = int(best_rank_focus["edited_rank"])
        return metrics

    def _current_answer_readout_canary(
        self,
        *,
        top_k: int = 20,
        promoted_cache_surfaces: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any] | None:
        focus_terms = self._feedback_terms(
            ("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms")
        )
        if not focus_terms:
            return None
        tokens = self._current_token_tensor()
        saved_last_tokens = None if getattr(self.runtime_state, "last_tokens", None) is None else self.runtime_state.last_tokens.detach().clone()
        saved_last_logits = None if getattr(self.runtime_state, "last_logits", None) is None else self.runtime_state.last_logits.detach().clone()
        saved_last_cache = None
        if getattr(self.runtime_state, "last_cache", None) is not None:
            saved_last_cache = {str(name): tensor.detach().clone() for name, tensor in self.runtime_state.last_cache.items()}
        try:
            logits, _cache = self.runtime_state.run_with_cache(tokens, return_type="logits")
            next_logits = self._apply_token_constraints(logits[0, -1].detach())
            next_logits, _decoder_state = self._apply_decoder_control(next_logits)
        except Exception:
            return None
        finally:
            self.runtime_state.last_tokens = saved_last_tokens
            self.runtime_state.last_logits = saved_last_logits
            self.runtime_state.last_cache = saved_last_cache

        next_logits = next_logits.detach().cpu().float()
        probs = torch.softmax(next_logits, dim=-1)
        vocab_size = int(next_logits.shape[-1])
        top_width = min(max(1, int(top_k)), vocab_size)
        top_ids = torch.topk(next_logits, k=top_width).indices.detach().cpu().tolist()
        top_id_set = {int(token_id) for token_id in top_ids}
        target_sequences = self._target_token_sequences(focus_terms, vocab_size=vocab_size)
        target_token_ids = sorted({int(sequence.token_ids[0]) for sequence in target_sequences if sequence.token_ids})
        best_focus: dict[str, Any] | None = None
        for sequence in target_sequences:
            token_id = int(sequence.token_ids[0])
            rank = self._token_rank_for_logits(next_logits, token_id)
            row = {
                "term": str(sequence.term),
                "piece": self.codec.decode([token_id]),
                "rank": int(rank),
                "logit": float(next_logits[token_id].item()),
                "prob": float(probs[token_id].item()),
                "variant": str(sequence.variant),
            }
            if best_focus is None or (
                -int(row["rank"]),
                float(row["prob"]),
                float(row["logit"]),
                1 if str(sequence.variant).startswith(" ") else 0,
            ) > (
                -int(best_focus["rank"]),
                float(best_focus["prob"]),
                float(best_focus["logit"]),
                1 if str(best_focus["variant"]).startswith(" ") else 0,
            ):
                best_focus = row

        semantic_focus = self._semantic_focus_summary(promoted_cache_surfaces=promoted_cache_surfaces)
        recent_output_ids = []
        seen_recent_output_ids: set[int] = set()
        recent_output_piece_keys: set[str] = set()
        for token_id in reversed(self._output_token_ids()[-6:]):
            token_value = int(token_id)
            if token_value in seen_recent_output_ids or token_value < 0 or token_value >= vocab_size:
                continue
            seen_recent_output_ids.add(token_value)
            recent_output_ids.append(token_value)
            piece_key = self._canonical_effect_family_piece(self.codec.decode([token_value]))
            if piece_key:
                recent_output_piece_keys.add(piece_key)
        recent_output_ids.reverse()
        attractor_family_mass = sum(float(probs[token_id].item()) for token_id in recent_output_ids)
        attractor_family_overlap_tokens: list[str] = []
        for token_id in top_ids[:5]:
            token_piece = self.codec.decode([int(token_id)])
            token_key = self._canonical_effect_family_piece(token_piece)
            if token_key and token_key in recent_output_piece_keys and token_piece not in attractor_family_overlap_tokens:
                attractor_family_overlap_tokens.append(token_piece)
        attractor_family_top_overlap = len(attractor_family_overlap_tokens)
        attractor_family_top_overlap_exact = sum(1 for token_id in top_ids[:5] if int(token_id) in seen_recent_output_ids)

        return {
            "semantic_focus_term": semantic_focus.get("semantic_focus_term"),
            "semantic_focus_source": semantic_focus.get("semantic_focus_source"),
            "target_mass": round(sum(float(probs[token_id].item()) for token_id in target_token_ids), 6),
            "target_top20_hits": sum(1 for token_id in target_token_ids if token_id in top_id_set),
            "reachable_focus_term": None if best_focus is None else str(best_focus["term"]),
            "reachable_focus_piece": None if best_focus is None else str(best_focus["piece"]),
            "reachable_focus_rank": None if best_focus is None else int(best_focus["rank"]),
            "reachable_focus_prob": None if best_focus is None else round(float(best_focus["prob"]), 6),
            "focus_term": None if best_focus is None else str(best_focus["term"]),
            "focus_piece": None if best_focus is None else str(best_focus["piece"]),
            "focus_rank": None if best_focus is None else int(best_focus["rank"]),
            "focus_prob": None if best_focus is None else round(float(best_focus["prob"]), 6),
            "attractor_family_mass": round(float(attractor_family_mass), 6),
            "attractor_family_top_overlap": int(attractor_family_top_overlap),
            "attractor_family_top_overlap_exact": int(attractor_family_top_overlap_exact),
            "attractor_family_overlap_tokens": list(attractor_family_overlap_tokens),
            "attractor_family_tokens": [self.codec.decode([int(token_id)]) for token_id in recent_output_ids[:5]],
            "top_tokens": [self.codec.decode([int(token_id)]) for token_id in top_ids],
        }

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
            record["provenance_class"] = "misc_prompt"

        prompt_records = [record for record in records if str(record.get("segment_kind", "")) == "prompt"]
        prompt_text = ""
        prompt_offset = 0
        for record in prompt_records:
            piece = str(record.get("piece", ""))
            record["decoded_start"] = int(prompt_offset)
            prompt_offset += len(piece)
            record["decoded_end"] = int(prompt_offset)
            prompt_text += piece

        source_marker_index = prompt_text.find("SOURCE:")
        answer_marker_index = prompt_text.find("ANSWER:")
        source_marker_end = source_marker_index + len("SOURCE:") if source_marker_index >= 0 else -1
        for record in records:
            segment_kind = str(record.get("segment_kind", "") or "")
            if segment_kind == "output":
                record["provenance_class"] = "output"
                continue
            if segment_kind != "prompt":
                record["provenance_class"] = "misc_prompt"
                continue
            decoded_start = int(record.get("decoded_start", 0) or 0)
            decoded_end = int(record.get("decoded_end", decoded_start) or decoded_start)
            midpoint = decoded_start if decoded_end <= decoded_start else (decoded_start + ((decoded_end - decoded_start) / 2.0))
            if source_marker_index >= 0 and midpoint < source_marker_index:
                record["provenance_class"] = "constraint_header"
            elif source_marker_index >= 0 and midpoint >= source_marker_end and (answer_marker_index < 0 or midpoint < answer_marker_index):
                record["provenance_class"] = "source_body"
            elif answer_marker_index >= 0 and midpoint >= answer_marker_index:
                record["provenance_class"] = "answer_prefix"
            else:
                record["provenance_class"] = "misc_prompt"
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
        result = {
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
            "required_term_span_progress_delta": round(
                float(edited_score.get("required_term_span_progress") or 0.0)
                - float(baseline_score.get("required_term_span_progress") or 0.0),
                6,
            ),
            "semantic_progress_delta": round(
                float(edited_score.get("semantic_progress_score") or 0.0)
                - float(baseline_score.get("semantic_progress_score") or 0.0),
                6,
            ),
        }
        focus_terms: list[str] = []
        feature = str(candidate_edit.get("focus_feature", "") or "")
        if feature:
            focus_terms.append(feature)
        for term in self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms")):
            if term not in focus_terms:
                focus_terms.append(term)
        readout_metrics = self._first_token_target_readout_metrics(
            baseline["first_logits"],
            edited["first_logits"],
            focus_terms=focus_terms,
        )
        if readout_metrics:
            result.update(readout_metrics)
        probe_summary = self._classify_probe_result(result)
        if probe_summary is not None:
            result["probe_family"] = probe_summary.get("probe_family")
            result["probe_phase_profile"] = probe_summary.get("probe_phase_profile")
            result["probe_label"] = probe_summary.get("label")
            result["probe_score"] = probe_summary.get("score")
            result["readout_probe_score"] = probe_summary.get("readout_score")
            result["constraint_probe_score"] = probe_summary.get("constraint_score")
            result["trajectory_probe_score"] = probe_summary.get("trajectory_score")
            result["positive_axes"] = list(probe_summary.get("positive_axes", [])[:4])
            result["actionable_axes"] = list(probe_summary.get("actionable_axes", [])[:4])
            result["probe_summary"] = dict(probe_summary)
        return result

    def replay_candidate_edits_actual_delta(
        self,
        candidate_edits: Sequence[Mapping[str, Any]],
        *,
        max_new_tokens: int = 3,
        top_k: int = 8,
        max_edits_per_step_override: int | None = None,
        label: str | None = None,
        ownership_terms: Sequence[str] | None = None,
        intended_bundle_key: str | None = None,
        intended_term: str | None = None,
        contrast_partner_bundle_key: str | None = None,
        contrast_partner_term: str | None = None,
    ) -> dict[str, Any]:
        edits = [dict(item) for item in candidate_edits if isinstance(item, Mapping)]
        if not edits:
            return {"status": "error", "error": "missing_candidate_edits", "label": label}
        baseline = self._simulate_decode(max_new_tokens=max_new_tokens, top_k=top_k)
        if baseline is None:
            return {
                "status": "error",
                "error": "baseline_simulate_failed",
                "label": label,
                "simulate_error": self._last_simulate_decode_error,
            }
        prepared_edits: list[dict[str, Any]] = []
        focus_terms: list[str] = []
        operator_family_keys: list[str] = []
        operator_recipe_ids: list[str] = []
        for index, raw_edit in enumerate(edits):
            edit = dict(raw_edit)
            edit.setdefault("id", f"replay_{self._steps}_{index}")
            edit.setdefault("target", {"surface_id": raw_edit.get("surface_id")})
            prepared_edits.append(edit)
            feature = str(raw_edit.get("focus_feature", "") or "")
            if feature and feature not in focus_terms:
                focus_terms.append(feature)
            operator_family_keys.append(self._operator_family_key(raw_edit))
            operator_recipe_ids.append(self._operator_recipe_id(raw_edit))
        for term in self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms")):
            if term not in focus_terms:
                focus_terms.append(term)
        if isinstance(ownership_terms, SequenceABC) and not isinstance(ownership_terms, (str, bytes, bytearray)):
            for term in ownership_terms:
                normalized_term = str(term or "")
                if normalized_term and normalized_term not in focus_terms:
                    focus_terms.append(normalized_term)
        command = {"version": "0.1", "decision": "apply", "edits": prepared_edits}
        policy_override = self._replay_policy(max_edits_per_step_override=max_edits_per_step_override)
        effective_max_edits_per_step = (
            int(policy_override.global_budget.max_edits_per_step)
            if policy_override is not None
            else int(self.max_edits_per_step)
        )
        edited = self._simulate_decode(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            command=command,
            policy_override=policy_override,
        )
        if edited is None:
            return {
                "status": "error",
                "error": "simulate_decode_failed",
                "label": label,
                "simulate_error": self._last_simulate_decode_error,
                "bundle_keys": sorted({str(item.get("bundle_key", "") or "") for item in edits if str(item.get("bundle_key", "") or "")}),
                "operator_family_keys": sorted({str(key) for key in operator_family_keys if str(key)}),
            }
        baseline_score = baseline.get("scoring", {})
        edited_score = edited.get("scoring", {})
        result = {
            "status": "ok",
            "label": label,
            "edit_count": len(prepared_edits),
            "candidate_edits": [dict(item) for item in edits],
            "bundle_keys": sorted({str(item.get("bundle_key", "") or "") for item in edits if str(item.get("bundle_key", "") or "")}),
            "intended_bundle_key": str(intended_bundle_key or "") or None,
            "intended_term": str(intended_term or "") or None,
            "contrast_partner_bundle_key": str(contrast_partner_bundle_key or "") or None,
            "contrast_partner_term": str(contrast_partner_term or "") or None,
            "operator_family_key": operator_family_keys[0] if operator_family_keys else "unknown",
            "operator_family_keys": sorted({str(key) for key in operator_family_keys if str(key)}),
            "operator_recipe_id": operator_recipe_ids[0] if operator_recipe_ids else "unknown",
            "operator_recipe_ids": sorted({str(key) for key in operator_recipe_ids if str(key)}),
            "focus_terms": list(focus_terms),
            "continuation_baseline": baseline["continuation"],
            "continuation_candidate": edited["continuation"],
            "topk_token_diff": self._topk_token_diff(baseline["first_logits"], edited["first_logits"], top_k=top_k),
            "entropy_delta": round(float(edited["entropy"]) - float(baseline["entropy"]), 6),
            "repeat_flag_delta": int(bool(edited["repeat_flag"])) - int(bool(baseline["repeat_flag"])),
            "top1_margin_delta": round(float(edited.get("top1_margin", 0.0) or 0.0) - float(baseline.get("top1_margin", 0.0) or 0.0), 6),
            "repetition_score_delta": round(
                float(edited.get("repetition_score", 0.0) or 0.0) - float(baseline.get("repetition_score", 0.0) or 0.0),
                6,
            ),
            "required_term_recall_delta": round(
                float(edited_score.get("required_term_recall") or 0.0) - float(baseline_score.get("required_term_recall") or 0.0),
                6,
            ),
            "required_term_span_progress_delta": round(
                float(edited_score.get("required_term_span_progress") or 0.0)
                - float(baseline_score.get("required_term_span_progress") or 0.0),
                6,
            ),
            "semantic_progress_delta": round(
                float(edited_score.get("semantic_progress_score") or 0.0)
                - float(baseline_score.get("semantic_progress_score") or 0.0),
                6,
            ),
        }
        readout_metrics = self._first_token_target_readout_metrics(
            baseline["first_logits"],
            edited["first_logits"],
            focus_terms=focus_terms,
        )
        if readout_metrics:
            result.update(readout_metrics)
        term_readout_deltas = self._term_readout_deltas(
            baseline["first_logits"],
            edited["first_logits"],
            focus_terms=list(focus_terms),
        )
        if term_readout_deltas:
            result["term_readout_deltas"] = dict(term_readout_deltas)
        result["candidate_fingerprint"] = self._candidate_fingerprint(
            edits,
            intended_bundle_key=intended_bundle_key,
            label=label,
        )
        result["eval_context_fingerprint"] = self._eval_context_fingerprint(
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            max_edits_per_step=effective_max_edits_per_step,
            focus_terms=focus_terms,
        )
        probe_summary = self._classify_probe_result(
            {
                **result,
                "candidate_edit": prepared_edits[0],
            }
        )
        if probe_summary is not None:
            result["probe_summary"] = dict(probe_summary)
        result["actual_delta_class"] = self._classify_actual_delta_result(result)
        return result

    def _default_operator_recipe_specs(self) -> list[dict[str, Any]]:
        return [
            {
                "recipe_name": "baseline_span_mean",
                "localization": "exact_prompt_span_mean",
                "pooling": "mean",
                "contrast_mode": "none",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_token",
                "localization": "exact_term_token",
                "pooling": "single",
                "contrast_mode": "none",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_fused",
                "localization": "exact_term_fused",
                "pooling": "fused",
                "contrast_mode": "none",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_centered_pm1",
                "localization": "exact_term_centered_pm1",
                "pooling": "centered_mean",
                "contrast_mode": "none",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_window_pm1_weighted",
                "localization": "exact_term_window_pm1_weighted",
                "pooling": "weighted_mean",
                "contrast_mode": "none",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_window_pm1_weighted_minus_base",
                "localization": "exact_term_window_pm1_weighted",
                "pooling": "weighted_mean",
                "contrast_mode": "minus_base",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "span_mean_minus_stealer_l025",
                "localization": "exact_prompt_span_mean",
                "pooling": "mean",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.25,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "span_mean_minus_stealer_l050",
                "localization": "exact_prompt_span_mean",
                "pooling": "mean",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.5,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_token_minus_stealer_l025",
                "localization": "exact_term_token",
                "pooling": "single",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.25,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_token_minus_stealer_l050",
                "localization": "exact_term_token",
                "pooling": "single",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.5,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_fused_minus_stealer_l025",
                "localization": "exact_term_fused",
                "pooling": "fused",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.25,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_fused_orthogonal_stealer",
                "localization": "exact_term_fused",
                "pooling": "fused",
                "contrast_mode": "orthogonal_stealer",
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_centered_pm1_minus_stealer_l025",
                "localization": "exact_term_centered_pm1",
                "pooling": "centered_mean",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.25,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_centered_pm1_orthogonal_stealer",
                "localization": "exact_term_centered_pm1",
                "pooling": "centered_mean",
                "contrast_mode": "orthogonal_stealer",
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_window_pm1_weighted_minus_stealer_l025",
                "localization": "exact_term_window_pm1_weighted",
                "pooling": "weighted_mean",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.25,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_window_pm1_weighted_minus_stealer_l050",
                "localization": "exact_term_window_pm1_weighted",
                "pooling": "weighted_mean",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.5,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_window_pm1_weighted_minus_stealer_l075",
                "localization": "exact_term_window_pm1_weighted",
                "pooling": "weighted_mean",
                "contrast_mode": "minus_stealer",
                "contrast_scale": 0.75,
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
            {
                "recipe_name": "term_window_pm1_weighted_orthogonal_stealer",
                "localization": "exact_term_window_pm1_weighted",
                "pooling": "weighted_mean",
                "contrast_mode": "orthogonal_stealer",
                "competitor_strategy": "stealer",
                "modes": ("kv_pair",),
            },
        ]

    def replay_operator_recipe_matrix(
        self,
        *,
        bundle_keys: Sequence[str] | None = None,
        recipe_specs: Sequence[Mapping[str, Any]] | None = None,
        max_new_tokens: int = 3,
        top_k: int = 8,
        max_edits_per_step_override: int = 2,
    ) -> dict[str, Any]:
        packet = self.build_controller_packet()
        strategy_hints = packet.get("strategy_hints") if isinstance(packet.get("strategy_hints"), Mapping) else {}
        candidate_edits = strategy_hints.get("kv_candidate_edits")
        if not isinstance(candidate_edits, SequenceABC) or isinstance(candidate_edits, (str, bytes, bytearray)):
            candidate_edits = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in candidate_edits:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("bundle_key", "") or "")
            if not bundle_key:
                continue
            grouped.setdefault(bundle_key, []).append(dict(item))
        chosen_bundle_keys = [str(item) for item in (bundle_keys or ()) if str(item)]
        if not chosen_bundle_keys:
            for key in (
                strategy_hints.get("base_winner_bundle_key"),
                strategy_hints.get("challenger_bundle_key"),
            ):
                if key not in (None, "") and str(key) not in chosen_bundle_keys:
                    chosen_bundle_keys.append(str(key))
        specs = [dict(item) for item in (recipe_specs or self._default_operator_recipe_specs()) if isinstance(item, Mapping)]
        bundle_term_by_key = {
            str(bundle_key): self._bundle_focus_term(str(bundle_key), grouped.get(str(bundle_key), []))
            for bundle_key in chosen_bundle_keys
            if grouped.get(str(bundle_key), [])
        }
        ownership_terms = [str(term) for term in bundle_term_by_key.values() if str(term)]
        base_bundle_key = str(strategy_hints.get("base_winner_bundle_key", "") or "")
        base_members = grouped.get(base_bundle_key, [])
        primary_specs = [
            dict(spec)
            for spec in specs
            if str(spec.get("competitor_strategy", "") or "") != "stealer"
            and str(spec.get("contrast_mode", "none") or "none") not in {"minus_stealer", "orthogonal_stealer"}
        ]
        stealer_specs = [
            dict(spec)
            for spec in specs
            if spec not in primary_specs
        ]

        def _evaluate_specs(
            spec_list: Sequence[Mapping[str, Any]],
            *,
            stealer_bundle_map: Mapping[tuple[str, str], str] | None = None,
        ) -> list[dict[str, Any]]:
            local_evaluations: list[dict[str, Any]] = []
            raw_self_min = 0.005
            for bundle_key in chosen_bundle_keys:
                members = grouped.get(bundle_key, [])
                if not members:
                    continue
                members_by_site = {
                    str(item.get("site", "") or ""): dict(item)
                    for item in members
                    if isinstance(item, Mapping)
                }
                intended_term = str(bundle_term_by_key.get(str(bundle_key), "") or "")
                for spec in spec_list:
                    localization = str(spec.get("localization", "exact_prompt_span_mean") or "exact_prompt_span_mean")
                    pooling = str(spec.get("pooling", "mean") or "mean")
                    contrast_mode = str(spec.get("contrast_mode", "none") or "none")
                    contrast_scale = float(spec.get("contrast_scale", 1.0) or 1.0)
                    competitor_strategy = str(spec.get("competitor_strategy", "") or "base")
                    recipe_name = str(spec.get("recipe_name", localization) or localization)
                    v_alpha_override = (
                        float(spec["v_alpha"])
                        if isinstance(spec.get("v_alpha"), (int, float)) and not isinstance(spec.get("v_alpha"), bool)
                        else None
                    )
                    k_alpha_override = (
                        float(spec["k_alpha"])
                        if isinstance(spec.get("k_alpha"), (int, float)) and not isinstance(spec.get("k_alpha"), bool)
                        else None
                    )
                    for mode in tuple(spec.get("modes", ("kv_pair",))):
                        recipe_seed_key = self._operator_recipe_seed_key(
                            mode=str(mode),
                            localization=localization,
                            pooling=pooling,
                        )
                        if competitor_strategy == "stealer":
                            raw_seed = raw_seed_ownership.get((recipe_seed_key, str(bundle_key)))
                            if raw_seed is None or float(raw_seed.get("self_delta", 0.0) or 0.0) <= raw_self_min:
                                continue
                        competitor_bundle_key = ""
                        if competitor_strategy == "stealer":
                            competitor_bundle_key = str((stealer_bundle_map or {}).get((recipe_seed_key, str(bundle_key)), "") or "")
                            if not competitor_bundle_key:
                                competitor_bundle_key = str((stealer_bundle_map or {}).get(("*", str(bundle_key)), "") or "")
                        elif base_bundle_key and str(bundle_key) != base_bundle_key:
                            competitor_bundle_key = base_bundle_key
                        competitor_members = grouped.get(competitor_bundle_key, []) if competitor_bundle_key else []
                        competitor_by_site = {
                            str(item.get("site", "") or ""): dict(item)
                            for item in competitor_members
                            if isinstance(item, Mapping)
                        }
                        competitor_term = str(bundle_term_by_key.get(competitor_bundle_key, "") or "")
                        if mode == "kv_pair":
                            pair_members: list[dict[str, Any]] = []
                            for site in ("v_cache", "k_cache"):
                                member = members_by_site.get(site)
                                if member is None:
                                    pair_members = []
                                    break
                                pair_members.append(
                                    self._materialize_recipe_candidate(
                                        member,
                                        localization=localization,
                                        pooling=pooling,
                                        contrast_mode=contrast_mode,
                                        competitor_candidate=competitor_by_site.get(site),
                                        contrast_scale=contrast_scale,
                                        alpha_override=v_alpha_override if site == "v_cache" else k_alpha_override,
                                    )
                                )
                            if pair_members and all(member is not None for member in pair_members):
                                prepared_pair = [dict(member) for member in pair_members if isinstance(member, Mapping)]
                                if len(prepared_pair) == 2:
                                    pair_recipe_id = self._operator_recipe_id(
                                        {
                                            "phase_objective": prepared_pair[0].get("phase_objective"),
                                            "bundle_family": "kv_pair_source_anchor",
                                            "provenance_class": prepared_pair[0].get("provenance_class"),
                                            "recipe_localization": localization,
                                            "recipe_pooling": pooling,
                                            "contrast_mode": contrast_mode,
                                            "recipe_contrast_scale": contrast_scale,
                                            "recipe_k_alpha": prepared_pair[1].get("recipe_alpha", prepared_pair[1].get("op", {}).get("alpha", 0.0)),
                                            "recipe_v_alpha": prepared_pair[0].get("recipe_alpha", prepared_pair[0].get("op", {}).get("alpha", 0.0)),
                                        }
                                    )
                                    result = self.replay_candidate_edits_actual_delta(
                                        prepared_pair,
                                        max_new_tokens=max_new_tokens,
                                        top_k=top_k,
                                        max_edits_per_step_override=max_edits_per_step_override,
                                        label=f"{recipe_name}:pair:{bundle_key}",
                                        ownership_terms=ownership_terms,
                                        intended_bundle_key=bundle_key,
                                        intended_term=intended_term,
                                        contrast_partner_bundle_key=competitor_bundle_key or None,
                                        contrast_partner_term=competitor_term or None,
                                    )
                                    result["operator_recipe_id"] = pair_recipe_id
                                    result["operator_recipe_seed_key"] = recipe_seed_key
                                    result["recipe_mode"] = "kv_pair"
                                    result["recipe_name"] = recipe_name
                                    result["competitor_strategy"] = competitor_strategy
                                    local_evaluations.append(result)
                        elif mode == "kv_pair_asymmetric":
                            v_member = members_by_site.get("v_cache")
                            k_member = members_by_site.get("k_cache")
                            if v_member is None or k_member is None:
                                continue
                            prepared_v = self._materialize_recipe_candidate(
                                v_member,
                                localization=localization,
                                pooling=pooling,
                                contrast_mode=contrast_mode,
                                competitor_candidate=competitor_by_site.get("v_cache"),
                                contrast_scale=contrast_scale,
                                alpha_override=0.045 if v_alpha_override is None else v_alpha_override,
                            )
                            prepared_k = self._materialize_recipe_candidate(
                                k_member,
                                localization=localization,
                                pooling=pooling,
                                contrast_mode=contrast_mode,
                                competitor_candidate=competitor_by_site.get("k_cache"),
                                contrast_scale=contrast_scale,
                                alpha_override=0.03 if k_alpha_override is None else k_alpha_override,
                            )
                            if prepared_v is None or prepared_k is None:
                                continue
                            asym_pair = [prepared_v, prepared_k]
                            pair_recipe_id = self._operator_recipe_id(
                                {
                                    "phase_objective": prepared_v.get("phase_objective"),
                                    "bundle_family": "kv_pair_source_anchor",
                                    "provenance_class": prepared_v.get("provenance_class"),
                                    "recipe_localization": localization,
                                    "recipe_pooling": pooling,
                                    "contrast_mode": contrast_mode,
                                    "recipe_contrast_scale": contrast_scale,
                                    "recipe_k_alpha": prepared_k.get("recipe_alpha", prepared_k.get("op", {}).get("alpha", 0.0)),
                                    "recipe_v_alpha": prepared_v.get("recipe_alpha", prepared_v.get("op", {}).get("alpha", 0.0)),
                                }
                            )
                            result = self.replay_candidate_edits_actual_delta(
                                asym_pair,
                                max_new_tokens=max_new_tokens,
                                top_k=top_k,
                                max_edits_per_step_override=max_edits_per_step_override,
                                label=f"{recipe_name}:pair_asym:{bundle_key}",
                                ownership_terms=ownership_terms,
                                intended_bundle_key=bundle_key,
                                intended_term=intended_term,
                                contrast_partner_bundle_key=competitor_bundle_key or None,
                                contrast_partner_term=competitor_term or None,
                            )
                            result["operator_recipe_id"] = pair_recipe_id
                            result["operator_recipe_seed_key"] = recipe_seed_key
                            result["recipe_mode"] = "kv_pair_asymmetric"
                            result["recipe_name"] = recipe_name
                            result["competitor_strategy"] = competitor_strategy
                            local_evaluations.append(result)
                        else:
                            site = "v_cache" if mode == "kv_v" else "k_cache"
                            member = members_by_site.get(site)
                            if member is None:
                                continue
                            prepared = self._materialize_recipe_candidate(
                                member,
                                localization=localization,
                                pooling=pooling,
                                contrast_mode=contrast_mode,
                                competitor_candidate=competitor_by_site.get(site),
                                contrast_scale=contrast_scale,
                                alpha_override=v_alpha_override if site == "v_cache" else k_alpha_override,
                            )
                            if prepared is None:
                                continue
                            result = self.replay_candidate_edits_actual_delta(
                                [prepared],
                                max_new_tokens=max_new_tokens,
                                top_k=top_k,
                                label=f"{recipe_name}:{mode}:{bundle_key}",
                                ownership_terms=ownership_terms,
                                intended_bundle_key=bundle_key,
                                intended_term=intended_term,
                                contrast_partner_bundle_key=competitor_bundle_key or None,
                                contrast_partner_term=competitor_term or None,
                            )
                            result["operator_recipe_id"] = str(prepared.get("operator_recipe_id", self._operator_recipe_id(prepared)))
                            result["operator_recipe_seed_key"] = recipe_seed_key
                            result["recipe_mode"] = str(mode)
                            result["recipe_name"] = recipe_name
                            result["competitor_strategy"] = competitor_strategy
                            local_evaluations.append(result)
            return local_evaluations

        evaluations = _evaluate_specs(primary_specs)
        ownership = self._summarize_operator_recipe_bundle_ownership(
            evaluations,
            bundle_term_by_key=bundle_term_by_key,
        )
        raw_seed_ownership = {
            (str(item.get("operator_recipe_seed_key", "") or ""), str(item.get("intended_bundle_key", "") or "")): dict(item)
            for item in ownership
            if str(item.get("contrast_mode", "") or "none") == "none"
        }
        stealer_bundle_map = self._infer_recipe_stealer_bundles(ownership)
        if stealer_specs:
            evaluations.extend(
                _evaluate_specs(
                    stealer_specs,
                    stealer_bundle_map=stealer_bundle_map,
                )
            )
        recipe_certifications = self._summarize_operator_certifications(
            evaluations,
            key_field="operator_recipe_id",
        )
        ownership = self._summarize_operator_recipe_bundle_ownership(
            evaluations,
            bundle_term_by_key=bundle_term_by_key,
        )
        bridge_plan_recommendations = self._summarize_bridge_plan_recommendations(ownership)
        self._operator_bridge_plan_table = {
            str(item.get("objective_bundle_key", "") or ""): dict(item)
            for item in bridge_plan_recommendations
            if str(item.get("objective_bundle_key", "") or "")
        }
        return {
            "base_winner_bundle_key": strategy_hints.get("base_winner_bundle_key"),
            "challenger_bundle_key": strategy_hints.get("challenger_bundle_key"),
            "selected_bundle_key": strategy_hints.get("selected_bundle_key"),
            "selection_source": strategy_hints.get("selection_source"),
            "evaluations": evaluations,
            "operator_recipe_certifications": recipe_certifications,
            "operator_recipe_bundle_ownership": ownership,
            "bridge_plan_recommendations": bridge_plan_recommendations,
            "recipe_stealer_bundle_keys": {
                f"{str(seed_key)}::{str(bundle_key)}": str(stealer_key)
                for (seed_key, bundle_key), stealer_key in sorted(stealer_bundle_map.items(), key=lambda item: (str(item[0][0]), str(item[0][1])))
            },
        }

    def replay_operator_certification(
        self,
        *,
        bundle_keys: Sequence[str] | None = None,
        max_new_tokens: int = 3,
        top_k: int = 8,
        max_edits_per_step_override: int = 2,
    ) -> dict[str, Any]:
        packet = self.build_controller_packet()
        strategy_hints = packet.get("strategy_hints") if isinstance(packet.get("strategy_hints"), Mapping) else {}
        candidate_edits = strategy_hints.get("kv_candidate_edits")
        if not isinstance(candidate_edits, SequenceABC) or isinstance(candidate_edits, (str, bytes, bytearray)):
            candidate_edits = []
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in candidate_edits:
            if not isinstance(item, Mapping):
                continue
            bundle_key = str(item.get("bundle_key", "") or "")
            if not bundle_key:
                continue
            grouped.setdefault(bundle_key, []).append(dict(item))
        chosen_bundle_keys = [str(item) for item in (bundle_keys or ()) if str(item)]
        if not chosen_bundle_keys:
            for key in (
                strategy_hints.get("base_winner_bundle_key"),
                strategy_hints.get("challenger_bundle_key"),
            ):
                if key not in (None, "") and str(key) not in chosen_bundle_keys:
                    chosen_bundle_keys.append(str(key))
        evaluations: list[dict[str, Any]] = []
        for bundle_key in chosen_bundle_keys:
            members = grouped.get(bundle_key, [])
            if not members:
                continue
            for member in members:
                evaluations.append(
                    self.replay_candidate_edits_actual_delta(
                        [member],
                        max_new_tokens=max_new_tokens,
                        top_k=top_k,
                        label=f"single:{bundle_key}:{member.get('site')}",
                    )
                )
            if len(members) > 1:
                evaluations.append(
                    self.replay_candidate_edits_actual_delta(
                        members,
                        max_new_tokens=max_new_tokens,
                        top_k=top_k,
                        max_edits_per_step_override=max_edits_per_step_override,
                        label=f"pair:{bundle_key}",
                    )
                )
        certifications = self._summarize_operator_certifications(evaluations)
        self._operator_certification_table = {
            str(item.get("operator_family_key", "") or ""): dict(item)
            for item in certifications
            if str(item.get("operator_family_key", "") or "")
        }
        self._operator_bridge_plan_table = {}
        return {
            "base_winner_bundle_key": strategy_hints.get("base_winner_bundle_key"),
            "challenger_bundle_key": strategy_hints.get("challenger_bundle_key"),
            "selected_bundle_key": strategy_hints.get("selected_bundle_key"),
            "selection_source": strategy_hints.get("selection_source"),
            "evaluations": evaluations,
            "operator_certifications": certifications,
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
        policy_override: HarnessPolicy | None = None,
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
                if (
                    policy_override is not None
                    and isinstance(packet, Mapping)
                    and isinstance(packet.get("budget"), Mapping)
                ):
                    packet = dict(packet)
                    budget = dict(packet["budget"])
                    budget["edits_left_this_step"] = max(
                        int(budget.get("edits_left_this_step", 0) or 0),
                        int(policy_override.global_budget.max_edits_per_step),
                    )
                    packet["budget"] = budget
                ctx = StepContext(packet=packet, runtime_state=self.runtime_state, traces={}, stats={}, adapter=self.adapter, active_edits={})
                compiled = compile_command(command, packet, ctx, policy=policy_override)
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
                "continuation_token_ids": [int(token_id) for token_id in continuation_ids],
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
            "surface_family_key": None if metadata.get("surface_family_key") is None else str(metadata.get("surface_family_key")),
            "operator_recipe_id": None if metadata.get("operator_recipe_id") is None else str(metadata.get("operator_recipe_id")),
            "operator_recipe_seed_key": None
            if metadata.get("operator_recipe_seed_key") is None
            else str(metadata.get("operator_recipe_seed_key")),
            "bundle_key": None if metadata.get("bundle_key") is None else str(metadata.get("bundle_key")),
            "objective_bundle_key": None
            if metadata.get("objective_bundle_key") is None
            else str(metadata.get("objective_bundle_key")),
            "step_actuator_bundle_key": None
            if metadata.get("step_actuator_bundle_key") is None
            else str(metadata.get("step_actuator_bundle_key")),
            "apply_kind": None if metadata.get("apply_kind") is None else str(metadata.get("apply_kind")),
            "production_apply_allowed": None
            if metadata.get("production_apply_allowed") is None
            else bool(metadata.get("production_apply_allowed")),
            "production_policy_would_apply": None
            if metadata.get("production_policy_would_apply") is None
            else bool(metadata.get("production_policy_would_apply")),
            "certified_for_apply": None
            if metadata.get("certified_for_apply") is None
            else bool(metadata.get("certified_for_apply")),
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

    def _resolve_token_selector_position(self, token_selector: Mapping[str, Any]) -> int | None:
        if not isinstance(token_selector, Mapping):
            return None
        token_count = int(self._current_token_tensor().shape[-1])
        if token_count <= 0:
            return None
        mode = str(token_selector.get("mode", "") or "")
        if mode == "last":
            return token_count - 1
        if mode == "index":
            value = token_selector.get("value")
            if isinstance(value, bool) or not isinstance(value, int):
                return None
            index = int(value)
            if index < 0:
                index += token_count
            if 0 <= index < token_count:
                return index
        return None

    def _readout_sidecar_context(self) -> StepContext | None:
        if getattr(self.runtime_state, "last_cache", None) is None:
            return None
        if hasattr(self.runtime_state, "set_trace_alignment"):
            self.runtime_state.set_trace_alignment(self._steps)
        return StepContext(
            packet={"step": self._steps},
            runtime_state=self.runtime_state,
            traces={},
            stats={},
            adapter=self.adapter,
            active_edits={},
        )

    def _readout_sidecar_boundary_sites(self, *, ctx: StepContext) -> tuple[ReadoutSidecarSiteCapture, ...]:
        captures: list[ReadoutSidecarSiteCapture] = []
        for surface in self.surface_catalog:
            target = surface.target
            if getattr(target, "kind", None) != "activation" or getattr(target, "site", None) != "resid_pre":
                continue
            token_selector = self._token_to_dict(target.token)
            token_mode = str(token_selector.get("mode", "") or "")
            token_value = token_selector.get("value")
            if token_mode not in {"last", "index"}:
                continue
            if token_mode == "index" and token_value != -2:
                continue
            bound = self.adapter.bind_surface(surface)
            vector = self._surface_probe_vector(surface, bound, ctx)
            if vector is None:
                continue
            captures.append(
                ReadoutSidecarSiteCapture(
                    role="answer_boundary_prev" if token_mode == "index" else "answer_boundary_last",
                    layer=int(getattr(target, "layer", 0) or 0),
                    token_selector=token_selector,
                    surface_id=str(surface.surface_id),
                    position=self._resolve_token_selector_position(token_selector),
                    piece=None,
                    vector=vector.detach().reshape(-1).cpu().float(),
                    metadata={"site": str(getattr(target, "site", "") or "")},
                )
            )
        captures.sort(
            key=lambda item: (
                int(item.layer),
                0 if item.role.endswith("prev") else 1,
                str(item.surface_id or ""),
            )
        )
        return tuple(captures[:6])

    def _readout_sidecar_source_sites(
        self,
        *,
        ctx: StepContext,
        control_phase_hint: str,
        answer_readout_canary: Mapping[str, Any] | None,
        boundary_sites: Sequence[ReadoutSidecarSiteCapture],
    ) -> tuple[ReadoutSidecarSiteCapture, ...]:
        target_terms = self._ordered_missing_terms_for_phase(
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
            max_terms=2,
        )
        layers = sorted({int(site.layer) for site in boundary_sites})
        if not layers:
            layers = list(self._feature_scan_layers()[:2])
        captures: list[ReadoutSidecarSiteCapture] = []
        seen_keys: set[tuple[str, int, int, int]] = set()
        for term in target_terms:
            spans = self._prompt_term_spans(term, max_spans=3)
            source_body_spans = [span for span in spans if str(span.get("provenance_class", "")) == "source_body"]
            selected_spans = source_body_spans[:2] if source_body_spans else spans[:1]
            for span in selected_spans:
                start = int(span.get("start", 0) or 0)
                end = int(span.get("end", start + 1) or (start + 1))
                for layer in layers[:2]:
                    capture_key = (str(term), int(layer), start, end)
                    if capture_key in seen_keys:
                        continue
                    seen_keys.add(capture_key)
                    ref = {
                        "scope": "runtime",
                        "worker": self.worker_id,
                        "tensor": "hidden",
                        "layer": int(layer),
                        "token": {"mode": "span", "start": start, "end": end, "pool": "mean"},
                    }
                    try:
                        vector = self.adapter.read_ref(ref, ctx).detach().reshape(-1).cpu().float()
                    except Exception:
                        continue
                    captures.append(
                        ReadoutSidecarSiteCapture(
                            role="source_body_exact_span_mean",
                            layer=int(layer),
                            token_selector={"mode": "span", "start": start, "end": end, "pool": "mean"},
                            vector=vector,
                            term=str(term),
                            provenance_class=str(span.get("provenance_class", "misc_prompt") or "misc_prompt"),
                            piece=str(span.get("text", "") or ""),
                            span=(start, end),
                            metadata={
                                "span_kind": str(span.get("span_kind", "") or ""),
                                "segment_kind": str(span.get("segment_kind", "") or ""),
                            },
                        )
                    )
        captures.sort(
            key=lambda item: (
                -self._provenance_weight(str(item.provenance_class or "misc_prompt")),
                str(item.term or ""),
                int(item.layer),
                0 if item.role.endswith("mean") else 1,
                int(item.span[0] if item.span is not None else 0),
            )
        )
        return tuple(captures[:8])

    def _build_readout_sidecar_capture(
        self,
        *,
        control_phase_hint: str,
        answer_readout_canary: Mapping[str, Any] | None,
    ) -> ReadoutSidecarCapture | None:
        self._latest_readout_sidecar_capture = None
        self._latest_readout_sidecar_capture_summary = None
        should_capture = self.readout_sidecar_analyzer is not None or control_phase_hint in {"readout_escape", "shot_mode"}
        if not should_capture:
            return None
        if not isinstance(answer_readout_canary, Mapping):
            return None
        if not self._feedback_terms(("entity_recall_terms", "missing_required_terms", "missing_keywords", "missing_summary_terms")):
            return None
        ctx = self._readout_sidecar_context()
        if ctx is None:
            return None
        boundary_sites = self._readout_sidecar_boundary_sites(ctx=ctx)
        source_sites = self._readout_sidecar_source_sites(
            ctx=ctx,
            control_phase_hint=control_phase_hint,
            answer_readout_canary=answer_readout_canary,
            boundary_sites=boundary_sites,
        )
        if not boundary_sites and not source_sites:
            return None
        capture = ReadoutSidecarCapture(
            run_id=self.run_id,
            episode_id=self.episode_id,
            worker_id=self.worker_id,
            step=int(self._steps),
            control_phase_hint=str(control_phase_hint),
            answer_readout_canary=dict(answer_readout_canary),
            answer_sites=tuple(boundary_sites),
            source_sites=tuple(source_sites),
            metadata={
                "prompt_hash": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
                "target_terms": self._ordered_missing_terms_for_phase(
                    control_phase_hint=control_phase_hint,
                    answer_readout_canary=answer_readout_canary,
                    max_terms=3,
                ),
            },
        )
        self._latest_readout_sidecar_capture = capture
        self._latest_readout_sidecar_capture_summary = capture.summary()
        return capture

    def _analyze_readout_sidecar_capture(self, capture: ReadoutSidecarCapture | None) -> dict[str, Any]:
        self._latest_readout_sidecar_hints = {}
        if capture is None or self.readout_sidecar_analyzer is None:
            return {}
        try:
            raw_hints = self.readout_sidecar_analyzer(capture)
        except Exception as exc:
            analyzer_name = getattr(self.readout_sidecar_analyzer, "__name__", type(self.readout_sidecar_analyzer).__name__)
            raw_hints = {"analyzer_name": analyzer_name, "analyzer_error": str(exc)}
        hints = normalize_readout_sidecar_hints(raw_hints)
        if "analyzer_name" not in hints:
            hints["analyzer_name"] = getattr(
                self.readout_sidecar_analyzer,
                "__name__",
                type(self.readout_sidecar_analyzer).__name__,
            )
        self._latest_readout_sidecar_hints = hints
        return hints

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
