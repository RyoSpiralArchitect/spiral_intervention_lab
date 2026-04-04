from __future__ import annotations

import hashlib
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .adapter import ModelAdapter
from .codecs import TextCodec, resolve_text_codec
from .compiler import StepContext
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
}
_DECODER_CONTROL_MODES = set(_DECODER_CONTROL_MODE_SPECS)
_CONTROLLER_MEMORY_STRING_FIELDS = (
    "hypothesis",
    "expected_effect",
    "observed_outcome",
    "why_failed_or_helped",
    "next_change",
    "stop_condition",
)
_CONTROLLER_MEMORY_ALLOWED_OUTCOMES = {"unknown", "helpful", "harmful", "neutral", "mixed"}
_SOFT_CONSTRAINT_FEEDBACK_KEYS = ("missing_required_terms", "missing_keywords", "missing_summary_terms")


@dataclass
class _TokenSegment:
    kind: str
    token_ids: list[int]


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
    confidence = value.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        entry["confidence"] = max(0.0, min(1.0, float(confidence)))
    if decision is not None:
        entry["decision"] = str(decision)
    if recorded_step is not None:
        entry["recorded_step"] = int(recorded_step)
    return entry or None


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
        trace_recorder: StepAlignedTraceRecorder | None = None,
        controller_reflection_mode: str = "off",
        controller_memory_window: int = 3,
        decoder_control_mode: str = "off",
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
        self.controller_reflection_mode = _normalize_controller_reflection_mode(controller_reflection_mode)
        self.controller_memory_window = max(1, int(controller_memory_window))
        self.decoder_control_mode = _normalize_decoder_control_mode(decoder_control_mode)

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
                "decoder_control_mode": self.decoder_control_mode,
                "decoder_control_track": str(self._last_decoder_control.get("track", "baseline")),
                "decoder_rescue_active": bool(self._last_decoder_control.get("active", False)),
                "decoder_candidate_prune_active": bool(self._last_decoder_control.get("candidate_prune_active", False)),
                "decoder_constraint_target_count": int(self._last_decoder_control.get("constraint_target_count", 0) or 0),
            },
            "surface_catalog": [dict(surface) for surface in self._surface_catalog_raw],
            "probe_frames": self._build_probe_frames(),
            "trace_bank": self._build_trace_bank(),
            "active_edits": active_edits,
            "recent_effects": [dict(effect) for effect in self._recent_effects],
            "recent_effect_summary": dict(self._recent_effect_summary),
            "budget": self._budget_state(active_edits),
            "task_feedback": dict(self._last_task_feedback),
        }
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
        }
        if self.decoder_control_mode == "off":
            return next_logits, state

        tokens = self._output_token_ids()
        rescue_active = cycle_length is not None or self._repeat_flag() or self._no_progress_steps > 0
        if not tokens or not rescue_active:
            return next_logits, state

        adjusted, loop_state = self._apply_loop_aware_penalties(next_logits.clone(), cycle_length=cycle_length)
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

    def _soft_constraint_target_token_ids(self, *, vocab_size: int) -> tuple[list[int], list[str]]:
        if bool(self._last_task_feedback.get("done", False)):
            return [], []
        terms: list[str] = []
        seen_terms: set[str] = set()
        for key in _SOFT_CONSTRAINT_FEEDBACK_KEYS:
            value = self._last_task_feedback.get(key)
            if not isinstance(value, SequenceABC) or isinstance(value, (str, bytes, bytearray)):
                continue
            for item in value:
                term = " ".join(str(item).split()).strip()
                if not term or term in seen_terms:
                    continue
                seen_terms.add(term)
                terms.append(term)
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

    def _constraint_token_variants(self, term: str) -> tuple[str, ...]:
        stripped = " ".join(str(term).split()).strip()
        variants: list[str] = []
        for candidate in (
            stripped,
            stripped.lower(),
            f" {stripped}",
            f" {stripped.lower()}",
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

    def _task_feedback(self) -> dict[str, Any]:
        return dict(self._last_task_feedback)

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
        label = str(feedback.get("progress_label", "") or "").strip().lower()
        if label == "progressing":
            return 0.25
        if label == "regressing":
            return -0.5
        return 0.0

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
