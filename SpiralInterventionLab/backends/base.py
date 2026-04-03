from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import torch


@dataclass(frozen=True)
class BackendCapabilities:
    backend_name: str
    device: str | None = None
    supports_logits: bool = True
    supports_hidden_state: bool = False
    supports_kv_cache: bool = False
    supports_activation_hooks: bool = False
    supports_rank1_patch: bool = False
    supports_step_trace: bool = False
    supports_prompt_hints: bool = True
    supports_mps: bool = False
    supports_mlx: bool = False


@dataclass
class BackendStepResult:
    token_id: int
    token_text: str
    logits: torch.Tensor | None = None
    hidden_state: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AutoregressiveBackend(ABC):
    capabilities: BackendCapabilities

    @abstractmethod
    def reset(self, prompt: str) -> None:
        ...

    @abstractmethod
    def step(self) -> BackendStepResult:
        ...

    @abstractmethod
    def append_prompt_hint(self, hint: str) -> bool:
        ...

    @abstractmethod
    def current_tokens(self) -> torch.Tensor:
        ...

    @abstractmethod
    def output_token_ids(self) -> list[int]:
        ...

    @abstractmethod
    def decode_tokens(self, token_ids: Sequence[int] | torch.Tensor) -> str:
        ...

    @abstractmethod
    def final_text(self) -> str:
        ...

    @abstractmethod
    def last_logits_tensor(self) -> torch.Tensor | None:
        ...

    def backend_done(self) -> bool:
        return False


class LocalBackendWorkerRuntime:
    def __init__(
        self,
        *,
        backend: AutoregressiveBackend,
        run_id: str = "run_local",
        episode_id: str = "episode_local",
        worker_id: str = "worker_local",
        task_id: str = "task_local",
        task_view_mode: str = "redacted",
        goal_hint: str | None = None,
        constraints: Sequence[str] | None = None,
        max_generated_tokens: int = 32,
        min_generated_tokens: int = 0,
        generated_tail_chars: int = 80,
        recent_token_count: int = 6,
        stop_token_ids: Sequence[int] | None = None,
        stop_checker: Callable[[str], bool] | None = None,
        task_feedback_fn: Callable[[str], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self.backend = backend
        self.run_id = run_id
        self.episode_id = episode_id
        self.worker_id = worker_id
        self.task_id = task_id
        self.task_view_mode = task_view_mode
        self.goal_hint = goal_hint
        self.constraints = tuple(constraints or ())
        self.max_generated_tokens = int(max_generated_tokens)
        self.min_generated_tokens = max(0, int(min_generated_tokens))
        self.generated_tail_chars = int(generated_tail_chars)
        self.recent_token_count = int(recent_token_count)
        self.stop_token_ids = {int(token_id) for token_id in (stop_token_ids or ())}
        self.stop_checker = stop_checker
        self.task_feedback_fn = task_feedback_fn

        self.prompt = ""
        self._steps = 0
        self._last_metrics: dict[str, float] = {
            "entropy": 0.0,
            "top1_margin": 0.0,
            "repetition_score": 0.0,
        }
        self._last_status = "thinking"
        self._no_progress_steps = 0

    def reset(self, prompt: str) -> None:
        self.prompt = prompt
        self._steps = 0
        self._last_metrics = {"entropy": 0.0, "top1_margin": 0.0, "repetition_score": 0.0}
        self._last_status = "thinking"
        self._no_progress_steps = 0
        self.backend.reset(prompt)

    def step(self) -> None:
        result = self.backend.step()
        self._steps += 1
        logits = result.logits if isinstance(result.logits, torch.Tensor) else self.backend.last_logits_tensor()
        self._last_metrics = self._compute_metrics(logits)
        self._update_status()

    def done(self) -> bool:
        output_tokens = self.backend.output_token_ids()
        generated_tokens = len(output_tokens)
        if self.max_generated_tokens > 0 and generated_tokens >= self.max_generated_tokens:
            return True
        if generated_tokens < self.min_generated_tokens:
            return False
        if self.backend.backend_done():
            return True
        if output_tokens and output_tokens[-1] in self.stop_token_ids:
            return True
        if self.stop_checker is not None and self.stop_checker(self.final_text()):
            return True
        return False

    def append_prompt_hint(self, hint: str) -> bool:
        return self.backend.append_prompt_hint(hint)

    def build_controller_packet(self) -> dict[str, Any]:
        return {
            "version": "0.1",
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "worker_id": self.worker_id,
            "step": self._steps,
            "horizon": {
                "generated_tokens": len(self.backend.output_token_ids()),
                "max_generated_tokens": self.max_generated_tokens,
                "done": self.done(),
            },
            "task_view": self._task_view(),
            "worker_view": self._worker_view(),
            "telemetry": dict(self._last_metrics) | {"repeat_flag": self._repeat_flag(), "no_progress_steps": self._no_progress_steps},
            "surface_catalog": [],
            "probe_frames": [],
            "trace_bank": [],
            "active_edits": [],
            "recent_effects": [],
            "budget": {
                "edits_left_this_step": 0,
                "edits_left_this_run": 0,
                "alpha_left_total": 0.0,
                "active_patch_slots_left": 0,
                "rollbackable_ids": [],
            },
            "task_feedback": self._task_feedback(),
        }

    def observe_recent_effects(self) -> None:
        return None

    def tick_ttl(self) -> None:
        return None

    def cleanup_expired(self) -> None:
        return None

    def final_text(self) -> str:
        return self.backend.final_text()

    def current_tokens(self) -> torch.Tensor:
        return self.backend.current_tokens().detach().clone()

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
        output_tokens = self.backend.output_token_ids()
        recent_tokens = output_tokens[-self.recent_token_count :]
        return {
            "generated_tail": self.final_text()[-self.generated_tail_chars :],
            "recent_tokens": [self.backend.decode_tokens([token_id]) for token_id in recent_tokens],
            "status": self._last_status,
        }

    def _task_feedback(self) -> dict[str, Any]:
        feedback = {"done": self.done(), "progress_label": "stalled" if self._last_status == "looping" else "progressing"}
        if self.task_feedback_fn is None:
            return feedback
        custom = self.task_feedback_fn(self.final_text()) or {}
        return feedback | dict(custom)

    def _compute_metrics(self, next_logits: torch.Tensor | None) -> dict[str, float]:
        if next_logits is None:
            return {
                "entropy": 0.0,
                "top1_margin": 0.0,
                "repetition_score": self._repetition_score(),
            }
        logits = next_logits.detach().float().reshape(-1)
        probs = torch.softmax(logits, dim=-1)
        entropy = float((-(probs * probs.clamp_min(1e-8).log()).sum()).item())
        top_probs = torch.topk(probs, k=min(2, probs.shape[-1]), dim=-1).values
        margin = float((top_probs[0] - top_probs[1]).item()) if top_probs.shape[0] >= 2 else float(top_probs[0].item())
        return {
            "entropy": entropy,
            "top1_margin": margin,
            "repetition_score": self._repetition_score(),
        }

    def _repetition_score(self) -> float:
        tokens = self.backend.output_token_ids()
        if len(tokens) < 2:
            return 0.0
        window = tokens[-4:]
        repeated = max(window.count(token_id) for token_id in set(window))
        return float((repeated - 1) / max(1, len(window) - 1))

    def _repeat_flag(self) -> bool:
        tokens = self.backend.output_token_ids()
        if len(tokens) >= 3 and len(set(tokens[-3:])) == 1:
            return True
        return self._repetition_score() >= 0.66

    def _update_status(self) -> None:
        if self._repeat_flag():
            self._no_progress_steps += 1
            self._last_status = "looping"
            return
        self._no_progress_steps = 0
        self._last_status = "acting" if self.backend.output_token_ids() else "thinking"
