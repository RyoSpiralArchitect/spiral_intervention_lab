from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from .adapter import ModelAdapter
from .codecs import TextCodec, resolve_text_codec
from .compiler import StepContext
from .effects import build_edit_effect
from .schema import SurfaceInfo
from .trace_recorder import StepAlignedTrace, StepAlignedTraceRecorder


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
        max_edits_per_step: int = 1,
        max_edits_per_run: int = 4,
        max_total_alpha: float = 0.5,
        max_active_patch_slots: int = 1,
        generated_tail_chars: int = 80,
        recent_token_count: int = 6,
        stop_token_ids: Sequence[int] | None = None,
        stop_checker: Callable[[str], bool] | None = None,
        trace_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        task_feedback_fn: Callable[[str], Mapping[str, Any] | None] | None = None,
        trace_recorder: StepAlignedTraceRecorder | None = None,
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
        self.max_edits_per_step = max_edits_per_step
        self.max_edits_per_run = max_edits_per_run
        self.max_total_alpha = max_total_alpha
        self.max_active_patch_slots = max_active_patch_slots
        self.generated_tail_chars = generated_tail_chars
        self.recent_token_count = recent_token_count
        self.stop_token_ids = set(int(token_id) for token_id in (stop_token_ids or ()))
        self.stop_checker = stop_checker
        self.trace_metadata = {str(trace_id): dict(meta) for trace_id, meta in (trace_metadata or {}).items()}
        self.task_feedback_fn = task_feedback_fn

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
        self._last_status = "thinking"
        self._previous_probe_vectors: dict[str, torch.Tensor] = {}
        self._recent_effects: list[dict[str, Any]] = []
        self._pending_effects: list[dict[str, Any]] = []
        self._seen_registration_ids: set[str] = set()
        self._spent_budget_alpha: dict[str, float] = {}
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
        next_logits = logits[0, -1].detach()
        next_token = int(torch.argmax(next_logits).item())
        self._append_output_token(next_token)
        self._steps += 1
        self._last_metrics = self._compute_metrics(next_logits)
        self._update_status()
        self._record_trace_step(emitted_token_id=next_token)
        self._last_packet = None

    def done(self) -> bool:
        output_tokens = self._output_token_ids()
        if self.max_generated_tokens > 0 and len(output_tokens) >= self.max_generated_tokens:
            return True
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
            "telemetry": dict(self._last_metrics) | {"repeat_flag": self._repeat_flag(), "no_progress_steps": self._no_progress_steps},
            "surface_catalog": [dict(surface) for surface in self._surface_catalog_raw],
            "probe_frames": self._build_probe_frames(),
            "trace_bank": self._build_trace_bank(),
            "active_edits": active_edits,
            "recent_effects": [dict(effect) for effect in self._recent_effects],
            "budget": self._budget_state(active_edits),
            "task_feedback": self._task_feedback(),
        }
        self._last_packet = packet
        return packet

    def observe_recent_effects(self) -> None:
        current_metrics = dict(self._last_metrics)
        completed = [
            build_edit_effect(
                edit_id=pending["edit_id"],
                surface_id=pending["surface_id"],
                observed_window_steps=1,
                before=pending["before"],
                after=current_metrics,
            )
            for pending in self._pending_effects
        ]
        if completed:
            self._recent_effects.extend(completed)
            self._recent_effects = self._recent_effects[-8:]
        self._pending_effects = []

        for active in self._collect_active_edits():
            edit_id = str(active["edit_id"])
            if edit_id in self._seen_registration_ids:
                continue
            self._seen_registration_ids.add(edit_id)
            budget_key = str(active.get("budget_key", edit_id.split(":", 1)[0]))
            self._spent_budget_alpha.setdefault(budget_key, float(active["alpha"]))
            self._pending_effects.append(
                {
                    "edit_id": edit_id,
                    "surface_id": str(active["surface_id"]),
                    "before": current_metrics,
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
        self._last_status = "thinking"
        self._previous_probe_vectors = {}
        self._recent_effects = []
        self._pending_effects = []
        self._seen_registration_ids = set()
        self._spent_budget_alpha = {}
        self._last_packet = None
        self._no_progress_steps = 0
        if self.trace_recorder is not None:
            self.trace_recorder.reset()

    def _current_token_tensor(self) -> torch.Tensor:
        tokens: list[int] = []
        for segment in self._segments:
            tokens.extend(segment.token_ids)
        return torch.tensor([tokens], dtype=torch.long)

    def _append_output_token(self, token_id: int) -> None:
        if self._segments and self._segments[-1].kind == "output":
            self._segments[-1].token_ids.append(int(token_id))
            return
        self._segments.append(_TokenSegment(kind="output", token_ids=[int(token_id)]))

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
        return self._repetition_score() >= 0.66

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
        feedback = {"done": self.done(), "progress_label": "stalled" if self._last_status == "looping" else "progressing"}
        if self.task_feedback_fn is None:
            return feedback
        custom = self.task_feedback_fn(self.final_text()) or {}
        return feedback | dict(custom)

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
            "budget_key": str(metadata.get("budget_key", str(edit_id).split(":", 1)[0])),
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
        return {
            "edits_left_this_step": self.max_edits_per_step,
            "edits_left_this_run": max(0, self.max_edits_per_run - len(self._spent_budget_alpha)),
            "alpha_left_total": max(0.0, self.max_total_alpha - spent_alpha),
            "active_patch_slots_left": max(0, self.max_active_patch_slots - active_patch_slots),
            "rollbackable_ids": [str(edit["edit_id"]) for edit in active_edits if bool(edit.get("revertible", True))],
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
