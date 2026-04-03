from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TYPE_CHECKING

import torch

from .overlays import OverlayHandle

if TYPE_CHECKING:
    from .trace_recorder import StepAlignedTrace

try:
    from transformer_lens.hook_points import LensHandle
except Exception:  # pragma: no cover - optional dependency at import time
    LensHandle = Any  # type: ignore[assignment]


@dataclass
class HookRegistration:
    edit_id: str
    hook_name: str
    lens_handle: LensHandle
    ttl_steps: int
    revertible: bool
    metadata: dict[str, Any]


@dataclass
class OverlayRegistration:
    edit_id: str
    handle: OverlayHandle
    ttl_steps: int
    revertible: bool
    metadata: dict[str, Any]


class HookedTransformerRuntimeState:
    def __init__(
        self,
        model: Any,
        *,
        seed: int = 0,
        trace_caches: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
        running_stats: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self.model = model
        self.seed = seed
        self.last_tokens: torch.Tensor | None = None
        self.last_logits: torch.Tensor | None = None
        self.last_cache: dict[str, torch.Tensor] | None = None
        self.trace_caches: dict[str, dict[str, torch.Tensor]] = {
            trace_id: self._freeze_cache(cache) for trace_id, cache in (trace_caches or {}).items()
        }
        self.trace_sequences: dict[str, "StepAlignedTrace"] = {}
        self.trace_alignment_step: int | None = None
        self.running_stats: dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in (running_stats or {}).items()
        }
        self.hooks: dict[str, HookRegistration] = {}
        self.overlays: dict[str, OverlayRegistration] = {}

    def _freeze_cache(self, cache: Mapping[str, torch.Tensor] | Any) -> dict[str, torch.Tensor]:
        if cache is None:
            return {}
        return {str(name): tensor.detach().clone() for name, tensor in cache.items()}

    def run_with_cache(self, model_input: Any, *, return_type: str = "logits", **kwargs: Any) -> tuple[Any, Any]:
        output, cache = self.model.run_with_cache(model_input, return_type=return_type, **kwargs)
        if isinstance(model_input, torch.Tensor):
            self.last_tokens = model_input.detach().clone()
        else:
            self.last_tokens = None
        if isinstance(output, torch.Tensor):
            self.last_logits = output.detach().clone()
        else:
            self.last_logits = None
        self.last_cache = self._freeze_cache(cache)
        return output, cache

    def snapshot_last_cache(self, trace_id: str) -> None:
        if self.last_cache is None:
            raise ValueError("cannot snapshot trace before run_with_cache has populated last_cache")
        self.trace_caches[trace_id] = self._freeze_cache(self.last_cache)

    def put_trace_cache(self, trace_id: str, cache: Mapping[str, torch.Tensor] | Any) -> None:
        self.trace_caches[trace_id] = self._freeze_cache(cache)

    def put_step_trace(self, trace_id: str, trace: "StepAlignedTrace") -> None:
        self.trace_sequences[trace_id] = trace

    def set_trace_alignment(self, step: int | None) -> None:
        self.trace_alignment_step = None if step is None else int(step)

    def get_cache(
        self,
        scope: str,
        trace_id: str | None = None,
        *,
        step: int | None = None,
    ) -> Mapping[str, torch.Tensor]:
        if scope == "runtime":
            if self.last_cache is None:
                raise KeyError("runtime cache is empty; call run_with_cache first")
            return self.last_cache
        if scope == "trace":
            if trace_id is None:
                raise KeyError("trace scope requires trace_id")
            trace = self.trace_sequences.get(trace_id)
            if trace is not None:
                aligned_step = self.trace_alignment_step if step is None else step
                return trace.aligned_cache(aligned_step)
            try:
                return self.trace_caches[trace_id]
            except KeyError as exc:
                raise KeyError(f"trace cache '{trace_id}' is not available") from exc
        raise KeyError(f"unsupported cache scope '{scope}'")

    def read_stat_tensor(self, ref: Mapping[str, Any]) -> torch.Tensor:
        stat = ref.get("stat")
        if stat is None:
            raise KeyError("stats scope requires ref.stat")
        key = self._stat_key(ref)
        try:
            return self.running_stats[key]
        except KeyError as exc:
            raise KeyError(f"running stat '{key}' is not available") from exc

    def _stat_key(self, ref: Mapping[str, Any]) -> str:
        token = ref.get("token") or {"mode": "last"}
        token_mode = token.get("mode", "last")
        head = ref.get("head")
        head_suffix = "" if head is None else f":head={head}"
        return f"{ref['tensor']}@L{ref['layer']}:{token_mode}{head_suffix}:{ref['stat']}"

    def register_hook(
        self,
        *,
        hook_name: str,
        hook_fn: Any,
        edit_id: str,
        ttl_steps: int,
        revertible: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        hook_point = self.model.hook_dict[hook_name]
        before_len = len(hook_point.fwd_hooks)
        hook_point.add_hook(hook_fn, dir="fwd")
        if len(hook_point.fwd_hooks) <= before_len:
            raise RuntimeError(f"failed to register forward hook on '{hook_name}'")
        lens_handle = hook_point.fwd_hooks[-1]
        self.hooks[edit_id] = HookRegistration(
            edit_id=edit_id,
            hook_name=hook_name,
            lens_handle=lens_handle,
            ttl_steps=ttl_steps,
            revertible=revertible,
            metadata=dict(metadata or {}),
        )

    def register_overlay(
        self,
        *,
        edit_id: str,
        handle: OverlayHandle,
        ttl_steps: int,
        revertible: bool,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        handle.attach()
        self.overlays[edit_id] = OverlayRegistration(
            edit_id=edit_id,
            handle=handle,
            ttl_steps=ttl_steps,
            revertible=revertible,
            metadata=dict(metadata or {}),
        )

    def remove_edit(self, edit_id: str) -> None:
        registration = self.hooks.pop(edit_id, None)
        if registration is not None:
            registration.lens_handle.hook.remove()
            hook_point = self.model.hook_dict[registration.hook_name]
            hook_point.fwd_hooks = [handle for handle in hook_point.fwd_hooks if handle is not registration.lens_handle]
            return
        overlay = self.overlays.pop(edit_id, None)
        if overlay is not None:
            overlay.handle.detach()

    def tick_ttl(self) -> None:
        expired: list[str] = []
        for edit_id, registration in list(self.hooks.items()):
            registration.ttl_steps -= 1
            if registration.ttl_steps <= 0:
                expired.append(edit_id)
        for edit_id, overlay in list(self.overlays.items()):
            overlay.ttl_steps -= 1
            overlay.handle.tick()
            if overlay.ttl_steps <= 0:
                expired.append(edit_id)
        for edit_id in expired:
            self.remove_edit(edit_id)

    def cleanup_expired(self) -> None:
        return None

    def clear_edits(self) -> None:
        for edit_id in list(self.hooks):
            self.remove_edit(edit_id)
        for edit_id in list(self.overlays):
            self.remove_edit(edit_id)
