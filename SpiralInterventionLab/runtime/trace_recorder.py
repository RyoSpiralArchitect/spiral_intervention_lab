from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch

from .adapter import ModelAdapter
from .schema import SurfaceInfo


def _freeze_cache(cache: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {str(name): tensor.detach().clone() for name, tensor in cache.items()}


def _freeze_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): item for key, item in value.items()}


@dataclass(frozen=True)
class StepTraceFrame:
    step: int
    generated_tokens: int
    emitted_token_id: int | None
    output_text: str
    telemetry: Mapping[str, Any] = field(default_factory=dict)
    active_edits: tuple[Mapping[str, Any], ...] = ()
    cache: Mapping[str, torch.Tensor] = field(default_factory=dict)

    def frozen_cache(self) -> dict[str, torch.Tensor]:
        return _freeze_cache(self.cache)


@dataclass(frozen=True)
class StepAlignedTrace:
    trace_id: str
    frames: tuple[StepTraceFrame, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def aligned_frame(self, step: int | None = None) -> StepTraceFrame:
        if not self.frames:
            raise KeyError(f"trace '{self.trace_id}' has no recorded frames")
        if step is None:
            return self.frames[-1]
        chosen = self.frames[0]
        for frame in self.frames:
            if frame.step > step:
                break
            chosen = frame
        return chosen

    def aligned_cache(self, step: int | None = None) -> Mapping[str, torch.Tensor]:
        return self.aligned_frame(step).frozen_cache()

    @property
    def step_count(self) -> int:
        return len(self.frames)


class StepAlignedTraceRecorder:
    def __init__(
        self,
        *,
        surface_catalog: Sequence[SurfaceInfo | Mapping[str, Any]],
        adapter: ModelAdapter,
    ) -> None:
        self.adapter = adapter
        self.surface_catalog = tuple(
            surface if isinstance(surface, SurfaceInfo) else SurfaceInfo.from_dict(surface) for surface in surface_catalog
        )
        hook_names: list[str] = []
        for surface in self.surface_catalog:
            bound = self.adapter.bind_surface(surface)
            if bound.kind in {"activation", "cache"} and bound.hook_name is not None:
                hook_names.append(bound.hook_name)
        self.hook_names = tuple(sorted(set(hook_names)))
        self._frames: list[StepTraceFrame] = []

    def reset(self) -> None:
        self._frames = []

    def record_step(
        self,
        *,
        runtime_state: Any,
        step: int,
        generated_tokens: int,
        emitted_token_id: int | None,
        output_text: str,
        telemetry: Mapping[str, Any],
        active_edits: Sequence[Mapping[str, Any]],
    ) -> None:
        cache = getattr(runtime_state, "last_cache", None)
        if cache is None:
            return
        selected_cache = {
            hook_name: cache[hook_name].detach().clone()
            for hook_name in self.hook_names
            if hook_name in cache
        }
        self._frames.append(
            StepTraceFrame(
                step=int(step),
                generated_tokens=int(generated_tokens),
                emitted_token_id=None if emitted_token_id is None else int(emitted_token_id),
                output_text=str(output_text),
                telemetry=_freeze_mapping(telemetry),
                active_edits=tuple(dict(edit) for edit in active_edits),
                cache=selected_cache,
            )
        )

    def snapshot(self, trace_id: str, *, metadata: Mapping[str, Any] | None = None) -> StepAlignedTrace:
        merged_metadata = dict(metadata or {})
        merged_metadata.setdefault("step_count", len(self._frames))
        return StepAlignedTrace(
            trace_id=str(trace_id),
            frames=tuple(self._frames),
            metadata=merged_metadata,
        )

    @property
    def step_count(self) -> int:
        return len(self._frames)
