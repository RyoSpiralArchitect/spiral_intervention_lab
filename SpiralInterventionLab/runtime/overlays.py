from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


class OverlayHandle:
    def attach(self) -> None:
        raise NotImplementedError

    def detach(self) -> None:
        raise NotImplementedError

    def tick(self) -> None:
        return None


@dataclass
class NoOpOverlayHandle(OverlayHandle):
    attached: bool = False

    def attach(self) -> None:
        self.attached = True

    def detach(self) -> None:
        self.attached = False


@dataclass
class LinearRank1OverlayHandle(OverlayHandle):
    module: Any
    ctx_getter: Callable[[], Any]
    u_fn: Callable[[Any], torch.Tensor]
    v_fn: Callable[[Any], torch.Tensor]
    alpha: float
    handle: Any | None = field(default=None, init=False)

    def attach(self) -> None:
        if self.handle is not None:
            return

        def forward_hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
            if not isinstance(output, torch.Tensor) or not inputs:
                return output
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                return output
            ctx = self.ctx_getter()
            u = self.u_fn(ctx).to(device=output.device, dtype=output.dtype).reshape(-1)
            v = self.v_fn(ctx).to(device=x.device, dtype=x.dtype).reshape(-1)
            if x.shape[-1] != v.shape[0] or output.shape[-1] != u.shape[0]:
                raise ValueError(
                    "rank1 overlay dimension mismatch: "
                    f"x={tuple(x.shape)}, v={tuple(v.shape)}, output={tuple(output.shape)}, u={tuple(u.shape)}"
                )
            coeff = torch.matmul(x, v)
            patch = coeff.unsqueeze(-1) * u
            return output + (self.alpha * patch)

        self.handle = self.module.register_forward_hook(forward_hook)

    def detach(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


@dataclass
class ParameterRank1OverlayHandle(OverlayHandle):
    parameter: torch.nn.Parameter
    ctx_getter: Callable[[], Any]
    u_fn: Callable[[Any], torch.Tensor]
    v_fn: Callable[[Any], torch.Tensor]
    alpha: float
    delta: torch.Tensor | None = field(default=None, init=False)
    attached: bool = field(default=False, init=False)

    def _build_delta(self) -> torch.Tensor:
        ctx = self.ctx_getter()
        u = self.u_fn(ctx).to(device=self.parameter.device, dtype=self.parameter.dtype).reshape(-1)
        v = self.v_fn(ctx).to(device=self.parameter.device, dtype=self.parameter.dtype).reshape(-1)
        target = self.parameter.data

        if target.ndim == 2:
            rows, cols = target.shape
            if rows != u.numel() or cols != v.numel():
                raise ValueError(
                    "rank1 parameter overlay dimension mismatch: "
                    f"target={tuple(target.shape)}, u={tuple(u.shape)}, v={tuple(v.shape)}"
                )
            delta = torch.outer(u, v)
            return self.alpha * delta

        rows = int(target.numel() // target.shape[-1])
        cols = int(target.shape[-1])
        if rows != u.numel() or cols != v.numel():
            raise ValueError(
                "rank1 parameter overlay dimension mismatch after flattening prefix dims: "
                f"target={tuple(target.shape)}, flat_matrix=({rows}, {cols}), "
                f"u={tuple(u.shape)}, v={tuple(v.shape)}"
            )
        delta = torch.outer(u, v).reshape_as(target)
        return self.alpha * delta

    def attach(self) -> None:
        if self.attached:
            return
        self.delta = self._build_delta()
        self.parameter.data.add_(self.delta)
        self.attached = True

    def detach(self) -> None:
        if not self.attached:
            return
        assert self.delta is not None
        self.parameter.data.sub_(self.delta)
        self.delta = None
        self.attached = False
