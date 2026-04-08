from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

import torch

from .edit_budget import prepare_direction
from .overlays import LinearRank1OverlayHandle, OverlayHandle, ParameterRank1OverlayHandle
from .rank1_bridge import HybridRank1VectorBridge, Rank1Geometry, Rank1VectorBridge
from .schema import (
    ActivationTarget,
    CacheTarget,
    ControllerObservationPacket,
    SurfaceInfo,
    SurfaceTargetRef,
    Target,
    TargetRef,
    WeightTarget,
    parse_observation_packet,
)

if TYPE_CHECKING:
    from .compiler import StepContext


@dataclass
class BoundSurface:
    surface_id: str
    kind: str
    hook_name: str | None
    module_ref: Any | None
    layer: int
    site: str
    token_selector: Any | None
    head: int | None
    caps: Any
    allow_ops: list[str]
    target: Target


class ModelAdapter:
    def __init__(self) -> None:
        self._current_step_ctx: StepContext | None = None

    def set_step_context(self, ctx: "StepContext") -> None:
        self._current_step_ctx = ctx

    def resolve_surface(self, packet: ControllerObservationPacket | dict[str, Any], target_ref: TargetRef) -> BoundSurface:
        packet_obj = parse_observation_packet(packet) if isinstance(packet, dict) else packet
        surface_info = self._find_surface(packet_obj, target_ref)
        return self.bind_surface(surface_info)

    def _find_surface(self, packet: ControllerObservationPacket, target_ref: TargetRef) -> SurfaceInfo:
        surface_map = packet.surface_map()
        if isinstance(target_ref, SurfaceTargetRef):
            try:
                return surface_map[target_ref.surface_id]
            except KeyError as exc:
                raise KeyError(f"unknown surface_id '{target_ref.surface_id}'") from exc
        for surface in packet.surface_catalog:
            if surface.target == target_ref:
                return surface
        raise KeyError(f"target is not present in surface_catalog: {target_ref!r}")

    def bind_surface(self, surface: SurfaceInfo) -> BoundSurface:
        raise NotImplementedError

    def read_ref(self, ref: dict[str, Any], ctx: "StepContext") -> torch.Tensor:
        raise NotImplementedError

    def make_activation_hook(
        self,
        surface: BoundSurface,
        op_kind: str,
        tensor_fn: Callable[["StepContext"], torch.Tensor],
        alpha: float,
        budget: dict[str, Any],
    ) -> tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]:
        raise NotImplementedError

    def make_kv_hook(
        self,
        surface: BoundSurface,
        tensor_fn: Callable[["StepContext"], torch.Tensor],
        alpha: float,
        which: str,
        budget: dict[str, Any],
    ) -> tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]:
        raise NotImplementedError

    def make_rank1_overlay(
        self,
        surface: BoundSurface,
        u_fn: Callable[["StepContext"], torch.Tensor],
        v_fn: Callable[["StepContext"], torch.Tensor],
        alpha: float,
        budget: dict[str, Any],
    ) -> OverlayHandle:
        raise NotImplementedError

    def _select_positions(
        self,
        act: torch.Tensor,
        selector: Any | None,
        *,
        position_axis: int = -2,
    ) -> list[int]:
        if act.ndim < 2:
            raise ValueError(f"expected tensor with position axis, got shape {tuple(act.shape)}")
        axis = int(position_axis)
        if axis < 0:
            axis = act.ndim + axis
        if axis < 0 or axis >= act.ndim:
            raise IndexError(f"position axis {position_axis} out of range for shape {tuple(act.shape)}")
        seq_len = act.shape[axis]
        if selector is None or selector.mode == "last":
            return [seq_len - 1]
        if selector.mode == "index":
            index = selector.value
            if index is None or not (-seq_len <= index < seq_len):
                raise IndexError(f"token index {index} out of range for length {seq_len}")
            return [index % seq_len]
        if selector.mode == "span":
            assert selector.start is not None and selector.end is not None
            start = max(0, selector.start)
            end = min(seq_len, selector.end)
            if start >= end:
                raise IndexError(f"empty token span [{selector.start}, {selector.end}) for length {seq_len}")
            return list(range(start, end))
        raise ValueError(f"unsupported token selector mode: {selector.mode}")

    def _context_step(self, ctx: "StepContext") -> int | None:
        packet = getattr(ctx, "packet", None)
        if isinstance(packet, dict):
            step = packet.get("step")
        else:
            step = getattr(packet, "step", None)
        if step is None:
            return None
        return int(step)

    def _coerce_vector(self, value: torch.Tensor, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        vec = value.to(device=device, dtype=dtype).reshape(-1)
        if vec.shape[0] != width:
            raise ValueError(f"expected vector of width {width}, got {tuple(vec.shape)}")
        return vec

    def _add_to_selected_tokens(
        self,
        act: torch.Tensor,
        vec: torch.Tensor,
        token_selector: Any | None,
        alpha: float,
    ) -> torch.Tensor:
        out = act.clone()
        width = out.shape[-1]
        vec = self._coerce_vector(vec, width, device=out.device, dtype=out.dtype)
        for pos in self._select_positions(out, token_selector):
            out[..., pos, :] = out[..., pos, :] + (alpha * vec)
        return out

    def _mix_selected_tokens(
        self,
        act: torch.Tensor,
        vec: torch.Tensor,
        token_selector: Any | None,
        alpha: float,
    ) -> torch.Tensor:
        out = act.clone()
        width = out.shape[-1]
        vec = self._coerce_vector(vec, width, device=out.device, dtype=out.dtype)
        for pos in self._select_positions(out, token_selector):
            out[..., pos, :] = ((1.0 - alpha) * out[..., pos, :]) + (alpha * vec)
        return out

    def _mix_cache_selected_tokens(
        self,
        act: torch.Tensor,
        vec: torch.Tensor,
        token_selector: Any | None,
        alpha: float,
        head: int | None = None,
    ) -> torch.Tensor:
        if act.ndim != 4:
            return self._mix_selected_tokens(act, vec, token_selector, alpha)

        out = act.clone()
        positions = self._select_positions(out, token_selector, position_axis=-3)
        heads = out.shape[-2]
        width = out.shape[-1]

        if head is not None:
            vec_1d = self._coerce_vector(vec, width, device=out.device, dtype=out.dtype)
            for pos in positions:
                out[..., pos, head, :] = ((1.0 - alpha) * out[..., pos, head, :]) + (alpha * vec_1d)
            return out

        vec_cast = vec.to(device=out.device, dtype=out.dtype)
        if vec_cast.ndim == 1:
            if vec_cast.numel() != heads * width:
                raise ValueError(
                    f"expected flattened cache vector of width {heads * width}, got {tuple(vec_cast.shape)}"
                )
            vec_cast = vec_cast.reshape(heads, width)
        elif vec_cast.ndim == 2:
            if tuple(vec_cast.shape) != (heads, width):
                raise ValueError(
                    f"expected cache matrix of shape {(heads, width)}, got {tuple(vec_cast.shape)}"
                )
        else:
            raise ValueError(f"expected cache source tensor with rank 1 or 2, got shape {tuple(vec_cast.shape)}")

        for pos in positions:
            out[..., pos, :, :] = ((1.0 - alpha) * out[..., pos, :, :]) + (alpha * vec_cast)
        return out


class HookedTransformerAdapter(ModelAdapter):
    HOOK_SITE_TEMPLATES = {
        "resid_pre": "blocks.{layer}.hook_resid_pre",
        "resid_post": "blocks.{layer}.hook_resid_post",
        "mlp_out": "blocks.{layer}.hook_mlp_out",
        "k_cache": "blocks.{layer}.attn.hook_k",
        "v_cache": "blocks.{layer}.attn.hook_v",
    }

    def __init__(self, model: Any, *, rank1_bridge: Rank1VectorBridge | None = None) -> None:
        super().__init__()
        self.model = model
        self.rank1_bridge = rank1_bridge or HybridRank1VectorBridge()

    def bind_surface(self, surface: SurfaceInfo) -> BoundSurface:
        target = surface.target
        if isinstance(target, ActivationTarget):
            hook_name = self.HOOK_SITE_TEMPLATES[target.site].format(layer=target.layer)
            return BoundSurface(
                surface_id=surface.surface_id,
                kind=target.kind,
                hook_name=hook_name,
                module_ref=None,
                layer=target.layer,
                site=target.site,
                token_selector=target.token,
                head=None,
                caps=surface.caps,
                allow_ops=list(surface.allow_ops),
                target=target,
            )
        if isinstance(target, CacheTarget):
            hook_name = self.HOOK_SITE_TEMPLATES[target.site].format(layer=target.layer)
            return BoundSurface(
                surface_id=surface.surface_id,
                kind=target.kind,
                hook_name=hook_name,
                module_ref=None,
                layer=target.layer,
                site=target.site,
                token_selector=target.token,
                head=target.head,
                caps=surface.caps,
                allow_ops=list(surface.allow_ops),
                target=target,
            )
        if isinstance(target, WeightTarget):
            module_ref = self._resolve_weight_module(target)
            return BoundSurface(
                surface_id=surface.surface_id,
                kind=target.kind,
                hook_name=None,
                module_ref=module_ref,
                layer=target.layer,
                site=target.module,
                token_selector=None,
                head=None,
                caps=surface.caps,
                allow_ops=list(surface.allow_ops),
                target=target,
            )
        raise TypeError(f"unsupported target type: {type(target)!r}")

    def _resolve_weight_module(self, target: WeightTarget) -> Any:
        block = self.model.blocks[target.layer]
        if target.module == "attn_out":
            module = getattr(getattr(block, "attn", None), "W_O", None)
        else:
            mlp = getattr(block, "mlp", None)
            module = getattr(mlp, "W_out", None)
            if module is None:
                module = getattr(mlp, "linear_out", None)
            if module is None:
                module = mlp
        return module

    def read_ref(self, ref: dict[str, Any], ctx: "StepContext") -> torch.Tensor:
        if hasattr(ctx.runtime_state, "read_tensor"):
            return ctx.runtime_state.read_tensor(ref, ctx)
        scope = ref["scope"]
        if scope == "stats":
            return ctx.runtime_state.read_stat_tensor(ref)
        step = self._context_step(ctx)
        if scope == "trace" and hasattr(ctx.runtime_state, "set_trace_alignment"):
            ctx.runtime_state.set_trace_alignment(step)
        try:
            cache = ctx.runtime_state.get_cache(scope, ref.get("trace_id"), step=step)
        except TypeError:
            cache = ctx.runtime_state.get_cache(scope, ref.get("trace_id"))
        hook_name = self._hook_name_for_ref(ref["tensor"], ref["layer"])
        try:
            raw_tensor = cache[hook_name]
        except KeyError as exc:
            raise KeyError(f"tensor '{hook_name}' is not present in cache scope '{scope}'") from exc
        return self._select_ref_tensor(raw_tensor, ref)

    def _hook_name_for_ref(self, tensor_name: str, layer: int) -> str:
        site = "resid_post" if tensor_name == "hidden" else tensor_name
        try:
            template = self.HOOK_SITE_TEMPLATES[site]
        except KeyError as exc:
            raise KeyError(f"unsupported tensor ref '{tensor_name}' for HookedTransformerAdapter") from exc
        return template.format(layer=layer)

    def _select_ref_tensor(self, raw_tensor: torch.Tensor, ref: dict[str, Any]) -> torch.Tensor:
        token_selector = ref.get("token")
        selector = token_selector if token_selector is None or hasattr(token_selector, "mode") else None
        if selector is None and token_selector is not None:
            from .schema import TokenSelector

            selector = TokenSelector.from_dict(token_selector)
        batch0 = raw_tensor[0]
        if batch0.ndim == 2:
            positions = self._select_positions(batch0.unsqueeze(0), selector)
            selected = batch0[positions]
            if selected.shape[0] == 1 or getattr(selector, "mode", None) != "span" or getattr(selector, "pool", "last") == "last":
                return selected[-1]
            return selected.mean(dim=0)

        if batch0.ndim == 3:
            positions = self._select_positions(raw_tensor, selector, position_axis=-3)
            selected = batch0[positions]
            pooled = (
                selected[-1]
                if selected.shape[0] == 1 or getattr(selector, "mode", None) != "span" or getattr(selector, "pool", "last") == "last"
                else selected.mean(dim=0)
            )
            head = ref.get("head")
            if head is not None:
                return pooled[head]
            return pooled.reshape(-1)

        raise ValueError(f"unsupported cached tensor shape for ref selection: {tuple(raw_tensor.shape)}")

    def make_activation_hook(
        self,
        surface: BoundSurface,
        op_kind: str,
        tensor_fn: Callable[["StepContext"], torch.Tensor],
        alpha: float,
        budget: dict[str, Any],
    ) -> tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]:
        if surface.hook_name is None:
            raise ValueError(f"surface '{surface.surface_id}' does not expose an activation hook")
        norm_clip = budget.get("norm_clip")
        step_size = budget.get("step_size")

        def hook_fn(act: torch.Tensor, hook: Any | None = None) -> torch.Tensor:
            if self._current_step_ctx is None:
                raise RuntimeError("step context is not bound before hook execution")
            vec = tensor_fn(self._current_step_ctx)
            vec = prepare_direction(vec, alpha=alpha, norm_clip=norm_clip, step_size=step_size)
            return self._add_to_selected_tokens(act, vec, surface.token_selector, alpha)

        return surface.hook_name, hook_fn

    def make_kv_hook(
        self,
        surface: BoundSurface,
        tensor_fn: Callable[["StepContext"], torch.Tensor],
        alpha: float,
        which: str,
        budget: dict[str, Any],
    ) -> tuple[str, Callable[[torch.Tensor, Any], torch.Tensor]]:
        if surface.hook_name is None:
            raise ValueError(f"surface '{surface.surface_id}' does not expose a cache hook")
        norm_clip = budget.get("norm_clip")
        step_size = budget.get("step_size")

        def hook_fn(act: torch.Tensor, hook: Any | None = None) -> torch.Tensor:
            if self._current_step_ctx is None:
                raise RuntimeError("step context is not bound before hook execution")
            vec = tensor_fn(self._current_step_ctx)
            vec = prepare_direction(vec, alpha=alpha, norm_clip=norm_clip, step_size=step_size)
            return self._mix_cache_selected_tokens(act, vec, surface.token_selector, alpha, head=surface.head)

        return surface.hook_name, hook_fn

    def make_rank1_overlay(
        self,
        surface: BoundSurface,
        u_fn: Callable[["StepContext"], torch.Tensor],
        v_fn: Callable[["StepContext"], torch.Tensor],
        alpha: float,
        budget: dict[str, Any],
    ) -> OverlayHandle:
        if surface.module_ref is None:
            raise ValueError(
                f"surface '{surface.surface_id}' does not resolve to a weight module; "
                "provide a module_ref-capable adapter for rank1 overlays"
            )
        if isinstance(surface.module_ref, torch.nn.Parameter):
            geometry = self._parameter_geometry(surface)

            def bridged_u_fn(step_ctx: "StepContext") -> torch.Tensor:
                raw = u_fn(step_ctx)
                return self.rank1_bridge.adapt(raw, side="row", geometry=geometry)

            def bridged_v_fn(step_ctx: "StepContext") -> torch.Tensor:
                raw = v_fn(step_ctx)
                return self.rank1_bridge.adapt(raw, side="col", geometry=geometry)

            return ParameterRank1OverlayHandle(
                parameter=surface.module_ref,
                ctx_getter=lambda: self._current_step_ctx,
                u_fn=bridged_u_fn,
                v_fn=bridged_v_fn,
                alpha=alpha,
                step_size=budget.get("step_size"),
            )
        return LinearRank1OverlayHandle(
            module=surface.module_ref,
            ctx_getter=lambda: self._current_step_ctx,
            u_fn=u_fn,
            v_fn=v_fn,
            alpha=alpha,
            step_size=budget.get("step_size"),
        )

    def _parameter_geometry(self, surface: BoundSurface) -> Rank1Geometry:
        parameter = surface.module_ref
        if not isinstance(parameter, torch.nn.Parameter):
            raise TypeError("parameter geometry requires a torch.nn.Parameter surface")
        target = parameter.data
        rows = int(target.numel() // target.shape[-1])
        cols = int(target.shape[-1])
        matrix = target.reshape(rows, cols)
        return Rank1Geometry(
            target_shape=tuple(target.shape),
            rows=rows,
            cols=cols,
            matrix=matrix,
            surface_id=surface.surface_id,
            site=surface.site,
        )
