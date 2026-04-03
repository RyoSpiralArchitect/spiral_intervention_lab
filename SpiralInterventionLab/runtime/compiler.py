from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .adapter import ModelAdapter
from .edit_budget import budget_metadata
from .policy import HarnessPolicy, validate_command_against_packet
from .schema import (
    CachePairSource,
    ControllerCommand,
    ControllerObservationPacket,
    KvMixOp,
    Rank1PatchOp,
    ResidAddOp,
    SchemaError,
    SurfaceTargetRef,
    VectorSource,
    parse_controller_command,
    parse_observation_packet,
)

TensorThunk = Callable[["StepContext"], torch.Tensor]


@dataclass
class CompiledEdit:
    edit_id: str
    ttl_steps: int
    revertible: bool
    kind: str
    apply: Callable[["StepContext"], None]
    rollback: Callable[["StepContext"], None]


@dataclass
class StepContext:
    packet: ControllerObservationPacket
    runtime_state: Any
    traces: dict[str, Any]
    stats: Any
    adapter: ModelAdapter
    active_edits: dict[str, CompiledEdit] = field(default_factory=dict)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm() + 1e-8)


def _zero_mean(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean()


def _clip_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = x.norm()
    if float(norm) <= max_norm:
        return x
    return x * (max_norm / (float(norm) + 1e-8))


def _project_parallel(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    flat_x = x.reshape(-1)
    flat_basis = _normalize(basis.reshape(-1))
    coeff = torch.dot(flat_x, flat_basis)
    return coeff * flat_basis.reshape_as(x)


def _project_orthogonal(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return x - _project_parallel(x, basis)


def compile_expr(expr: dict[str, Any] | Any) -> TensorThunk:
    if not isinstance(expr, dict):
        raise SchemaError(f"expression must be a mapping, got {type(expr)!r}")
    if "ref" in expr:
        ref = expr["ref"]
        return lambda ctx: ctx.adapter.read_ref(ref, ctx)

    fn = expr["fn"]

    if fn == "add":
        items = [compile_expr(item) for item in expr["args"]]
        return lambda ctx: torch.stack([item(ctx) for item in items], dim=0).sum(dim=0)

    if fn == "sub":
        left, right = [compile_expr(item) for item in expr["args"]]
        return lambda ctx: left(ctx) - right(ctx)

    if fn == "mean":
        items = [compile_expr(item) for item in expr["args"]]
        return lambda ctx: torch.stack([item(ctx) for item in items], dim=0).mean(dim=0)

    if fn == "scale":
        item = compile_expr(expr["arg"])
        factor = float(expr["by"])
        return lambda ctx: factor * item(ctx)

    if fn == "normalize":
        item = compile_expr(expr["arg"])
        return lambda ctx: _normalize(item(ctx))

    if fn == "zero_mean":
        item = compile_expr(expr["arg"])
        return lambda ctx: _zero_mean(item(ctx))

    if fn == "sign":
        item = compile_expr(expr["arg"])
        return lambda ctx: torch.sign(item(ctx))

    if fn == "clip_norm":
        item = compile_expr(expr["arg"])
        max_norm = float(expr["max_norm"])
        return lambda ctx: _clip_norm(item(ctx), max_norm)

    if fn == "mix":
        left = compile_expr(expr["left"])
        right = compile_expr(expr["right"])
        alpha = float(expr["alpha"])
        return lambda ctx: (alpha * left(ctx)) + ((1.0 - alpha) * right(ctx))

    if fn == "project_parallel":
        item = compile_expr(expr["arg"])
        basis = compile_expr(expr["basis"])
        return lambda ctx: _project_parallel(item(ctx), basis(ctx))

    if fn == "project_orthogonal":
        item = compile_expr(expr["arg"])
        basis = compile_expr(expr["basis"])
        return lambda ctx: _project_orthogonal(item(ctx), basis(ctx))

    raise ValueError(f"unknown expr fn: {fn}")


def compile_command(
    command: ControllerCommand | dict[str, Any],
    packet: ControllerObservationPacket | dict[str, Any],
    ctx: StepContext,
    *,
    policy: HarnessPolicy | None = None,
) -> list[CompiledEdit]:
    command_obj = parse_controller_command(command) if isinstance(command, dict) else command
    packet_obj = parse_observation_packet(packet) if isinstance(packet, dict) else packet
    ctx.packet = packet_obj
    validate_command_against_packet(command_obj, packet_obj, policy=policy)

    if command_obj.decision == "noop":
        return []
    if command_obj.decision == "rollback":
        return [_compile_rollback(rollback_id) for rollback_id in command_obj.rollback_ids]
    return [compile_edit(edit, packet_obj, ctx) for edit in command_obj.edits]


def compile_edit(edit: Any, packet: ControllerObservationPacket, ctx: StepContext) -> CompiledEdit:
    surface = ctx.adapter.resolve_surface(packet, edit.target)
    op = edit.op
    if isinstance(op, ResidAddOp):
        if not isinstance(edit.source, VectorSource):
            raise ValueError("resid_add requires a vector source")
        expr_fn = compile_expr(dict(edit.source.expr))
        return _compile_resid_add(edit, surface, expr_fn, ctx)
    if isinstance(op, KvMixOp):
        return _compile_kv_mix(edit, surface, ctx)
    if isinstance(op, Rank1PatchOp):
        if edit.source.dtype != "rank1":
            raise ValueError("rank1_patch requires a rank1 source")
        u_fn = compile_expr(dict(edit.source.u))
        v_fn = compile_expr(dict(edit.source.v))
        return _compile_rank1_patch(edit, surface, u_fn, v_fn, ctx)
    raise ValueError(f"unsupported op kind: {op.kind}")


def _compile_rollback(edit_id: str) -> CompiledEdit:
    def apply(step_ctx: StepContext) -> None:
        compiled = step_ctx.active_edits.pop(edit_id, None)
        if compiled is not None:
            compiled.rollback(step_ctx)
        else:
            step_ctx.runtime_state.remove_edit(edit_id)

    return CompiledEdit(
        edit_id=edit_id,
        ttl_steps=0,
        revertible=True,
        kind="rollback",
        apply=apply,
        rollback=lambda _ctx: None,
    )


def _resolved_step_size(edit: Any, surface: Any) -> float | None:
    if edit.budget.step_size is not None:
        return float(edit.budget.step_size)
    surface_step_size = getattr(getattr(surface, "caps", None), "step_size", None)
    if surface_step_size is None:
        return None
    return float(surface_step_size)


def _runtime_budget(edit: Any, surface: Any) -> dict[str, Any]:
    return {
        "ttl_steps": edit.budget.ttl_steps,
        "revertible": edit.budget.revertible,
        "norm_clip": edit.budget.norm_clip,
        "step_size": _resolved_step_size(edit, surface),
        "rank_cap": edit.budget.rank_cap,
    }


def _registration_metadata(edit: Any, surface: Any) -> dict[str, Any]:
    runtime_budget = _runtime_budget(edit, surface)
    metadata = {
        "surface_id": surface.surface_id,
        "op": edit.op.kind,
        "alpha": float(edit.op.alpha),
        "budget_key": edit.id,
    }
    metadata.update(
        budget_metadata(
            op_kind=edit.op.kind,
            alpha=edit.op.alpha,
            ttl_steps=edit.budget.ttl_steps,
            step_size=runtime_budget["step_size"],
            rank_cap=edit.budget.rank_cap,
        )
    )
    return metadata


def _compile_resid_add(edit: Any, surface: Any, expr_fn: TensorThunk, ctx: StepContext) -> CompiledEdit:
    alpha = float(edit.op.alpha)
    budget = _runtime_budget(edit, surface)
    metadata = _registration_metadata(edit, surface)
    hook_name, hook_fn = ctx.adapter.make_activation_hook(
        surface=surface,
        op_kind="resid_add",
        tensor_fn=expr_fn,
        alpha=alpha,
        budget=budget,
    )

    def apply(step_ctx: StepContext) -> None:
        step_ctx.adapter.set_step_context(step_ctx)
        step_ctx.runtime_state.register_hook(
            hook_name=hook_name,
            hook_fn=hook_fn,
            edit_id=edit.id,
            ttl_steps=edit.budget.ttl_steps,
            revertible=edit.budget.revertible,
            metadata=metadata,
        )

    def rollback(step_ctx: StepContext) -> None:
        step_ctx.runtime_state.remove_edit(edit.id)

    return CompiledEdit(
        edit_id=edit.id,
        ttl_steps=edit.budget.ttl_steps,
        revertible=edit.budget.revertible,
        kind="resid_add",
        apply=apply,
        rollback=rollback,
    )


def _compile_kv_mix(edit: Any, surface: Any, ctx: StepContext) -> CompiledEdit:
    alpha = float(edit.op.alpha)
    budget = _runtime_budget(edit, surface)
    metadata = _registration_metadata(edit, surface)

    if edit.op.which == "kv":
        if not isinstance(edit.source, CachePairSource) or edit.source.k is None or edit.source.v is None:
            raise ValueError("kv_mix with which='kv' requires cache_pair source with both k and v")
        k_fn = compile_expr(dict(edit.source.k))
        v_fn = compile_expr(dict(edit.source.v))
        k_hook_name, k_hook_fn = ctx.adapter.make_kv_hook(surface, k_fn, alpha, "k", budget)
        v_hook_name, v_hook_fn = ctx.adapter.make_kv_hook(surface, v_fn, alpha, "v", budget)

        def apply(step_ctx: StepContext) -> None:
            step_ctx.adapter.set_step_context(step_ctx)
            step_ctx.runtime_state.register_hook(
                hook_name=k_hook_name,
                hook_fn=k_hook_fn,
                edit_id=f"{edit.id}:k",
                ttl_steps=edit.budget.ttl_steps,
                revertible=edit.budget.revertible,
                metadata=metadata,
            )
            step_ctx.runtime_state.register_hook(
                hook_name=v_hook_name,
                hook_fn=v_hook_fn,
                edit_id=f"{edit.id}:v",
                ttl_steps=edit.budget.ttl_steps,
                revertible=edit.budget.revertible,
                metadata=metadata,
            )

        def rollback(step_ctx: StepContext) -> None:
            step_ctx.runtime_state.remove_edit(f"{edit.id}:k")
            step_ctx.runtime_state.remove_edit(f"{edit.id}:v")

        return CompiledEdit(
            edit_id=edit.id,
            ttl_steps=edit.budget.ttl_steps,
            revertible=edit.budget.revertible,
            kind="kv_mix",
            apply=apply,
            rollback=rollback,
        )

    if isinstance(edit.source, CachePairSource):
        expr = edit.source.k if edit.op.which == "k" else edit.source.v
    else:
        expr = None
    if expr is None:
        raise ValueError(f"kv_mix which='{edit.op.which}' requires a matching cache source")
    expr_fn = compile_expr(dict(expr))
    hook_name, hook_fn = ctx.adapter.make_kv_hook(surface, expr_fn, alpha, edit.op.which, budget)

    def apply(step_ctx: StepContext) -> None:
        step_ctx.adapter.set_step_context(step_ctx)
        step_ctx.runtime_state.register_hook(
            hook_name=hook_name,
            hook_fn=hook_fn,
            edit_id=edit.id,
            ttl_steps=edit.budget.ttl_steps,
            revertible=edit.budget.revertible,
            metadata=metadata,
        )

    def rollback(step_ctx: StepContext) -> None:
        step_ctx.runtime_state.remove_edit(edit.id)

    return CompiledEdit(
        edit_id=edit.id,
        ttl_steps=edit.budget.ttl_steps,
        revertible=edit.budget.revertible,
        kind="kv_mix",
        apply=apply,
        rollback=rollback,
    )


def _compile_rank1_patch(edit: Any, surface: Any, u_fn: TensorThunk, v_fn: TensorThunk, ctx: StepContext) -> CompiledEdit:
    alpha = float(edit.op.alpha)
    budget = _runtime_budget(edit, surface)
    metadata = _registration_metadata(edit, surface)

    def apply(step_ctx: StepContext) -> None:
        step_ctx.adapter.set_step_context(step_ctx)
        handle = step_ctx.adapter.make_rank1_overlay(
            surface=surface,
            u_fn=u_fn,
            v_fn=v_fn,
            alpha=alpha,
            budget=budget,
        )
        step_ctx.runtime_state.register_overlay(
            edit_id=edit.id,
            handle=handle,
            ttl_steps=edit.budget.ttl_steps,
            revertible=edit.budget.revertible,
            metadata=metadata,
        )

    def rollback(step_ctx: StepContext) -> None:
        step_ctx.runtime_state.remove_edit(edit.id)

    return CompiledEdit(
        edit_id=edit.id,
        ttl_steps=edit.budget.ttl_steps,
        revertible=edit.budget.revertible,
        kind="rank1_patch",
        apply=apply,
        rollback=rollback,
    )
