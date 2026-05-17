from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from .adapter import ModelAdapter
from .edit_budget import MAIN_EDIT_BUDGET_POOL, budget_metadata, classify_edit_budget_pool
from .policy import HarnessPolicy, validate_command_against_packet
from .schema import (
    CachePairSource,
    ControllerCommand,
    ControllerObservationPacket,
    KvMixOp,
    Rank1PatchOp,
    READOUT_DIRECTION_MAX_NEGATIVE_TOKEN_IDS,
    READOUT_DIRECTION_MAX_SCALE,
    READOUT_DIRECTION_MAX_TARGET_TOKEN_IDS,
    READOUT_DIRECTION_MIN_SCALE,
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


def _model_from_ctx(ctx: "StepContext") -> Any:
    runtime_model = getattr(ctx.runtime_state, "model", None)
    if runtime_model is not None:
        return runtime_model
    return getattr(ctx.adapter, "model", None)


def _as_token_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    token_ids: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise SchemaError("readout_direction token ids must be integers, not booleans")
        try:
            token_id = int(item)
        except Exception:
            raise SchemaError("readout_direction token ids must be integers")
        if token_id < 0:
            raise SchemaError("readout_direction token ids must be >= 0")
        if token_id not in token_ids:
            token_ids.append(token_id)
    return tuple(token_ids)


def _readout_scale(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"readout_direction.{name} must be numeric")
    scale = float(value)
    if scale < READOUT_DIRECTION_MIN_SCALE or scale > READOUT_DIRECTION_MAX_SCALE:
        raise SchemaError(
            f"readout_direction.{name} must be between "
            f"{READOUT_DIRECTION_MIN_SCALE:g} and {READOUT_DIRECTION_MAX_SCALE:g}"
        )
    return scale


def _readout_matrix(ctx: "StepContext") -> torch.Tensor:
    model = _model_from_ctx(ctx)
    if model is None:
        raise KeyError("readout_direction requires a model on runtime_state or adapter")
    candidates = (
        getattr(model, "W_U", None),
        getattr(getattr(model, "unembed", None), "W_U", None),
        getattr(getattr(model, "unembed", None), "weight", None),
        getattr(getattr(model, "lm_head", None), "weight", None),
    )
    for candidate in candidates:
        if isinstance(candidate, torch.nn.Parameter):
            return candidate.detach().float()
        if isinstance(candidate, torch.Tensor):
            return candidate.detach().float()
    raise KeyError("readout_direction could not find W_U, unembed.weight, or lm_head.weight")


def _readout_vectors(matrix: torch.Tensor, token_ids: tuple[int, ...]) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError(f"readout_direction expected a matrix, got shape {tuple(matrix.shape)}")
    if not token_ids:
        raise ValueError("readout_direction requires at least one token id")
    max_token_id = max(token_ids)
    ids_for_columns = max_token_id < int(matrix.shape[1])
    ids_for_rows = max_token_id < int(matrix.shape[0])
    ids = torch.tensor(list(token_ids), dtype=torch.long, device=matrix.device)
    if ids_for_columns and (not ids_for_rows or matrix.shape[1] >= matrix.shape[0]):
        return matrix.index_select(1, ids).transpose(0, 1).contiguous()
    if ids_for_rows:
        return matrix.index_select(0, ids).contiguous()
    raise IndexError(f"readout_direction token id {max_token_id} is outside matrix shape {tuple(matrix.shape)}")


def _mean_readout_vector(matrix: torch.Tensor, token_ids: tuple[int, ...]) -> torch.Tensor | None:
    if not token_ids:
        return None
    vectors = _readout_vectors(matrix, token_ids)
    return vectors.reshape(len(token_ids), -1).mean(dim=0)


def _readout_direction(expr: Mapping[str, Any], ctx: "StepContext") -> torch.Tensor:
    matrix = _readout_matrix(ctx)
    target_ids = _as_token_ids(expr.get("target_token_ids"))
    negative_ids = _as_token_ids(expr.get("negative_token_ids"))
    if len(target_ids) > READOUT_DIRECTION_MAX_TARGET_TOKEN_IDS:
        raise SchemaError("readout_direction exceeds target_token_ids cap")
    if len(negative_ids) > READOUT_DIRECTION_MAX_NEGATIVE_TOKEN_IDS:
        raise SchemaError("readout_direction exceeds negative_token_ids cap")
    target_scale = _readout_scale(expr.get("target_scale", 1.0), name="target_scale")
    negative_scale = _readout_scale(expr.get("negative_scale", 1.0), name="negative_scale")
    pieces: list[torch.Tensor] = []
    target = _mean_readout_vector(matrix, target_ids)
    if target is not None:
        pieces.append(float(target_scale) * target)
    negative = _mean_readout_vector(matrix, negative_ids)
    if negative is not None:
        pieces.append(-float(negative_scale) * negative)
    if not pieces:
        raise ValueError("readout_direction produced no vector")
    direction = torch.stack(pieces, dim=0).sum(dim=0)
    if bool(expr.get("normalize", True)):
        direction = _normalize(direction)
    return direction


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

    if fn == "readout_direction":
        payload = dict(expr)
        return lambda ctx: _readout_direction(payload, ctx)

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
    return [compile_edit(edit, packet_obj, ctx, command_meta=command_obj.meta) for edit in command_obj.edits]


def compile_edit(
    edit: Any,
    packet: ControllerObservationPacket,
    ctx: StepContext,
    *,
    command_meta: Mapping[str, Any] | None = None,
) -> CompiledEdit:
    surface = ctx.adapter.resolve_surface(packet, edit.target)
    op = edit.op
    if isinstance(op, ResidAddOp):
        if not isinstance(edit.source, VectorSource):
            raise ValueError("resid_add requires a vector source")
        expr_fn = compile_expr(dict(edit.source.expr))
        return _compile_resid_add(edit, surface, expr_fn, ctx, command_meta=command_meta)
    if isinstance(op, KvMixOp):
        return _compile_kv_mix(edit, surface, ctx, command_meta=command_meta)
    if isinstance(op, Rank1PatchOp):
        if edit.source.dtype != "rank1":
            raise ValueError("rank1_patch requires a rank1 source")
        u_fn = compile_expr(dict(edit.source.u))
        v_fn = compile_expr(dict(edit.source.v))
        return _compile_rank1_patch(edit, surface, u_fn, v_fn, ctx, command_meta=command_meta)
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


def _registration_metadata(
    edit: Any,
    surface: Any,
    packet: ControllerObservationPacket,
    command_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_budget = _runtime_budget(edit, surface)
    budget_pool = classify_edit_budget_pool(edit=edit, packet=packet, surface=surface, command_meta=command_meta)
    edit_meta = edit.meta if isinstance(getattr(edit, "meta", None), Mapping) else {}
    metadata = {
        "surface_id": surface.surface_id,
        "op": edit.op.kind,
        "alpha": float(edit.op.alpha),
        "budget_key": edit.id,
        "budget_pool": budget_pool or MAIN_EDIT_BUDGET_POOL,
        "hypothesis": None if command_meta is None else command_meta.get("hypothesis"),
        "controller_confidence": None if command_meta is None else command_meta.get("confidence"),
        "expected_effect": edit.meta.get("expected_effect"),
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
    passthrough_fields = (
        "surface_family_key",
        "operator_recipe_id",
        "operator_recipe_seed_key",
        "bundle_key",
        "objective_bundle_key",
        "step_actuator_bundle_key",
        "apply_kind",
        "production_trial_allowed",
        "production_trial_budget_class",
        "production_trial_followup_allowed",
        "production_trial_contract",
        "promotion_reason",
        "latest_failed_operator_recipe_id",
        "production_apply_allowed",
        "production_policy_would_apply",
        "certified_for_apply",
        "source_localization",
        "patch_mode",
        "base_localization",
        "contrast_mode",
        "contrast_scale",
        "stealer_bundle_key",
        "stealer_term",
    )
    for key in passthrough_fields:
        if key in edit_meta and edit_meta.get(key) is not None:
            metadata[key] = edit_meta.get(key)
        elif command_meta is not None and command_meta.get(key) is not None:
            metadata[key] = command_meta.get(key)
    return metadata


def _compile_resid_add(
    edit: Any,
    surface: Any,
    expr_fn: TensorThunk,
    ctx: StepContext,
    *,
    command_meta: Mapping[str, Any] | None = None,
) -> CompiledEdit:
    alpha = float(edit.op.alpha)
    budget = _runtime_budget(edit, surface)
    metadata = _registration_metadata(edit, surface, ctx.packet, command_meta)
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


def _compile_kv_mix(
    edit: Any,
    surface: Any,
    ctx: StepContext,
    *,
    command_meta: Mapping[str, Any] | None = None,
) -> CompiledEdit:
    alpha = float(edit.op.alpha)
    budget = _runtime_budget(edit, surface)
    metadata = _registration_metadata(edit, surface, ctx.packet, command_meta)

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


def _compile_rank1_patch(
    edit: Any,
    surface: Any,
    u_fn: TensorThunk,
    v_fn: TensorThunk,
    ctx: StepContext,
    *,
    command_meta: Mapping[str, Any] | None = None,
) -> CompiledEdit:
    alpha = float(edit.op.alpha)
    budget = _runtime_budget(edit, surface)
    metadata = _registration_metadata(edit, surface, ctx.packet, command_meta)

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
