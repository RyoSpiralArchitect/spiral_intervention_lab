from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .edit_budget import (
    LOOP_RESCUE_EDIT_BUDGET_POOL,
    MAIN_EDIT_BUDGET_POOL,
    classify_edit_budget_pool,
    estimate_edit_cost,
)
from .schema import (
    CachePairSource,
    ControllerCommand,
    ControllerObservationPacket,
    HarnessControllerView,
    KvMixOp,
    Rank1PatchOp,
    SchemaError,
    Source,
    SurfaceInfo,
    SurfaceTargetRef,
    Target,
    TargetRef,
    VectorSource,
    parse_controller_command,
    parse_observation_packet,
)


class PolicyViolation(ValueError):
    pass


@dataclass(frozen=True)
class DenyTargetRule:
    kind: str
    layers: str | None = None


@dataclass(frozen=True)
class GlobalBudget:
    max_edits_per_step: int = 1
    max_edits_per_run: int = 4
    max_total_alpha: float = 0.5
    max_total_edit_cost: float = 0.5
    max_loop_rescue_edits_per_run: int = 4
    max_loop_rescue_total_alpha: float = 0.24
    max_loop_rescue_total_edit_cost: float = 0.24
    max_rank_per_edit: int = 1


@dataclass(frozen=True)
class HarnessPolicy:
    version: str = "0.1"
    controller_view: HarnessControllerView = field(default_factory=HarnessControllerView)
    allow_ops: tuple[str, ...] = ("resid_add", "kv_mix", "rank1_patch")
    deny_targets: tuple[DenyTargetRule, ...] = (
        DenyTargetRule(kind="embedding"),
        DenyTargetRule(kind="logits"),
        DenyTargetRule(kind="weight", layers="last2"),
    )
    global_budget: GlobalBudget = field(default_factory=GlobalBudget)
    max_expr_depth: int = 6
    max_expr_args: int = 4

    @classmethod
    def default_v0(cls) -> "HarnessPolicy":
        return cls()


def _packet_and_command(
    command: ControllerCommand | Mapping[str, Any],
    packet: ControllerObservationPacket | Mapping[str, Any],
) -> tuple[ControllerCommand, ControllerObservationPacket]:
    cmd = parse_controller_command(command) if isinstance(command, Mapping) else command
    pkt = parse_observation_packet(packet) if isinstance(packet, Mapping) else packet
    return cmd, pkt


def _resolve_surface(packet: ControllerObservationPacket, target_ref: TargetRef) -> SurfaceInfo:
    if isinstance(target_ref, SurfaceTargetRef):
        surface_map = packet.surface_map()
        if target_ref.surface_id not in surface_map:
            raise PolicyViolation(f"surface_id '{target_ref.surface_id}' is not available")
        return surface_map[target_ref.surface_id]
    for surface in packet.surface_catalog:
        if surface.target == target_ref:
            return surface
    raise PolicyViolation(f"target is not available in this observation packet: {target_ref!r}")


def _iter_expr_nodes(expr: Mapping[str, Any], depth: int = 0) -> Iterable[tuple[int, Mapping[str, Any]]]:
    yield depth, expr
    if "ref" in expr:
        return
    fn = expr.get("fn")
    if fn in {"add", "sub", "mean"}:
        for arg in expr.get("args", []):
            yield from _iter_expr_nodes(arg, depth + 1)
    elif fn in {"scale", "normalize", "zero_mean", "sign", "clip_norm"}:
        yield from _iter_expr_nodes(expr["arg"], depth + 1)
    elif fn == "mix":
        yield from _iter_expr_nodes(expr["left"], depth + 1)
        yield from _iter_expr_nodes(expr["right"], depth + 1)
    elif fn in {"project_parallel", "project_orthogonal"}:
        yield from _iter_expr_nodes(expr["arg"], depth + 1)
        yield from _iter_expr_nodes(expr["basis"], depth + 1)


def _walk_source_exprs(source: Source) -> Iterable[Mapping[str, Any]]:
    if isinstance(source, VectorSource):
        yield source.expr
    elif source.dtype == "rank1":
        yield source.u
        yield source.v
    elif isinstance(source, CachePairSource):
        if source.k is not None:
            yield source.k
        if source.v is not None:
            yield source.v


def _validate_expr_budget(source: Source, policy: HarnessPolicy) -> None:
    for expr in _walk_source_exprs(source):
        for depth, node in _iter_expr_nodes(expr):
            if depth > policy.max_expr_depth:
                raise PolicyViolation(f"expression exceeds max depth {policy.max_expr_depth}")
            if node.get("fn") in {"add", "sub", "mean"} and len(node.get("args", [])) > policy.max_expr_args:
                raise PolicyViolation(f"expression exceeds max arg count {policy.max_expr_args}")


def _validate_trace_access(source: Source, packet: ControllerObservationPacket) -> None:
    trace_map = packet.trace_map()
    for expr in _walk_source_exprs(source):
        for _depth, node in _iter_expr_nodes(expr):
            ref = node.get("ref")
            if not ref or ref.get("scope") != "trace":
                continue
            trace_id = ref.get("trace_id")
            trace = trace_map.get(trace_id)
            if trace is None:
                raise PolicyViolation(f"trace '{trace_id}' is not available")
            if not trace.compatible:
                raise PolicyViolation(f"trace '{trace_id}' is marked incompatible")


def _check_deny_rules(target: Target, packet: ControllerObservationPacket, policy: HarnessPolicy) -> None:
    max_weight_layer = max(
        (surface.target.layer for surface in packet.surface_catalog if getattr(surface.target, "kind", None) == "weight"),
        default=None,
    )
    for rule in policy.deny_targets:
        if getattr(target, "kind", None) != rule.kind:
            continue
        if rule.kind != "weight":
            raise PolicyViolation(f"target kind '{rule.kind}' is denied by policy")
        if rule.layers == "last2" and max_weight_layer is not None and target.layer >= max_weight_layer - 1:
            raise PolicyViolation("editing the last two weight layers is denied by policy")


def validate_command_against_packet(
    command: ControllerCommand | Mapping[str, Any],
    packet: ControllerObservationPacket | Mapping[str, Any],
    policy: HarnessPolicy | None = None,
) -> None:
    cmd, pkt = _packet_and_command(command, packet)
    pol = policy or HarnessPolicy.default_v0()

    if cmd.decision == "noop":
        return

    if cmd.decision == "rollback":
        unknown = [rollback_id for rollback_id in cmd.rollback_ids if rollback_id not in pkt.budget.rollbackable_ids]
        if unknown:
            raise PolicyViolation(f"rollback ids are not currently revertible: {unknown}")
        return

    usage = command_budget_usage(cmd, pkt, policy=pol)
    budget_reason = budget_violation_reason(cmd, pkt, policy=pol, usage=usage)
    if budget_reason is not None:
        raise PolicyViolation(budget_reason)

    rank_patch_count = 0

    for edit in cmd.edits:
        surface = _resolve_surface(pkt, edit.target)
        _check_deny_rules(surface.target, pkt, pol)
        if edit.op.kind not in pol.allow_ops:
            raise PolicyViolation(f"op '{edit.op.kind}' is denied by policy")
        if edit.op.kind not in surface.allow_ops:
            raise PolicyViolation(f"surface '{surface.surface_id}' does not allow op '{edit.op.kind}'")
        if edit.budget.ttl_steps > surface.caps.max_ttl_steps:
            raise PolicyViolation(f"edit '{edit.id}' exceeds surface ttl cap")
        if surface.caps.revertible_only and not edit.budget.revertible:
            raise PolicyViolation(f"edit '{edit.id}' must be revertible on this surface")

        alpha = float(edit.op.alpha)
        if alpha > surface.caps.max_alpha:
            raise PolicyViolation(f"edit '{edit.id}' exceeds surface alpha cap")
        if edit.budget.norm_clip is not None and surface.caps.norm_clip is not None:
            if edit.budget.norm_clip > surface.caps.norm_clip:
                raise PolicyViolation(f"edit '{edit.id}' exceeds surface norm clip cap")
        if edit.budget.step_size is not None and surface.caps.step_size is not None:
            if edit.budget.step_size > surface.caps.step_size:
                raise PolicyViolation(f"edit '{edit.id}' exceeds surface step_size cap")

        if isinstance(edit.op, Rank1PatchOp):
            rank_patch_count += 1
            requested_rank = edit.budget.rank_cap or 1
            if requested_rank > pol.global_budget.max_rank_per_edit:
                raise PolicyViolation(f"edit '{edit.id}' exceeds policy rank cap")
            if surface.caps.rank_cap is not None and requested_rank > surface.caps.rank_cap:
                raise PolicyViolation(f"edit '{edit.id}' exceeds surface rank cap")

        if isinstance(edit.op, KvMixOp):
            if edit.op.which == "k" and not getattr(edit.source, "k", None) and edit.source.dtype == "cache_pair":
                raise PolicyViolation(f"edit '{edit.id}' requires source.k for kv_mix which='k'")
            if edit.op.which == "v" and not getattr(edit.source, "v", None) and edit.source.dtype == "cache_pair":
                raise PolicyViolation(f"edit '{edit.id}' requires source.v for kv_mix which='v'")
            if edit.op.which == "kv" and edit.source.dtype != "cache_pair":
                raise PolicyViolation(f"edit '{edit.id}' requires cache_pair source for kv_mix which='kv'")

        _validate_expr_budget(edit.source, pol)
        _validate_trace_access(edit.source, pkt)
    if rank_patch_count > pkt.budget.active_patch_slots_left:
        raise PolicyViolation("command exceeds packet active_patch_slots_left")


def command_budget_usage(
    command: ControllerCommand | Mapping[str, Any],
    packet: ControllerObservationPacket | Mapping[str, Any],
    *,
    policy: HarnessPolicy | None = None,
) -> dict[str, dict[str, float]]:
    cmd, pkt = _packet_and_command(command, packet)
    del policy
    usage = {
        MAIN_EDIT_BUDGET_POOL: {"edit_count": 0.0, "alpha": 0.0, "edit_cost": 0.0},
        LOOP_RESCUE_EDIT_BUDGET_POOL: {"edit_count": 0.0, "alpha": 0.0, "edit_cost": 0.0},
    }
    for edit in cmd.edits:
        surface = _resolve_surface(pkt, edit.target)
        pool = classify_edit_budget_pool(edit=edit, packet=pkt, surface=surface, command_meta=cmd.meta)
        if pool not in usage:
            pool = MAIN_EDIT_BUDGET_POOL
        usage[pool]["edit_count"] += 1.0
        usage[pool]["alpha"] += float(edit.op.alpha)
        effective_step_size = edit.budget.step_size if edit.budget.step_size is not None else surface.caps.step_size
        usage[pool]["edit_cost"] += estimate_edit_cost(
            op_kind=edit.op.kind,
            alpha=edit.op.alpha,
            ttl_steps=edit.budget.ttl_steps,
            step_size=effective_step_size,
            rank_cap=edit.budget.rank_cap,
        )
    return usage


def budget_violation_reason(
    command: ControllerCommand | Mapping[str, Any],
    packet: ControllerObservationPacket | Mapping[str, Any],
    *,
    policy: HarnessPolicy | None = None,
    usage: dict[str, dict[str, float]] | None = None,
) -> str | None:
    cmd, pkt = _packet_and_command(command, packet)
    pol = policy or HarnessPolicy.default_v0()

    if cmd.decision != "apply":
        return None

    usage = usage or command_budget_usage(cmd, pkt, policy=pol)

    if len(cmd.edits) > pkt.budget.edits_left_this_step:
        return "command exceeds packet.edits_left_this_step"
    if len(cmd.edits) > pol.global_budget.max_edits_per_step:
        return "command exceeds policy.max_edits_per_step"
    if usage[MAIN_EDIT_BUDGET_POOL]["edit_count"] > pkt.budget.edits_left_this_run:
        return "command exceeds packet.edits_left_this_run"
    if usage[MAIN_EDIT_BUDGET_POOL]["edit_count"] > pol.global_budget.max_edits_per_run:
        return "command exceeds policy.max_edits_per_run"
    if usage[LOOP_RESCUE_EDIT_BUDGET_POOL]["edit_count"] > pkt.budget.loop_rescue_edits_left_this_run:
        return "command exceeds packet.loop_rescue_edits_left_this_run"
    if usage[LOOP_RESCUE_EDIT_BUDGET_POOL]["edit_count"] > pol.global_budget.max_loop_rescue_edits_per_run:
        return "command exceeds policy.max_loop_rescue_edits_per_run"

    main_alpha = float(usage[MAIN_EDIT_BUDGET_POOL]["alpha"])
    main_edit_cost = float(usage[MAIN_EDIT_BUDGET_POOL]["edit_cost"])
    if main_alpha > pkt.budget.alpha_left_total:
        return "command exceeds packet alpha budget"
    if main_alpha > pol.global_budget.max_total_alpha:
        return "command exceeds policy total alpha budget"
    if pkt.budget.edit_cost_left_total is not None and main_edit_cost > pkt.budget.edit_cost_left_total:
        return "command exceeds packet edit cost budget"
    if main_edit_cost > pol.global_budget.max_total_edit_cost:
        return "command exceeds policy total edit cost budget"

    rescue_alpha = float(usage[LOOP_RESCUE_EDIT_BUDGET_POOL]["alpha"])
    rescue_edit_cost = float(usage[LOOP_RESCUE_EDIT_BUDGET_POOL]["edit_cost"])
    if rescue_alpha > pkt.budget.loop_rescue_alpha_left_total:
        return "command exceeds packet loop_rescue alpha budget"
    if rescue_alpha > pol.global_budget.max_loop_rescue_total_alpha:
        return "command exceeds policy loop_rescue total alpha budget"
    if rescue_edit_cost > pkt.budget.loop_rescue_edit_cost_left_total:
        return "command exceeds packet loop_rescue edit cost budget"
    if rescue_edit_cost > pol.global_budget.max_loop_rescue_total_edit_cost:
        return "command exceeds policy loop_rescue total edit cost budget"

    return None
