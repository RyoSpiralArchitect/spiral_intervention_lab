from __future__ import annotations

from typing import Any, Mapping

import torch

MAIN_EDIT_BUDGET_POOL = "main"
LOOP_RESCUE_EDIT_BUDGET_POOL = "loop_rescue"


OP_COST_FACTORS: dict[str, float] = {
    "resid_add": 1.0,
    "kv_mix": 1.15,
    "rank1_patch": 1.35,
}


def clip_norm(x: torch.Tensor, max_norm: float | None) -> torch.Tensor:
    if max_norm is None:
        return x
    norm = x.norm()
    if float(norm) <= float(max_norm):
        return x
    return x * (float(max_norm) / (float(norm) + 1e-8))


def enforce_step_size(
    direction: torch.Tensor,
    *,
    alpha: float,
    step_size: float | None,
) -> torch.Tensor:
    if step_size is None:
        return direction
    if alpha == 0.0:
        return torch.zeros_like(direction)
    max_direction_norm = float(step_size) / max(abs(float(alpha)), 1e-8)
    return clip_norm(direction, max_direction_norm)


def prepare_direction(
    direction: torch.Tensor,
    *,
    alpha: float,
    norm_clip: float | None,
    step_size: float | None,
) -> torch.Tensor:
    clipped = clip_norm(direction, norm_clip)
    return enforce_step_size(clipped, alpha=alpha, step_size=step_size)


def clip_delta_matrix(delta: torch.Tensor, *, step_size: float | None) -> torch.Tensor:
    return clip_norm(delta, step_size)


def estimate_edit_cost(
    *,
    op_kind: str,
    alpha: float,
    ttl_steps: int,
    step_size: float | None = None,
    rank_cap: int | None = None,
) -> float:
    op_factor = OP_COST_FACTORS.get(op_kind, 1.0)
    magnitude = float(step_size) if step_size is not None else abs(float(alpha))
    rank_factor = max(1, int(rank_cap or 1))
    return float(op_factor * max(1, int(ttl_steps)) * magnitude * rank_factor)


def budget_metadata(
    *,
    op_kind: str,
    alpha: float,
    ttl_steps: int,
    step_size: float | None,
    rank_cap: int | None,
) -> dict[str, Any]:
    edit_cost = estimate_edit_cost(
        op_kind=op_kind,
        alpha=alpha,
        ttl_steps=ttl_steps,
        step_size=step_size,
        rank_cap=rank_cap,
    )
    return {
        "step_size": None if step_size is None else float(step_size),
        "edit_cost": edit_cost,
    }


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _field(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _numeric(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _packet_is_looping(packet: Any) -> bool:
    telemetry = _as_mapping(_field(packet, "telemetry")) or {}
    worker_view = _as_mapping(_field(packet, "worker_view")) or {}
    if _numeric(telemetry.get("repeat_flag")) > 0.5:
        return True
    if _numeric(telemetry.get("no_progress_steps")) > 0.0:
        return True
    if _numeric(telemetry.get("loop_cycle_length")) > 0.0:
        return True
    return str(worker_view.get("status", "") or "").strip().lower() == "looping"


def classify_edit_budget_pool(
    *,
    edit: Any,
    packet: Any,
    surface: Any | None = None,
    command_meta: Mapping[str, Any] | None = None,
) -> str:
    del command_meta
    if not _packet_is_looping(packet):
        return MAIN_EDIT_BUDGET_POOL

    op = _field(edit, "op")
    if str(_field(op, "kind") or "").strip().lower() != "resid_add":
        return MAIN_EDIT_BUDGET_POOL

    budget = _field(edit, "budget")
    ttl_steps = int(_numeric(_field(budget, "ttl_steps"), default=0.0))
    if ttl_steps <= 0 or ttl_steps > 2:
        return MAIN_EDIT_BUDGET_POOL

    alpha = abs(_numeric(_field(op, "alpha"), default=0.0))
    if alpha <= 0.0 or alpha > 0.08:
        return MAIN_EDIT_BUDGET_POOL

    step_size = _field(budget, "step_size")
    if step_size is None and surface is not None:
        caps = _field(surface, "caps")
        step_size = _field(caps, "step_size")
    if step_size is not None and _numeric(step_size, default=0.0) > 0.08:
        return MAIN_EDIT_BUDGET_POOL

    target = surface if surface is not None else _field(edit, "target")
    if _field(target, "kind") is None:
        nested_target = _field(target, "target")
        if nested_target is not None:
            target = nested_target
    if str(_field(target, "kind") or "").strip().lower() != "activation":
        return MAIN_EDIT_BUDGET_POOL

    return LOOP_RESCUE_EDIT_BUDGET_POOL
