from __future__ import annotations

from typing import Any

import torch


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
