from __future__ import annotations

import math
from typing import Any, Mapping

import torch

MAIN_EDIT_BUDGET_POOL = "main"
LOOP_RESCUE_EDIT_BUDGET_POOL = "loop_rescue"
PRODUCTION_TRIAL_EDIT_BUDGET_POOL = "production_trial"
PRODUCTION_TRIAL_FOLLOWUP_EDIT_BUDGET_POOL = "production_trial_followup"
NORM_RELATIVE_WRITE_CONTRACT = "norm_relative_structural_v1"


OP_COST_FACTORS: dict[str, float] = {
    "resid_add": 1.0,
    "kv_mix": 1.15,
    "rank1_patch": 1.35,
    "activation_patch": 1.25,
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


def clip_relative_write(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    relative_cap: float,
    telemetry: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Limit each position's write to relative_cap times its own pre-edit norm.

    The guard sees only the hook input and output, so it holds for any adapter.
    An unsaturated write is returned unchanged rather than recomputed. The
    clip is exact in float32; casting back to a half-precision activation can
    move the realized write by that dtype's rounding, which telemetry reports.
    """
    if before.shape != after.shape:
        raise ValueError("relative write guard requires matching activation shapes")
    base = before.detach().float()
    delta = after.detach().float() - base
    if not torch.isfinite(delta).all():
        raise RuntimeError("non-finite write under the relative write contract")
    base_norm = base.norm(dim=-1, keepdim=True)
    delta_norm = delta.norm(dim=-1, keepdim=True)
    limit = float(relative_cap) * base_norm
    saturated = delta_norm > limit
    result = after
    if bool(saturated.any()):
        scale = torch.where(saturated, limit / delta_norm.clamp_min(1e-12), torch.ones_like(delta_norm))
        result = (base + delta * scale).to(after.dtype)
    if telemetry is not None:
        written = (delta_norm > 0).reshape(-1)
        telemetry["written_positions"] = int(written.sum())
        telemetry["relative_clip_saturated"] = bool(saturated.any())
        if bool(written.any()):
            # Report the position whose raw request was largest relative to its norm.
            ratio = (delta_norm / base_norm.clamp_min(1e-12)).reshape(-1)
            index = int(torch.argmax(torch.where(written, ratio, torch.full_like(ratio, -1.0))))
            site = float(base_norm.reshape(-1)[index])
            realized = float((result.detach().float() - base).norm(dim=-1).reshape(-1)[index])
            for key, value in (("site_norm", site), ("write_limit_norm", float(limit.reshape(-1)[index])),
                               ("raw_write_norm", float(delta_norm.reshape(-1)[index])),
                               ("write_norm", realized),
                               ("relative_write_norm", realized / site if site > 0 else 0.0)):
                telemetry[key] = round(value, 8)
    return result


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


def relative_write_cap(edit: Any) -> float | None:
    """The cap an edit declares under the opt-in norm-relative write contract.

    Either meta key opts in; a partial or unknown declaration is an error,
    never a silent fallback to the absolute surface caps.
    """
    meta = _as_mapping(_field(edit, "meta")) or {}
    if "write_contract" not in meta and "relative_write_cap" not in meta:
        return None
    if meta.get("write_contract") != NORM_RELATIVE_WRITE_CONTRACT:
        raise ValueError("declares an unknown write contract")
    cap = meta.get("relative_write_cap")
    if (isinstance(cap, bool) or not isinstance(cap, (int, float))
            or not math.isfinite(cap) or not 0 < cap <= 1):
        raise ValueError("relative_write_cap must be a finite number in (0, 1]")
    return float(cap)


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
    edit_meta = _as_mapping(_field(edit, "meta")) or {}
    apply_kind = str(
        edit_meta.get("apply_kind")
        or (command_meta or {}).get("apply_kind")
        or ""
    ).strip().lower()
    if apply_kind == "production_trial":
        trial_budget_class = str(
            edit_meta.get("production_trial_budget_class")
            or (command_meta or {}).get("production_trial_budget_class")
            or ""
        ).strip().lower()
        if trial_budget_class in {"alternate_followup", "followup", "production_trial_followup"}:
            return PRODUCTION_TRIAL_FOLLOWUP_EDIT_BUDGET_POOL
        return PRODUCTION_TRIAL_EDIT_BUDGET_POOL

    # Contract edits are charged in relative units, which the absolute
    # loop-rescue thresholds below cannot compare against.
    if "write_contract" in edit_meta or "relative_write_cap" in edit_meta:
        return MAIN_EDIT_BUDGET_POOL

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
