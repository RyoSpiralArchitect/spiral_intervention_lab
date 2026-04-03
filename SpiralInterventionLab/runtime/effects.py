from __future__ import annotations

from typing import Any, Mapping


def _coerce_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def compute_metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key in ("entropy", "top1_margin", "repetition_score", "partial_score"):
        before_value = _coerce_metric(before, key)
        after_value = _coerce_metric(after, key)
        if before_value is None or after_value is None:
            continue
        delta[key] = after_value - before_value
    return delta


def classify_effect(delta: Mapping[str, float]) -> str:
    score = 0.0
    score += delta.get("top1_margin", 0.0)
    score += delta.get("partial_score", 0.0)
    score -= delta.get("entropy", 0.0)
    score -= delta.get("repetition_score", 0.0)
    if score > 0.02:
        return "helpful"
    if score < -0.02:
        return "harmful"
    return "neutral"


def build_edit_effect(
    *,
    edit_id: str,
    surface_id: str,
    observed_window_steps: int,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    delta = compute_metric_delta(before, after)
    return {
        "edit_id": edit_id,
        "surface_id": surface_id,
        "observed_window_steps": observed_window_steps,
        "before": dict(before),
        "after": dict(after),
        "delta": delta,
        "verdict": classify_effect(delta) if delta else "unknown",
    }
