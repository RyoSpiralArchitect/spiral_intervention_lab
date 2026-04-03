from __future__ import annotations

from typing import Any, Mapping, Sequence


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
    hypothesis: str | None = None,
    expected_effect: str | None = None,
    controller_confidence: float | None = None,
    op: str | None = None,
    step_size: float | None = None,
    edit_cost: float | None = None,
) -> dict[str, Any]:
    delta = compute_metric_delta(before, after)
    effect = {
        "edit_id": edit_id,
        "surface_id": surface_id,
        "observed_window_steps": observed_window_steps,
        "before": dict(before),
        "after": dict(after),
        "delta": delta,
        "verdict": classify_effect(delta) if delta else "unknown",
    }
    if hypothesis is not None:
        effect["hypothesis"] = str(hypothesis)
    if expected_effect is not None:
        effect["expected_effect"] = str(expected_effect)
    if controller_confidence is not None:
        effect["controller_confidence"] = float(controller_confidence)
    if op is not None:
        effect["op"] = str(op)
    if step_size is not None:
        effect["step_size"] = float(step_size)
    if edit_cost is not None:
        effect["edit_cost"] = float(edit_cost)
    return effect


def summarize_effects(effects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verdict_counts = {"helpful": 0, "neutral": 0, "harmful": 0, "unknown": 0}
    grouped: dict[str, dict[str, Any]] = {}
    latest: list[dict[str, Any]] = []

    for effect in effects:
        verdict = str(effect.get("verdict", "unknown"))
        verdict_counts.setdefault(verdict, 0)
        verdict_counts[verdict] += 1

        hypothesis = str(effect.get("hypothesis") or "unlabeled")
        stats = grouped.setdefault(
            hypothesis,
            {
                "hypothesis": hypothesis,
                "attempts": 0,
                "helpful": 0,
                "neutral": 0,
                "harmful": 0,
                "unknown": 0,
                "sum_entropy_delta": 0.0,
                "sum_top1_margin_delta": 0.0,
                "sum_repetition_delta": 0.0,
                "sum_partial_score_delta": 0.0,
                "last_expected_effect": None,
            },
        )
        stats["attempts"] += 1
        stats.setdefault(verdict, 0)
        stats[verdict] += 1
        delta = effect.get("delta", {})
        if isinstance(delta, Mapping):
            stats["sum_entropy_delta"] += float(delta.get("entropy", 0.0) or 0.0)
            stats["sum_top1_margin_delta"] += float(delta.get("top1_margin", 0.0) or 0.0)
            stats["sum_repetition_delta"] += float(delta.get("repetition_score", 0.0) or 0.0)
            stats["sum_partial_score_delta"] += float(delta.get("partial_score", 0.0) or 0.0)
        if effect.get("expected_effect") is not None:
            stats["last_expected_effect"] = str(effect["expected_effect"])

        latest.append(
            {
                "edit_id": effect.get("edit_id"),
                "surface_id": effect.get("surface_id"),
                "hypothesis": effect.get("hypothesis"),
                "expected_effect": effect.get("expected_effect"),
                "verdict": verdict,
            }
        )

    hypothesis_stats: list[dict[str, Any]] = []
    for hypothesis, stats in grouped.items():
        attempts = max(1, int(stats["attempts"]))
        hypothesis_stats.append(
            {
                "hypothesis": hypothesis,
                "attempts": attempts,
                "helpful": int(stats.get("helpful", 0)),
                "neutral": int(stats.get("neutral", 0)),
                "harmful": int(stats.get("harmful", 0)),
                "unknown": int(stats.get("unknown", 0)),
                "mean_entropy_delta": stats["sum_entropy_delta"] / attempts,
                "mean_top1_margin_delta": stats["sum_top1_margin_delta"] / attempts,
                "mean_repetition_delta": stats["sum_repetition_delta"] / attempts,
                "mean_partial_score_delta": stats["sum_partial_score_delta"] / attempts,
                "last_expected_effect": stats["last_expected_effect"],
            }
        )

    hypothesis_stats.sort(key=lambda item: (-int(item["attempts"]), str(item["hypothesis"])))
    return {
        "window_size": len(effects),
        "verdict_counts": verdict_counts,
        "hypothesis_stats": hypothesis_stats,
        "latest_effects": latest[-4:],
    }
