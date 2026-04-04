from __future__ import annotations

from typing import Any, Mapping, Sequence

_DELTA_KEYS = (
    "entropy",
    "top1_margin",
    "repetition_score",
    "partial_score",
    "repeat_flag",
    "no_progress_steps",
    "progress_score",
    "task_violation_count",
    "done",
)


def _coerce_metric(metrics: Mapping[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _metric(metrics: Mapping[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not isinstance(metrics, Mapping):
        return default
    value = _coerce_metric(metrics, key)
    return default if value is None else value


def _after_is_looping(after: Mapping[str, Any] | None) -> bool:
    return _metric(after, "repeat_flag") > 0.5 or _metric(after, "no_progress_steps") >= 3.0


def compute_metric_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key in _DELTA_KEYS:
        before_value = _coerce_metric(before, key)
        after_value = _coerce_metric(after, key)
        if before_value is None or after_value is None:
            continue
        delta[key] = after_value - before_value
    return delta


def classify_effect(
    delta: Mapping[str, float],
    *,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> str:
    partial_delta = float(delta.get("partial_score", 0.0) or 0.0)
    progress_delta = float(delta.get("progress_score", 0.0) or 0.0)
    repetition_delta = float(delta.get("repetition_score", 0.0) or 0.0)
    repeat_flag_delta = float(delta.get("repeat_flag", 0.0) or 0.0)
    no_progress_delta = float(delta.get("no_progress_steps", 0.0) or 0.0)
    violation_delta = float(delta.get("task_violation_count", 0.0) or 0.0)
    done_delta = float(delta.get("done", 0.0) or 0.0)
    entropy_delta = float(delta.get("entropy", 0.0) or 0.0)
    margin_delta = float(delta.get("top1_margin", 0.0) or 0.0)

    made_task_progress = partial_delta > 1e-6 or progress_delta > 0.2 or done_delta > 0.5 or _metric(after, "done") > 0.5
    lost_task_progress = partial_delta < -1e-6 or progress_delta < -0.2
    loop_worsened = (
        repetition_delta > 0.05
        or repeat_flag_delta > 0.5
        or no_progress_delta > 0.5
        or (_after_is_looping(after) and partial_delta <= 1e-6 and progress_delta <= 0.0 and _metric(after, "done") < 0.5)
    )
    loop_relieved = repetition_delta < -0.05 or repeat_flag_delta < -0.5 or no_progress_delta < -0.5
    telemetry_support = (-entropy_delta) + margin_delta

    if made_task_progress:
        return "helpful"
    if lost_task_progress or violation_delta > 0.5:
        return "harmful"
    if loop_worsened and partial_delta <= 1e-6 and progress_delta <= 0.0:
        return "harmful"
    if loop_relieved and violation_delta <= 0.0:
        return "helpful"
    if violation_delta < -0.5 and not _after_is_looping(after):
        return "helpful"
    if telemetry_support < -0.2 and partial_delta <= 1e-6 and progress_delta <= 0.0:
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
        "verdict": classify_effect(delta, before=before, after=after) if delta else "unknown",
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
                "sum_progress_delta": 0.0,
                "sum_no_progress_delta": 0.0,
                "sum_task_violation_delta": 0.0,
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
            stats["sum_progress_delta"] += float(delta.get("progress_score", 0.0) or 0.0)
            stats["sum_no_progress_delta"] += float(delta.get("no_progress_steps", 0.0) or 0.0)
            stats["sum_task_violation_delta"] += float(delta.get("task_violation_count", 0.0) or 0.0)
        if effect.get("expected_effect") is not None:
            stats["last_expected_effect"] = str(effect["expected_effect"])

        effect_delta = effect.get("delta", {})
        effect_after = effect.get("after", {})
        latest.append(
            {
                "edit_id": effect.get("edit_id"),
                "surface_id": effect.get("surface_id"),
                "hypothesis": effect.get("hypothesis"),
                "expected_effect": effect.get("expected_effect"),
                "verdict": verdict,
                "partial_score_delta": float(effect_delta.get("partial_score", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "progress_delta": float(effect_delta.get("progress_score", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "repetition_delta": float(effect_delta.get("repetition_score", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "task_violation_delta": float(effect_delta.get("task_violation_count", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "after_is_looping": _after_is_looping(effect_after if isinstance(effect_after, Mapping) else None),
                "after_no_progress_steps": int(_metric(effect_after if isinstance(effect_after, Mapping) else None, "no_progress_steps")),
                "after_task_violation_count": int(
                    _metric(effect_after if isinstance(effect_after, Mapping) else None, "task_violation_count")
                ),
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
                "mean_progress_delta": stats["sum_progress_delta"] / attempts,
                "mean_no_progress_delta": stats["sum_no_progress_delta"] / attempts,
                "mean_task_violation_delta": stats["sum_task_violation_delta"] / attempts,
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
