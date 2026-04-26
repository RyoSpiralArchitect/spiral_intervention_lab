from __future__ import annotations

from typing import Any, Mapping, Sequence

_DELTA_KEYS = (
    "entropy",
    "top1_margin",
    "repetition_score",
    "partial_score",
    "semantic_progress_score",
    "required_term_recall",
    "required_term_span_progress",
    "forbidden_term_clean",
    "word_budget_score",
    "budget_ok",
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


def _coverage_progress(delta: Mapping[str, float], after: Mapping[str, Any] | None = None) -> bool:
    return (
        float(delta.get("required_term_recall", 0.0) or 0.0) > 1e-6
        or float(delta.get("required_term_span_progress", 0.0) or 0.0) > 0.01
        or float(delta.get("partial_score", 0.0) or 0.0) > 1e-6
        or float(delta.get("semantic_progress_score", 0.0) or 0.0) > 0.03
        or float(delta.get("progress_score", 0.0) or 0.0) > 0.2
        or float(delta.get("done", 0.0) or 0.0) > 0.5
        or _metric(after, "done") > 0.5
    )


def _constraint_cleanup_progress(delta: Mapping[str, float]) -> bool:
    return (
        float(delta.get("forbidden_term_clean", 0.0) or 0.0) > 1e-6
        or float(delta.get("budget_ok", 0.0) or 0.0) > 0.5
        or float(delta.get("word_budget_score", 0.0) or 0.0) > 1e-6
    )


def _loop_worsened(delta: Mapping[str, float], after: Mapping[str, Any] | None = None) -> bool:
    partial_delta = float(delta.get("partial_score", 0.0) or 0.0)
    progress_delta = float(delta.get("progress_score", 0.0) or 0.0)
    return (
        float(delta.get("repetition_score", 0.0) or 0.0) > 0.05
        or float(delta.get("repeat_flag", 0.0) or 0.0) > 0.5
        or float(delta.get("no_progress_steps", 0.0) or 0.0) > 0.5
        or (_after_is_looping(after) and partial_delta <= 1e-6 and progress_delta <= 0.0 and _metric(after, "done") < 0.5)
    )


def _loop_relieved(delta: Mapping[str, float]) -> bool:
    return (
        float(delta.get("repetition_score", 0.0) or 0.0) < -0.05
        or float(delta.get("repeat_flag", 0.0) or 0.0) < -0.5
        or float(delta.get("no_progress_steps", 0.0) or 0.0) < -0.5
    )


def classify_signal_profile(
    delta: Mapping[str, float],
    *,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> str:
    del before
    coverage_progress = _coverage_progress(delta, after)
    constraint_cleanup = _constraint_cleanup_progress(delta)
    loop_relieved = _loop_relieved(delta)
    loop_worsened = _loop_worsened(delta, after)
    violation_delta = float(delta.get("task_violation_count", 0.0) or 0.0)
    semantic_delta = float(delta.get("semantic_progress_score", 0.0) or 0.0)
    partial_delta = float(delta.get("partial_score", 0.0) or 0.0)
    progress_delta = float(delta.get("progress_score", 0.0) or 0.0)

    if coverage_progress:
        return "coverage_progress"
    if constraint_cleanup:
        return "constraint_cleanup"
    if loop_relieved and partial_delta <= 1e-6 and progress_delta <= 0.0:
        return "stabilizing_only"
    if loop_worsened or violation_delta > 0.5 or semantic_delta < -0.03:
        return "regressing"
    return "flat"


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
    semantic_delta = float(delta.get("semantic_progress_score", 0.0) or 0.0)
    required_delta = float(delta.get("required_term_recall", 0.0) or 0.0)
    required_span_delta = float(delta.get("required_term_span_progress", 0.0) or 0.0)
    forbidden_clean_delta = float(delta.get("forbidden_term_clean", 0.0) or 0.0)
    budget_ok_delta = float(delta.get("budget_ok", 0.0) or 0.0)
    budget_score_delta = float(delta.get("word_budget_score", 0.0) or 0.0)
    progress_delta = float(delta.get("progress_score", 0.0) or 0.0)
    repetition_delta = float(delta.get("repetition_score", 0.0) or 0.0)
    repeat_flag_delta = float(delta.get("repeat_flag", 0.0) or 0.0)
    no_progress_delta = float(delta.get("no_progress_steps", 0.0) or 0.0)
    violation_delta = float(delta.get("task_violation_count", 0.0) or 0.0)
    done_delta = float(delta.get("done", 0.0) or 0.0)
    entropy_delta = float(delta.get("entropy", 0.0) or 0.0)
    margin_delta = float(delta.get("top1_margin", 0.0) or 0.0)

    made_task_progress = _coverage_progress(delta, after) or _constraint_cleanup_progress(delta)
    lost_task_progress = (
        required_delta < -1e-6
        or required_span_delta < -0.01
        or partial_delta < -1e-6
        or semantic_delta < -0.03
        or forbidden_clean_delta < -1e-6
        or budget_ok_delta < -0.5
        or budget_score_delta < -1e-6
        or progress_delta < -0.2
    )
    loop_worsened = _loop_worsened(delta, after)
    loop_relieved = _loop_relieved(delta)
    telemetry_support = (-entropy_delta) + margin_delta

    if made_task_progress:
        return "helpful"
    if lost_task_progress or violation_delta > 0.5:
        return "harmful"
    if loop_worsened and partial_delta <= 1e-6 and progress_delta <= 0.0:
        return "harmful"
    if loop_relieved and violation_delta <= 0.0:
        return "neutral"
    if violation_delta < -0.5 and not _after_is_looping(after):
        return "helpful"
    if telemetry_support < -0.2 and partial_delta <= 1e-6 and progress_delta <= 0.0:
        return "harmful"
    return "neutral"


def classify_actuator_effect(
    delta: Mapping[str, float],
    *,
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> str:
    del before
    partial_delta = float(delta.get("partial_score", 0.0) or 0.0)
    semantic_delta = float(delta.get("semantic_progress_score", 0.0) or 0.0)
    recall_delta = float(delta.get("required_term_recall", 0.0) or 0.0)
    span_delta = float(delta.get("required_term_span_progress", 0.0) or 0.0)
    progress_delta = float(delta.get("progress_score", 0.0) or 0.0)
    forbidden_delta = float(delta.get("forbidden_term_clean", 0.0) or 0.0)
    budget_score_delta = float(delta.get("word_budget_score", 0.0) or 0.0)
    budget_ok_delta = float(delta.get("budget_ok", 0.0) or 0.0)
    repetition_delta = float(delta.get("repetition_score", 0.0) or 0.0)
    repeat_flag_delta = float(delta.get("repeat_flag", 0.0) or 0.0)
    entropy_delta = float(delta.get("entropy", 0.0) or 0.0)
    top1_margin_delta = float(delta.get("top1_margin", 0.0) or 0.0)

    made_term_progress = _coverage_progress(delta, after) or _constraint_cleanup_progress(delta)
    if made_term_progress:
        return "self_or_progress_actuator"

    no_term_progress = (
        recall_delta <= 1e-6
        and span_delta <= 0.01
        and partial_delta <= 1e-6
        and semantic_delta <= 0.03
        and progress_delta <= 0.0
        and forbidden_delta <= 1e-6
        and budget_score_delta <= 1e-6
        and budget_ok_delta <= 0.0
    )
    collapse_like = (
        repeat_flag_delta > 0.0
        or repetition_delta > 0.05
        or _after_is_looping(after)
    )
    sharpens_wrong_basin = entropy_delta < -0.02 and top1_margin_delta > 0.002
    if no_term_progress and collapse_like and sharpens_wrong_basin:
        return "collapse_sharpener"

    if (
        no_term_progress
        and abs(repetition_delta) <= 0.05
        and abs(repeat_flag_delta) <= 0.5
        and abs(entropy_delta) <= 0.02
        and abs(top1_margin_delta) <= 0.002
    ):
        return "dead_actuator"

    return "unknown"


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
    surface_family_key: str | None = None,
    operator_recipe_id: str | None = None,
    operator_recipe_seed_key: str | None = None,
    bundle_key: str | None = None,
    objective_bundle_key: str | None = None,
    step_actuator_bundle_key: str | None = None,
    apply_kind: str | None = None,
    production_apply_allowed: bool | None = None,
    production_policy_would_apply: bool | None = None,
    certified_for_apply: bool | None = None,
) -> dict[str, Any]:
    delta = compute_metric_delta(before, after)
    effect = {
        "edit_id": edit_id,
        "surface_id": surface_id,
        "observed_window_steps": observed_window_steps,
        "before": dict(before),
        "after": dict(after),
        "delta": delta,
        "signal_profile": classify_signal_profile(delta, before=before, after=after) if delta else "flat",
        "verdict": classify_effect(delta, before=before, after=after) if delta else "unknown",
        "actuator_class": classify_actuator_effect(delta, before=before, after=after) if delta else "unknown",
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
    if surface_family_key is not None:
        effect["surface_family_key"] = str(surface_family_key)
    if operator_recipe_id is not None:
        effect["operator_recipe_id"] = str(operator_recipe_id)
    if operator_recipe_seed_key is not None:
        effect["operator_recipe_seed_key"] = str(operator_recipe_seed_key)
    if bundle_key is not None:
        effect["bundle_key"] = str(bundle_key)
    if objective_bundle_key is not None:
        effect["objective_bundle_key"] = str(objective_bundle_key)
    if step_actuator_bundle_key is not None:
        effect["step_actuator_bundle_key"] = str(step_actuator_bundle_key)
    if apply_kind is not None:
        effect["apply_kind"] = str(apply_kind)
    if production_apply_allowed is not None:
        effect["production_apply_allowed"] = bool(production_apply_allowed)
    if production_policy_would_apply is not None:
        effect["production_policy_would_apply"] = bool(production_policy_would_apply)
    if certified_for_apply is not None:
        effect["certified_for_apply"] = bool(certified_for_apply)
    return effect


def summarize_effects(effects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    verdict_counts = {"helpful": 0, "neutral": 0, "harmful": 0, "unknown": 0}
    grouped: dict[str, dict[str, Any]] = {}
    latest: list[dict[str, Any]] = []
    recent_harmful_surface_family_keys: list[str] = []
    recent_collapse_sharpener_surface_family_keys: list[str] = []
    recent_harmful_operator_recipe_ids: list[str] = []
    recent_collapse_sharpener_operator_recipe_ids: list[str] = []
    recent_harmful_bundle_keys: list[str] = []
    recent_collapse_sharpener_bundle_keys: list[str] = []

    def _append_unique(items: list[str], value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in items:
            items.append(text)

    for effect in effects:
        verdict = str(effect.get("verdict", "unknown"))
        actuator_class = str(effect.get("actuator_class", "unknown") or "unknown")
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
                "sum_semantic_progress_score_delta": 0.0,
                "sum_required_term_recall_delta": 0.0,
                "sum_required_term_span_progress_delta": 0.0,
                "sum_forbidden_term_clean_delta": 0.0,
                "sum_word_budget_score_delta": 0.0,
                "sum_budget_ok_delta": 0.0,
                "sum_progress_delta": 0.0,
                "sum_no_progress_delta": 0.0,
                "sum_task_violation_delta": 0.0,
                "stabilizing_only_count": 0,
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
            stats["sum_semantic_progress_score_delta"] += float(delta.get("semantic_progress_score", 0.0) or 0.0)
            stats["sum_required_term_recall_delta"] += float(delta.get("required_term_recall", 0.0) or 0.0)
            stats["sum_required_term_span_progress_delta"] += float(delta.get("required_term_span_progress", 0.0) or 0.0)
            stats["sum_forbidden_term_clean_delta"] += float(delta.get("forbidden_term_clean", 0.0) or 0.0)
            stats["sum_word_budget_score_delta"] += float(delta.get("word_budget_score", 0.0) or 0.0)
            stats["sum_budget_ok_delta"] += float(delta.get("budget_ok", 0.0) or 0.0)
            stats["sum_progress_delta"] += float(delta.get("progress_score", 0.0) or 0.0)
            stats["sum_no_progress_delta"] += float(delta.get("no_progress_steps", 0.0) or 0.0)
            stats["sum_task_violation_delta"] += float(delta.get("task_violation_count", 0.0) or 0.0)
        if str(effect.get("signal_profile", "flat")) == "stabilizing_only":
            stats["stabilizing_only_count"] += 1
        if effect.get("expected_effect") is not None:
            stats["last_expected_effect"] = str(effect["expected_effect"])

        effect_delta = effect.get("delta", {})
        effect_after = effect.get("after", {})
        latest.append(
            {
                "edit_id": effect.get("edit_id"),
                "surface_id": effect.get("surface_id"),
                "surface_family_key": effect.get("surface_family_key"),
                "operator_recipe_id": effect.get("operator_recipe_id"),
                "operator_recipe_seed_key": effect.get("operator_recipe_seed_key"),
                "bundle_key": effect.get("bundle_key"),
                "objective_bundle_key": effect.get("objective_bundle_key"),
                "step_actuator_bundle_key": effect.get("step_actuator_bundle_key"),
                "apply_kind": effect.get("apply_kind"),
                "production_apply_allowed": effect.get("production_apply_allowed"),
                "certified_for_apply": effect.get("certified_for_apply"),
                "hypothesis": effect.get("hypothesis"),
                "expected_effect": effect.get("expected_effect"),
                "verdict": verdict,
                "actuator_class": actuator_class,
                "signal_profile": str(effect.get("signal_profile", "flat")),
                "partial_score_delta": float(effect_delta.get("partial_score", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "semantic_progress_score_delta": float(effect_delta.get("semantic_progress_score", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "required_term_recall_delta": float(effect_delta.get("required_term_recall", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "required_term_span_progress_delta": float(effect_delta.get("required_term_span_progress", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "forbidden_term_clean_delta": float(effect_delta.get("forbidden_term_clean", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "word_budget_score_delta": float(effect_delta.get("word_budget_score", 0.0) or 0.0)
                if isinstance(effect_delta, Mapping)
                else 0.0,
                "budget_ok_delta": float(effect_delta.get("budget_ok", 0.0) or 0.0)
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
        if verdict == "harmful" or actuator_class == "collapse_sharpener":
            _append_unique(recent_harmful_surface_family_keys, effect.get("surface_family_key"))
            _append_unique(recent_harmful_operator_recipe_ids, effect.get("operator_recipe_id"))
            _append_unique(recent_harmful_bundle_keys, effect.get("bundle_key"))
        if actuator_class == "collapse_sharpener":
            _append_unique(recent_collapse_sharpener_surface_family_keys, effect.get("surface_family_key"))
            _append_unique(recent_collapse_sharpener_operator_recipe_ids, effect.get("operator_recipe_id"))
            _append_unique(recent_collapse_sharpener_bundle_keys, effect.get("bundle_key"))

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
                "mean_semantic_progress_score_delta": stats["sum_semantic_progress_score_delta"] / attempts,
                "mean_required_term_recall_delta": stats["sum_required_term_recall_delta"] / attempts,
                "mean_required_term_span_progress_delta": stats["sum_required_term_span_progress_delta"] / attempts,
                "mean_forbidden_term_clean_delta": stats["sum_forbidden_term_clean_delta"] / attempts,
                "mean_word_budget_score_delta": stats["sum_word_budget_score_delta"] / attempts,
                "mean_budget_ok_delta": stats["sum_budget_ok_delta"] / attempts,
                "mean_progress_delta": stats["sum_progress_delta"] / attempts,
                "mean_no_progress_delta": stats["sum_no_progress_delta"] / attempts,
                "mean_task_violation_delta": stats["sum_task_violation_delta"] / attempts,
                "stabilizing_only_count": int(stats["stabilizing_only_count"]),
                "last_expected_effect": stats["last_expected_effect"],
            }
        )

    hypothesis_stats.sort(key=lambda item: (-int(item["attempts"]), str(item["hypothesis"])))
    return {
        "window_size": len(effects),
        "verdict_counts": verdict_counts,
        "hypothesis_stats": hypothesis_stats,
        "latest_effects": latest[-4:],
        "recent_harmful_surface_family_keys": recent_harmful_surface_family_keys[-8:],
        "recent_collapse_sharpener_surface_family_keys": recent_collapse_sharpener_surface_family_keys[-8:],
        "recent_harmful_operator_recipe_ids": recent_harmful_operator_recipe_ids[-8:],
        "recent_collapse_sharpener_operator_recipe_ids": recent_collapse_sharpener_operator_recipe_ids[-8:],
        "recent_harmful_bundle_keys": recent_harmful_bundle_keys[-8:],
        "recent_collapse_sharpener_bundle_keys": recent_collapse_sharpener_bundle_keys[-8:],
    }
