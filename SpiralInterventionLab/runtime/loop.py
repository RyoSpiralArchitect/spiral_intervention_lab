from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .compiler import StepContext, compile_command
from .edit_budget import LOOP_RESCUE_EDIT_BUDGET_POOL, MAIN_EDIT_BUDGET_POOL
from .policy import budget_violation_reason, command_budget_usage


class TaskEnv(Protocol):
    def reset(self, seed: int) -> str:
        ...

    def score(self, output: str) -> float:
        ...

    def done(self, output: str) -> bool:
        ...


class ControllerClient(Protocol):
    def invoke(self, packet: dict[str, Any]) -> dict[str, Any]:
        ...


class WorkerRuntime(Protocol):
    def reset(self, prompt: str) -> None:
        ...

    def step(self) -> None:
        ...

    def done(self) -> bool:
        ...

    def build_controller_packet(self) -> dict[str, Any]:
        ...

    def observe_recent_effects(self) -> None:
        ...

    def tick_ttl(self) -> None:
        ...

    def cleanup_expired(self) -> None:
        ...

    def final_text(self) -> str:
        ...


class StructuredLogger(Protocol):
    def log(self, event: dict[str, Any]) -> None:
        ...


@dataclass
class InMemoryStructuredLogger:
    events: list[dict[str, Any]] = field(default_factory=list)

    def log(self, event: dict[str, Any]) -> None:
        self.events.append(_coerce_jsonable(dict(event)))


@dataclass
class JSONLStructuredLogger:
    path: Path

    def log(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_coerce_jsonable(event), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


@dataclass
class EpisodeResult:
    prompt: str
    output: str
    score: float
    steps: int


def _coerce_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _coerce_jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _coerce_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_coerce_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return {str(key): _coerce_jsonable(item) for key, item in vars(value).items()}
    except Exception:
        pass
    return value


def _latest_controller_trace(controller_client: Any) -> Mapping[str, Any] | None:
    getter = getattr(controller_client, "latest_trace", None)
    if not callable(getter):
        return None
    trace = getter()
    if not isinstance(trace, Mapping):
        return None
    return trace


def _log_controller_trace(
    logger: StructuredLogger | None,
    *,
    step: int,
    trace: Mapping[str, Any] | None,
) -> None:
    if logger is None or trace is None:
        return

    observation = trace.get("observation")
    if isinstance(observation, Mapping):
        logger.log({"event": "controller_observation", "step": step, **dict(observation)})

    for attempt in trace.get("attempts", []) if isinstance(trace.get("attempts"), Sequence) else []:
        if isinstance(attempt, Mapping):
            logger.log({"event": "controller_provider_attempt", "step": step, **dict(attempt)})

    decision = trace.get("decision")
    if isinstance(decision, Mapping):
        logger.log({"event": "controller_decision", "step": step, **dict(decision)})


def _latest_effect_trace(worker_runtime: Any) -> Mapping[str, Any] | None:
    getter = getattr(worker_runtime, "latest_effect_trace", None)
    if not callable(getter):
        return None
    trace = getter()
    if not isinstance(trace, Mapping):
        return None
    return trace


def _log_effect_trace(
    logger: StructuredLogger | None,
    *,
    step: int,
    worker_runtime: Any,
) -> None:
    if logger is None:
        return
    trace = _latest_effect_trace(worker_runtime)
    if trace is None:
        return

    completed = trace.get("completed_effects")
    if isinstance(completed, Sequence):
        for effect in completed:
            if isinstance(effect, Mapping):
                logger.log({"event": "controller_effect", "step": step, **dict(effect)})

    summary = trace.get("summary")
    if isinstance(summary, Mapping):
        logger.log({"event": "controller_effect_summary", "step": step, **dict(summary)})


def _command_field(command: Any, name: str) -> Any:
    if isinstance(command, Mapping):
        return command.get(name)
    return getattr(command, name, None)


def _command_decision(command: Any) -> str:
    return str(_command_field(command, "decision") or "").strip().lower()


def _command_meta(command: Any) -> dict[str, Any] | None:
    meta = _command_field(command, "meta")
    if not isinstance(meta, Mapping):
        return None
    return dict(meta)


def _command_edit_count(command: Any) -> int:
    edits = _command_field(command, "edits")
    if isinstance(edits, SequenceABC) and not isinstance(edits, (str, bytes, bytearray)):
        return len(edits)
    return 0


def _command_surface_family_key(command: Any) -> str | None:
    edits = _command_field(command, "edits")
    if not isinstance(edits, SequenceABC) or isinstance(edits, (str, bytes, bytearray)):
        return None
    for edit in edits:
        if isinstance(edit, Mapping):
            target = edit.get("target")
            op = edit.get("op")
        else:
            target = getattr(edit, "target", None)
            op = getattr(edit, "op", None)
        surface_id = None
        if isinstance(target, Mapping):
            surface_id = target.get("surface_id")
        elif target is not None:
            surface_id = getattr(target, "surface_id", None)
        op_kind = None
        if isinstance(op, Mapping):
            op_kind = op.get("kind")
        elif op is not None:
            op_kind = getattr(op, "kind", None)
        if surface_id:
            if op_kind:
                return f"{str(op_kind)}_{str(surface_id)}"
            return str(surface_id)
    return None


def _guarded_noop_command(
    command: Any,
    *,
    noop_reason: str | None = None,
    apply_block_reason: str | None = None,
    finish_budget_reserved: int | bool | None = None,
    evidence_bullets: Sequence[str] = (),
) -> dict[str, Any]:
    guarded_command = {
        "version": str(_command_field(command, "version") or "0.1"),
        "decision": "noop",
        "edits": [],
        "rollback_ids": [],
    }
    meta = _command_meta(command) or {}
    controller_memory = meta.get("controller_memory")
    normalized_memory = dict(controller_memory) if isinstance(controller_memory, Mapping) else {}
    if noop_reason is not None:
        meta.setdefault("noop_reason", noop_reason)
        normalized_memory.setdefault("noop_reason", noop_reason)
    if apply_block_reason is not None:
        meta.setdefault("apply_block_reason", apply_block_reason)
        normalized_memory.setdefault("apply_block_reason", apply_block_reason)
    if finish_budget_reserved is not None:
        meta.setdefault("finish_budget_reserved", finish_budget_reserved)
        normalized_memory.setdefault("finish_budget_reserved", finish_budget_reserved)
    surface_family_key = _command_surface_family_key(command)
    if surface_family_key is not None:
        meta.setdefault("surface_family_key", surface_family_key)
        normalized_memory.setdefault("surface_family_key", surface_family_key)
    bullets: list[str] = []
    raw_existing_bullets = normalized_memory.get("evidence_bullets")
    if isinstance(raw_existing_bullets, SequenceABC) and not isinstance(raw_existing_bullets, (str, bytes, bytearray)):
        for item in raw_existing_bullets:
            text = " ".join(str(item).split()).strip()
            if text and text not in bullets:
                bullets.append(text)
    for item in evidence_bullets:
        text = " ".join(str(item).split()).strip()
        if text and text not in bullets:
            bullets.append(text)
    if bullets:
        normalized_memory["evidence_bullets"] = bullets[:4]
    if normalized_memory:
        meta["controller_memory"] = normalized_memory
    if meta:
        guarded_command["meta"] = meta
    return guarded_command


def _guard_exhausted_apply_command(
    command: Any,
    packet: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    decision = _command_decision(command)
    if decision != "apply":
        return command, None
    budget = packet.get("budget")
    if not isinstance(budget, Mapping):
        return command, None
    edits_left_this_run = budget.get("edits_left_this_run")
    try:
        edits_left_this_run = int(edits_left_this_run)
    except Exception:
        return command, None
    if edits_left_this_run > 0:
        return command, None
    try:
        usage = command_budget_usage(command, packet)
    except Exception:
        usage = {
            MAIN_EDIT_BUDGET_POOL: {"edit_count": 1.0},
            LOOP_RESCUE_EDIT_BUDGET_POOL: {"edit_count": 0.0},
        }
    loop_rescue_edits_left_this_run = budget.get("loop_rescue_edits_left_this_run")
    try:
        loop_rescue_edits_left_this_run = int(loop_rescue_edits_left_this_run)
    except Exception:
        loop_rescue_edits_left_this_run = 0
    if (
        int(usage.get(MAIN_EDIT_BUDGET_POOL, {}).get("edit_count", 0.0) or 0.0) == 0
        and int(usage.get(LOOP_RESCUE_EDIT_BUDGET_POOL, {}).get("edit_count", 0.0) or 0.0) > 0
        and loop_rescue_edits_left_this_run > 0
    ):
        return command, None
    guarded_command = _guarded_noop_command(
        command,
        noop_reason="budget_exhausted",
        apply_block_reason="edits_left_this_run_exhausted",
        evidence_bullets=(
            "main_edit_budget exhausted",
            f"requested_edit_count={_command_edit_count(command)}",
        ),
    )
    return guarded_command, {
        "reason": "edits_left_this_run_exhausted",
        "original_decision": decision,
        "edits_left_this_run": edits_left_this_run,
        "requested_edit_count": _command_edit_count(command),
    }


def _guard_budget_violating_apply_command(
    command: Any,
    packet: Mapping[str, Any],
    *,
    policy: Any | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    decision = _command_decision(command)
    if decision != "apply":
        return command, None
    try:
        usage = command_budget_usage(command, packet, policy=policy)
        reason = budget_violation_reason(command, packet, policy=policy, usage=usage)
    except Exception:
        return command, None
    if reason is None:
        return command, None
    guarded_command = _guarded_noop_command(
        command,
        noop_reason="guardrail_blocked",
        apply_block_reason=reason,
        evidence_bullets=(
            "runtime_guardrail converted apply to noop",
            str(reason),
        ),
    )
    return guarded_command, {
        "reason": reason,
        "original_decision": decision,
        "requested_edit_count": _command_edit_count(command),
        "requested_main_alpha": float(usage.get(MAIN_EDIT_BUDGET_POOL, {}).get("alpha", 0.0) or 0.0),
        "requested_loop_rescue_alpha": float(
            usage.get(LOOP_RESCUE_EDIT_BUDGET_POOL, {}).get("alpha", 0.0) or 0.0
        ),
    }


def _extract_controller_memory(command: Any) -> tuple[Mapping[str, Any] | None, str | None]:
    if isinstance(command, Mapping):
        meta = command.get("meta")
        decision = command.get("decision")
    else:
        meta = getattr(command, "meta", None)
        decision = getattr(command, "decision", None)
    if not isinstance(meta, Mapping):
        return None, None if decision is None else str(decision)
    entry = meta.get("controller_memory")
    if isinstance(entry, Mapping):
        normalized = dict(entry)
    else:
        normalized = {}
    if meta.get("hypothesis") is not None and "hypothesis" not in normalized:
        normalized["hypothesis"] = meta.get("hypothesis")
    if meta.get("micro_rationale") is not None and "micro_rationale" not in normalized:
        normalized["micro_rationale"] = meta.get("micro_rationale")
    if meta.get("next_trigger") is not None and "next_trigger" not in normalized:
        normalized["next_trigger"] = meta.get("next_trigger")
    if meta.get("next_action") is not None and "next_action" not in normalized:
        normalized["next_action"] = meta.get("next_action")
    for key in (
        "noop_reason",
        "apply_block_reason",
        "apply_kind",
        "surface_family_key",
        "focus_term",
        "objective_bundle_key",
        "step_actuator_bundle_key",
        "blocked_by",
        "next_evidence_needed",
        "why_not_apply",
        "transfer_confidence",
        "same_family_escalation_risk",
        "finish_budget_reserved",
        "production_apply_allowed",
        "certified_for_apply",
        "evidence_bullets",
    ):
        if meta.get(key) is not None and key not in normalized:
            normalized[key] = meta.get(key)
    shadow_proposals = meta.get("shadow_proposals")
    if shadow_proposals is not None and "shadow_proposals" not in normalized:
        normalized["shadow_proposals"] = shadow_proposals
    if "surface_family_key" not in normalized:
        derived_surface_family_key = _command_surface_family_key(command)
        if derived_surface_family_key is not None:
            normalized["surface_family_key"] = derived_surface_family_key
    if not normalized:
        return None, None if decision is None else str(decision)
    return normalized, None if decision is None else str(decision)


def _extract_observer_check_request(command: Any) -> Mapping[str, Any] | None:
    meta = _command_meta(command)
    if not isinstance(meta, Mapping):
        return None
    request = meta.get("observer_check_request")
    if request is True:
        return {"kind": "semantic_progress"}
    if isinstance(request, Mapping):
        return dict(request)
    return None


def _extract_tool_requests(command: Any) -> list[dict[str, Any]]:
    meta = _command_meta(command)
    if not isinstance(meta, Mapping):
        return []
    requests = meta.get("tool_requests")
    if isinstance(requests, Mapping):
        return [dict(requests)]
    if not isinstance(requests, SequenceABC) or isinstance(requests, (str, bytes, bytearray)):
        return []
    normalized: list[dict[str, Any]] = []
    for item in requests:
        if isinstance(item, Mapping):
            normalized.append(dict(item))
    return normalized


def _packet_strategy_hints(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    strategy_hints = packet.get("strategy_hints")
    if isinstance(strategy_hints, Mapping):
        return strategy_hints
    return {}


def _candidate_bundle_signature(item: Mapping[str, Any]) -> tuple[str, str] | None:
    surface_id = item.get("surface_id")
    if surface_id in (None, "") and isinstance(item.get("target"), Mapping):
        surface_id = item["target"].get("surface_id")
    op = item.get("op")
    op_kind = None
    if isinstance(op, Mapping):
        op_kind = op.get("kind")
    elif op is not None:
        op_kind = getattr(op, "kind", None)
    if surface_id in (None, "") or op_kind in (None, ""):
        return None
    return str(surface_id), str(op_kind)


def _command_edit_signature(edit: Any) -> tuple[str, str] | None:
    if isinstance(edit, Mapping):
        target = edit.get("target")
        op = edit.get("op")
    else:
        target = getattr(edit, "target", None)
        op = getattr(edit, "op", None)
    surface_id = None
    if isinstance(target, Mapping):
        surface_id = target.get("surface_id")
    elif target is not None:
        surface_id = getattr(target, "surface_id", None)
    op_kind = None
    if isinstance(op, Mapping):
        op_kind = op.get("kind")
    elif op is not None:
        op_kind = getattr(op, "kind", None)
    if surface_id in (None, "") or op_kind in (None, ""):
        return None
    return str(surface_id), str(op_kind)


def _packet_bundle_candidates(packet: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    strategy_hints = _packet_strategy_hints(packet)
    by_signature: dict[tuple[str, str], str] = {}
    for key in ("kv_candidate_edits", "kv_retry_candidate_edits", "shot_candidate_edits"):
        items = strategy_hints.get(key)
        if not isinstance(items, SequenceABC) or isinstance(items, (str, bytes, bytearray)):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            signature = _candidate_bundle_signature(item)
            bundle_key = item.get("bundle_key")
            if signature is None or bundle_key in (None, ""):
                continue
            by_signature.setdefault(signature, str(bundle_key))
    return by_signature


def _extract_controller_selected_bundle_key(packet: Mapping[str, Any], command: Any) -> str | None:
    if _command_decision(command) != "apply":
        return None
    bundle_map = _packet_bundle_candidates(packet)
    edits = _command_field(command, "edits")
    if not isinstance(edits, SequenceABC) or isinstance(edits, (str, bytes, bytearray)):
        return None
    matched_bundle_keys: list[str] = []
    for edit in edits:
        signature = _command_edit_signature(edit)
        if signature is None:
            continue
        bundle_key = bundle_map.get(signature)
        if bundle_key and bundle_key not in matched_bundle_keys:
            matched_bundle_keys.append(bundle_key)
    if len(matched_bundle_keys) == 1:
        return matched_bundle_keys[0]
    return None


def _dedup_text_items(*groups: Any) -> list[str]:
    items: list[str] = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, SequenceABC) and not isinstance(group, (str, bytes, bytearray)):
            values = group
        else:
            values = [group]
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in items:
                items.append(text)
    return items


def _optional_meta_text(meta: Mapping[str, Any], key: str) -> str | None:
    value = meta.get(key)
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text or None


def _normalized_shadow_proposals(meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = meta.get("shadow_proposals")
    if not isinstance(raw, SequenceABC) or isinstance(raw, (str, bytes, bytearray)):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw[:3]:
        if not isinstance(item, Mapping):
            continue
        proposal: dict[str, Any] = {}
        for key in (
            "kind",
            "bundle_key",
            "objective_bundle_key",
            "actuator_bundle_key",
            "term",
            "reason",
            "decision",
            "status",
        ):
            if item.get(key) not in (None, ""):
                proposal[key] = item.get(key)
        if proposal:
            normalized.append(proposal)
    return normalized


def _recent_effect_summary(packet: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = packet.get("recent_effect_summary")
    if isinstance(summary, Mapping):
        return summary
    return {}


def _recent_effect_veto_context(
    packet: Mapping[str, Any],
    *,
    frontier_bundle_key: str | None,
    surface_family_key: str | None,
    operator_recipe_id: str | None,
    objective_bundle_key: str | None,
    step_actuator_bundle_key: str | None,
) -> tuple[list[str], str | None]:
    summary = _recent_effect_summary(packet)
    harmful_families = {
        str(item)
        for item in summary.get("recent_harmful_surface_family_keys", ())
        if str(item)
    }
    collapse_families = {
        str(item)
        for item in summary.get("recent_collapse_sharpener_surface_family_keys", ())
        if str(item)
    }
    harmful_recipe_ids = {
        str(item)
        for item in summary.get("recent_harmful_operator_recipe_ids", ())
        if str(item)
    }
    collapse_recipe_ids = {
        str(item)
        for item in summary.get("recent_collapse_sharpener_operator_recipe_ids", ())
        if str(item)
    }
    harmful_bundle_keys = {
        str(item)
        for item in summary.get("recent_harmful_bundle_keys", ())
        if str(item)
    }
    collapse_bundle_keys = {
        str(item)
        for item in summary.get("recent_collapse_sharpener_bundle_keys", ())
        if str(item)
    }
    candidate_bundle_keys = {
        str(item)
        for item in (frontier_bundle_key, objective_bundle_key, step_actuator_bundle_key)
        if item not in (None, "") and str(item)
    }
    latest_effects = summary.get("latest_effects")
    matched_effects: list[Mapping[str, Any]] = []
    if isinstance(latest_effects, SequenceABC) and not isinstance(latest_effects, (str, bytes, bytearray)):
        for effect in latest_effects:
            if not isinstance(effect, Mapping):
                continue
            effect_family = str(effect.get("surface_family_key", "") or "")
            effect_recipe = str(effect.get("operator_recipe_id", "") or "")
            effect_bundle = str(effect.get("bundle_key", "") or "")
            effect_objective = str(effect.get("objective_bundle_key", "") or "")
            effect_step_actuator = str(effect.get("step_actuator_bundle_key", "") or "")
            if (
                (surface_family_key not in (None, "") and effect_family == str(surface_family_key))
                or (operator_recipe_id not in (None, "") and effect_recipe == str(operator_recipe_id))
                or (effect_bundle and effect_bundle in candidate_bundle_keys)
                or (effect_objective and effect_objective in candidate_bundle_keys)
                or (effect_step_actuator and effect_step_actuator in candidate_bundle_keys)
            ):
                matched_effects.append(effect)
    matched_harmful = bool(
        (surface_family_key not in (None, "") and str(surface_family_key) in harmful_families)
        or (operator_recipe_id not in (None, "") and str(operator_recipe_id) in harmful_recipe_ids)
        or any(bundle_key in harmful_bundle_keys for bundle_key in candidate_bundle_keys)
        or any(str(effect.get("verdict", "") or "") == "harmful" for effect in matched_effects)
    )
    matched_collapse = bool(
        (surface_family_key not in (None, "") and str(surface_family_key) in collapse_families)
        or (operator_recipe_id not in (None, "") and str(operator_recipe_id) in collapse_recipe_ids)
        or any(bundle_key in collapse_bundle_keys for bundle_key in candidate_bundle_keys)
        or any(str(effect.get("actuator_class", "") or "") == "collapse_sharpener" for effect in matched_effects)
    )
    if not matched_harmful and not matched_collapse:
        return [], None

    signals: list[str] = []
    if matched_harmful:
        signals.append("recent_harmful_family")
    if matched_collapse:
        signals.append("collapse_sharpener_veto")
    signals.append("no_certified_actuator_for_frontier")

    reason = None
    for effect in reversed(matched_effects):
        effect_family = str(effect.get("surface_family_key", "") or "")
        actuator_class = str(effect.get("actuator_class", "unknown") or "unknown")
        frontier_label = next((bundle for bundle in candidate_bundle_keys if bundle), "frontier candidate")
        if effect_family:
            reason = f"{frontier_label} was vetoed after observed {actuator_class} on {effect_family}"
        else:
            reason = f"{frontier_label} was vetoed after observed {actuator_class}"
        break
    return _dedup_text_items(signals), reason


def _diagnostic_evidence_context(strategy_hints: Mapping[str, Any]) -> dict[str, Any]:
    ledger_raw = strategy_hints.get("diagnostic_evidence_ledger")
    ledger = [
        dict(item)
        for item in ledger_raw[:12]
        if isinstance(item, Mapping)
    ] if isinstance(ledger_raw, Sequence) and not isinstance(ledger_raw, (str, bytes, bytearray)) else []
    status_raw = strategy_hints.get("bundle_diagnostic_status")
    bundle_status = {
        str(key): dict(value)
        for key, value in status_raw.items()
        if str(key) and isinstance(value, Mapping)
    } if isinstance(status_raw, Mapping) else {}
    frontier_bundle_key = str(
        strategy_hints.get(
            "diagnostic_frontier_bundle_key",
            strategy_hints.get(
                "gate_report_frontier_bundle_key",
                strategy_hints.get("selected_bundle_key", ""),
            ),
        )
        or ""
    )
    frontier_status = dict(bundle_status.get(frontier_bundle_key, {})) if frontier_bundle_key else {}
    raw_frontier_status = strategy_hints.get("diagnostic_frontier_status")
    if not frontier_status and isinstance(raw_frontier_status, Mapping):
        frontier_status = dict(raw_frontier_status)
    try:
        ledger_count = int(strategy_hints.get("diagnostic_evidence_ledger_count", len(ledger)) or len(ledger))
    except Exception:
        ledger_count = len(ledger)
    next_evidence = str(
        strategy_hints.get(
            "diagnostic_frontier_next_evidence",
            frontier_status.get("next_evidence_needed", ""),
        )
        or ""
    )
    request = str(
        strategy_hints.get(
            "diagnostic_frontier_request",
            frontier_status.get("diagnostic_request", ""),
        )
        or ""
    )
    reason_text = str(
        strategy_hints.get(
            "diagnostic_frontier_reason_text",
            frontier_status.get("reason_text", ""),
        )
        or ""
    )
    return {
        "diagnostic_evidence_ledger": ledger,
        "diagnostic_evidence_ledger_count": ledger_count,
        "bundle_diagnostic_status": bundle_status,
        "diagnostic_frontier_bundle_key": frontier_bundle_key or None,
        "diagnostic_frontier_status": frontier_status,
        "diagnostic_frontier_next_evidence": next_evidence or None,
        "diagnostic_frontier_request": request or None,
        "diagnostic_frontier_reason_text": reason_text or None,
    }


def _visible_no_safe_actuator_guidance(
    *,
    strategy_hints: Mapping[str, Any],
    controller_rejected_signals: Sequence[str],
    controller_objective_bundle_key: str | None,
    controller_step_actuator_bundle_key: str | None,
    controller_shadow_proposals: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    frontier_bundle_key = str(
        strategy_hints.get(
            "gate_report_frontier_bundle_key",
            strategy_hints.get("selected_bundle_key", ""),
        )
        or ""
    )
    unavailable_reason = str(strategy_hints.get("bridge_plan_unavailable_reason", "") or "")
    no_safe_reasons = {
        "all_dead_actuator",
        "all_collapse_sharpener",
        "no_safe_actuator",
        "no_objective_lift_to_frontier",
        "bridge_plan_not_certified",
    }
    rejected_set = {str(item) for item in controller_rejected_signals if str(item)}
    visible_no_safe = bool(
        frontier_bundle_key
        and (
            "no_certified_actuator_for_frontier" in rejected_set
            or "collapse_sharpener_veto" in rejected_set
            or unavailable_reason in no_safe_reasons
        )
    )
    if not visible_no_safe:
        return {}

    objective_key = controller_objective_bundle_key or frontier_bundle_key
    actuator_key = controller_step_actuator_bundle_key or frontier_bundle_key
    suggested_shadow = [
        {
            "kind": "frontier_shadow",
            "bundle_key": frontier_bundle_key,
            "objective_bundle_key": objective_key,
            "actuator_bundle_key": actuator_key,
            "reason": "frontier_visible_but_no_certified_actuator",
            "decision": "shadow",
        }
    ]
    if controller_shadow_proposals:
        suggested_shadow = []

    diagnostic_options = [
        "attention_head_ablation_on_frontier",
        "readout_logit_adjacent_probe",
        "sae_feature_emitter_scan",
    ]
    extra_count = strategy_hints.get("bridge_eval_extra_operator_diagnostic_count")
    try:
        extra_count_int = int(extra_count)
    except Exception:
        extra_count_int = 0
    if extra_count_int > 0:
        diagnostic_options.append("compare_extra_operator_diagnostics")
    diagnostic_context = _diagnostic_evidence_context(strategy_hints)
    frontier_next_evidence = diagnostic_context.get("diagnostic_frontier_next_evidence")
    frontier_request = diagnostic_context.get("diagnostic_frontier_request")
    frontier_reason = diagnostic_context.get("diagnostic_frontier_reason_text")
    next_evidence_options = [
        "certified_self_or_bridge_actuator",
        "non_dead_attention_ablation_signal",
        "readout_feature_emitter_support",
    ]
    if frontier_next_evidence not in (None, ""):
        next_evidence_options = [
            str(frontier_next_evidence),
            *[item for item in next_evidence_options if item != str(frontier_next_evidence)],
        ]
    if frontier_request not in (None, "", "none") and str(frontier_request) not in diagnostic_options:
        diagnostic_options.insert(0, str(frontier_request))

    return {
        "controller_loop_pressure_mode": "investigate_visible_no_safe_actuator",
        "visible_frontier_status": "visible_but_no_safe_actuator",
        "controller_next_evidence_options": next_evidence_options,
        "controller_diagnostic_request_options": diagnostic_options,
        "controller_recommended_next_evidence": frontier_next_evidence,
        "controller_recommended_diagnostic_request": frontier_request,
        "controller_diagnostic_reason_text": frontier_reason,
        "controller_suggested_shadow_proposals": suggested_shadow,
        "controller_next_action_options": [
            "request_operator_diagnostic",
            "request_attention_head_ablation",
            "request_sae_feature_scan",
            "noop_until_certified",
        ],
    }


def _build_controller_selection_report(packet: Mapping[str, Any], command: Any) -> dict[str, Any]:
    strategy_hints = _packet_strategy_hints(packet)
    meta = _command_meta(command) or {}
    helper_frontier_bundle_key = strategy_hints.get(
        "gate_report_frontier_bundle_key",
        strategy_hints.get("selected_bundle_key"),
    )
    sidecar_suggested_bundle_key = strategy_hints.get(
        "readout_analyzer_suggested_bundle_key",
        strategy_hints.get("readout_sidecar_suggested_bundle_key"),
    )
    analyzer_hints = strategy_hints.get("readout_analyzer_hints")
    if not isinstance(analyzer_hints, Mapping):
        analyzer_hints = strategy_hints.get("readout_sidecar_hints")
    if not isinstance(analyzer_hints, Mapping):
        analyzer_hints = {}
    analyzer_name = analyzer_hints.get("analyzer_name")
    analyzer_feature_backend = analyzer_hints.get("feature_backend")
    analyzer_sae_status = analyzer_hints.get("sae_status")
    analyzer_sae_feature_hints = analyzer_hints.get("sae_feature_hints")
    analyzer_sae_feature_hint_count = (
        len(analyzer_sae_feature_hints)
        if isinstance(analyzer_sae_feature_hints, Sequence) and not isinstance(analyzer_sae_feature_hints, (str, bytes, bytearray))
        else 0
    )
    bridge_plan_objective_bundle_key = strategy_hints.get("bridge_plan_objective_bundle_key")
    bridge_plan_actuator_bundle_key = strategy_hints.get("bridge_plan_actuator_bundle_key")
    bridge_plan_reason = strategy_hints.get("bridge_plan_reason")
    bridge_plan_unavailable_reason = strategy_hints.get("bridge_plan_unavailable_reason")
    bridge_plan_unavailable_objective_bundle_key = strategy_hints.get("bridge_plan_unavailable_objective_bundle_key")
    bridge_plan_unavailable_objective_reasons = strategy_hints.get("bridge_plan_unavailable_objective_reasons")
    bridge_eval_context_drift = strategy_hints.get("bridge_eval_context_drift")
    controller_selected_bundle_key = _extract_controller_selected_bundle_key(packet, command)
    controller_objective_bundle_key = _optional_meta_text(meta, "objective_bundle_key")
    if controller_objective_bundle_key is None and controller_selected_bundle_key not in (None, ""):
        controller_objective_bundle_key = str(controller_selected_bundle_key)
    controller_step_actuator_bundle_key = _optional_meta_text(meta, "step_actuator_bundle_key")
    if controller_step_actuator_bundle_key is None and controller_selected_bundle_key not in (None, ""):
        controller_step_actuator_bundle_key = str(controller_selected_bundle_key)
    controller_shadow_proposals = _normalized_shadow_proposals(meta)
    controller_why_not_apply = _optional_meta_text(meta, "why_not_apply")
    controller_apply_kind = _optional_meta_text(meta, "apply_kind")
    production_apply_allowed = meta.get("production_apply_allowed")
    certified_for_apply = meta.get("certified_for_apply")
    operator_recipe_id = _optional_meta_text(meta, "operator_recipe_id")
    controller_selection_source = "controller_apply"
    if _command_decision(command) == "apply" and controller_apply_kind == "diagnostic_probe":
        controller_selection_source = "forced_diagnostic_apply"
    elif controller_selected_bundle_key is None:
        controller_selection_source = "controller_noop" if _command_decision(command) == "noop" else "controller_unresolved"
    elif controller_selected_bundle_key == helper_frontier_bundle_key:
        controller_selection_source = str(
            strategy_hints.get("gate_report_selection_source", strategy_hints.get("selection_source", "controller_apply"))
            or "controller_apply"
        )
    rejected_signal_groups: list[Any] = [meta.get("noop_reason"), meta.get("apply_block_reason")]
    if controller_selected_bundle_key is None or (
        helper_frontier_bundle_key not in (None, "")
        and controller_selected_bundle_key not in (None, "")
        and str(controller_selected_bundle_key) != str(helper_frontier_bundle_key)
    ):
        rejected_signal_groups.extend(
            [
                strategy_hints.get("gate_report_rejected_signals"),
                strategy_hints.get("gate_report_reasons"),
                strategy_hints.get("bundle_rerank_gate_reasons"),
                strategy_hints.get("rerank_vetoes"),
            ]
        )
    recent_effect_signals: list[str] = []
    recent_effect_reason: str | None = None
    if controller_selected_bundle_key is None or _command_decision(command) == "noop":
        recent_effect_signals, recent_effect_reason = _recent_effect_veto_context(
            packet,
            frontier_bundle_key=None if helper_frontier_bundle_key in (None, "") else str(helper_frontier_bundle_key),
            surface_family_key=_optional_meta_text(meta, "surface_family_key") or _command_surface_family_key(command),
            operator_recipe_id=operator_recipe_id,
            objective_bundle_key=controller_objective_bundle_key,
            step_actuator_bundle_key=controller_step_actuator_bundle_key,
        )
        rejected_signal_groups.append(recent_effect_signals)
    controller_rejected_signals = _dedup_text_items(*rejected_signal_groups)
    if controller_why_not_apply is None and recent_effect_reason is not None and _command_decision(command) == "noop":
        controller_why_not_apply = recent_effect_reason
    no_safe_guidance = _visible_no_safe_actuator_guidance(
        strategy_hints=strategy_hints,
        controller_rejected_signals=controller_rejected_signals,
        controller_objective_bundle_key=controller_objective_bundle_key,
        controller_step_actuator_bundle_key=controller_step_actuator_bundle_key,
        controller_shadow_proposals=controller_shadow_proposals,
    )
    diagnostic_context = _diagnostic_evidence_context(strategy_hints)
    if (
        controller_why_not_apply is None
        and no_safe_guidance
        and _command_decision(command) == "noop"
    ):
        controller_why_not_apply = "frontier visible but no certified self or bridge actuator is available"
    bridge_plan_visible = (
        bridge_plan_objective_bundle_key not in (None, "")
        and bridge_plan_actuator_bundle_key not in (None, "")
    )
    controller_plan_mode = (
        "dual_layer"
        if controller_objective_bundle_key not in (None, "")
        and controller_step_actuator_bundle_key not in (None, "")
        and str(controller_objective_bundle_key) != str(controller_step_actuator_bundle_key)
        else "single_layer"
    )
    controller_bridge_dual_layer_missing = bool(
        bridge_plan_visible
        and controller_plan_mode != "dual_layer"
    )
    controller_bridge_dual_layer_reason = (
        "bridge_available_but_single_layer"
        if controller_bridge_dual_layer_missing
        else None
    )
    return {
        "sidecar_suggested_bundle_key": None if sidecar_suggested_bundle_key in (None, "") else str(sidecar_suggested_bundle_key),
        "readout_analyzer_name": None if analyzer_name in (None, "") else str(analyzer_name),
        "readout_analyzer_feature_backend": None
        if analyzer_feature_backend in (None, "")
        else str(analyzer_feature_backend),
        "readout_analyzer_sae_status": None
        if analyzer_sae_status in (None, "")
        else str(analyzer_sae_status),
        "readout_analyzer_sae_feature_hint_count": int(analyzer_sae_feature_hint_count),
        "gate_report_frontier_bundle_key": None if helper_frontier_bundle_key in (None, "") else str(helper_frontier_bundle_key),
        "bridge_plan_objective_bundle_key": (
            None if bridge_plan_objective_bundle_key in (None, "") else str(bridge_plan_objective_bundle_key)
        ),
        "bridge_plan_actuator_bundle_key": (
            None if bridge_plan_actuator_bundle_key in (None, "") else str(bridge_plan_actuator_bundle_key)
        ),
        "bridge_plan_reason": None if bridge_plan_reason in (None, "") else str(bridge_plan_reason),
        "bridge_plan_unavailable_reason": (
            None if bridge_plan_unavailable_reason in (None, "") else str(bridge_plan_unavailable_reason)
        ),
        "bridge_plan_unavailable_objective_bundle_key": (
            None
            if bridge_plan_unavailable_objective_bundle_key in (None, "")
            else str(bridge_plan_unavailable_objective_bundle_key)
        ),
        "bridge_plan_unavailable_objective_reasons": (
            {
                str(key): str(value)
                for key, value in bridge_plan_unavailable_objective_reasons.items()
                if str(key) and str(value)
            }
            if isinstance(bridge_plan_unavailable_objective_reasons, Mapping)
            else {}
        ),
        "bridge_eval_context_drift": None if bridge_eval_context_drift is None else bool(bridge_eval_context_drift),
        "bridge_plan_used": bool(
            bridge_plan_actuator_bundle_key not in (None, "")
            and controller_selected_bundle_key not in (None, "")
            and str(bridge_plan_actuator_bundle_key) == str(controller_selected_bundle_key)
        ),
        "controller_objective_bundle_key": controller_objective_bundle_key,
        "controller_step_actuator_bundle_key": controller_step_actuator_bundle_key,
        "controller_plan_mode": controller_plan_mode,
        "controller_bridge_plan_visible": bool(bridge_plan_visible),
        "controller_bridge_plan_unavailable_reason": (
            None
            if bridge_plan_visible or bridge_plan_unavailable_reason in (None, "")
            else str(bridge_plan_unavailable_reason)
        ),
        "controller_bridge_dual_layer_missing": controller_bridge_dual_layer_missing,
        "controller_bridge_dual_layer_reason": controller_bridge_dual_layer_reason,
        "controller_shadow_proposals": controller_shadow_proposals,
        "controller_shadow_proposal_count": len(controller_shadow_proposals),
        "controller_apply_kind": controller_apply_kind,
        "production_apply_allowed": None if production_apply_allowed is None else bool(production_apply_allowed),
        "certified_for_apply": None if certified_for_apply is None else bool(certified_for_apply),
        "controller_why_not_apply": controller_why_not_apply,
        "controller_selected_bundle_key": controller_selected_bundle_key,
        "controller_rejected_signals": controller_rejected_signals,
        "controller_selection_source": controller_selection_source,
        **diagnostic_context,
        **no_safe_guidance,
    }


def _log_pending_observer_check_events(
    logger: StructuredLogger | None,
    *,
    step: int,
    worker_runtime: Any,
) -> None:
    if logger is None:
        return
    reader = getattr(worker_runtime, "pop_observer_check_events", None)
    if not callable(reader):
        return
    for event in reader():
        logger.log({"event": "observer_check", "step": step, **dict(event)})


def _log_pending_tool_events(
    logger: StructuredLogger | None,
    *,
    step: int,
    worker_runtime: Any,
) -> None:
    if logger is None:
        return
    reader = getattr(worker_runtime, "pop_tool_events", None)
    if not callable(reader):
        return
    for event in reader():
        logger.log({"event": "controller_tool_result", "step": step, **dict(event)})


def _record_controller_memory(
    worker_runtime: Any,
    command: Any,
    *,
    step: int,
    logger: StructuredLogger | None = None,
) -> None:
    recorder = getattr(worker_runtime, "record_controller_memory", None)
    if not callable(recorder):
        return
    entry, decision = _extract_controller_memory(command)
    if entry is None:
        return
    try:
        recorded = recorder(entry, decision=decision)
    except TypeError:
        recorded = recorder(entry)
    if logger is not None and isinstance(recorded, Mapping):
        logger.log({"event": "controller_memory", "step": step, **dict(recorded)})


def run_episode(
    task_env: TaskEnv,
    worker_runtime: WorkerRuntime,
    controller_client: ControllerClient,
    ctx: StepContext,
    *,
    logger: StructuredLogger | None = None,
    policy: Any | None = None,
) -> EpisodeResult:
    prompt = task_env.reset(seed=ctx.runtime_state.seed)
    worker_runtime.reset(prompt)
    if logger is not None:
        logger.log({"event": "episode_start", "seed": ctx.runtime_state.seed, "prompt": prompt})

    step_count = 0
    while not worker_runtime.done():
        worker_runtime.step()
        _log_pending_observer_check_events(logger, step=step_count, worker_runtime=worker_runtime)
        packet = worker_runtime.build_controller_packet()
        try:
            command = controller_client.invoke(packet)
        except Exception as exc:
            _log_controller_trace(logger, step=step_count, trace=_latest_controller_trace(controller_client))
            if logger is not None:
                logger.log({"event": "controller_error", "step": step_count, "phase": "invoke", "error": str(exc)})
            raise

        command, guard_event = _guard_exhausted_apply_command(command, packet)
        command, budget_guard_event = _guard_budget_violating_apply_command(command, packet, policy=policy)
        _log_controller_trace(logger, step=step_count, trace=_latest_controller_trace(controller_client))

        if logger is not None:
            selection_report = _build_controller_selection_report(packet, command)
            if guard_event is not None:
                logger.log({"event": "controller_guardrail", "step": step_count, **guard_event})
            if budget_guard_event is not None:
                logger.log({"event": "controller_guardrail", "step": step_count, **budget_guard_event})
            logger.log({"event": "controller_command", "step": step_count, "command": command, **selection_report})
            logger.log({"event": "controller_selection", "step": step_count, **selection_report})
        _record_controller_memory(worker_runtime, command, step=step_count, logger=logger)
        observer_check_request = _extract_observer_check_request(command)
        if observer_check_request is not None:
            requester = getattr(worker_runtime, "request_observer_check", None)
            result = None
            if callable(requester):
                try:
                    result = requester(observer_check_request, source="controller")
                except TypeError:
                    result = requester(observer_check_request)
            if logger is not None:
                request_event = {"event": "controller_observer_check_request", "step": step_count, **dict(observer_check_request)}
                request_event["executed"] = bool(result)
                if isinstance(result, Mapping):
                    request_event["result_trigger"] = result.get("trigger")
                    request_event["result_verdict"] = result.get("verdict")
                    request_event["result_score"] = result.get("score")
                logger.log(request_event)
            _log_pending_observer_check_events(logger, step=step_count, worker_runtime=worker_runtime)

        tool_requests = _extract_tool_requests(command)
        if tool_requests:
            requester = getattr(worker_runtime, "request_controller_tools", None)
            results = []
            if callable(requester):
                try:
                    results = requester(tool_requests, source="controller") or []
                except TypeError:
                    results = requester(tool_requests) or []
            if logger is not None:
                for request in tool_requests:
                    request_event = {"event": "controller_tool_request", "step": step_count, **dict(request)}
                    request_event["executed"] = bool(results)
                    logger.log(request_event)
            _log_pending_tool_events(logger, step=step_count, worker_runtime=worker_runtime)

        try:
            compiled_edits = compile_command(command, packet, ctx, policy=policy)
        except Exception as exc:
            if logger is not None:
                logger.log({"event": "controller_error", "step": step_count, "phase": "compile", "error": str(exc)})
            raise

        for compiled in compiled_edits:
            compiled.apply(ctx)
            if compiled.kind == "rollback":
                ctx.active_edits.pop(compiled.edit_id, None)
            else:
                ctx.active_edits[compiled.edit_id] = compiled
            if logger is not None:
                logger.log(
                    {
                        "event": "compiled_edit",
                        "step": step_count,
                        "edit_id": compiled.edit_id,
                        "kind": compiled.kind,
                        "ttl_steps": compiled.ttl_steps,
                    }
                )

        worker_runtime.observe_recent_effects()
        _log_effect_trace(logger, step=step_count, worker_runtime=worker_runtime)
        worker_runtime.tick_ttl()
        worker_runtime.cleanup_expired()
        step_count += 1

    output = worker_runtime.final_text()
    score = task_env.score(output)
    if logger is not None:
        logger.log(
            {
                "event": "episode_end",
                "steps": step_count,
                "output": output,
                "score": score,
                "task_done": task_env.done(output),
            }
        )
    return EpisodeResult(prompt=prompt, output=output, score=score, steps=step_count)
