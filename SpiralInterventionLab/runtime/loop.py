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
        "surface_family_key",
        "focus_term",
        "transfer_confidence",
        "same_family_escalation_risk",
        "finish_budget_reserved",
        "evidence_bullets",
    ):
        if meta.get(key) is not None and key not in normalized:
            normalized[key] = meta.get(key)
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
    controller_selected_bundle_key = _extract_controller_selected_bundle_key(packet, command)
    controller_selection_source = "controller_apply"
    if controller_selected_bundle_key is None:
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
    controller_rejected_signals = _dedup_text_items(*rejected_signal_groups)
    return {
        "sidecar_suggested_bundle_key": None if sidecar_suggested_bundle_key in (None, "") else str(sidecar_suggested_bundle_key),
        "gate_report_frontier_bundle_key": None if helper_frontier_bundle_key in (None, "") else str(helper_frontier_bundle_key),
        "controller_selected_bundle_key": controller_selected_bundle_key,
        "controller_rejected_signals": controller_rejected_signals,
        "controller_selection_source": controller_selection_source,
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
