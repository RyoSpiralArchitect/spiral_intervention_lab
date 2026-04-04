from __future__ import annotations

import json
from collections.abc import Sequence as SequenceABC
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .compiler import StepContext, compile_command


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


def _guard_exhausted_apply_command(
    command: Any,
    packet: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None]:
    if not isinstance(command, Mapping):
        return command, None
    decision = str(command.get("decision", "") or "").strip().lower()
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
    guarded_command = {
        "version": str(command.get("version", "0.1") or "0.1"),
        "decision": "noop",
        "edits": [],
        "rollback_ids": [],
    }
    if isinstance(command.get("meta"), Mapping):
        guarded_command["meta"] = dict(command["meta"])
    return guarded_command, {
        "reason": "edits_left_this_run_exhausted",
        "original_decision": decision,
        "edits_left_this_run": edits_left_this_run,
        "requested_edit_count": len(command.get("edits", ()))
        if isinstance(command.get("edits"), SequenceABC)
        else 0,
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
    if not isinstance(entry, Mapping):
        return None, None if decision is None else str(decision)
    return entry, None if decision is None else str(decision)


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
        packet = worker_runtime.build_controller_packet()
        try:
            command = controller_client.invoke(packet)
        except Exception as exc:
            _log_controller_trace(logger, step=step_count, trace=_latest_controller_trace(controller_client))
            if logger is not None:
                logger.log({"event": "controller_error", "step": step_count, "phase": "invoke", "error": str(exc)})
            raise

        command, guard_event = _guard_exhausted_apply_command(command, packet)
        _log_controller_trace(logger, step=step_count, trace=_latest_controller_trace(controller_client))

        if logger is not None:
            if guard_event is not None:
                logger.log({"event": "controller_guardrail", "step": step_count, **guard_event})
            logger.log({"event": "controller_command", "step": step_count, "command": command})
        _record_controller_memory(worker_runtime, command, step=step_count, logger=logger)

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
