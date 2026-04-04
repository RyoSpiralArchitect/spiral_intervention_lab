from __future__ import annotations

import json
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

        _log_controller_trace(logger, step=step_count, trace=_latest_controller_trace(controller_client))

        if logger is not None:
            logger.log({"event": "controller_command", "step": step_count, "command": command})

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
