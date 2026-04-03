from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

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
        self.events.append(dict(event))


@dataclass
class JSONLStructuredLogger:
    path: Path

    def log(self, event: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


@dataclass
class EpisodeResult:
    prompt: str
    output: str
    score: float
    steps: int


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
        command = controller_client.invoke(packet)
        compiled_edits = compile_command(command, packet, ctx, policy=policy)

        if logger is not None:
            logger.log({"event": "controller_command", "step": step_count, "command": command})

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
