from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol

from .compiler import StepContext
from .loop import EpisodeResult, StructuredLogger, TaskEnv, WorkerRuntime, run_episode
from .policy import DenyTargetRule, HarnessPolicy
from .trace_recorder import StepAlignedTrace


class PromptControllerClient(Protocol):
    def invoke(self, packet: dict[str, Any]) -> str | None:
        ...


@dataclass(frozen=True)
class BaselineSuiteResult:
    b0: EpisodeResult
    b1: EpisodeResult | None = None
    c1: EpisodeResult | None = None
    paired_trace_id: str | None = None


def activation_only_policy(base: HarnessPolicy | None = None) -> HarnessPolicy:
    root = base or HarnessPolicy.default_v0()
    deny_targets = tuple(root.deny_targets) + (DenyTargetRule(kind="weight"),)
    # Activation-only includes cache-space edits; it excludes persistent weight surgery.
    return replace(root, allow_ops=("resid_add", "kv_mix"), deny_targets=deny_targets)


def run_b0(
    task_env: TaskEnv,
    worker_runtime: WorkerRuntime,
    *,
    logger: StructuredLogger | None = None,
    seed: int | None = None,
    trace_snapshot_id: str | None = None,
) -> EpisodeResult:
    return _run_promptless_episode(
        task_env,
        worker_runtime,
        logger=logger,
        seed=seed,
        trace_snapshot_id=trace_snapshot_id,
    )


def run_b1(
    task_env: TaskEnv,
    worker_runtime: WorkerRuntime,
    prompt_controller: PromptControllerClient,
    *,
    logger: StructuredLogger | None = None,
    seed: int | None = None,
    max_hints_per_run: int = 4,
    trace_snapshot_id: str | None = None,
) -> EpisodeResult:
    hints_used = 0

    def on_packet(packet: dict[str, Any]) -> None:
        nonlocal hints_used
        if hints_used >= max_hints_per_run:
            return
        try:
            advice = prompt_controller.invoke(packet)
        except Exception as exc:
            _log_prompt_controller_trace(logger, prompt_controller, step=packet["step"])
            if logger is not None:
                logger.log({"event": "controller_error", "step": packet["step"], "phase": "invoke", "error": str(exc)})
            raise
        _log_prompt_controller_trace(logger, prompt_controller, step=packet["step"])
        if advice is None:
            return
        text = str(advice).strip()
        if not text:
            return
        append_hint = getattr(worker_runtime, "append_prompt_hint", None)
        if append_hint is None:
            raise AttributeError("worker_runtime does not support prompt hints required for B1")
        if append_hint(text):
            hints_used += 1
            if logger is not None:
                logger.log({"event": "prompt_advice", "step": packet["step"], "advice": text})

    return _run_promptless_episode(
        task_env,
        worker_runtime,
        logger=logger,
        seed=seed,
        trace_snapshot_id=trace_snapshot_id,
        on_packet=on_packet,
    )


def run_c1(
    task_env: TaskEnv,
    worker_runtime: WorkerRuntime,
    controller_client: Any,
    *,
    ctx: StepContext | None = None,
    logger: StructuredLogger | None = None,
    policy: HarnessPolicy | None = None,
) -> EpisodeResult:
    resolved_ctx = ctx or _infer_step_context(worker_runtime)
    return run_episode(
        task_env,
        worker_runtime,
        controller_client,
        resolved_ctx,
        logger=logger,
        policy=policy or activation_only_policy(),
    )


def run_minimal_baseline_suite(
    task_env: TaskEnv,
    *,
    make_worker_runtime: Callable[[], WorkerRuntime],
    c1_controller: Any,
    b1_controller: PromptControllerClient | None = None,
    logger_factory: Callable[[str], StructuredLogger | None] | None = None,
    paired_trace_id: str = "paired_baseline",
    c1_policy: HarnessPolicy | None = None,
) -> BaselineSuiteResult:
    b0_worker = make_worker_runtime()
    b0 = run_b0(task_env, b0_worker, logger=_make_logger(logger_factory, "b0"), trace_snapshot_id=paired_trace_id)
    baseline_trace = _export_trace_artifact(b0_worker, paired_trace_id)

    b1 = None
    if b1_controller is not None:
        b1_worker = make_worker_runtime()
        _seed_trace_artifact(b1_worker, paired_trace_id, baseline_trace)
        b1 = run_b1(task_env, b1_worker, b1_controller, logger=_make_logger(logger_factory, "b1"))

    c1_worker = make_worker_runtime()
    _seed_trace_artifact(c1_worker, paired_trace_id, baseline_trace)
    c1 = run_c1(
        task_env,
        c1_worker,
        c1_controller,
        logger=_make_logger(logger_factory, "c1"),
        policy=c1_policy,
    )
    return BaselineSuiteResult(b0=b0, b1=b1, c1=c1, paired_trace_id=paired_trace_id)


def _run_promptless_episode(
    task_env: TaskEnv,
    worker_runtime: WorkerRuntime,
    *,
    logger: StructuredLogger | None = None,
    seed: int | None = None,
    trace_snapshot_id: str | None = None,
    on_packet: Callable[[dict[str, Any]], None] | None = None,
) -> EpisodeResult:
    episode_seed = _resolve_seed(worker_runtime, seed)
    prompt = task_env.reset(seed=episode_seed)
    worker_runtime.reset(prompt)
    if logger is not None:
        logger.log({"event": "episode_start", "seed": episode_seed, "prompt": prompt})

    step_count = 0
    while not worker_runtime.done():
        worker_runtime.step()
        packet = worker_runtime.build_controller_packet()
        if on_packet is not None:
            on_packet(packet)
        worker_runtime.observe_recent_effects()
        worker_runtime.tick_ttl()
        worker_runtime.cleanup_expired()
        step_count += 1

    output = worker_runtime.final_text()
    score = task_env.score(output)
    if trace_snapshot_id is not None:
        snapshot_trace = getattr(worker_runtime, "snapshot_trace", None)
        if snapshot_trace is not None:
            snapshot_trace(trace_snapshot_id)
        else:
            runtime_state = getattr(worker_runtime, "runtime_state", None)
            if runtime_state is not None and getattr(runtime_state, "last_cache", None) is not None:
                if hasattr(runtime_state, "snapshot_last_cache"):
                    runtime_state.snapshot_last_cache(trace_snapshot_id)

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


def _infer_step_context(worker_runtime: WorkerRuntime) -> StepContext:
    runtime_state = getattr(worker_runtime, "runtime_state", None)
    adapter = getattr(worker_runtime, "adapter", None)
    if runtime_state is None or adapter is None:
        raise ValueError("ctx was not provided and worker_runtime does not expose runtime_state + adapter")
    return StepContext(packet={}, runtime_state=runtime_state, traces={}, stats={}, adapter=adapter, active_edits={})


def _resolve_seed(worker_runtime: WorkerRuntime, seed: int | None) -> int:
    if seed is not None:
        return seed
    runtime_state = getattr(worker_runtime, "runtime_state", None)
    return int(getattr(runtime_state, "seed", 0))


def _freeze_last_cache(cache: Any) -> Any:
    if cache is None:
        return None
    return {str(name): tensor.detach().clone() for name, tensor in cache.items()}


def _export_trace_artifact(worker_runtime: WorkerRuntime, trace_id: str) -> StepAlignedTrace | Any | None:
    export_step_trace = getattr(worker_runtime, "export_step_trace", None)
    if export_step_trace is not None:
        trace = export_step_trace(trace_id)
        if trace is not None:
            return trace
    runtime_state = getattr(worker_runtime, "runtime_state", None)
    if runtime_state is None:
        return None
    trace_sequences = getattr(runtime_state, "trace_sequences", {})
    if trace_id in trace_sequences:
        return trace_sequences[trace_id]
    trace_caches = getattr(runtime_state, "trace_caches", {})
    if trace_id in trace_caches:
        return _freeze_last_cache(trace_caches[trace_id])
    return _freeze_last_cache(getattr(runtime_state, "last_cache", None))


def _seed_trace_artifact(worker_runtime: WorkerRuntime, trace_id: str, trace: StepAlignedTrace | Any | None) -> None:
    if trace is None:
        return
    runtime_state = getattr(worker_runtime, "runtime_state", None)
    if runtime_state is None:
        return
    if isinstance(trace, StepAlignedTrace):
        if hasattr(runtime_state, "put_step_trace"):
            runtime_state.put_step_trace(trace_id, trace)
            return
        if hasattr(runtime_state, "put_trace_cache"):
            runtime_state.put_trace_cache(trace_id, trace.aligned_cache())
            return
    if hasattr(runtime_state, "put_trace_cache"):
        runtime_state.put_trace_cache(trace_id, trace)


def _make_logger(
    logger_factory: Callable[[str], StructuredLogger | None] | None,
    baseline_name: str,
) -> StructuredLogger | None:
    if logger_factory is None:
        return None
    return logger_factory(baseline_name)


def _prompt_controller_trace(prompt_controller: Any) -> dict[str, Any] | None:
    getter = getattr(prompt_controller, "latest_trace", None)
    if not callable(getter):
        return None
    trace = getter()
    if not isinstance(trace, dict):
        return None
    return trace


def _log_prompt_controller_trace(
    logger: StructuredLogger | None,
    prompt_controller: Any,
    *,
    step: int,
) -> None:
    if logger is None:
        return
    trace = _prompt_controller_trace(prompt_controller)
    if trace is None:
        return

    observation = trace.get("observation")
    if isinstance(observation, dict):
        logger.log({"event": "controller_observation", "step": step, **observation})

    attempts = trace.get("attempts")
    if isinstance(attempts, list):
        for attempt in attempts:
            if isinstance(attempt, dict):
                logger.log({"event": "controller_provider_attempt", "step": step, **attempt})

    decision = trace.get("decision")
    if isinstance(decision, dict):
        logger.log({"event": "controller_decision", "step": step, **decision})
