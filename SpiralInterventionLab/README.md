# SpiralInterventionLab

This package is the minimal experiment scaffold for:

"Can a controller LLM improve a smaller worker model only through bounded internal interventions?"

The code is intentionally narrow.

- `schema.py`
  Parses the JSON DSL and controller observation packet into typed dataclasses.
- `policy.py`
  Enforces the harness: bounded ops, bounded alpha, rollback-only surfaces, trace compatibility.
- `compiler.py`
  Compiles the DSL into executable hooks or overlays.
- `codecs.py`
  Keeps text/token conversion swappable so the runner does not depend on one tokenizer path.
- `adapter.py`
  Defines the adapter interface and a HookedTransformer-oriented binding layer.
- `rank1_bridge.py`
  Absorbs `u/v` length mismatches before parameter patches land on a concrete weight tensor.
- `tlens_runtime.py`
  Manages live TransformerLens hook registrations, cache snapshots, traces, TTL, and rollback.
- `overlays.py`
  Holds temporary runtime-only overlays such as rank-1 linear patches.
- `effects.py`
  Computes compact edit verdicts from before/after telemetry.
- `trace_recorder.py`
  Records step-aligned selected-surface traces and exposes aligned replay for paired baselines.
- `worker.py`
  Runs a token-by-token worker loop, builds controller packets, tracks budgets, and records recent edit effects.
- `baselines.py`
  Provides the minimal B0/B1/C1 execution paths plus an activation-only C1 harness.
- `loop.py`
  Runs the episode loop with structured logging.
- `backends/`
  Holds capability-based local worker backends plus a generic worker runtime for B0/B1 on non-mechanistic stacks.
- `controllers/`
  Holds black-box API provider adapters for controller models.
- `bridge/`
  Loads prompt assets and converts provider text into validated controller commands or prompt hints.
- `examples/`
  Holds runnable end-to-end examples. The first one is `digit_transform_e2e.py`.
- `tasks/`
  Holds real, seed-reproducible `TaskEnv` implementations. The first one is `SpiralDigitTransformEnv`.

## Current cut

This is a scaffold, not a full benchmark suite yet.

- The controller DSL supports `resid_add`, `kv_mix`, and `rank1_patch`.
- The runtime assumes edits are reversible and TTL-scoped to an episode.
- The default policy follows the v0 constraints: one edit per step, small alpha budget, no free-form answer channel.
- The HookedTransformer path now has a real runtime state for cache reads and hook lifecycle management.
- Paired traces can now be recorded step-by-step and replayed at the controller's current step instead of collapsing to a final-cache snapshot.
- Rank-1 parameter patches now pass through a bridge layer: exact-size match first, parameter-aware lift second, deterministic resample last. This makes `mlp_out` and other non-square-ish cases much more portable without exposing raw tensor writes.
- The worker runtime can now step autoregressively, build the observation packet schema directly, and track edit spend / recent effects across an episode.
- The baseline layer now has concrete runners for B0, B1, and C1. B1 appends bounded text hints into prompt context, while C1 locks the harness down to `resid_add` only.
- The package now includes a first real task environment: `SpiralDigitTransformEnv`, a seeded exact-match rewrite task with `task_feedback()` and `stop_checker()` helpers for worker wiring.
- There is now a capability-based local backend layer. `HFTransformersBackend` is the main local path, while `MLXLMBackend` and `LlamaCppBackend` start as shallow comparison backends.
- There is now a black-box controller layer plus bridge clients, so OpenAI / Anthropic / Mistral / Google responses can be normalized into the same `ControllerCommand` path used by the compiler.
- There is now a first end-to-end example runner that wires a black-box controller provider into a HookedTransformer worker on the digit-transform task.
- That runner now executes `B1` by default and can load the worker from a fully local Hugging Face directory with `local_files_only` semantics.

## Smoke path

Run the targeted unit tests:

```bash
python3 -m pytest -q SpiralInterventionLab/tests
```

Or with stdlib only:

```bash
python3 -m unittest discover -s SpiralInterventionLab/tests -q
```
