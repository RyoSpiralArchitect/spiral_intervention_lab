# spiral_intervention_lab

Minimal scaffold for bounded internal intervention experiments.

Core question:

> Can a controller LLM improve a smaller worker model only through bounded internal interventions?

This repository currently focuses on the v0 backbone:

- task-swappable runtime interfaces
- a typed JSON edit DSL
- strict harness and budget validation
- TransformerLens-compatible activation and rank-1 intervention plumbing
- step-aligned trace recording for paired baseline replay
- minimal baseline runners for `B0`, `B1`, and `C1`

## Layout

- `SpiralInterventionLab/`
  Main Python package.
- `SpiralInterventionLab/runtime/`
  Runtime, compiler, adapter, worker, and baseline code.
- `SpiralInterventionLab/backends/`
  Capability-based local worker backends for HF Transformers, MLX, and llama.cpp.
- `SpiralInterventionLab/controllers/`
  Black-box controller provider adapters for OpenAI, Anthropic, Mistral, and Google.
- `SpiralInterventionLab/bridge/`
  Strict JSON controller clients that bridge provider responses into the runtime/compiler loop.
- `SpiralInterventionLab/examples/`
  End-to-end runnable examples that wire tasks, worker runtimes, and black-box controllers together.
- `SpiralInterventionLab/tasks/`
  Seeded exact-match task environments.
- `SpiralInterventionLab/prompts/`
  Controller prompt assets.
- `SpiralInterventionLab/tests/`
  Focused smoke and integration tests.

## Current cut

The current implementation includes:

- `resid_add`, `kv_mix`, and `rank1_patch` in the controller DSL
- reversible, TTL-scoped runtime edits
- a HookedTransformer runtime state with cache snapshots, trace bank support, rollback, and live hook management
- a step-aligned trace recorder that stores selected-surface snapshots per generation step and replays them through the same trace channel
- a rank-1 bridge layer that absorbs `u/v` dimensional mismatch before parameter overlays land
- an autoregressive worker runtime that emits the controller observation packet and tracks recent edit effects
- minimal `B0`, `B1`, and `C1` runners
- a first real `TaskEnv`: `SpiralDigitTransformEnv`, a seeded exact-match digit rewrite task with strict scoring plus partial progress feedback
- capability-based local worker backends, including a practical HF Transformers path and shallow MLX / llama.cpp paths
- strict black-box controller adapters that normalize OpenAI / Anthropic / Mistral / Google responses into `ControllerCommand`
- a first end-to-end example runner that binds `ProviderControllerClient + HookedTransformerWorkerRuntime + SpiralDigitTransformEnv`

## Quick start

```bash
python3 -m pytest -q SpiralInterventionLab/tests
```

To run the first end-to-end experiment once your controller API key is set:

```bash
python3 -m SpiralInterventionLab.examples.digit_transform_e2e \
  --provider openai \
  --controller-model gpt-4.1-mini \
  --worker-model gpt2-small \
  --seed 7
```

To force a fully local Hugging Face worker load:

```bash
python3 -m SpiralInterventionLab.examples.digit_transform_e2e \
  --provider openai \
  --controller-model gpt-4.1-mini \
  --worker-model local-worker \
  --worker-model-path /absolute/path/to/hf-export \
  --worker-hf-offline \
  --seed 7
```

or:

```bash
python3 -m unittest discover -s SpiralInterventionLab/tests -q
```

## Notes

- `torch` is a base dependency.
- `transformers`, `mlx-lm`, `openai`, `anthropic`, `mistralai`, and `google-genai` now live behind optional extras so local/backend and controller stacks can be installed independently.
- `transformer-lens` is optional, but required for the real HookedTransformer path.
- The package directory is still `SpiralInterventionLab/` for now to keep the initial extraction low-risk. We can normalize that later once the dedicated repo settles.
