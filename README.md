# spiral_intervention_lab

Minimal scaffold for bounded internal intervention experiments.

Core question:

> Can a controller LLM improve a smaller worker model only through bounded internal interventions?

This repository currently focuses on the v0 backbone:

- task-swappable runtime interfaces
- a typed JSON edit DSL
- strict harness and budget validation
- TransformerLens-compatible activation and rank-1 intervention plumbing
- minimal baseline runners for `B0`, `B1`, and `C1`

## Layout

- `SpiralInterventionLab/`
  Main Python package.
- `SpiralInterventionLab/runtime/`
  Runtime, compiler, adapter, worker, and baseline code.
- `SpiralInterventionLab/prompts/`
  Controller prompt assets.
- `SpiralInterventionLab/tests/`
  Focused smoke and integration tests.

## Current cut

The current implementation includes:

- `resid_add`, `kv_mix`, and `rank1_patch` in the controller DSL
- reversible, TTL-scoped runtime edits
- a HookedTransformer runtime state with cache snapshots, trace bank support, rollback, and live hook management
- a rank-1 bridge layer that absorbs `u/v` dimensional mismatch before parameter overlays land
- an autoregressive worker runtime that emits the controller observation packet and tracks recent edit effects
- minimal `B0`, `B1`, and `C1` runners

## Quick start

```bash
python3 -m pytest -q SpiralInterventionLab/tests/test_controller_runtime.py
```

or:

```bash
python3 -m unittest -q SpiralInterventionLab.tests.test_controller_runtime
```

## Notes

- `torch` is a base dependency.
- `transformer-lens` is optional, but required for the real HookedTransformer path.
- The package directory is still `SpiralInterventionLab/` for now to keep the initial extraction low-risk. We can normalize that later once the dedicated repo settles.
