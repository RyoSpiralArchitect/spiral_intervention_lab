# spiral_intervention_lab

Research scaffold for bounded runtime intervention experiments.

Core question:

> Can a controller LLM improve a smaller worker model only through bounded internal interventions, without directly generating the task answer itself?

The current project framing is closer to an inference-time intervention system or runtime control architecture than to ordinary prompting, RAG, or agent orchestration:

- the worker model remains the surface actor
- the controller emits typed intervention commands, not task answers
- the runtime/compiler resolves those commands into bounded edits
- task feedback and effect summaries are used to search over interventions
- auxiliary controls and sidecars are allowed only when they stay bounded, auditable, and clearly secondary to the worker

The repo has now moved beyond the initial v0 scaffold. It includes the intervention loop, structured reflection, multiple task environments, readout-collapse diagnostics, provenance-aware candidate compilation, and an offline readout sidecar path.

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
  Seeded task environments, including digit, rewrite, ordering, entailment, and structured-summary tasks.
- `SpiralInterventionLab/prompts/`
  Controller prompt assets.
- `SpiralInterventionLab/tests/`
  Focused smoke and integration tests.
- `docs/`
  Research notes and control-plane design docs.

## Current Status

The current implementation includes:

- a typed controller DSL with `resid_add`, `kv_mix`, and `rank1_patch`
- reversible, TTL-scoped runtime edits with strict budget validation
- structured controller reflection via `controller_memory`
- task-grounded effect labeling driven by `partial_score`, coverage, repetition, and violations
- multiple local bounded tools such as `dry_run_decode`, `constraint_scorer`, and `tokenize_terms`
- readout diagnostics for answer-start collapse, including attractor-family and reachable-focus signals
- provenance-aware candidate compilation with `source_body > answer_prefix > constraint_header > misc_prompt`
- same-term / same-family dominance pruning and clean `kv_pair` bundle metadata
- shot-mode and readout-escape phases, separated from loop-break / stabilization phases
- a minimal offline readout sidecar scaffold that can analyze captured sites and feed small hints back into candidate ranking

## Current Bottleneck

The project is no longer stuck in a “nothing is visible” failure mode. The current bottleneck is narrower and more interesting:

- semantic support can often be detected
- source positions and candidate spans can be resolved
- some probes are weakly positive
- but first-token answer readout still often collapses into attractor basins such as repetition or junk subword families

In practice, this means the current system is better at:

- detecting collapse
- opening `readout_escape`
- compiling cleaner candidates
- reranking candidates with auxiliary evidence

than it is at:

- making those candidates win at the answer boundary
- escaping the attractor basin strongly enough to re-enter a healthy answer-start manifold

The main research question has therefore sharpened from “can we intervene at all?” to:

> Which bounded, auditable runtime controls actually improve readout competitiveness at the answer boundary?

## What We Have Learned

Several concrete lessons have come out of the recent runs:

- Stability is not progress. Lower entropy or higher margin can still reinforce a bad local attractor.
- Loop relief is useful but insufficient. It can improve local behavior without moving task score.
- Semantic focus and reachable focus are different. A term can be semantically relevant but unreachable at the current decode state.
- Candidate quality matters more than candidate count. Many nominally different candidates collapse into the same effect family.
- Source provenance matters. `SOURCE:` body spans are usually more useful than header-only constraint spans for escape and recall.
- Sidecars are promising when kept bounded. The best use so far is to improve candidate ranking and vetoes, not to become a hidden answer channel.

## Readout Sidecar Status

The new sidecar path is intentionally minimal and offline-oriented.

It currently:

- captures answer-boundary and source-body sites
- allows an external analyzer to inspect them
- feeds back only small hints such as:
  - `focus_term_override`
  - `term_anchor_strength_by_term`
  - `candidate_support_terms`
  - `candidate_support_scores`
  - optional family/key vetoes

It currently does **not**:

- inject raw feature vectors into the controller
- force task answers
- replace runtime actuation with a heavyweight analysis stack

This is deliberate. The sidecar is meant to improve the candidate compiler’s eyes, not to smuggle in a second generator.

## Decision Ownership

The current runtime is intentionally moving toward a stricter separation of powers:

- the `controller` is the only policy owner and is responsible for the final `apply / noop / rollback` decision
- the `readout analyzer` is an evidence emitter that may suggest focus terms or bundle support, but does not own final selection
- the `gate` is treated as an evaluation/report layer rather than a shadow selector
- runtime guardrails remain outside the controller for purely mechanical invariants such as budget, compile validity, and tensor safety

This split now also shows up in logs:

- `sidecar_suggested_*` records what the analyzer suggested
- `gate_report_*` records what the bounded runtime evaluation reported
- `controller_selected_bundle_key` and `controller_rejected_signals` record what the controller actually adopted or rejected

That distinction matters for research hygiene. If a bundle changes and we cannot explain the change from controller-visible evidence, the helper stack is too strong.

## Near-Term Roadmap

The next steps are now fairly concrete:

1. Improve candidate quality after escape opens.
   Focus on stronger post-bundle reranking, family quotas, and support-aware selection.
2. Compare `off` vs `heuristic` sidecar in more realistic direct-scan and replay settings.
   The current smoke already shows that sidecar hints can flip target-term priority.
3. Strengthen readout-escape scoring.
   Prefer metrics that measure “can this candidate win at the boundary?” over purely semantic similarity.
4. Only after that, consider a heavier offline analyzer such as a SAELens-style sidecar.
   If used, it should remain a sidecar that returns small hints rather than becoming a runtime dependency.

## Design Guardrails

The current design tries hard not to drift into hidden-answer injection.

- The worker remains the main actor.
- The controller chooses bounded interventions, not free-form answers.
- Auxiliary controls stay soft, local, and inspectable.
- Sidecars are allowed to emit evidence, focus suggestions, and veto signals, but not to become covert channels or de facto policy owners.

For the current “healthy tweaking” framing, see [healthy_tweaking_control_planes.md](docs/healthy_tweaking_control_planes.md).

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

To enable the current heuristic readout analyzer while keeping it strictly bounded:

```bash
python3 -m SpiralInterventionLab.examples.digit_transform_e2e \
  --provider openai \
  --controller-model gpt-4.1-mini \
  --worker-model gpt2-small \
  --task constrained_rewrite \
  --readout-analyzer heuristic \
  --seed 7
```

`--readout-sidecar-analyzer` remains available as a backward-compatible alias, but
`--readout-analyzer` is now the preferred public name.

To force a fully local Hugging Face worker load:

```bash
python3 -m SpiralInterventionLab.examples.digit_transform_e2e \
  --provider openai \
  --controller-model gpt-4.1-mini \
  --worker-model gpt2 \
  --worker-model-path /absolute/path/to/hf-export \
  --worker-hf-offline \
  --seed 7
```

When using `--worker-model-path`, keep `--worker-model` set to the matching
TransformerLens-supported family name such as `gpt2`.

or:

```bash
python3 -m unittest discover -s SpiralInterventionLab/tests -q
```

## Notes

- `torch` is a base dependency.
- `transformers`, `mlx-lm`, `openai`, `anthropic`, `mistralai`, and `google-genai` now live behind optional extras so local/backend and controller stacks can be installed independently.
- `transformer-lens` is optional, but required for the real HookedTransformer path.
- The package directory is still `SpiralInterventionLab/` for now to keep the initial extraction low-risk. We can normalize that later once the dedicated repo settles.
