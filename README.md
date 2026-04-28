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

The repo has now moved beyond the initial v0 scaffold. It includes the intervention loop, structured reflection, multiple task environments, readout-collapse diagnostics, provenance-aware candidate compilation, ownership-aware operator replay, diagnostic evidence ledgers, and an offline readout analyzer path.

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
- a scaffolded SAE-style readout analyzer backend that emits feature hints without becoming a runtime dependency
- diagnostic-only readout tools, including first-piece logit probes and attention head ablation
- a bounded controller diagnostic request loop:
  - controller emits `meta.diagnostic_request`
  - runtime executes cached/bounded diagnostics
  - next packets expose `latest_diagnostic_results` and `recent_diagnostic_results`
  - production apply remains blocked unless independently certified
- a stricter `controller / analyzer / gate / runtime guardrail` separation with explicit logging of:
  - `sidecar_suggested_*`
  - `gate_report_*`
  - `controller_selected_bundle_key`
  - `controller_rejected_signals`
- operator-family and operator-recipe certification via actual-delta replay
- ownership-aware replay summaries that distinguish:
  - `self_actuator`
  - `bridge_actuator`
  - `cross_bound`
  - `dead_actuator`
  - `noisy_or_harmful`
- diagnostic evidence ledgers that summarize, per bundle:
  - `readout_reachable`
  - `head_sensitive`
  - `feature_supported`
  - `operator_certified`
  - `blocked_by`
  - the next diagnostic request the controller should ask for

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
- ensuring that the selected bundle is also the bundle that actually receives the lift

The main research question has therefore sharpened from “can we intervene at all?” to:

> Which bounded, auditable runtime controls actually improve readout competitiveness at the answer boundary, and can that lift stay owned by the intended bundle?

The latest runs narrow this further. For the `budget` frontier in the constrained rewrite direct-scan replay, the diagnostic ledger reports:

- `readout_reachable = true`
- `head_sensitive = true`
- `feature_supported = true`
- `operator_certified = false`
- `blocked_by = dead_actuator / all_dead_actuator`

That means the current bottleneck is not visibility. The stack can now see a plausible frontier and collect non-dead diagnostic signals, but it still lacks a certified self or bridge actuator that can safely move the worker at the answer boundary.

The newest diagnostic-request replay closes one more control-plane loop:

- the controller sees the `budget` frontier
- the controller asks for `operator_diagnostic_replay`
- runtime returns one diagnostic result in `latest_diagnostic_results`
- the result says `production_apply_allowed = false`
- the reason remains `blocked_by = dead_actuator / all_dead_actuator`

This is an important negative result. The controller can now request the next measurement without converting that measurement into apply authority.

## What We Have Learned

Several concrete lessons have come out of the recent runs:

- Stability is not progress. Lower entropy or higher margin can still reinforce a bad local attractor.
- Loop relief is useful but insufficient. It can improve local behavior without moving task score.
- Semantic focus and reachable focus are different. A term can be semantically relevant but unreachable at the current decode state.
- Candidate quality matters more than candidate count. Many nominally different candidates collapse into the same effect family.
- Source provenance matters. `SOURCE:` body spans are usually more useful than header-only constraint spans for escape and recall.
- Sidecars are promising when kept bounded. The best use so far is to improve candidate ranking and vetoes, not to become a hidden answer channel.
- Selector quality and operator quality are not the same problem. The current stack can now nominate challengers more cleanly than it can certify that those challengers actually win when applied.
- “Helpful” is not enough. We now explicitly distinguish between lift that belongs to the intended bundle and lift that gets stolen by another bundle.
- Ownership matters. A recipe can move logits while still being the wrong actuator if `realized_lift_bundle != intended_bundle`.
- Diagnostic signals are not apply permission. `logit_adjacent`, attention ablation, and SAE-style feature hints improve the controller's map, but they do not certify a runtime edit.
- Operator certification is starting to look slightly over-powered if treated as a decision layer. It should be kept as evidence for the controller, not as a second policy owner.

## Readout Analyzer Status

The readout analyzer path is intentionally minimal and offline-oriented.

It currently:

- captures answer-boundary and source-body sites
- allows an external analyzer to inspect them
- feeds back only small hints such as:
  - `focus_term_override`
  - `term_anchor_strength_by_term`
  - `candidate_support_terms`
  - `candidate_support_scores`
  - `bundle_support_scores`
  - `bundle_evidence_vectors`
  - `sae_feature_hints`
  - optional family/key vetoes

It currently does **not**:

- inject raw feature vectors into the controller
- force task answers
- replace runtime actuation with a heavyweight analysis stack

This is deliberate. The analyzer is meant to improve the candidate compiler’s eyes, not to smuggle in a second generator.

The current `sae_scaffold` mode is not a full SAELens integration. It is a feature-emitter scaffold that marks evidence as `feature_backend="sae_sidecar"` and keeps policy ownership in the controller.

## Diagnostic Ledger

The latest control-plane addition is a diagnostic evidence ledger.

It collects bridge eval, operator replay, readout-local probes, attention head ablation, and analyzer feature hints into a bundle-indexed status table. A typical frontier status looks like:

```json
{
  "bundle_key": "kv_pair:budget:source_body:72:73",
  "readout_reachable": true,
  "head_sensitive": true,
  "feature_supported": true,
  "operator_certified": false,
  "blocked_by": ["dead_actuator", "all_dead_actuator"],
  "next_evidence_needed": "certified_self_or_bridge_actuator",
  "diagnostic_request": "operator_diagnostic_replay"
}
```

This ledger gives the controller more room to reason about what evidence to request next while preserving the hard boundary:

- diagnostics may improve visibility
- diagnostics may suggest the next measurement
- diagnostics may block unsafe application
- diagnostics do not grant production apply permission

That last point is important. The current design intentionally lets the controller become more curious without letting helper modules become covert selectors.

The replay harness now has a `diagnostic_request` controller mode for this contract. In direct-scan GPT-2 replay it produces:

```json
{
  "replay_mode": "directscan_diagnostic_request",
  "diagnostic_request_event_count": 1,
  "diagnostic_result_event_count": 1,
  "controller_diagnostic_request_names": ["operator_diagnostic_replay"],
  "latest_diagnostic_results": [
    {
      "diagnostic": "operator_diagnostic_replay",
      "bundle_key": "kv_pair:budget:source_body:72:73",
      "production_apply_allowed": false,
      "blocked_by": ["dead_actuator", "all_dead_actuator"]
    }
  ]
}
```

## Decision Ownership

The current runtime is intentionally moving toward a stricter separation of powers:

- the `controller` is the only policy owner and is responsible for the final `apply / noop / rollback` decision
- the `readout analyzer` is an evidence emitter that may suggest focus terms or bundle support, but does not own final selection
- the `gate` is treated as an evaluation/report layer rather than a shadow selector
- runtime guardrails remain outside the controller for purely mechanical invariants such as budget, compile validity, and tensor safety

This split now also shows up in logs:

- `sidecar_suggested_*` records what the analyzer suggested
- `gate_report_*` records what the bounded runtime evaluation reported
- `diagnostic_evidence_ledger` records what has been measured about each frontier bundle
- `bundle_diagnostic_status` records what is still missing before a bundle can be treated as safe
- `controller_selected_bundle_key` and `controller_rejected_signals` record what the controller actually adopted or rejected

That distinction matters for research hygiene. If a bundle changes and we cannot explain the change from controller-visible evidence, the helper stack is too strong.

The operator layer is especially sensitive here. Operator replay and certification are allowed to say “this recipe is dead,” “this recipe sharpens collapse,” or “this recipe looks like a self/bridge actuator.” They should not silently decide the episode. The controller remains responsible for turning that evidence into `apply`, `noop`, `rollback`, or a diagnostic request.

## Near-Term Roadmap

The next steps are now fairly concrete:

1. Keep operator certification under controller ownership.
   Operator replay is now powerful enough to shape the agenda, so it must remain an auditable evidence source rather than an implicit policy layer.
2. Keep selector and gate mostly frozen while operator quality is the active workstream.
   The main measurement axis is now operator certification and ownership, not candidate churn.
3. Search for a `budget` self-actuator recipe.
   Current real-packet replay shows that several recipes move something, but the lift is often stolen by `send`.
4. Continue ownership-first operator sweeps.
   Current promising directions are:
   - more local positive seeds such as `exact_term_token`, fused seeds, and centered local windows
   - contrastive recipes such as `minus_stealer` and `orthogonal_stealer`
   - recipe-level certification keyed by operator recipe rather than broad family alone
5. If no `budget` self-actuator appears, promote an explicit bridge plan.
   In that branch the controller would treat `budget` as the objective term and a certified `send` bridge as the readout-escape actuator, rather than pretending the lift already belongs to `budget`.
6. Expand the readout analyzer backend carefully.
   SAELens can move from scaffold to real backend only if it stays a feature emitter that returns small, auditable hints rather than becoming a runtime dependency or hidden generator.

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

To enable the current SAE-style scaffold without adding SAELens as a dependency:

```bash
python3 -m SpiralInterventionLab.examples.digit_transform_e2e \
  --provider openai \
  --controller-model gpt-4.1-mini \
  --worker-model gpt2-small \
  --task constrained_rewrite \
  --readout-analyzer sae_scaffold \
  --seed 7
```

This mode emits feature-like evidence only. It does not perform production SAE steering.

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
