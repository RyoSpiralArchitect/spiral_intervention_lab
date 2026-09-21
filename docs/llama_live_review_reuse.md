# Live Review Reuse Follow-Up

## Scope

This 2026-09-21 run tests the review-reuse interface in a fresh Luna trajectory,
after the [fixed-prefix competition/completion experiment](llama_competition_completion.md).
It does **not** feed those measurements, a supplied name prefix, or the previous
debrief into the controller. Conditional name completion remains offline-only.

The original paired B0/C1 CLI is retained: full-depth 28-layer local Llama,
MPS/FP16, raw prompt, greedy decoding, L1/review/seed 0, 64-token ceiling,
sentence/EOS stopping, compact Luna prompt/packet, SAE scaffold, expanded
activation surfaces, twelve diagnostic calls and four inspection slots. The
existing analyzer rerank setting and production/trial limits are unchanged.
The ordinary B0 trace is supplied to C1 by the existing paired harness.

Checkpoint, tokenizer, selected task and baseline output/score are checked
before controller calls. The 0.7 MPS memory fraction and cleanup of unoccupied
buffers match the successful preceding pair. The new manifest enumerates all
code changes relative to that pair. This is not a same-request-sequence ablation
and cannot isolate a causal API-token or wall-clock saving.

## Live Result

Both episodes completed. The pair plus debrief took 288.83 seconds.

| Observation | Result |
| --- | --- |
| B0 / C1 score | 0.938889 / 0.938889 |
| C1 generated steps / words | 18 / 15 |
| Missing required term | Selim |
| Forbidden terms / word-budget violations | none / none |
| Luna commands | 18 parsed noops |
| Compiled rollout edits | 0 |
| Charged diagnostics / inspections | 12 / 3 |
| Historical review reuse events | 4 |
| Explicit `matched_response_probe` requests | 0 |

B0, the new C1 and the preceding C1 produce the same sentence:

> Nora will take the sample to Ivo before sending the report to Yara before dusk.

All eighteen recorded provider attempts identify `gpt-5.6-luna` and parse
successfully. The ordinary task still fails; functioning review reuse is not
an intervention success. The legacy substring scorer accepts `sending` for
`send` but does not certify preservation of both source actors.

### What Reuse Did and Did Not Change

At controller steps 5, 7, 10 and 12, a closed review was returned as
`review_reused`. Every receipt identifies a changed prefix, reports zero new
measurements/physical replays, and has identical before/after diagnostic budget.
All carry `historical_review_not_current_measurement` and false permission flags.
These are four observed avoided review executions, not four saved model replays:
the alternative review might itself only have returned cached rows.

The total diagnostic budget is nevertheless exhausted. Different intent fields
can reopen a review: step 8 asks for `activation_patch_candidate_review_on_current_prefix`
and step 11 for `current_activation_patch_candidate_review`. Subsequent charged
reviews still return `already_replayed`. Later requests also combine inspection
with operator replay. Request events and diagnostic-result events retain their
recorded step numbers; inspection result bookkeeping can use the worker step.

Crucially, describing a review as "current" does not request an actual new-prefix
measurement. Luna never emits the explicit `matched_response_probe` diagnostic.
The cache accounting works in this run, but the semantic distinction between
**read a review** and **remeasure this candidate here** has not become reliable
controller behavior. Do not remove context checks or collapse genuine measurement
requests into historical reuse to improve the count.

### Measurement Continuity

The newly executed matched matrix at controller step 4 uses the same recorded
source, prefix, token bindings and caps as the preceding pair. Its four
observables exactly match on the identity and response fields enumerated in the
audit. Two intervention settings, repeats and controls produce six physical
replays; four observables are not four independent candidates.

At `Nora will take the sample`, cap 0.16 gives canonical ` Sel` logit delta
`0.0078125`, probability delta `3.428826339e-8`, and zero top-20 hit delta.
State restoration and no-edit/repeat controls pass. This is the earlier anchor,
not the separately measured `before` position. The live run does not retest the
`sending` versus ` Sel` competition at that later position.

### API Usage and Debrief

The recorded controller calls use 351,988 input and 4,945 output tokens, including
39,882 cached input tokens. The preceding trajectory used 336,608 input and
4,225 output tokens. Therefore this run does **not** demonstrate API-token savings.
The separate debrief uses 6,698 input and 679 output tokens.

The debrief suggests fresh matched measurements and requests more stepwise
trajectory/review detail. All three wishes are classified `present_but_not_shown`:
they describe its bounded exposure, not proven absence from the runtime. Its
claim that output was unchanged "across the run" must not be read as a stalled
token trajectory; the output grew normally, and the terminal outputs match
across B0/C1. Citation checks establish exposure, not truth of the prose.
The debrief is retained as qualitative feedback, never certification or an
automatic instruction for another run.

## Next Boundary

Keep the fixed-prefix competition results and this independent live run as
separate evidence. The next interface change should make a current-prefix
measurement a concrete, controller-selected candidate/context request rather
than a new description of the same review. Inspect intent differences before
normalizing synonyms: do not merge changed dose, binding, source or context.

On the operator side, keep comparing the actual competitor gap, natural complete
name generation and deadline retention at the fixed prefix. Do not make the
conditional ` Sel` branch a production hint or raise dose to manufacture success.
This PR establishes the measurement and accounting boundary, not a safe actuator.

## Artifacts and Validation

- [Run manifest](../results/llama_l1_luna_reuse_20260921/manifest.json),
  [completion status](../results/llama_l1_luna_reuse_20260921/run_status.json),
  [audit](../results/llama_l1_luna_reuse_20260921/audit.json).
- [B0 JSONL](../results/llama_l1_luna_reuse_20260921/live/b0.jsonl) and
  [C1 JSONL](../results/llama_l1_luna_reuse_20260921/live/c1.jsonl).
- [Qualitative debrief](../results/llama_l1_luna_reuse_20260921/live/post_run_debrief.md)
  and [bounded debrief input](../results/llama_l1_luna_reuse_20260921/live/post_run_debrief_input.json).
- [Publication inventory](../results/llama_l1_luna_reuse_20260921/publication_manifest.json)
  records included source/evidence hashes and large raw tensors/operational logs
  retained locally. Original results are not rewritten or removed.

`python3 -m pytest SpiralInterventionLab/tests -q`: **356 passed, 2 subtests
passed, 16 warnings**. Working-tree and staged whitespace checks pass.
No new operator, conditional-prefix routing or production permission is added.

Reproduce through the original CLI with a fresh output directory and a local
checkpoint path. The new run's `manifest.json` records the complete argv. To
rerun its stricter sealed wrapper, use a fresh copy of the evidence wrapper
directory, keeping its repository-relative depth, with `--log-dir` pointing to
that directory's `live` child. The wrapper refuses a reused log directory,
checks the reference controls, and never records a CLI API key.
