# Llama Rewrite Difficulty Ladder v1

Follow-up: [frozen candidate across connective boundaries](llama_frozen_prefix_response.md)
separates cached review reuse from fresh prefix-local measurements. The same
patch has different probability effects across prefixes, but still no task gain.

## Question and Frozen Design

Full-depth Llama solved the original seed-7 rewrite without intervention, and
Luna C1 returned the identical output with no rollout edits. Keep that condition
as a no-regression control; do not force it into readout collapse.

This opt-in ladder increases lexical constraint difficulty without changing
controller policy, operator permissions, model depth, or decoder controls.
The ten conditions are fixed before observing their outputs:

| Level | CLI task | Required phrases | Word limit | Forbidden phrases |
| --- | --- | --- | --- | --- |
| Control | `constrained_rewrite`, seed 7 | Original 4 | Original 9 | Original 2 |
| L1 | `constrained_rewrite_l1` | 9 (8 entity/action/object terms plus deadline) | 18 | `in order to`, `should` |
| L2 | `constrained_rewrite_l2` | Same 9 | 12 | Same 2 |
| L3 | `constrained_rewrite_l3` | Same 9 | 12 | Previous 2 plus `the`, `will`, `must` |

Each new level uses three fixed SOURCE cases: delivery, review, and workshop.
Seeds `1, 0, 5` cover these in that order. SOURCE and required terms are identical
across L1/L2/L3 within a case. L1 introduces a richer two-action task, not a
single-variable comparison against the legacy source. L1 to L2 changes only the
word constraint; L2 to L3 changes only lexical exclusions. Actual success need
not be monotonic: wording changes can change a greedy trajectory.

All new levels allow 64 generated tokens and stop at sentence punctuation or
EOS, not at `max_words + 3`. This avoids turning excess words into artificial
truncation. The old task retains its original 18-token budget and stop rule.
Future B0/C1 pairs must use the same selected level, seed, prompt, precision,
tokenizer, 64-token ceiling, and stop rule.

## Scoring Boundary

The inherited scorer uses case-insensitive substring term matching and a
whitespace word count. Success here means lexical constraint satisfaction,
not certified relation preservation, grammar, single-sentence structure, or
semantic faithfulness. The three cases share a template and do not establish
cross-template generalization. Feasibility witnesses live only in tests; they
are not included in prompts, feedback, candidate sources, or controller hints.

Keep missing terms, forbidden terms, excess words, and decode truncation
separate. A finite, complete lexical failure is not automatically a readout
collapse or an operator opportunity. Read the output before choosing a causal
experiment.

## Local Screen

```bash
python3 -m SpiralInterventionLab.examples.rewrite_ladder_baseline \
  --worker-model-path "${WORKER_MODEL_PATH:?Set a local HF checkpoint path}" \
  --worker-model meta-llama/Llama-3.2-3B-Instruct \
  --device mps --dtype float16 \
  --output-dir results/llama_rewrite_ladder_v1
```

The CLI is local-only and runs the full TransformerLens worker in raw-prompt,
greedy, no-edit mode, without observer/controller orchestration or API calls.
It writes a hash-bound fixture/checkpoint/source manifest before model
generation, a row per condition in `conditions.jsonl`, and `summary.json`.
Existing output directories are rejected. Model paths remain CLI-supplied.

The prespecified selection rule requires a successful legacy control and picks
the first nonempty, non-truncated lexical failure in ascending level and case
order. It is a candidate for human coherence review and **full-runtime B0
confirmation**, not an automatically certified intervention target. If every
case passes, report that result rather than quietly strengthening the ladder.
Do not stop the screen after the first failure; retain all ten conditions.

## Next Comparison

After that confirmation, use the ordinary full-run CLI with the selected
`--task constrained_rewrite_l1|l2|l3`, seed, local model path, and Luna controller
(`gpt-5.6-luna`). Keep B0 in the suite and keep existing gates intact. Count
actual rollout edits separately from diagnostic replay. A score increase needs
a paired baseline and cannot be inferred from gap/rank movement or from a
post-run interview.

## Observations: 2026-09-14

The full ten-condition screen completed on the local 28-layer Llama checkpoint,
MPS/FP16, with no API calls and no intervention. Every output ended at sentence
punctuation before its token ceiling; no adjacent repeated words were recorded.

| Level | Success | Missing-required failures | Word-budget failures | Forbidden failures |
| --- | --- | --- | --- | --- |
| Legacy control | 1/1 | 0 | 0 | 0 |
| L1 | 2/3 | 1 | 0 | 0 |
| L2 | 0/3 | 2 | 2 | 0 |
| L3 | 1/3 | 1 | 1 | 0 |

Failure categories overlap. These are observations of three fixed lexical
variants, not estimates of population success rates.

The selected condition is **L1 / review / seed 0**. It produced:

> Nora will take the sample to Ivo before sending the report to Yara before dusk.

Score: `0.938889`; 15 words under an 18-word limit; only `Selim` is missing.
Reading the sentence suggests the two source actors were compressed into Nora
alone. This is a human interpretation of the output, not a causal mechanism
established by the lexical scorer. A separate full-runtime B0, including normal
observer/packet construction, reproduced the exact output and score in 18
generated tokens, with zero controller calls and zero rollout edits.

Two useful contrasts:

- Under L2, delivery drops the second action and four required terms entirely;
  workshop keeps every term but uses 15 rather than 12 words. Coverage loss and
  compression failure should not be treated as the same bottleneck.
- Under L3, workshop succeeds in 12 words after removing articles and a
  connective. Extra exclusions can steer a different, shorter greedy answer;
  empirical difficulty is not monotonic even though constraints are cumulative.
  Review still drops `Selim` at all three levels.

One measurement caveat is newly visible: inherited span-progress gives `Selim`
`0.4` credit from a shared character prefix in other words even when that name
is absent. Likewise, substring scoring accepts `sending` for `send`. Preserve
these legacy semantics for comparison, but do not mistake them for entity
realization or ownership evidence. The decisive L1 failure is the binary
missing-required-term check, not that partial-progress proxy.

### Artifacts

- [Sealed manifest](../results/llama_rewrite_ladder_20260914/manifest.json):
  all prompts, token ceilings, checkpoint/source hashes, and selection rule.
- [Ten outputs and token steps](../results/llama_rewrite_ladder_20260914/conditions.jsonl).
- [Screen summary](../results/llama_rewrite_ladder_20260914/summary.json).
- [Full-runtime B0 confirmation](../results/llama_rewrite_ladder_20260914/runtime_b0/summary.json)
  and [episode JSONL](../results/llama_rewrite_ladder_20260914/runtime_b0/b0.jsonl).
  B0's stock logger records episode start/end, not full per-step observation
  packets; the screen JSONL retains the no-edit token trace.

The manifest was sealed before model loading/generation; its SHA-256 is
`67c2c36c663287a80a5ac01ce888edd0a557a6c3c7763c4ecdd526c033bff48c`.
The screen and confirmation summaries hash their corresponding raw JSONL.
Artifacts describe the current working tree, including the previously validated
Llama loader/observer changes, not an already committed release.

To repeat the full-runtime confirmation after creating a fresh screen directory:

```bash
PYTHONPATH=. python3 results/llama_rewrite_ladder_20260914/confirm_b0.py \
  --screen-dir results/llama_rewrite_ladder_v1 \
  --worker-model-path "${WORKER_MODEL_PATH:?Set a local HF checkpoint path}" \
  --output-dir results/llama_rewrite_ladder_v1/runtime_b0
```

### Next Experiment

Freeze `--task constrained_rewrite_l1 --seed 0` for a paired B0/Luna C1 run.
Retain the original seed-7 task as a no-regression control. Ask whether C1 adds
the missing actor while retaining the other eight required phrases, avoiding
forbidden text and staying within 18 words. Keep all apply gates unchanged.
At this initial screen, no C1 difficulty-ladder result had been measured.
L2 compression and L3 exclusion can be subsequent axes, not simultaneous
changes while evaluating an operator on the selected L1 fixture.

## Verification

`python3 -m pytest SpiralInterventionLab/tests -q`: **333 passed**, 17 warnings.
`git diff --check` passed. Artifact hashes, current source hashes, all ten
manifest conditions, and the full-runtime output/score match were verified.
Existing loader tests cover the local full-depth Llama conversion; this ladder
screen is not a new cross-device or all-prompt numerical fidelity proof.

## Matched Luna Observation: 2026-09-21

The prespecified L1/review/seed-0 B0/C1 pair completed with `gpt-5.6-luna`.
Checkpoint, tokenizer, prompt, 28 layers, FP16, greedy decoding, 64-token
ceiling, stopping rule, and existing apply gates were unchanged. Fresh B0
matched the sealed screen exactly before any C1 provider call. The original
seed-7 success control remains a historical reference, not a fresh result here.

| Measurement | B0 | Luna C1 |
| --- | --- | --- |
| Score | 0.938889 | 0.938889 |
| Generated tokens / words | 18 / 15 | 18 / 15 |
| Missing required terms | Selim | Selim |
| Forbidden / over-budget violations | None | None |
| Rollout edits | 0 | 0 |

Both outputs are the same sentence quoted above. C1 produced 18 parsed no-op
commands, consumed 12 diagnostic calls and one separate evidence inspection.
Thus 13 diagnostic-result events do **not** mean 13 physical experiments or a
diagnostic-budget overrun. No `compiled_edit` event occurred. This is a
completed diagnostic-controller run with no task improvement, not a successful
intervention or evidence that production operators cannot ever help.

### What the Probes Showed

At the prefix `Nora will take`, a readout-direction diagnostic for Selim
reported `target_mass_delta=0.000426`, bound-piece probability delta `0.000420`,
and gap delta `-0.0625`. The frozen piece ` Sel` was already rank 6 before the
edit; its top-20 hit count did not increase. This row lacks the later matched
repeat-control protocol, so it is exploratory evidence, not a certified
actuator. Its legacy `collapse_suppressor` label coexists with
`effect_role=target_actuator`; neither establishes escape from repetition in
this coherent Llama trajectory. Keep the raw metrics and label mismatch visible.

An activation-patch review then reached late layer 27 with live hooks. At the
later prefix `Nora will take the sample`, the frozen canonical piece was rank
119. A matched `resid_pre/source_term_token` comparison used two step-size
caps, `0.04` and `0.16`, and two token-form observables per execution:

| Cap | Canonical logit delta | Canonical probability delta | Gap delta | Top-20 hit delta |
| --- | --- | --- | --- | --- |
| 0.04 | 0 | 3.39e-9 | 0 | 0 |
| 0.16 | 0.0078125 | 3.43e-8 | -0.0078125 | 0 |

The matched summary records two new intervention settings, four observables,
six physical replays including controls/repeats, identical no-edit logits and
mask, zero repeat variation, and restored runtime state. These are **one
prefix/context**, not four independent candidates. The small positive response
at the larger cap is above the measured repeat variation but remains a
quantized FP16, gap-only observation; rounded term-mass delta stays zero.
Neither cap supplies certified target actuation or production permission.

Later reviews repeatedly returned `already_replayed`. Four budget-consuming
requests at controller steps 5-8 revisited this matrix without a new matched
measurement; one tried runtime support but remained blocked by
`rank_carrier_not_target_actuator`. Counts of repeated evidence rows must not
be read as additional successes or independently replicated failures.

### Resource Boundary and Artifacts

The first attempt was killed by macOS after B0, with one Luna response and one
diagnostic recorded. The kernel reported paging exhaustion and a 17,889 MB
compressed Python process. Preserve that incomplete attempt; it is not a task
outcome. The successful retry releases completed baseline workers after trace
export and frees only unused MPS allocator buffers at cleanup boundaries. The
paired trace contents remain intact. A 0.7 allocator memory fraction makes
excess allocation fail rather than encouraging further system paging; this
changes resource handling, not model precision, task, or intervention policy.

- [Completed comparison and hashes](../results/llama_l1_luna_20260921_r2/audit.json).
- [Run manifest](../results/llama_l1_luna_20260921_r2/manifest.json),
  [B0 JSONL](../results/llama_l1_luna_20260921_r2/live/b0.jsonl), and
  [C1 JSONL](../results/llama_l1_luna_20260921_r2/live/c1.jsonl).
- [Qualitative debrief](../results/llama_l1_luna_20260921_r2/live/post_run_debrief.md).
  Its request for current-prefix evidence and a stepwise task view is useful
  design feedback, not a measurement or an instruction for the next run.
  It saw only 14 selected event anchors out of 173 event lines. Its three
  wishes were tagged `present_but_not_shown`; they do not establish that the
  runtime or live controller lacked those fields.
- [Interrupted attempt](../results/llama_l1_luna_20260921/attempt_status.json)
  and [kernel evidence](../results/llama_l1_luna_20260921/os_memory_exit.txt).
- [Memory-only retry wrapper](../results/llama_l1_luna_20260921_r2/run_memory_bounded_pair.py)
  retains the original sealed-pair checks and accepts the model path via CLI.

The completed controller calls used 336,608 input and 4,225 output tokens
(57,974 cached input tokens reported). The separate debrief used 6,654 input
and 782 output tokens. These exclude the interrupted attempt; reasoning tokens
are included in output totals, not an additional token charge here.

Validation after the worker-lifetime fix: **334 passed**, 16 warnings, two
subtests passed. The new regression checks that both B0-only and B0/B1 suites
release completed workers while C1 still receives the intact paired trace.

### Next Narrow Experiment

Do not respond to this result by opening apply gates or simply raising alpha.
First distinguish review reuse from new measurements and expose prefix-scoped
evidence age. Then compare the same bounded candidates at a few frozen
sentence-transition prefixes, retaining no-edit controls and token bindings.
Measure the complete missing name and source actor/action relations alongside
piece mass, gap, and task constraints. The hypothesis is timing/composition
failure rather than universal target invisibility; the present run does not
prove that mechanism. Do not compare the two prefixes above as a controlled
operator comparison, since both context and recipe changed.
