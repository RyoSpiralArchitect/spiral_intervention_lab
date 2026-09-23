# Entry Competition Versus Name Completion

The subsequent [live Luna pair](llama_live_review_reuse.md) tests review reuse
separately; this conditional branch was not supplied to that controller.

## Fixed Follow-Up

The previous [prefix comparison](llama_frozen_prefix_response.md) found a larger
bound-piece probability response immediately after `before`, without changing
the generated sentence. This follow-up fixes that **previously selected**
prefix. It is an exploratory follow-up, not an independent search or a claim
that this is the globally best insertion site.

The question is whether failure lies in winning the first token, completing the
name after its first piece, or preserving the rest of the task. These are
measured separately; neither a conditional branch nor a positive probability
delta grants apply permission.

Fixed controls:

- Full-depth 28-layer Llama, MPS/FP16, raw prompt, greedy decoding,
  `constrained_rewrite_l1`, seed 0, original 64-token ceiling and stop rule.
- Worker step 9: ` Nora will take the sample to Ivo before`.
- The same frozen `resid_pre L27 / source_term_token` activation patch:
  `alpha=0.04`, step-size cap `0.16`, TTL 1. This is the previously measured
  diagnostic dose, not an increased dose or a newly permitted production edit.
- The same source tensor and target token: ` Sel` / 24082. The canonical name
  tokenization is `[24082, 318]`, decoded as `[' Sel', 'im']`.
- Competitor ` sending` / 11889 comes from the earlier no-edit greedy output,
  not from selecting a favorable new comparison after measurement.

The checkpoint/tokenizer hashes, candidate ID, source tensor hash, original
anchor state, selected-prefix state, and reference JSONL hash are checked before
measurement. No controller API call or candidate-selection rule was changed.

## Measurements

The ordinary arm has two no-edit and two edited continuations. The patch affects
only the next token; the suffix is generated without it. Both conditions use
the same maximum 16-token lookahead.

The diagnostic branch then **supplies only the first piece ` Sel`** and measures
the unedited continuation twice. This is explicitly counterfactual conditioning,
not spontaneous name generation or a runtime steering success. The remainder
`im` is not supplied in the greedy branch. For longer names the helper can
separately score teacher-forced partial-name factors, bounded to eight name
tokens; those factors are not conflated with greedy completion.

Under this runtime's full-recompute, TTL-1 semantics, once the hook is rolled
back and the same output tokens are given, the suffix distribution is shared
between the two entry conditions. The product of entry and conditional suffix
probabilities therefore describes **one canonical token-sequence prefix**.
It excludes the following word boundary and alternative tokenizations. It is
not a greedy success rate, semantic score, or certified whole-name event mass.

## Result: 2026-09-21

All six replay trajectories completed in 74.55 seconds, using 52 generated-token
forward steps inside the probes. The ordinary arm used 36 and the conditioned
branch 16. All primary and conditional repeat controls had zero maximum logit
difference; original-prefix and conditional-state restoration checks passed.
The ordinary readout metrics exactly reproduced the previous step-9 row.

### First-Token Competition

| Observable | No edit | Same activation patch |
| --- | --- | --- |
| ` Sel` probability | 1.028502% | 1.044719% |
| ` sending` probability | 50.731179% | 50.337385% |
| ` Sel` rank | 5 | 5 |
| ` sending` rank | 1 | 1 |
| Competitor-minus-target logit gap | 3.8984375 | 3.8750000 |

The target logit rises `0.0078125` while the competitor logit falls `0.015625`.
The pairwise gap shrinks `0.0234375`, about 0.60% of the original gap. This is a
repeatable relative movement at the measured precision, but the greedy winner
does not change. Do not extrapolate this response linearly to a required dose.

Both ordinary continuations remain:

> Nora will take the sample to Ivo before sending the report to Yara before dusk.

The unchanged score is `0.938889`; Selim is still missing. Top-20 hit delta is
zero because the target piece was already in the top 20.

### Conditional Name Completion

Given the supplied first piece ` Sel`, the next-token probability of `im` is
`0.9999797135` (99.997971%), at rank 1. The repeated **unedited** greedy branch
generates `im sends the report to Yara.` and reaches a real word boundary after
Selim. Thus the conditional text is:

> Nora will take the sample to Ivo before Selim sends the report to Yara.

This is not scored or admitted as a task result: its first name piece was
supplied by the diagnostic. It also drops the required deadline `before dusk`.
Even this favorable conditional branch does not establish complete repair.

For the single canonical name token sequence, prefix probabilities are
`0.01028480945` without the entry patch and `0.01044697906` with it. Their delta
is almost the same as the first-piece probability delta because the conditional
suffix is already highly probable. The probability product is not evidence of
an observed natural Selim output; all ordinary continuations omit the name.

## Reading and Next Boundary

Within this one prefix and this operator, the sharpest remaining split is:

1. **Entry competition remains unresolved.** The patch moves the relative
   readout in the intended direction, but `sending` remains much more probable.
2. **Name completion is available conditionally.** There is no evidence here
   that the `im` continuation is the main obstacle once ` Sel` is present.
3. **Task preservation remains separate.** Reaching the actor-bearing branch
   can lose the deadline. Entity presence alone cannot certify success.

These findings do not prove a general mechanism or an applicable intervention.
The checkpoint, precision, prefix, source, dose, and binding are fixed; persistent
edited-KV trajectories, other tokenizations, other sites, and other tasks were
not tested. Deterministic FP16 controls do not establish higher-precision
fidelity. The diagnostic cap remains distinct from normal-budget permission.

The next comparison can keep this prefix and budget fixed and ask which
existing source/direction recipe most reduces the **actual competitor gap**,
while still tracking natural whole-name realization and deadline retention.
Neither teacher forcing the name nor raising alpha until it wins is a substitute
for that comparison. Conditional results stay out of controller memory,
candidate sources, and production certification.

## Artifacts and Reproduction

- [Manifest and code/checkpoint hashes](../results/llama_l1_completion_20260921/manifest.json).
- [Sealed binding and reference plan](../results/llama_l1_completion_20260921/completion_plan.json).
- [Raw row](../results/llama_l1_completion_20260921/rows.jsonl),
  [summary](../results/llama_l1_completion_20260921/summary.json), and
  [completion status](../results/llama_l1_completion_20260921/status.json).
- [Artifact, checkpoint, code, and state verification](../results/llama_l1_completion_20260921/verification.json).

```bash
python3 -m SpiralInterventionLab.examples.replay_prefix_response \
  --source-jsonl results/llama_l1_luna_20260921_r2/live/c1.jsonl \
  --source-manifest results/llama_l1_luna_20260921_r2/manifest.json \
  --worker-model-path "${WORKER_MODEL_PATH:?Set the matching local HF checkpoint}" \
  --controller-step 4 --recorded-dose 0.16 --horizon 16 \
  --completion-worker-step 9 \
  --reference-prefix-dir results/llama_l1_frozen_prefix_20260921_r2 \
  --output-dir results/llama_l1_completion_repeat
```

Output directories must be fresh. The optional completion mode lives in
`runtime/completion_probe.py` and the existing offline replay CLI, not in a new
controller decision path. The ordinary prefix-replay mode remains available.

Validation: `python3 -m pytest SpiralInterventionLab/tests -q` passed **356 tests
and 2 subtests**, with 16 warnings. `git diff --check` passed. Results refer to
the current working tree, not an already published release.
