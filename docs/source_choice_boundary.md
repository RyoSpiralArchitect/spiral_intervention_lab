# Source response versus actual next-token choice

## Scope and implementation

On 2026-10-24, one predeclared CPU/float32 diagnostic moved from the retained
position-8 boundary to exactly one natural token later. A fresh static review
found no actionable defect in the existing source-component experiment. Its
coverage gaps motivated partial-failure and multi-surface regression tests;
its single-boundary limitation motivated this one adjacent-boundary measurement.

`runtime/clause_choice.py` adds read-only analysis of saved raw logits. It fixes
the competitor to the first no-edit full-vocabulary argmax, records repeat
stability and ties, and reports whether that winner belongs to the structural
readout set. Each structural token's logit margin against that fixed winner is
kept separate from its probability change. The edited winner never redefines
the reference. No new forward, candidate selection, or permission originates
from the analyzer. The earlier component-summary schema is unchanged.

The trajectory CLI has an explicit `--next-natural-boundary` option. It verifies
the original reference's source-file hashes and exact position-8 token identity
before naturally stepping once. Position 9 must match the recorded prefix and
be exactly one token longer. Exact SOURCE bindings, offered site and caps must
still match the reference. Missing candidates or mismatches stop the experiment;
there is no fallback position, forced token, or site search. Default same-prefix
verification remains strict.

## Fixed experiment

- Worker: full 28-layer Llama-3.2-3B-Instruct, same checkpoint hashes, seed 0.
- Numerics: CPU, float32, four threads; model parameters verified explicitly.
- Source IDs/positions: comma 11 at 76, ` while` 1418 at 86, both SOURCE body.
- Surface: `s_resid_pre_l14_last`; alpha 0.04, step cap 0.08, norm cap 1.0, TTL 1.
- Conditions: two sources, self-copy, radial shrink, common, positive difference,
  negative difference. Each condition has two deterministic paired replays.
- The unchanged 12-slot budget used nine slots. No provider call, controller
  choice, production apply, model retry, or full-continuation trial occurred.

The anchor hash at position 8 is
`404675115f71bd07b66be7bb325337ce9b6c743c94f543e7234a2cb6900bf870`.
The new report records both this verified anchor and the position-9 token hash.
The two experiments are **different-prefix observations**, not same-context
replication or independent samples.

| Boundary | Natural prefix ending | Raw baseline winner | Winner in structural set | Comma rank | Comma-to-winner logit gap |
| --- | --- | --- | --- | ---: | ---: |
| 8 | `... to Ivo` | ` before` (1603) | Yes | 2 | 1.114995956 |
| 9 | `... to Ivo before` | ` sending` (11889) | No | 75 | 8.263474464 |

At position 9, `while` is rank 786, `before` rank 114, and period rank 788.
The structural readout therefore does not contain the actual winner here.

## Observations

The table below uses **raw, pre-constraint CPU-float64 softmax** probabilities,
not the earlier receipts' post-constraint probabilities. Their small numerical
differences are preserved in the comparison rather than silently conflated.
Contrast means delta of `(comma logit - while logit)`. A positive comma margin
delta means moving toward the fixed no-edit winner, not overtaking it.

| Position-9 condition | Comma probability delta | While probability delta | Contrast delta | Comma margin delta vs sending |
| --- | ---: | ---: | ---: | ---: |
| Comma source | -4.138659e-7 | -8.281595e-9 | -0.001545906 | -0.000862122 |
| While source | -1.139840e-6 | -2.505944e-8 | -0.003831387 | -0.009928703 |
| Self-copy | 0 | 0 | 0 | 0 |
| Radial shrink | -4.004419e-7 | -2.177061e-8 | +0.001214981 | -0.000865936 |
| Common | -8.770136e-7 | -1.889305e-8 | -0.003019810 | -0.006057739 |
| Positive difference | +7.619663e-7 | +1.787558e-8 | +0.002312183 | +0.009622574 |
| Negative difference | -7.333113e-7 | -1.667991e-8 | -0.002348900 | -0.009336472 |

The signed contrast direction persists, but its magnitude is smaller than at
position 8: positive +0.021520615 becomes +0.002312183; negative -0.021354675
becomes -0.002348900. This is not uniform transfer of all effects:

- Common changes comma probability by +0.001517545 at position 8, but
  -8.770136e-7 at position 9. The earlier common-component lift does not persist.
- Positive difference lowers comma probability at position 8 (-0.000152790)
  while suppressing `while` more strongly. At position 9 both structural tokens
  increase slightly instead. The same contrast sign has a different composition.
- At position 8, positive difference improves comma versus while but worsens
  comma versus the actual winner `before` by -0.000331879. At position 9,
  positive difference improves comma versus `sending` by +0.009622574, but
  leaves a gap of 8.253851891. Improvement is not an actual choice reversal.
- Position-9 `sending` probability moves from 0.506642705 to 0.504718657 under
  positive difference, and to 0.508523507 under negative difference. It remains
  the raw greedy winner in every condition and repeat. Neither sign establishes
  target-owned generation or semantic success.

## Execution checks

All seven conditions completed. Each edited replay called the intended hook
once, state/caps were restored, and expected-versus-captured post-hook residual
norms were exactly zero. Captured baseline, preparation, delta-repeat and
self-copy floors were also zero. These verify execution, not semantic fidelity.

All six non-null writes saturated the unchanged cap; norms were 0.0799999800
through 0.0800000046. The positive and negative written differences sum to norm
6.41e-8, not exactly zero. Separately clipped nonlinear responses still do not
establish additive mediation or a semantic subspace.

Accounting: **30 explicit replays + 22 packet canaries = 52 diagnostic forwards**;
nine reconstruction forwards make **61 actual forwards**, all completed.
The seven condition captures contain 28 replays and 21 canaries; the remaining
alignment/observation work is separately accounted. No one-token repetition
increase or collapse-sharpening signal was observed. Full continuation safety,
task score, and language-quality improvement were not evaluated.

The single supervised run completed in 591.75 seconds. Free disk was 59.21 GiB
at start, at least 44.21 GiB in sampled checks, and 52.11 GiB at finish. Startup
32 GiB, stop 16 GiB and 900-second guards were unchanged; no implicit retry.

Tests: **697 passed, 9 subtests passed, 15 dependency warnings**. New tests cover
fixed-winner versus edited-winner identity, improved margins with falling target
probability, unstable baselines, readout rebinding, adjacent-boundary preflight,
multi-surface replay offsets, and saved partial failure with restored wrappers
and accurate charged/physical work counts. `git diff --check` passes.

## Interpretation and next question

At these two boundaries, relative comma/while control survives but absolute
probability movement and distance to the actual winner are prefix-dependent.
The structural-token set can also miss the dominant lexical competitor. It is
therefore insufficient to summarize this recipe as either globally useful or
globally dead. This does not identify an ownership circuit, establish broad
generalization, or authorize a stronger intervention.

A next bounded test would return to the fixed position-8 boundary and compare
the common component with a predeclared unrelated, dose-matched direction.
That would address source specificity versus generic sensitivity without also
changing time, site, cap, or production authority. It has **not** been run here.

## Artifacts

- [Pre-measurement plan](../results/source_clause_choice_boundary9_cpu32_20261024/plan.md)
- [Live receipt](../results/source_clause_choice_boundary9_cpu32_20261024/cpu32/measurement/trajectory.json)
- [Resource supervision](../results/source_clause_choice_boundary9_cpu32_20261024/cpu32/supervision.json)
- [Hash-checked cross-boundary comparison](../results/source_clause_choice_boundary9_cpu32_20261024/comparison.json)
- [Read-only comparison procedure](../results/source_clause_choice_boundary9_cpu32_20261024/build_comparison.py)
- [Tests](../results/source_clause_choice_boundary9_cpu32_20261024/tests.txt)
- [Earlier component results](source_response_components.md)

The comparison verifies vector hashes, compact records and the unchanged
component summaries. Checkpoint, numerics, source bindings and caps match across
positions. Code differences are limited to the opt-in harness/readout audit and
their tests, with exact hashes recorded. Reanalysis adds zero model forwards
and does not modify the earlier artifacts.
