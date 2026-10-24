# Common/difference source-response experiment

## Question and contract

The [trajectory audit](source_response_trajectory.md) found that comma-source and
while-source patches were distinct, yet both increased comma probability. This
follow-up intervenes on their average and signed half-difference at the same
boundary. It tests shared readout response versus relative token preference,
without calling either direction a semantic subspace.

For the same current activation `T`:

```text
a = S_comma - T; b = S_while - T
c = (a + b) / 2; d = (a - b) / 2
a = c + d; b = c - d
```

The blend sources are `(S_comma + S_while)/2`, `T + d` and `T - d`.
Adding `T` back for the difference is essential: `activation_patch` already
subtracts the current activation. Existing mean/sub/add/scale expressions suffice;
there is no new operator, controller card or production permission.

These raw displacement components are not normalized or guaranteed orthogonal.
Here their cosine is 0.01549 on MPS/float16 and 0.01556 on CPU/float32. Each
direction is clipped separately under the same alpha/cap. This is a matched-dose
direction test, not additive attribution to the
original patches. The pair was selected by the earlier screen, not held out.

The [pre-measurement plan](../results/source_clause_components_20261023/plan.md)
fixes seven conditions: original comma, original while, self-copy, radial shrink,
common, positive difference, negative difference. Each has two no-edit baselines
and two edited one-token replays, under these conditions:

- Full 28-layer Llama-3.2-3B-Instruct, seed 0, checkpoint hashes verified.
- Natural eight-token prefix: ` Nora will take the sample to Ivo`.
- Exact 108-token input identity matches the previous aligned experiment.
- SOURCE comma: ID 11, position 76; SOURCE ` while`: ID 1418, position 86.
- Offered `resid_pre L14`, alpha 0.04, step cap 0.08, norm cap 1.0, TTL 1.
- No provider calls, forced tokens, policy overrides or production applies.
- Semantic safety and clause correctness remain unassessed.

`runtime/clause_components.py` reuses the offline expression-control executor in
`runtime/clause_specificity.py`, with normal policy validation and restoration.
The CLI is the existing trajectory runner with `--decompose-components`:

```sh
python3 -m SpiralInterventionLab.examples.replay_source_clause_trajectory \
  --source-jsonl <completed-episode.jsonl> \
  --source-manifest <checkpoint-and-profile-manifest.json> \
  --reference-receipt <aligned-source-specificity.json> \
  --worker-model-path <local-model-directory> \
  --position 8 --source-token-ids 11,1418 --decompose-components \
  --device cpu --dtype float32 --torch-threads 4 \
  --output-dir <new-directory>
```

IDs are tokenizer-specific; model paths remain CLI-supplied. Existing output
directories, insufficient budget, degenerate components, stale bindings and
mismatched prefixes fail closed. No measured outcome selects a new dose or site.

## Measurement qualification: 2026-10-23

MPS/float16 completed all seven conditions. Every edited replay called its hook
once; state and cap checks passed. It used **nine of 12 diagnostic slots**, **30
explicit diagnostic replays**, **22 packet canaries**, and **52 actual diagnostic
forwards**. Eight natural prefix-reconstruction forwards bring the total to 60.

The final analyzer initially rejected the measurements: captured radial source
used `zeros_like(T)`, but the compiled expression used `0.0 * T`. Both are zero,
but 1,513 coordinates differ in the sign of zero, so byte hashes differ. This is
a capture/identity error, not a failed model replay.

The original error receipt and all vector artifacts remain unchanged. A separate
[qualified reanalysis](../results/source_clause_components_20261023/mps16/reanalysis.json)
accepts only this radial zero-source case: stored values must all be zero, and
reconstructing `0.0 * T` must match the compiled-source hash exactly. All other
source hashes and the ordinary post-hook checks must pass without qualification.
No model forward was repeated. The general identity checker was not weakened.

Future capture now uses `0.0 * T`. Regression tests cover preservation of signed
zero and rejection of unexplained hash mismatches. Replay accounting now precedes
analysis so an analyzer failure cannot hide a completed measurement's count.

After the narrow correction, all captured patch differences match dtype-rounded
blend arithmetic exactly. Baseline drift, repeat differences, preparation versus
baseline, and self-copy floors are zero at every captured stage. Repeatability is
not numerical accuracy, especially given the loader's MPS warning.

On 2026-10-23 CPU/float32 replication was **not started**: 22.18 GiB free was below its
predeclared 32 GiB startup floor. The blocked receipt is retained. No guard was
relaxed or other process stopped. The previous CPU trajectory study cannot stand
in for this new component experiment. The new replication below was run only
after storage headroom was restored.

## Observed readout: MPS/float16

Probability deltas use post-constraint CPU-float64 arithmetic on the float16
model outputs. The final column is the change in raw
`logit(comma) - logit(while)`, not a winner label. Both repeats agree exactly.

| Condition | Comma probability delta | While probability delta | Logit-contrast delta |
| --- | ---: | ---: | ---: |
| Comma source | +0.003940657 | +0.000048021 | +0.015625 |
| While source | +0.004022856 | +0.000182363 | 0 |
| Self-copy | 0 | 0 | 0 |
| Radial shrink | -0.000465106 | -0.000148603 | +0.015625 |
| Common component | +0.004015277 | +0.000182019 | 0 |
| Positive difference, comma minus while | +0.002230106 | -0.000155726 | +0.03125 |
| Negative difference, while minus comma | +0.001516135 | +0.000332674 | -0.03125 |

Original-source/control responses reproduce the earlier MPS receipt. All six
non-null conditions saturate the cap, with actual written norms 0.079908 to
0.080141. Self-copy is zero. Source-rounding errors are retained. The two written
difference vectors sum to norm 0.0004415, not exactly zero, due to rounding.

All seven conditions still greedily emit ` before`. Comma stays rank 2 and while
rank 7. No one-step repetition increase or collapse-sharpening signal was observed;
full-continuation safety and task performance were not evaluated.

## CPU/float32 replication: 2026-10-24

The [new pre-measurement plan](../results/source_clause_components_cpu32_20261024/plan.md)
retains all seven conditions, checkpoint, exact token-prefix identity, SOURCE
positions, site and caps. Two further preflights at approximately 19 GiB remained
blocked. After storage was freed, the supervisor started at 64.12 GiB free,
completed in 573.55 seconds and finished at 55.03 GiB free. The 32 GiB startup,
16 GiB stop and 900-second time guards were unchanged. There was no retry after
model execution began.

All 28 layers were verified on CPU in float32. The natural prefix and reference
bindings matched. All seven conditions completed without a capture exception:
**nine of 12 slots, 30 explicit replays, 22 packet canaries, 52 diagnostic
forwards and 60 total forwards**. Every edited replay called its hook once.
No production apply or provider call occurred.

| Condition | Comma probability delta | While probability delta | Logit-contrast delta |
| --- | ---: | ---: | ---: |
| Comma source | +0.001265603 | -0.000072357 | +0.015590668 |
| While source | +0.001429283 | +0.000096798 | -0.003755569 |
| Self-copy | 0 | 0 | 0 |
| Radial shrink | -0.000492017 | -0.000165610 | +0.017347336 |
| Common component | +0.001517558 | +0.000012840 | +0.006713867 |
| Positive difference, comma minus while | -0.000152789 | -0.000184556 | +0.021520615 |
| Negative difference, while minus comma | +0.000173783 | +0.000188283 | -0.021354675 |

Both repeats agree exactly. Captured baseline drift, preparation/baseline,
delta-repeat and self-copy floors are all zero. Expected versus actual post-hook
blend residual norms are zero. All six non-null conditions are cap-saturated;
written norms range from 0.0799999893 to 0.0800000084. The two signed written
differences sum to norm 7.47e-8, versus 4.41e-4 on MPS. Component-source rounding
errors are 2.77e-7 to 3.76e-7. These checks verify execution, not semantic fidelity.

All conditions still select raw greedy ` before` (ID 1603). The observed ranks
remain before=1, comma=2, while=7, period=33. No one-token repetition increase or
collapse-sharpening signal was observed. No full continuation was evaluated.

The [cross-numerics comparison](../results/source_clause_components_cpu32_20261024/comparison.json)
recomputes both summaries from hash-verified saved vectors without new model
forwards. MPS retains its original error receipt and exact signed-zero
qualification; CPU needs no qualification. Code hashes are identical except for
the predeclared signed-zero capture fix, replay-count recording order, and their
regression tests. Exact old/new hashes are recorded; no general mismatch bypass
is introduced. Backend and dtype still change together, so this does not isolate
a dtype-only or device-only effect.

## Interpretation

Within this MPS/float16 boundary, common retains a comma increase close to both
original sources. Signed difference reverses relative comma/while logit
preference. This is evidence of separable local readout effects, not proof that
common uniquely explains comma lift or isolates a semantic ownership circuit.

Negative difference still raises comma probability, and comma still has the
largest absolute probability increase among the observed structural tokens.
Reversed **relative logit contrast** is not reversed lift ownership or a different
greedy token. The radial control also shows that a favorable contrast can come
from competitor suppression rather than target lift.

The CPU replication preserves the two central observations at this one boundary:
common increases comma probability, and signed difference reverses relative
comma/while logit contrast. The latter is therefore not observed only on the
coarse MPS/float16 grid. Effect sizes and individual probability signs do not
replicate uniformly.

Most importantly, CPU positive difference **lowers** comma probability by
0.000152789, whereas MPS increased it by 0.002230106. Its positive contrast is
mainly while-logit suppression: comma logit changes by -0.000246048 and while by
-0.021766663. CPU negative difference raises while probability slightly more than
comma, but `before` has the largest increase among the observed structural tokens
(+0.000260217). Neither sign establishes target-owned generation.

The [2026-10-24 adjacent-boundary follow-up](source_choice_boundary.md) now tests
the next natural token in CPU32. Signed contrast reversal persists at smaller
magnitude, but common comma lift does not. A fixed-winner audit identifies
` sending` outside the structural readout set; all seven conditions still keep
that winner. This is a different-prefix observation, not same-context replication.

This separates common comma lift from relative preference control more clearly,
but it does not isolate a semantic carrier or an ownership circuit. At this
checkpoint, the next measurement question was which response survives a predeclared natural
boundary while tracking individual-token lift and the actual greedy competitor,
not merely a favorable pairwise contrast. It is not permission to enlarge caps,
add controller authority, or interpret separately clipped nonlinear responses as
additive mediation fractions.

Artifacts:

- [Original error receipt](../results/source_clause_components_20261023/mps16/measurement/trajectory.json)
- [Qualified retained-vector reanalysis](../results/source_clause_components_20261023/mps16/reanalysis.json)
- [Readout summary and blocked CPU status](../results/source_clause_components_20261023/comparison.json)
- [Reanalysis procedure](../results/source_clause_components_20261023/build_comparison.py)
- [MPS supervision](../results/source_clause_components_20261023/mps16/supervision.json)
- [CPU startup guard](../results/source_clause_components_20261023/cpu32/supervision.json)
- [Validation after the capture fix](../results/source_clause_components_20261023/tests_after_signed_zero_fix.txt)
- [Completed CPU measurement and vector hashes](../results/source_clause_components_cpu32_20261024/cpu32/measurement/trajectory.json)
- [CPU resource supervision](../results/source_clause_components_cpu32_20261024/cpu32/supervision.json)
- [Read-only cross-numerics comparison](../results/source_clause_components_cpu32_20261024/comparison.json)
- [Comparison procedure](../results/source_clause_components_cpu32_20261024/build_comparison.py)
- [CPU replication validation](../results/source_clause_components_cpu32_20261024/tests.txt)

Validation repeated on 2026-10-24: **686 passed, 9 subtests passed, 15 dependency
warnings**. The targeted component/trajectory tests also passed (33 tests).
`git diff --check` passes. The new tests cover component algebra/sign swaps,
actual dose/null reporting, zero-component rejection, seven-condition budget and
forward accounting, stale/policy-invalid bindings, restoration and signed zeros.
