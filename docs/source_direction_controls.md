# Fixed-coordinate direction controls

## Question

PR #36 separates source-dependent displacement, relative readout preference,
probability lift and unchanged next-token choice. Its position-8 common direction
increases comma probability under normal caps. Is that response specific to the
observed direction, or can a coordinate-permuted control of the same size do it?

This follow-up adds no controller request, public expression primitive, gate,
candidate certificate or production permission. It is an opt-in offline
extension of `replay_source_clause_trajectory`:

```text
--position 8 --source-token-ids 11,1418
--decompose-components --permuted-direction-controls
--device cpu --dtype float32 --torch-threads 4
```

Checkpoint, input files, reference receipt and output directory are CLI inputs.
The default replay remains strict. Permutation controls cannot be combined with
the adjacent-boundary switch. Checkpoint and source-input identities must match
the reference before model loading.

## Contract

Retain the seven existing conditions, then append a fixed permutation of the
rounded common source-minus-target displacement and its negative. The index
order is SHA-256-sorted from a versioned string, dimension and coordinate index;
it uses no process RNG, readout selection, resampling or oracle output.

Construct sources as T plus/minus that displacement, then use the unchanged
`activation_patch/blend` operator at the existing offered site/caps. Frozen
sources use a temporary diagnostic trace which is removed on success or error.
Packet and trace alignment are restored. All normal policy validation remains.

Nine conditions use nine diagnostic slots, plus one alignment and one observation
slot: 11/12. Preparation canaries, explicit replays and reconstruction forwards
are separately counted. Raw post-hook vectors validate actual dose and source
rounding; merely using the same cap is not proof of a matched dose.

The controls preserve the displacement's coordinate multiset and norm before
rounding. They do not preserve its angle to the target activation or learned
feature axes. They are **not** proven semantically unrelated directions, a random
null ensemble or independent samples. A smaller control response cannot by
itself prove source-specific semantic content, causal mediation or ownership.
Readout margins are still separate from probability lift and greedy choice.

## Evidence

The predeclared method and resource limits are in
[`plan.md`](../results/source_direction_controls_cpu32_20261024/plan.md).
The bounded run uses a 32-GiB startup floor, a 16-GiB stopping floor and a
900-second wall-clock limit, without implicit retries or external API calls.
Raw vectors are local artifacts, not implicitly published by this note.

## Observed Result

The CPU32 attempt completed in **662.68 seconds**, with **11/12 diagnostic slots**
and no retries. All nine conditions were measured. Each condition captured four
explicit replays and three packet canaries: 36 + 27 = 63 captured forwards.
Alignment and observation add three forwards, for **66 diagnostic forwards**;
eight natural reconstruction forwards make **74 total**, all completed.
Free disk started at 56.69 GiB, reached a sampled minimum of 42.68 GiB, and
ended at 50.55 GiB. Neither resource stop fired.

Every edit called its hook once per replay. State/caps were restored, both
temporary traces were removed, and expected-versus-captured post-hook residual
norms were zero. All baseline/preparation/repeat/self-copy floors were zero.
Actual nonzero write norms were approximately 0.08; both new controls matched
common within the predeclared 2% tolerance. Deterministic repeats are not new
samples. One-token repeat deltas were zero and collapse-sharpening flags false;
this does not certify a safe full continuation.

The original and repeated common condition have identical checkpoint, prefix,
token, source/target tensor, and captured-vector hashes. The new controls are
not being compared to a silently changed common seed.

Raw, pre-constraint readout deltas (one of two identical repeats):

| Condition | Comma probability | While probability | Comma minus while logit contrast | Comma margin vs fixed ` before` |
| --- | ---: | ---: | ---: | ---: |
| Common | +0.001517545 | +0.000012840 | +0.006713867 | +0.010457993 |
| Permuted common | +0.000216077 | +0.000095348 | -0.010177612 | +0.000213623 |
| Negative permuted common | -0.000205872 | -0.000094798 | +0.010295868 | -0.000158310 |
| Positive source difference | -0.000152790 | -0.000184555 | +0.021520615 | -0.000331879 |
| Negative source difference | +0.000173784 | +0.000188282 | -0.021354675 | +0.000482559 |

All nine conditions retain greedy **` before` (1603)**. The common comma lift is
about seven times this positive permutation control's lift, while its margin
movement toward the actual winner is much larger than the control's. This is
direction-dependent local response, not threshold crossing or task success.

The counterpoint matters: the permuted pair also reverses comma/while contrast.
Thus **contrast sign reversal alone is not a signature of source ownership**.
The control displacement is nearly orthogonal to common (cosine -0.01190) and
to the target state (cosine +0.00803); changing those angles is a confound for
semantic specificity, not a semantic-null guarantee. A single fixed permutation
cannot establish a general null distribution or identify a circuit.

## Interpretation and Next Question

The result supports separating three claims: a direction moves readout, it
preferentially raises one observed token, and it changes the actual token choice.
Only the first two have limited within-boundary evidence here. Task success,
ownership, semantic safety and production certification remain unestablished.

A next bounded question is whether common's advantage remains after separating
its radial component relative to the current activation, alongside a negative
common control. That is not yet executed or authorized as an expanded sweep;
this run did not raise caps, rotate sites or search for a favorable permutation.

Retained [summary](../results/source_direction_controls_cpu32_20261024/summary.json),
[full receipt](../results/source_direction_controls_cpu32_20261024/cpu32/measurement/trajectory.json),
[supervision](../results/source_direction_controls_cpu32_20261024/cpu32/supervision.json)
and [verification script](../results/source_direction_controls_cpu32_20261024/build_summary.py)
record the result. Raw-vector hashes, compact receipts and both numerical
summaries were recomputed in a separate offline pass from this run's retained vectors,
without loading a model or making more forwards. This is implementation-level
reanalysis, not independent experimental replication. Raw tensors remain local.

Validation: **713 passed, 9 subtests passed, 15 dependency warnings**. Tests cover
fixed permutation algebra, no RNG changes, degenerate-input refusal, normal
policy use, budget preflight, trace cleanup on failure, partial receipts and CLI
scope. `git diff --check` passes. No production permission was added.
