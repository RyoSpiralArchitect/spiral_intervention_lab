# Adjacent-boundary fixed-winner audit

Predeclared before model execution on 2026-10-24. This is one bounded offline
Llama diagnostic, not a controller episode, task-success trial, or production
promotion. A fresh static review found no actionable defect, but identified
partial-failure and multi-surface test gaps and single-boundary coverage limits.

## Question

Does the signed comma/while contrast observed at output position 8 persist at
the next natural boundary, and does either structural token approach the actual
full-vocabulary winner rather than merely outperforming the other token?

- Anchor: natural output position 8, ` Nora will take the sample to Ivo`.
- Measurement: exactly position 9, ` Nora will take the sample to Ivo before`.
- Refuse if the anchor token identity, one-token extension, recorded prefix,
  exact SOURCE positions, localization, offered sites, or normal caps disagree.
- No forced tokens, adaptive position search, fallback site, cap increase,
  negative-common/random controls, or implicit model retry.

## Fixed Conditions

Use the same local checkpoint hashes, task, seed 0, two SOURCE IDs 11 / 1418,
source positions 76 / 86, `resid_pre` L14, alpha 0.04, step size 0.08,
norm clip 1.0 and TTL 1 as the retained CPU32 position-8 run. The checkpoint
directory is supplied via CLI, not baked into library code. All 28 layers run
on CPU in float32, four threads. The normal 12-slot diagnostic budget remains.

Seven conditions in fixed order: the two source patches, self-copy, radial
shrink, common, positive half-difference, negative half-difference. Each has two
paired no-edit and two edited one-token replays per offered surface. Alignment,
observation, packet canaries and reconstruction forwards are counted separately.

The reference is the original complete single-prefix specificity receipt.
The existing strict same-prefix mode is unchanged. A separate opt-in mode
verifies position 8 before measuring position 9. Cross-position summaries must
be labelled **different-prefix comparisons**, not same-context replication.

## Readout Contract

Use retained raw logits without extra forwards. Anchor the winner to the first
no-edit full-vocabulary argmax, report repeat/condition stability and ties, and
never reanchor to the edited winner. Record whether baseline and edited winners
are inside the observed structural-token set. For each structural token and
the baseline winner, retain paired raw logit margins and full-vocabulary softmax
probability deltas. Raw probabilities are distinct from existing post-constraint
receipt probabilities. Favorable margins alone are not target lift.

Interpret unchanged greedy choice separately from relative contrast and raw
probability movement. Exact deterministic repeats are not independent samples.
No circuit, semantic ownership, generalization or production certificate is
inferred. Missing candidates or execution errors remain failures, not zero effects.

## Resource and Evidence Bounds

At least 32 GiB free disk before loading; stop only this experiment's child if
free space falls below 16 GiB or elapsed time exceeds 900 seconds. Keep partial
receipts and counts on failure. No provider calls, downloads, or interventions
in other processes. Preserve previous receipts untouched. Retain the source
snapshot, tests, vectors, summary, resource log and hashes for this attempt.
