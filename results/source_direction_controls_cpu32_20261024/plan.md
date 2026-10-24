# Fixed-permutation direction controls

Predeclared on 2026-10-24 before model execution, following merged PR #36.
One bounded Llama CPU32 diagnostic; no controller/provider calls, production
apply, model download, adaptive resampling, cap increase, or retries.

## Question and Fixed Conditions

At position 8 (` Nora will take the sample to Ivo`), does the observed common
displacement yield a different readout response from a fixed coordinate
permutation of the same displacement and its inverse, at comparable actual dose?

Reuse the original checkpoint/source receipt identities, seed 0, SOURCE IDs
11 / 1418 at positions 76 / 86, resid_pre L14, alpha 0.04, step cap 0.08,
norm clip 1.0 and TTL 1. All 28 layers, CPU float32, four threads. Refuse any
reference/prefix/source/site/cap mismatch. Model directory remains CLI-supplied.

First repeat the seven component conditions in their existing order. Then use
`permuted_common_positive` and `permuted_common_negative` in that order.
The permutation sorts coordinate indices by SHA-256 of
`sha256-index-sort:source-direction-control-v1:<width>:<index>` (index breaks
ties). This rule is fixed now, independent of model outputs and vocabulary.
There is no search over seeds or replacement for a degenerate permutation.

Use native-dtype common source C and target T, then construct sources
`T + permute(C - T)` and `T - permute(C - T)`. Source-addition rounding is
retained. Both pass the existing blend operator, normal validation and cap.
An ephemeral trace supplies the frozen vector only within diagnostic replay;
the trace, packet and trace alignment are restored even on failure.

Nine conditions cost nine diagnostic slots, plus alignment and observation:
11 of the existing 12 slots. Two no-edit and two edited one-token replays per
condition/site. Count preparation canaries and physical replays separately.

## Measurement and Limits

Capture actual post-hook displacements, all existing trajectory stages, raw
logits, fixed no-edit winner margins, raw softmax probabilities, repeat/null
floors, and one-token safety signals. Verify source identities and permutation
algebra from retained vectors. Report whether actual written norms match common
within 2%; an unmatched dose is not silently treated as matched.

The permutation preserves coordinates/norm before rounding, NOT the angle to
the current activation, learned axes, or a semantic representation. This is a
fixed coordinate control, not proof of an unrelated semantic direction or a
random null ensemble. Neither a weaker control nor a sign reversal establishes
SOURCE ownership, a circuit, task success, or production certification. One
boundary, one pair, and deterministic repeats cannot establish generality.

## Resource Bounds

Require 32 GiB free disk before loading, stop only our child below 16 GiB or
after 900 seconds. Preserve partial receipts on failure and do not implicitly
retry. Keep source snapshot, raw vectors, JSON, tests and hashes. Previous
receipts remain unchanged. Raw tensors stay local unless separately published.
