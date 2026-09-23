# Current-Prefix Trial Handoff and Harness Audit

## Scope

This change repairs the route from current-prefix readout evidence to the
existing bounded production-trial review. It does not optimize for a nonzero
edit count, lower the target-lift floor, or promote a diagnostic cap to a normal
runtime cap. A zero-edit episode can still be the correct result.

The preceding Llama run measured small bound-piece probability gains at some
positions but did not pass those observations to same-position trial review.
That is not retrospective certification: ownership, safety, the normal cap,
and the exact source still have to be checked.

## Handoff Contract

1. A restored canonical-bound measurement can expose a `candidate_trial_handoff`
   when bound probability gain is at least 0.0001 or top-20 entry improves,
   and its logit response exceeds repeat/no-edit variation. Rank/gap-only
   evidence cannot open this handoff. Old safety/ownership labels are not reused.
2. The controller may request the explicit evidence-ID review or decline.
   Confirmation resolves the same source under normal caps and compares baseline,
   edited, null, and repeated edited trajectories: four physical replays, two-token
   horizon, one-token edit TTL. Both edited runs must pass current ownership,
   bound-target response and nonregression checks before the existing gates run.
3. A passing review exposes one ephemeral `candidate_trial_offer`. Its authorization
   binds current state, source tensor and target/op/budget. The controller may apply
   exactly that edit or decline. Changed, stale, consumed or absent authorizations
   are rejected. After one generated token, the hook is removed before the next
   controller/diagnostic round. General production permission remains false.

`--candidate-handoff-rounds {0,1,2}` controls additional same-prefix controller
calls; the example CLI defaults to two, while the low-level loop defaults to zero.
The handoff is phase-independent, not limited to `readout_escape`. Calls occur
after backend-verified handoff/offer availability, or an explicit `hold_prefix`
with a usable diagnostic result. Both share this same round pool; the diagnostic
budget is unchanged. See [explicit prefix investigation](explicit_prefix_investigation.md)
for normal-cap variants and objective/frontier separation.

## Audit Corrections

- Diagnostic budget consumption is monotonic and independent of the retained
  result window. Truncating history no longer restores calls, including in hints.
- Multi-token `_simulate_decode` used to retain a TTL=1 hook for the whole horizon.
  It now expires temporary edits at their declared TTL and preserves telemetry.
  Old multi-token replay evidence is not silently relabeled as equivalent.
- Materialization no longer substitutes another activation site on the same layer
  when the requested site is missing or invalid.
- Matched-probe executable recipe IDs/names include actual dose; the original ID
  is preserved as seed lineage rather than falsely labeling a 0.16 edit as 0.08.
- Evidence-ID reviews are exclusive: stale `next_action` or memory text cannot
  append a second diagnostic to that request.
- Missing, malformed, nonfinite or exhausted trial budgets fail closed.
- Legacy summary-only review passes no longer advertise trial permission without
  exact current-prefix physical confirmation. Runtime validates the offer again.
- Parsed controller commands and compact packets preserve the authorization and
  complete concrete edit. Rejected trial commands receive guardrail and selection
  logs; review roles retain the actual requested diagnostic name.
- A masked target or fewer than 20 finite logits is no longer conflated with
  NaN/positive-infinity readouts. Failed prefix measurements retain mask/control
  counts and restoration facts; they are not assigned a zero-lift/dead label.

The positive scripted tests validate reachability of the handoff and one-token
execution, not worker-model improvement. Live model-local B0/C1 comparisons remain
the separate test for efficacy; GPT-2 and Llama scores are not pooled.

## GPT-2 Regression Observation (2026-09-22)

The full 12-layer FP32 GPT-2 / Luna pair completed on MPS in 204.7 seconds.
Fresh B0 matched the preceding recorded baseline exactly. Both B0 and C1 emitted
` In the case of a rewrite, the budget draft.` with score 0.5875, 11 steps and
`task_done=false`. There were 11 parsed controller calls, 10 diagnostic charges,
zero rollout edits and zero same-prefix handoff rounds. Four initial handoff
reports all failed the bound-target-lift requirement; two also had alternate
bindings and two failed the response-versus-variation check. No forced apply or
newly permissive cap was used.

One current-prefix remeasurement attempted four physical replays but was
incomplete at ` In the case of a rewrite, the budget draft`, targeting ` Mir`.
The old error named nonfinite bound/top-20 readout. The same word-budget guard
was subsequently reproduced in a regression test: this prefix already has nine
words, so the target is masked out and top-20 readout is not defined in the usual
vocabulary-wide sense. This error is not evidence of an ineffective operator or
a NaN model state. The post-run logging-only refinement now distinguishes masks
from numerical failures; it has unit/full-suite coverage, not a second live pair.

The model-library MPS numerical-correctness warning was preserved. This is one
matched-device regression observation, not a precision-validation study.

Evidence: [audit](../results/iteration_handoff_20260922/gpt2/audit.json),
[C1 JSONL](../results/iteration_handoff_20260922/gpt2/live/c1.jsonl),
[manifest](../results/iteration_handoff_20260922/gpt2/manifest.json), and
[qualitative debrief](../results/iteration_handoff_20260922/gpt2/live/post_run_debrief.md).
The run's source snapshot and artifacts are sealed independently of the later
error-label refinement.

## Llama Handoff Observation (2026-09-22)

The full 28-layer FP16 Llama / Luna pair completed in 456.3 seconds on MPS.
Fresh B0 matched the preceding baseline. B0 and C1 again produced
` Nora will take the sample to Ivo before sending the report to Yara before dusk.`
at score 0.938889, 18 steps, `task_done=false`. All 18 controller responses parsed;
all selected noop, with 12 diagnostic charges, three current-prefix measurements
(12 physical replays), zero confirmation offers and zero rollout edits.

The same frozen ` Sel` candidate at resid_pre L27, alpha 0.04, step_size 0.16
passed every readout handoff check after ` Nora will take the sample to`:
bound probability delta +0.0002275676, logit delta +0.015625, rank 3 -> 3,
top-20 entry delta 0, repeat/null controls 0. Its next token stayed ` I`.
The handoff then correctly blocked with `normal_trial_edit_differs_from_diagnostic`:
the normal surface cap is 0.12, not the measured diagnostic cap 0.16.
This is not a safety/ownership certification, nor a demonstration of task uplift.

The following two measurements failed the readout floor (one also failed the
positive raw-logit condition). Four anchor handoffs also failed the floor. Thus
seven recorded handoffs yielded one readout-qualified row, no normal-budget
confirmation and no same-prefix follow-up. The older Llama trace has no handoff
instrumentation; missing fields are not counted as failed gates.

Evidence: [audit](../results/iteration_handoff_20260922/llama/audit.json),
[C1 JSONL](../results/iteration_handoff_20260922/llama/live/c1.jsonl),
[manifest](../results/iteration_handoff_20260922/llama/manifest.json), and
[read-only comparison](../results/iteration_handoff_20260922/reachability_comparison.json).
This run predates the additional ordinary-edit TTL and debrief-projection fixes
in [the reachability audit](intervention_reachability_audit.md). Its sealed
source snapshot is retained; the later fixes are not credited to this live run.
