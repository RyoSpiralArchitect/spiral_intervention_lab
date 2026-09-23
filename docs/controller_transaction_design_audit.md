# Controller Transaction Design Audit

Date: 2026-09-23. Scope: historical design audit plus a synthetic control-plane repair;
not a new intervention experiment.

## Main Finding

The sealed traces exposed a scheduling problem: a controller-chosen normal-cap
repair could outlast the callbacks available at one prefix. The current code
now supports a bounded measure-and-conditionally-confirm transaction and
dispatches tool/observer requests in every controller round. This repairs
reachability, not operator strength. No safe, effective operator has been
established in those sealed live runs.

Do not optimize for a nonzero edit count. Distinguish candidate discovery,
current normal-budget measurement, confirmation, a visible trial offer,
controller acceptance, installed hook, changed decoding and task improvement.
Zero edits before any offer is not evidence that the controller rejected a
permitted intervention.

## Evidence Boundary

The pre-repair sealed traces are
[GPT-2 C1](../results/iteration_prefix_control_20260922/gpt2/live/c1.jsonl) and
[Llama C1](../results/iteration_prefix_control_20260922/llama/live/c1.jsonl).
The [reachability comparison](../results/iteration_prefix_control_20260922/reachability_comparison.json)
records their hashes and admission stages. Their source snapshots predate the
two final metadata/log-counter hardening fixes described in
[Explicit Prefix Investigation](explicit_prefix_investigation.md), and also
predate the transaction/dispatcher repair below. Do not reinterpret them as
post-fix live outcomes.

| Recorded stage | GPT-2 | Llama |
| --- | --- | --- |
| C1 score, identical to model-local B0 | 0.5875 | 0.938889 |
| Controller commands / generated tokens | 14 / 11 | 21 / 18 |
| Accepted prefix holds | 3 | 3 |
| Evidence-ID physical confirmation results | 0 | 0 |
| Visible exact trial offers | 0 | 0 |
| Compiled rollout edits | 0 | 0 |

Both used twelve diagnostic slots. GPT-2 exposed no frozen-candidate actions;
Llama exposed them and executed three measurements. They do not share an
identical failure stage merely because both have zero edits. These are two
model-local observations, not a multi-seed result or a controller comparison.

## Verified Structural Limits

### 1. Old Cap Repair Did Not Fit the Prefix Transaction

[`run_episode`](../SpiralInterventionLab/runtime/loop.py) permits at most two
candidate handoff/explicit-hold follow-ups per prefix. Outside the separate
readout-escape rethink mechanism, that gives three controller decisions:

| Controller call at one prefix | Direct normal-cap route | Diagnostic-cap repair route |
| --- | --- | --- |
| Initial | Measure normal-cap candidate | Measure diagnostic-cap candidate |
| Follow-up 1 | Confirm exact evidence | Measure new normal-cap candidate |
| Follow-up 2 | Accept or decline trial offer | Confirm exact evidence |
| Follow-up 3 | Not needed | Accept or decline: unavailable |

Before this repair, the short route was reachable but cap repair could lose the
prefix even if measurement and confirmation were positive. Replacing the
controller with a scripted, cooperative one did not fix the old call-count
mismatch.

The [scheduling audit tests](../SpiralInterventionLab/tests/test_controller_scheduling_audit.py)
exercise the real loop with a toy worker and deliberately successful toy
confirmation. The repair fixture initially failed as expected. It now passes:
the controller selects a single normal-cap investigation that measures the
new child, conditionally confirms it, and leaves the final callback for a
separate apply/decline decision. This isolates scheduling, not physical
certification or model efficacy.

Live Llama adds an important nuance: at JSONL line 85, `sample to`, the
controller requested the 0.16 measurement with an explicit hold. At line 95 it
chose a legacy promotion review rather than the offered new 0.12 measurement.
Even the correct measurement choice at that point would still need a further
confirmation and a final decision. One remaining follow-up would not cover
both. The live run did not actually obtain a positive 0.12 confirmation there.

### 2. Old Request Execution Depended on Controller Round

The old outer loop dispatched observer requests, tools and diagnostics; its
follow-up loop dispatched only diagnostics. A valid `tokenize_terms` request or
observer check in round one was logged but not executed.

The same request payloads now execute in both rounds through one dispatcher.
An explicit `hold_prefix` can also deliver a fresh usable tool or observer
result before another generated token, within the same two-round pool.
Blocked or missing results do not hold. The dispatch defect was not established
as the cause of either sealed live run's zero edits.

### 3. Reviews and Executable Evidence Still Share a Vocabulary

Llama JSONL lines 65, 77 and 98 contain three charged legacy reviews without
`evidence_id`. All return top-level `status=ok`; their next-evidence sequence
goes from trial review back to shadow review and then promotion review. The
first also reports `production_shadow_candidate_missing`. No recorded
evidence-ID confirmation or trial offer follows.

Here `ok` means that a report was produced, not that the next admission state
was reached. Physical replay counts are absent, not proven zero. The compact
prompt also exposes both legacy review names and the exact evidence-ID path.
The names alone do not tell the controller which operation can produce a
fresh offer at this prefix, or how much remaining work it entails.

This is an affordance problem, not proof of the controller's internal motive.
Repeated `production_apply_allowed=false` explanations are not proof that it
would refuse a valid scoped trial: no such offer was presented in these runs.

### 4. Token Time Is Still Partly Owned by the Harness

The first `worker.step()` occurs before the first controller call. The sealed
first prefixes are ` In` and ` Nora`, not the empty answer boundary. No later
hold can recover that first-token intervention opportunity.

Explicit `generation_action=advance` still means no additional explicit hold,
not cancellation of automatic handoff/rethink. `hold_prefix` now accepts a
fresh usable diagnostic, tool, or observer result. This remains a bounded
interface, not fully orthogonal generation control. A pre-first-token call or
changed advance semantics would be a new experimental condition and must be
named and compared separately.

## Measurement Versus Task Semantics

At `sample to`, Llama's 0.16 ` Sel` patch changed probability from approximately
0.0146293 to 0.0148568, with rank 3 unchanged and no top-20 entry. It passed the
readout checks but not normal-edit equivalence. The 0.12 child was only measured
later at `sample to Ivo`, where its probability delta was about +6.624885e-7.
These are not a same-prefix dose curve. The smaller-dose response at the
earlier position remains unmeasured in this trace.

Moreover, making Selim competitive in Ivo's recipient slot could harm argument
roles. Token binding/ownership is not sentence-role preservation, full-name
completion, a changed greedy token, or final-task success. Current readout
gates and two-token confirmation are narrower measurements than those claims.

The absolute probability floor (0.0001) plus positive raw-logit movement over
controls deserves calibration across token frequency and numerical precision.
That is a hypothesis for a fixed-policy sensitivity audit, not grounds to lower
the threshold until a candidate passes. Keep masked/nonfinite readings invalid.

## Implemented Control-Plane Repair

### One Round Dispatcher

The outer and follow-up controller rounds now use the same observer/tool
dispatcher. A requested action is executed under its existing budget or
returns no fresh result; only a fresh usable result can hold the prefix.
Diagnostic dedupe and state-restoration checks remain in place.

### Bounded Investigation Transaction

The controller may select an offered
`investigate_normal_cap_current_prefix` action ID. It measures the exact
normal-cap child and, only if the unchanged readout checks pass, physically
confirms that child in the same frozen context. Its outcome is explicitly
`measurement_only`, `confirmed_no_offer`, or `offer_available`; it never
applies an edit.

Preflight requires one remaining controller callback and up to two existing
diagnostic slots/eight physical replays for a fresh child, or one slot/four
replays if already measured at this context. Actual work is charged, including
only one slot when readout is weak and confirmation is skipped. The episode
budget is not refilled. Source/dose/binding are fixed by the selected action;
a changed dose gets a new ID and inherits no positive evidence. Existing
safety, effect, normal-cap and exact single-use grant checks stay mandatory.
An evidence-ID confirmation without a final decision round is blocked before
spending budget. A visible offer needs a separate controller accept/decline.

This shortens the interaction path without increasing alpha, general apply
permission, or the number of model diagnostics. There is no controller-owned
apply in the investigation itself.

### Action-Scoped Interface

The packet projects bounded exact actions with preconditions, candidate/context
IDs, expected child ID, remaining diagnostic/replay cost, and scope. Historical
inspection, new measurement, conditional confirmation, an actual trial offer,
and generation advance remain distinct. Missing evidence means `unmeasured`
or `blocked`, not a global statement that the controller cannot act.

Do not add a blanket "be more proactive" prompt. Do not force the normal-cap
action or sort terms toward a known answer. Helpers evaluate the controller's
chosen procedure; the controller retains objective and trial selection. Keep
raw evidence available for bounded inspection rather than copying several
overlapping permission dossiers into every decision.

## Next Experimental Split

1. Measure fixed-source/binding 0.04/0.12/0.16 at the SAME frozen prefix offline,
   with each dose independently identified and repeated controls. Use the
   existing `sample to` position as a historical diagnostic case, not a runtime
   preferred-position hint. Include prespecified neighboring positions and
   report all rows; 0.16 remains diagnostic-only under the current normal cap.
2. For a live reachability check of the new transaction, first obtain a fresh
   frozen-candidate card. Then distinguish `not_offered`, `offered_not_selected`,
   `measurement_only`, `confirmed_no_offer`, and `offer_available` before
   judging final acceptance. A narrowly scripted diagnostic replay, if used,
   must be labeled separately from ordinary C1 policy. Report installed hooks,
   changed continuations, semantic harm and task score separately.

The follow-up repair changes scheduling and controller affordances, not
operator physics, production permission, live-model results, or thresholds.

## Post-Repair Live Smoke (2026-09-23)

Two sequential, model-local B0/C1 pairs used the same checkpoint, task, seed,
precision, diagnostic limit, compact Luna controller profile and baseline
reference as their earlier profiles. Artifacts and source hashes are in the
[GPT-2 pair](../results/controller_transaction_live_20260923/gpt2/) and
[Llama pair](../results/controller_transaction_live_20260923/llama/).

| Observation | GPT-2 | Llama 3.2 3B |
| --- | ---: | ---: |
| B0 and C1 score | 0.5875 / 0.5875 | 0.938889 / 0.938889 |
| Parsed controller attempts | 20 | 28 |
| Charged diagnostics / accepted prefix holds | 12 / 9 | 12 / 10 |
| Exposed candidate-action positions | 0 | 0 |
| Exact trial offers / compiled rollout edits | 0 / 0 | 0 / 0 |

Both B0 runs matched their historical baseline reference; both C1 outputs
matched their own B0 output. In the Llama trace, the controller spent its
diagnostics on the entity/operator, readout-gap, objective-rotation and
carrier-conversion routes. The candidate-freezing matched-response matrix
remained `not_requested`, so no frozen action was offered. There were also no
controller tool or observer requests in either pair. Thus this run did not
exercise the new transaction or shared tool/observer dispatcher live; their
current coverage is synthetic. A changed controller trajectory is not evidence
that the new transaction failed mechanically. Nor do passing synthetic paths
show that a target-effective operator exists. Do not treat this single pair as
a causal policy ablation or pool scores across the two worker models.

## Validation

The pre-repair audit run reproduced two failures and two passing controls.
After the repair, focused controller scheduling, prefix-control and candidate
action tests pass with no expected failures. Full-suite validation:
`466 passed, 15 warnings, 2 subtests passed` (after the final offer-integrity
guard); no expected failures. `git diff --check` passes.

These tests use local toy fixtures and the real episode loop, not an API
controller, live Llama/GPT-2, or a claim that physical confirmation passed.

## Diagnostic Route Follow-Up (2026-09-23)

The diagnostic routing contract now filters already-measured objective/mode
expansions out of the available menu, prevents a requested diagnostic from
inheriting another frontier diagnostic's mode, and labels a replay with zero
new mode-specific rows as `no_new_measurement` or `already_replayed` with zero
diagnostic cost. A dedicated diagnostic name with a conflicting mode is an
`invalid_request`, also without physical execution or diagnostic charge.
The detailed request and `next_action` alias for the same objective/mode now
deduplicate before dispatch. None of these changes grants apply permission.

Final-code B0/C1 pairs used the same profile, checkpoint, seed, controller,
prompt and 12-slot diagnostic limit as the earlier live smoke. Baseline identity
checks passed; each pair sealed its source snapshot and hashes. The local
[GPT-2 pair](../results/diagnostic_route_fix_gpt2_r2_20260923/) and
[Llama pair](../results/diagnostic_route_fix_llama_r3_20260923/) retain the
JSONL and audits. These paths are local experiment artifacts, not committed
benchmark fixtures.

| Observation, old -> final | GPT-2 | Llama 3.2 3B |
| --- | ---: | ---: |
| Identical generated-prefix set | 11 / 11 | 18 / 18 |
| Charged diagnostics | 12 -> 11 | 12 -> 12 |
| Conversion requests carrying another mode | 3 -> 0 | 3 -> 0 |
| Rotation requests carrying readout mode | 0 -> 0 | 1 -> 0 |
| Maximum visible candidate cards | 0 -> 0 | 0 -> 0 |
| C1 score | 0.5875 -> 0.5875 | 0.938889 -> 0.938889 |

The generated prefixes match, but the controller's diagnostic histories do
not; this is a same-prefix routing comparison, not a causal operator-efficacy
ablation. In the final Llama run the recovered slots went to rotation and
readout deepening. GPT-2 retained one unused slot. Neither run requested
`matched_response_probe` or `activation_patch_candidate_review`, so neither
crossed the measurement-to-frozen-card handoff. GPT-2 did request an
activation-patch production-trial gate review, but it yielded no card or apply
permission. Zero-row replay accounting and invalid-mode rejection are covered
by tests; the final live paths did not exercise those no-charge statuses.
The remaining bottleneck is therefore candidate handoff selection, not proof
that another operator or a looser apply gate is needed.

Final validation: `473 passed, 15 warnings, 2 subtests passed` and
`git diff --check` passed.

## Seed-to-Card Handoff Follow-Up (2026-09-23)

An opt-in `candidate_handoff_mode=soft` now distinguishes `measurable`,
`unmeasurable`, and `carded`. A measurement offer is a preflight match against
recorded activation-patch evidence and the actual `matched_response_probe`
seed rules; it is not a binding check, frozen card, operator certification, or
apply permission. An already-present required term is control-only, not a
progress-oriented measurement offer. The controller can still select another
diagnostic, but its selection report records the reason given for deferring
measurement (or explicitly records that no reason was given). A small soft
opportunity cost counts repeated charged, non-safety deferrals at the same
prefix. Safety checks and zero-cost diagnostics are exempt. Measurement is
never forced, and production apply gates are unchanged. The mode defaults to
`off`.

The local [GPT-2 off pair](../results/candidate_handoff_off_gpt2_20260923/)
and [GPT-2 soft pair](../results/candidate_handoff_soft_gpt2_r2_20260923/)
used the same task, checkpoint, seed, Luna controller, baseline reference and
12 diagnostic slots. Their 11 generated prefixes and final score (`0.5875`)
matched. The off pair produced no measured card. The soft pair reached a
`measurable` Mira seed, ran one matched-response probe, and recorded two
frozen cards after 12 charged diagnostics. Neither card passed the bound
target-lift trial gate; there was no trial offer, compiled rollout edit, or
score improvement. The soft opportunity cost did not activate in this pair,
so the card handoff supports the measurement-offer plumbing, not a claim that
the penalty improved controller behavior or reduced diagnostics-to-card.

An initial Llama soft smoke recorded cards for `sample`, a required term
already present in the generated prefix. Those cards were valid measurements
but not progress toward the missing term, so the preflight now excludes such
terms from progress offers. In the corrected
[Llama soft pair](../results/candidate_handoff_soft_llama_r2_20260923/),
no recorded activation-patch seed became measurable before the diagnostic
budget was spent, and no card was produced. Its 18 generated prefixes matched
the [Llama reference](../results/diagnostic_route_fix_llama_r3_20260923/);
the output and score (`0.938889`) were unchanged. This run did not isolate
the already-present-term guard in live behavior; that guard is covered by a
focused synthetic test. Both reported pairs preserve the production
permission boundary. These are same-prefix routing observations, not a causal
operator-efficacy or task-success result. The next useful comparison is a
prefix where a missing-term seed is available *before* the final diagnostic
slot, so repeated optional deferral and its reason can be observed live.

Final validation: `479 passed, 15 warnings, 2 subtests passed` and
`git diff --check` passed.
