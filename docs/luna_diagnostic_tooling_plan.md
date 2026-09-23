# Luna-Centered Diagnostic Tooling Plan

Date: 2026-09-12. Implementation update: 2026-09-14.

The subsequent [source-construction experiment and evidence-ID trial bridge](source_direction_comparison.md)
build on this implementation. Historical local archive paths below are not part
of the Git repository; selected September 14 evidence is packaged with hashes.

## Implemented Contract

The first three items are implemented, without expanding production permission:

- `runtime/response_probe.py` measures direct/observed seeds at equal step-size
  doses (0.04/0.16), with one frozen pre-edit pair of readout bindings. Source
  tensor bytes and state distinguish executions; binding IDs distinguish
  observables. Identical materializations across seed labels reuse one replay,
  not independent votes. Forced canonical rows are excluded from observed seeds.
- Each unique edit has an identical-repeat control and shares a no-edit repeat
  control. The maximum is ten one-token simulations for four unique edits;
  identical seed materializations reduce it to six. Effective hook/norm telemetry
  and unrounded gap components are retained. CPU float64 is used only for audit
  arithmetic on recorded logits, not to change the model's execution precision.
- `inspect_evidence` reads up to six existing rows by evidence/execution/recipe
  ID. Its four-call episode cap is separate from the unchanged replay/request
  budget. New views, physical replays, and changes in the listed gate facts are
  counted separately. Historical evidence is not current-prefix certification.
- `runtime/debrief.py` selects deterministic event anchors rather than a tail
  window, under a 20,000-character digest cap including the coverage manifest.
  The exact supplied packet is saved as `post_run_debrief_input.json`. Wishes
  must reference exposed events and distinguish missing, hidden, ambiguous, and
  unknown signals. All prior quarantine flags remain unchanged.

The existing activation-patch review invokes the matched probe. Controllers can
also explicitly request `matched_response_probe` with an objective and optional
recipe IDs/dose subset; this spends the normal diagnostic request budget.
Site/layer/alpha/localization mismatches return an unavailable report, not an
unmatched comparison. No metric here is a new apply permission or a selector.

`actual_delta_class` on a matched row retains the execution-level canonical
binding/term-set interpretation. Use `bound_token_response` and the raw bound
logit/probability/rank fields for the other binding, not that aggregate label.
Repeat variation is a local control, not a statistical confidence interval.

## Operating Choice

Use `gpt-5.6-luna` for routine live controller runs and their opt-in qualitative
debriefs. Retain Opus for occasional, condition-matched cross-controller audits.
Keep model selection explicit in CLI/run manifests rather than changing the
library-wide default or silently falling back to another provider.

This is an operating preference, not a claim of model equivalence. The latest
pair contains one seed-7 episode per controller and different reasoning controls.

## Evidence And Its Limits

The local paired audit `results/luna_opus_pair_20260911_234846/pair_audit.json`
and `comparison_notes.md` in that same archive
show four matched diagnostic cells with identical readout metrics at stored
precision. Both episodes scored 0.5875, remained incomplete, and made zero
production applies.

The seed labels are confounded with step size: `direct_candidate` uses 0.04 and
`observed_gap_carrier` uses 0.16. Within a seed, canonical/alternate bindings are
different readout observables of the same edit, not different payloads. The
canonical `" Mir"` logit increases while alternate `" mir"` decreases, although
both top20 gaps narrow. All four target-rank and top20-hit deltas remain zero.

The matrix executes at controller step 4. Later reviews at steps 6/7/8/9/10
report `already_replayed`. That is not five fresh measurements, and does not by
itself prove those complete diagnostic requests were useless: other report
fields and gate context must be compared separately.

The following historical GPT-5.5 debriefs repeatedly request candidate-specific
lift/certification tables, budget visibility, materialization reasons, and
untruncated evidence:

- PR18: `results/pr18_real_run_20260613_173644/post_run_debrief.md`
- Suppress-then-target: `results/suppress_then_target_inline_live_20260613_235337/post_run_debrief.md`
- Two-stage variant: `results/two_stage_target_variant_live_20260614_001245/post_run_debrief.md`
- Two-stage saturation: `results/two_stage_saturation_live_20260614_002437/post_run_debrief.md`

These are qualitative suggestions, not causal diagnoses. The current Luna pair
has no debrief; do not attribute these wishes to Luna. Budget and diagnostic
ledger fields already exist, so a requested field can be a visibility problem
rather than a missing runtime capability.

The pre-update debrief builder mainly sampled the last 32 events per log. Applied
to the Luna log, that window covers steps 8-11 and excludes the original step-4
matrix execution. Cached views can survive in the tail, but they are not a
substitute for an explicitly linked execution record.

## 1. Matched Response Probe

Extend the existing binding/seed matrix, materializer, and actual-delta replay;
do not add a new production operator or selector. A controller request should
name candidate IDs, a bounded allowed dose grid, and pre-edit readout bindings.
The runtime validates and measures that request without choosing what to apply.

- Freeze the prefix, model/tokenizer identity, active patch state, site, layer,
  alpha, source localization, and seed-specific materialized source identity.
- Give both seed labels the same step-size grid. Start with already explored
  0.04 and 0.16, not another escalation in strength.
- Separate `execution_identity` from `observable_identity`. The former includes
  the resolved source, target, dose, dtype, and state; the latter adds binding.
  Provenance labels stay metadata, not proof of a different intervention.
- If equal-dose materialized edits are identical, report aliases rather than
  treating them as independent seed evidence.
- Evaluate both frozen token bindings from the same baseline/edited logits when
  execution identity matches. Do not silently reuse logits across worker states.
- Include no-edit and repeated-identical controls; record requested and effective
  edit norms, clipping, and replay count. Preserve raw precision for the audit.
- Expose gap components already available in readout telemetry:
  `gap_delta = threshold20_logit_delta - target_piece_logit_delta`.
  Keep term-aggregate mass distinct from bound-token probability.

Acceptance: matched-dose comparisons, execution aliases, noise-floor evidence,
binding honor, state restoration, and independent target/gap measurements are
machine-checkable. The probe can report failure or uncertainty; it never grants
production permission. The old dose-confounded matrix remains historical data.

## 2. Evidence Inspection And Novelty

Improve access to the existing ledger instead of adding more prose to every
controller packet. A bounded `inspect_evidence` request would fetch a small
candidate table or one execution record by ID, with its measurement context.

Suggested report fields:

- `execution_id`, `evidence_scope`, `measurement_context_id`
- `new_measurement_count`, `cached_measurement_count`, `new_gate_fact_count`
- `budget_before`, `budget_after`, `physical_replay_count`
- `blocked_axes`, `missing_evidence`, `executable_diagnostic_ids`
- `retry_preconditions`, with explicit context/dose/family changes

Keep review-only requests separate from new replay requests. A cached inspection
should not pretend to consume fresh experimental evidence, but still has bounded
tool/token cost. A new prefix is not silently considered certified by an old
prefix's result. Reconfirmation is explicit and budgeted, not automatic.

The controller decides whether to inspect, change the experiment, continue, or
stop. The helper reports availability and changed facts; it does not globally
ban repeat reasoning or nominate an answer. Compare all gate facts before
classifying a repeated request as uninformative.

Acceptance: a trace distinguishes new physical measurements, cached views, and
changed gate context. No clone evidence inflation, hidden state reuse, or new
apply authority is introduced. Large full rows stay behind on-demand access.

## 3. Event-Anchored Debrief

Keep the existing quarantine flags and use the run's requested Luna model for
the next opt-in interview. This is a new call supplied with public trajectory
records, not recovered private reasoning or a claim of continuous inner memory.

Replace tail-only selection with bounded deterministic event anchors: first
materialization, first actual matrix, first safety/ownership outcome, the first
unchanged review, and termination. Link each anchor to the relevant request,
result, structured decision, and budget state. Include opposing/negative results,
not just the best candidate. Keep the overall token cap fixed.

Provide an input coverage manifest identifying included and omitted evidence.
For each requested affordance, ask for a supporting event ID, what could not be
distinguished, and a minimal tool/field that would help. Separate:

- `missing_at_runtime`
- `present_but_not_shown`
- `shown_but_ambiguous`
- `unknown_from_supplied_evidence`

These categories are interview claims to verify against the exposure manifest,
not automatic engineering tickets. Do not force a recommendation if the model
has no grounded one. Do not feed the memo into the next controller context,
promotion gate, candidate score, or paper metric.

Acceptance: the step-4 execution is available alongside the late repeated
reviews under a bounded input budget. Suggestions have checkable references;
missing evidence is not confused with omitted evidence.

## Follow-Up Operator Work

Only after dose and binding are separated, decide whether to refine a source
recipe, site, or timing. Use existing source-token/centered/contrastive activation
patch recipes as controlled contrasts, not as allegedly new operators. Change
one execution axis at a time while maintaining the matched context and dose.

If target logit/probability improves above replay variation but stays below the
actuator gate, investigate a bounded local response curve. If only the competing
threshold falls, record that distinct effect rather than declaring target lift.
If the response is dead or wrong-direction at matched dose, change the source
or site before increasing strength again.

SAE/readout features and attention diagnostics can suggest contrasts as evidence
only. Do not add hard token forcing, answer-bearing memory, or a new authority
layer to compensate for a weak operator.

## Suggested Delivery Order

1. Matched-response probe and its regression fixtures, with minimal novelty IDs.
2. Evidence inspection plus event-anchored debrief, without expanding the prompt.
3. One Luna live episode with opt-in debrief; report task outcome separately from
   new-measurement yield, cached reviews, binding validity, and permission state.
4. Select a single operator-axis contrast from the measured result. Use Opus only
   when a new interpretation or controller-specific issue warrants another pair.

The original planning pass inspected artifacts only. Implementation preserves
the same runtime policy and production gates; live validation is recorded below
separately from the historical pair.

## 2026-09-14 Validation

Artifacts:

- [Lossless compressed live JSONL](../results/source_direction_comparison_20260914/source_episode.jsonl.gz)
- [Corrected local remeasurement](../results/source_direction_comparison_20260914/preceding_seed_remeasurement.json)
- [Machine-readable audit](../results/source_direction_comparison_20260914/preceding_seed_audit.json)
- [Original Luna debrief](../results/source_direction_comparison_20260914/source_episode_debrief.md)
- [Exact original debrief input](../results/source_direction_comparison_20260914/source_episode_debrief_input.json)

One Luna episode completed 11 steps with score 0.5875, `task_done=false`, and
zero apply commands. Output: `In the case of a rewrite, the budget draft.`
This is not a task-score improvement result.

At controller round 4 / worker step 5, prefix ` In the case of a`, the matched
probe produced eight lineage-labelled rows. They represented **two unique
executions and four unique observables**, not eight independent experiments.
Direct and observed seeds resolved to identical source/target/dose tensors at
both 0.04 and 0.16. Thus this prefix does not provide an independent seed-quality
contrast; provenance labels alone did not change the intervention.

| Step Size | Binding | Target Logit Delta | Top20 Threshold Delta | Gap Delta |
| --- | --- | --- | --- | --- |
| 0.04 | ` Mir` | +0.000392914 | -0.000415802 | -0.000808716 |
| 0.04 | ` mir` | -0.000257492 | -0.000415802 | -0.000158310 |
| 0.16 | ` Mir` | +0.001549721 | -0.001645088 | -0.003194809 |
| 0.16 | ` mir` | -0.001041412 | -0.001645088 | -0.000603676 |

All bound-token rank and target top20-hit deltas were zero. Objective-term mass
deltas were only 0.000001/0.000005. The lowercase binding demonstrates why gap
closure alone cannot be described as a target-logit lift: its logit fell, but
the competing threshold fell farther.

The initial live control metric contained `NaN`: four identically masked token
logits were `-inf`, so naive subtraction computed `-inf - -inf`. The original
JSONL and interview are preserved, not silently repaired. The fix compares only
finite logits **after verifying identical negative-infinity masks**; NaN,
positive infinity, changed masks, or an unmeasurable target fail explicitly.
The corrected local replay matched the live context and execution IDs exactly:
50,253 finite logits, four masked logits, no-edit max difference 0, identical-edit
repeat difference 0, six simulations, and state restoration true. These local
controls are not a confidence interval or evidence of cross-prefix generality.

The first local reconstruction failed closed because controller rounds are
zero-based while worker observations are one-based. A subsequent incomplete
materialization report is retained in `local_remeasurement.json`: compact pool
rows omitted original seed provenance, and lowercase bundle identity was not
the case-preserved source term. The final harness uses recorded seed descriptors
and preserves source-term spelling. It never invents a replacement candidate.

Luna did not choose `inspect_evidence` in this episode. A separate local test on
the real worker confirmed inspection with diagnostic budget 0: two cached rows,
one execution, zero new measurements/replays, inspection budget 4 -> 3, and
production permission false. This verifies availability, not autonomous use.

The original debrief saw the first matrix at `c1.jsonl:L50`, the cached review
at `L70`, and termination at `L111`, instead of just late events. Luna explicitly
noticed the invalid no-edit metric. Its five cited event IDs were exposed, but
the three affordance wishes lacked their own event IDs and two lacked an
availability category. They remain interview suggestions, not verified runtime
absences. The new reference audit records these gaps without rewriting the memo
or spending another provider call. Duplicate anchor events are now shared, and
negative rows are retained ahead of optional detail under the same digest cap.
`debrief_digest_rebuilt_not_sent.json` is a local exposure audit, not a second
interview or a changed original input.

Reproduction (model path is supplied by the caller):

Validation at this milestone: `302 passed` (17 dependency/runtime warnings), and
`git diff --check` passed. The six/ten simulation counts above cover only the
matched probe, not other replay work in the surrounding activation-patch review.

```sh
python3 -m pytest SpiralInterventionLab/tests -q
python3 -m SpiralInterventionLab.examples.replay_matched_response_probe \
  --source-jsonl results/source_direction_comparison_20260914/source_episode.jsonl.gz \
  --worker-model-path "$WORKER_MODEL_PATH" \
  --output /tmp/sil_matched_remeasurement.json
```

Next experiment, now [measured](source_direction_comparison.md): keep the prefix, binding, site/layer, alpha and dose
fixed, and vary only the actual source direction (for example source-term token
versus source-centered context). Do not compare aliases as different seeds or
raise the dose simply because the larger dose closes more gap. Apply remains
subject to the existing independent certification and trial contracts.
