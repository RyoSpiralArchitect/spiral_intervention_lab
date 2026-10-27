# Common: Current-State Radial and Nonradial Components

## Question and Scope

The fixed-permutation experiment found a larger comma probability response to
common than to one permuted direction at the same cap. It did not isolate whether
common benefited from shrinking the current activation. This follow-up separates
that geometric component from the rest, without treating either as a semantic
subspace, causal owner or safe production actuator.

Let C = mean(S_left, S_right), T = the current last-position activation, d = C-T.
The existing DSL already supplies `mean`, `sub`, `add`, `project_parallel` and
`project_orthogonal`; no new expression kind or operator permission is needed.

| Component | Source expression for activation blend |
| --- | --- |
| common | C |
| negative common | T - d |
| parallel | T + project_parallel(d, T) |
| orthogonal | T + project_orthogonal(d, T) |

Adding T back is essential: `activation_patch(mode="blend")` itself subtracts T.
Passing the projected displacement alone would subtract the current state a
second time and measure the wrong intervention.

The signed radial coefficient `dot(d,T)/dot(T,T)` identifies shrink versus
expansion. Zero/nonfinite bases and degenerate rounded displacements are
unavailable, not manufactured positive/dead evidence. Native DSL projection is
retained; its rounding is reported against CPU/float64 geometry. Each condition
uses normal alpha and step caps with no component renormalization. Independent
clipping and downstream nonlinearities preclude adding their effects as causal
contribution percentages.

## Controller Contract

With existing `clause_structure_operator_diagnostics` enabled, a current-prefix
structure observation and aligned cache can expose `source_common_radial` in
strategy hints. All pairs among the first three exact source tokens in SOURCE
occurrence order are offered: at most three pairs and twelve component recipes,
at the same existing activation sites. The original first-two pair stays first;
neither sources nor components are ranked by measured response. The offer records
the source count, limit and excluded count. Consuming a pair does not slide this
window to later sources, and displaying more choices does not execute a matrix.

The controller selects a current `operation_id`, or explicitly requests:

```json
{
  "diagnostic": "source_common_radial_probe",
  "component_recipe_id": "<current offered ID>"
}
```

Each `source_pair_id` binds both exact positions/pieces/provenance, the current
context/observation, surfaces, normal caps and both raw source tensor identities
plus the current target. Each `component_recipe_id` additionally binds the
component, compiled source/target identities and the canonical DSL expression.
Even opposite changes to the two raw sources that leave their mean unchanged
invalidate the old binding. Repeated pieces retain the explicit first-occurrence
rule; a different occurrence is not interchangeable evidence.
The request cannot supply a new expression, target, dose or policy override.
Prefix, cap and input-tensor changes invalidate the ID. Attempted recipes,
including failed physical attempts, cannot repeat at the same context; equivalent
offline measurements also consume the offer.

Offline radial replay and controller dispatch share this binding constructor.
Both record `source_pair_id` and `component_recipe_id` in results and operator
rows; the offline writer includes them before persisting each completed row.
Dispatch resolves the anchor from the selected pair, not the global first source.
Compact cards retain the pair, site/caps, formula and cost for all twelve choices.
Reanalysis checks new bindings against raw inputs and rejects relabeled receipts.
Older receipts remain analyzable with `source_pair_binding_status=legacy_unbound`;
no current ID or authority is retroactively assigned to them.

One selected recipe costs at most one diagnostic slot: two no-edit repeats plus
two edited repeats per offered site (four or six explicit physical replays).
Packet preparation canaries remain separate forwards and must be counted too.
Pure offer generation performs no model forward. No-cache/stale/degenerate
requests perform no replay and incur no diagnostic charge. Runtime dispatch owns
the charge; offline measurement charges its own slots.

Full receipts retain the actual DSL edit, source identity, geometry, hook
telemetry, readout rows, safety deltas and restoration checks. The compact prompt
retains a formula, source pair, geometry, ID and bounded cost, not raw vectors or
a long edit JSON. These controls have no intended semantic owner, never mint
frozen production candidates, and keep `production_apply_allowed=false`,
`certified_for_apply=false` and `policy_candidate_ready=false`. Existing
production/trial gates are unchanged.

## Fixed-Prefix Measurement

The opt-in `--radial-component-controls --decompose-components` replay keeps the
seven previous source/common/difference conditions and adds only the three new
controls. Ten condition slots plus cache alignment and structure observation use
the existing 12-slot budget. It cannot be combined with the permutation or
adjacent-boundary experiments. The same verified checkpoint and naturally
reconstructed prefix are required.

The predeclared protocol is in
[`results/source_common_radial_cpu32_20261025/plan.md`](../results/source_common_radial_cpu32_20261025/plan.md).
The worker model path is supplied by `--worker-model-path`. No external controller
API is called in this measurement; diagnostic-card integration is tested
separately and is not evidence of autonomous controller tool choice.

### First Attempt (Incomplete)

The single attempt stopped at the unchanged 900-second limit (903.48 seconds
including supervision polling/termination), with **five of ten conditions**
retained: source 11, source 1418, self-copy, radial shrink and common. The
negative-common, parallel and orthogonal conditions were not reached. No implicit
retry, cap increase or production apply occurred in that attempt.

Another local experiment was observed while this worker slowed and swap usage
increased. This is a resource-constrained incomplete measurement, not a dead or
harmful response from the new components. The other process was not modified.
Disk stayed above the 16 GiB stop floor: 54.68 GiB initially, minimum 39.63 GiB,
45.39 GiB at stop.

The retained five conditions have identical input and captured-vector hashes
to the previous CPU32 experiment, at the same checkpoint and natural prefix.
Common again gives comma probability delta +0.001517545, versus -0.000492017
for pure radial shrink. All five retain greedy token 1603 (` before`).
This reproduces the comparison, not task success.

Offline geometry from the retained source inputs, without a new model forward:

| Raw displacement property | Value |
| --- | ---: |
| radial coefficient, dot(d,T)/dot(T,T) | -0.686604 |
| common displacement norm | 11.152185 |
| parallel component norm | 7.186001 |
| orthogonal component norm | 8.528343 |

The coefficient describes **unclipped source-minus-target geometry**, not a
68.7% runtime shrink. The actual common write still obeys the 0.08 step cap.
Both a shrink component and a substantial orthogonal component exist; their
separate readout effects were **unmeasured in this attempt**. Do not infer effect ownership or
contribution percentages from these norms.

The five completed conditions contain 20 explicit replays plus 15 packet
preparation forwards. Total forwards, including work in flight when terminated,
are unknown: termination prevented final counter persistence. The original
trajectory retains its last `measuring` status; supervision, rather than that
stale status, records the stop. Neither receipt is rewritten as a success.

Retained [partial verification](../results/source_common_radial_cpu32_20261025/partial_verification.json),
[trajectory](../results/source_common_radial_cpu32_20261025/cpu32/measurement/trajectory.json)
and [supervision](../results/source_common_radial_cpu32_20261025/cpu32/supervision.json)
preserve this boundary. `verify_partial.py` checks the exact completed prefix,
raw-vector hashes, compact receipts and numerical summaries; it cannot certify
the unmeasured remainder. Its initial four-condition-only check correctly
rejected the final five-condition receipt; that analysis error log is retained
beside the corrected prefix-aware audit. `build_summary.py` continues to require
a fully completed experiment and is not used to bless this partial attempt.

Validation: **728 passed, 9 subtests passed, 15 dependency warnings**. Tests cover
DSL target add-back, radial sign, invalid/zero basis, current-ID/card selection,
stale prefix/cap/tensor rejection, single-charge accounting, duplicate suppression,
state restoration, no frozen production candidates and the ten-condition budget.
`git diff --check` passes.

## Completed Retry and GPT-2 Comparison (2026-10-25)

The cooperator explicitly authorized a new attempt after the competing run ended.
The original interrupted receipt is unchanged. Llama completed all ten conditions
in 842.52 seconds, within the same 900-second limit, at the same natural position-8
prefix, checkpoint, source pair, site, dtype and caps. Exact input and captured
vector hashes match the previous complete CPU32 run's first seven conditions
and the interrupted attempt's five retained conditions.

GPT-2 then ran separately, not concurrently. The cooperator explicitly selected
the same visible text, ` Nora will take the sample to Ivo`, as **diagnostic
prefill**. It is not naturally generated GPT-2 output. Both tokenizers happen to
encode this output prefix in eight tokens, but the full input is 108 tokens for
Llama versus 112 for GPT-2. SOURCE positions are 76/86 (IDs 11/1418) for Llama
and 79/89 (IDs 11/981) for GPT-2, corresponding to comma and ` while`.

The normal catalog selects `resid_pre L14` in full 28-layer Llama and
`resid_pre L6` in full 12-layer GPT-2. Both use CPU/float32, four threads,
alpha 0.04, step cap 0.08, norm clip 1 and TTL 1. No model download,
controller API, strength/site search or production apply occurred.

### Raw Readout

Full-vocabulary softmax is recomputed from retained logits in CPU/float64.
Numbers below are **absolute probability deltas**, not percentages. The two
deterministic repeats agree exactly; they are not independent samples.

| Model | Condition | Comma probability delta | While probability delta |
| --- | --- | ---: | ---: |
| Llama | Common | +0.001517545 | +0.000012840 |
| Llama | Negative common | -0.001485989 | -0.000010259 |
| Llama | Parallel | -0.000492807 | -0.000165582 |
| Llama | Orthogonal | +0.002361577 | +0.000158070 |
| GPT-2 | Common | -0.000260804 | +0.000018254 |
| GPT-2 | Negative common | +0.000253881 | -0.000018164 |
| GPT-2 | Parallel | -0.000199848 | +0.000002770 |
| GPT-2 | Orthogonal | -0.000180078 | +0.000021866 |

Llama's baseline comma probability is 0.183171009 and its fixed greedy winner
is ` before` (1603). GPT-2's baseline comma probability is 0.369785961 and comma
itself (11) is already the winner. All ten conditions retain each model's own
winner. GPT-2's comma margin against that fixed winner is necessarily zero;
it must not be interpreted as an absent probability response.

At this Llama boundary, the observed common comma lift is **not explained by
shrinking the current state alone**: parallel shrink lowers comma probability,
whereas the orthogonal component raises it more than common at the same cap.
Both comma and while rise under the orthogonal intervention, so this does not
identify semantic ownership. The GPT-2 sign difference also rules out treating
"orthogonal helps comma" as a model-independent rule from these data.

### Actual Geometry and Scope

All nonzero writes are approximately 0.08. The current activation norm is
10.466009 for Llama and 82.868653 for GPT-2, so the relative write is approximately
0.7644% versus 0.09654%. Equal absolute caps do not imply equal relative doses.
The raw common radial coefficients are -0.686604 and -0.522254 respectively.

The parallel write has cosine approximately -1 with current T in both models.
Its difference from the existing pure-radial control is only 5.03e-8 (Llama) or
1.69e-7 (GPT-2) in write norm, retaining native rounding rather than forcing
identical tensors. The orthogonal write's cosine is 4.90e-8 or -8.16e-6. Its
activation norm changes by +0.000305751 or +0.000037962, not a shrink.

No component renormalization was added. Each condition is independently bounded
by the existing step clip; the clipped common write is not the sum of the two
individually clipped component writes. Do not interpret response ratios as
additive causal contribution percentages.

Captured no-edit, self-copy, repeat-difference and preparation-drift floors are
zero in both measurements. The common/parallel/orthogonal one-step safety deltas
have no repeat increase or collapse-sharpening flag. This does not certify
continuation safety, clause correctness, a target actuator or task success.
Tokenizer, checkpoint, depth/site, activation scale, baseline distribution and
natural-generation versus prefill history differ across models; this is not a
model-only causal ablation.

### Accounting and Reanalysis

| Work | Llama | GPT-2 |
| --- | ---: | ---: |
| Diagnostic slots | 12 | 12 |
| Conditions / explicit condition replays | 10 / 40 | 10 / 40 |
| Captured packet-preparation forwards | 30 | 0 |
| Alignment/observation forwards outside condition captures | 3 | 2 |
| Natural prefix reconstruction forwards | 8 | 0 |
| Actual forwards attempted / completed | 81 / 81 | 42 / 42 |

GPT-2's output-segment counters describe the eight **supplied prefill tokens**;
`natural_generated_token_count=0` is recorded explicitly. The per-model
preparation counts differ and are not hidden in the 12-slot label.

Retained [Llama summary](../results/source_common_radial_cpu32_retry1_20261025/summary.json),
[GPT-2 summary](../results/source_common_radial_gpt2_cpu32_20261025/summary.json)
and [paired comparison](../results/source_common_radial_comparison_20261025/comparison.json)
link to hashed raw receipts. The [offline analyzer](../results/source_common_radial_comparison_20261025/analyze.py)
recomputes raw hashes, compact identities, component geometry and full readout
summaries without model calls; it refuses incomplete or code-drifted inputs.
Raw tensors remain local. The new fixed-prefill harness lives with the experiment
rather than changing the runtime during Llama's measurement.

The post-run full suite again reports **728 passed, 9 subtests passed, 15 dependency
warnings** in 50.44 seconds. The [test receipt](../results/source_common_radial_comparison_20261025/tests.txt)
and per-run checksum inventories retain verification separately from numerical
or semantic claims. `git diff --check` passes.

### Controller Binding Caveat

This offline experiment explicitly measures **comma + while**. At measurement
time the controller offer only used the first two SOURCE candidates, which at
this fixture are **before + comma**. Those remain different recipe identities.
The explicit-pair implementation now offers **before + comma**, **before + while**
and **comma + while**, with a shared offline/runtime binding contract. This closes
the selection-path mismatch, not the empirical question: the old receipt does
not become current evidence, and the controller has not yet demonstrated that
it selects or benefits from this pair. Remeasure the exact expression at the
current prefix; do not adopt a global "prefer orthogonal" rule or confer apply
permission from the historical comparison.

### Live Card-to-Execution Check

`replay_source_clause_trajectory --controller-radial-choice` replaces the full
offline matrix with **one** controller invocation and, only if selected, one
radial diagnostic. It requires `--radial-component-controls` and an explicit
`--offline-radial-receipt` from the same checkpoint, recorded prompt/prefix,
source pair, device/dtype, threads and caps. Model paths remain CLI arguments.

All current operation cards retain their runtime order and availability. No
preferred pair, component, offline response, rescue label or follow-up prompt
is supplied. A non-radial choice or abstention is recorded without forced
fallback or another provider call. An apply command is never dispatched by this
diagnostic harness. The legacy receipt is read only for the post-choice audit.
`selected_operation_card` retains any valid menu choice, while
`selected_radial_operation_card` is separate. A non-radial request is recorded
as `non_radial_selection_not_dispatched`, with the harness-scope reason. It is
not controller abstention, a parser failure or runtime unavailability.

`--controller-source-choice` is the broader diagnostic harness mode: it can
dispatch either the selected source-token patch or the selected radial recipe.
The original radial-only mode remains available for reproducing its contract.
`--recorded-controller-choice <receipt>` makes no provider call; it verifies the
checkpoint/numerics/prefix, the recorded raw and normalized response, and the
**entire current packet hash** before reusing the exact operation ID. It refuses
drift rather than rebinding a similarly named card. This is a replay of a prior
controller choice, not a new independent controller sample.

Single-source execution is checked against its prepared edit, input hash,
candidate fingerprint, repeated hooks and caps. A source token absent from the
historical offline matrix remains `not_covered`; source-token and radial-pair
effects cannot be equated by a shared surface or token label.

The new receipt separates:

- `selected_execution_comparison`: the selected recipe's exact source inputs,
  surface, DSL, blend dose/caps, two executed hooks and restored state.
- `offline_comparison`: the historical raw left/right/target hashes, compiled
  inputs, expression, context and actual edit. A different pair is `not_covered`,
  not an identity match or a failure of the operator.
- `readout_and_execution_exact_checks`: repeated structural response, physical
  write telemetry and repeat floors, separately from operation identity.

Old receipts remain `legacy_unbound` even if their raw physical evidence
matches. This is an interface/execution parity check, not a controller efficacy,
semantic rescue, production permission or task-success experiment. The 12-slot
ceiling and actual-forward accounting remain unchanged. The single-choice mode
defaults to a 32-GiB free-disk start floor. An explicitly authorized
`--min-free-disk-gib` override (minimum 10 GiB) is recorded in the receipt; the
supervisor must use the same floor, rather than silently lowering it on failure.

### Selected-Source Live Replay (2026-10-26)

The first Luna choice was a single-source ` before` patch, not a radial card.
The radial-only harness had recorded `no_radial_selection` and performed no
selected-operator replay. That was a dispatch-scope limitation, not controller
abstention. The original receipt is preserved.

The source-enabled replay then reconstructed the exact natural prefix and
**entire packet hash**, reused that recorded choice with zero new API calls, and
executed its normal-cap L14 resid-pre blend. Card, source index 74, input hash,
prepared edit, candidate fingerprint, actual hooks and restored state matched.
Two baseline plus two edited replays used one diagnostic slot; total run usage
was 3/12 slots and 19 actual forwards, including prefix/inspection overhead.

Both edited repeats lowered before probability by 0.000464914 and raised comma
by 0.001580512, with zero repeated-logit variation. The comma-to-winner gap fell
from 1.114996 to 1.105572, but greedy remained before. This is `cross_bound`
readout response, not semantic rescue. The before-bound `bound_gap_movement`
label does not imply its probability increased or that a task improved.

The old offline matrix lacks this exact single-source before candidate, so its
comparison is `not_covered`. The new local baseline measures its effect; it does
not establish independent historical parity or controller selection value.
Production authority remains unchanged. See the
[published derived audits](../results/published_intervention_20261027/README.md)
for the failed/interrupted attempts, lineage, resource bounds and full counts.

A fresh follow-up Luna call (attempt5) received the same packet/menu and again
selected the before card. Source-enabled dispatch executed it without fallback;
candidate binding, compiled inputs and complete repeated operator rows matched
attempt4 exactly. Usage remained 3/12 slots and 19 actual forwards, with greedy
before unchanged. This verifies one fresh selection-to-execution path, not
controller advantage or what the controller would do after receiving its result.

## Path Toward Use, Not Automatic Promotion

Geometry is available before measurement; actual response evidence is available
only afterward. The controller may choose which offered component to test and
compare its receipt with common and the radial control. The helper never chooses
a winning component or turns its coefficient into a veto or an apply bonus.

Before a measured component becomes a production candidate, a separate path must
bind an explicit output objective and remeasure that exact normal-cap expression
at the current prefix. Current geometric-control rows intentionally lack that
ownership assertion. Token lift, ownership, stability, continuation safety and
the existing trial/promotion contracts remain distinct requirements.
