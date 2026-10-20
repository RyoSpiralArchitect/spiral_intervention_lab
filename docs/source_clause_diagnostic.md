# Source-bound clause readout diagnostic

Date: 2026-10-20. Follows the merged #32-34 ceiling, dose-response and
[clause-structure survey](clause_structure_survey.md).

## Evidence carried forward

The stored survey has nine cases, six failed natural continuations and three
cases with a successful forced-connective branch. At the L1 review prefix
`Nora will take the sample to Ivo`, the stored `,` rank is 2 and its top-1
logit gap is 1.125. Seven of 90 source-patch curves win the structural token,
but only with released caps and large changes at L25-L27. These are historical
quarantined measurements, not new live results, controller choices or normal-cap
intervention success.

This suggests observing structural alternatives alongside missing-name
reachability. It does not justify promoting a known successful separator into
the controller's policy, raising caps, or treating a lexical score as role
preservation.

## New contract

`--clause-structure-diagnostics` is opt-in on the main example CLI (paired,
C1-only and seed-sweep paths). It offers `source_clause_structure_probe` in the
existing diagnostic menu. Default runs are unchanged.

The runtime extracts up to 12 distinct **actual source-body token IDs** whose
complete decoded pieces match a fixed connective/punctuation vocabulary.
Repeated occurrences share one row with source positions. Header and answer
prefix tokens, names, subpieces of longer words and unseen connectives are not
inferred or added. Natural connectives such as `before` and terminal punctuation
are retained; this is not a list of past rescue tokens. Source order is preserved.
For example, `;` is not offered merely because a past forced-token survey found
it useful if it does not occur in the current source.

The offer is available at any nonempty, nonterminal prefix with source
structural tokens, no active edits and remaining diagnostic budget. No role-frame annotation,
expected missing clause, successful position or future continuation determines
its availability. The controller chooses whether and where to request it.
One optional offer is appended without replacing the existing frontier or
displacing the original menu entries.

Execution uses one restored, unedited next-token replay. Active patches are
excluded in v1 so their effects or hook telemetry cannot be mistaken for a
no-edit observation. Each row reports its
exact vocabulary rank, probability, logit, gap to the current greedy token,
top-20 membership and source positions. This is the readout **after existing
decode constraints and decoder control**, not an unprocessed model logit.
Tied best rank is not called a greedy win; the argmax token is recorded
separately. Masked tokens have null rank/logit/gap rather than infinite JSON.

The request binds the prompt, exact current tokens, active edits, model config,
decode settings and source token catalog. A stale or missing ID is rejected
without work or charge. Each context is attempted once per episode, surviving
diagnostic history eviction; forward errors remain failed attempts rather than
positive evidence. A changed prefix permits another observation. Successful or
failed physical attempts cost one existing diagnostic slot, invalid/duplicate
requests cost zero, and generation/TTL do not advance. Exhausted budgets do not
offer executable cards.

Operation-card and legacy requests preserve this exact context instead of
inheriting another frontier's objective or expansion mode. Compact packets and
results retain binding, costs, masked rows, state restoration and the permission
boundary:

```text
current-prefix readout != operator effect
source structure visibility != clause rescue
diagnostic observation != apply permission
```

No operator candidate, frozen response card, trial offer or certification is
created by this diagnostic. It does not call the offline role checker or change
the legacy evaluator. Caps, alpha budgets, trial gates and controller prompts
are unchanged.

## Narrow verification

The replay harness below checks checkpoint hashes and the recorded prompt, then
decodes normally to the requested prefixes. Each full decoded prefix must equal
the recorded observation; clipped tails or drift fail closed. It resolves the
offered operation ID through the normal runtime dispatcher, asserts restored
state and writes a fresh receipt without overwriting an existing output.

```sh
SRC=results/entity_probe_refresh_llama_20261014_retry1
python3 -m SpiralInterventionLab.examples.replay_source_clause_diagnostic \
  --source-jsonl "$SRC/live/c1.jsonl" \
  --source-manifest "$SRC/manifest.json" \
  --worker-model-path "$LLAMA_MODEL_PATH" \
  --positions 7,8,9 \
  --output results/<new_dir>/source_clause_readout.json
```

There are no provider calls, forced tokens, raised caps or edited continuations.
This harness requests observations itself; it **does not measure Luna's choice**
and does not reconstruct the historical diagnostic ledger. The chosen positions
are an explicit retrospective fixture, not a discovered optimum.

Fresh Llama loading was deferred on 2026-10-20: available disk space was about
5.2 GiB, below the space available at an earlier swap-related reboot. Other
local experiments and saved outputs were not modified. Tests use synthetic
readouts and cover ID binding, source filtering, duplicate/error accounting,
state restoration, compact delivery and card dispatch; they are not new model
efficacy evidence.

Validation on 2026-10-20:
`python3 -m pytest SpiralInterventionLab/tests -q --tb=short` completed with
617 passed, 9 subtests passed and 14 dependency deprecation warnings.
`git diff --check` passed. No new Llama/GPT-2 checkpoint was loaded and no
provider/interview call was made in this verification.

## Matched-prefix observation on 2026-10-21

The provider-free Llama replay completed after disk space was freed. The local
checkpoint files matched the source manifest's SHA-256 values, and all three
decoded prefixes matched the recorded episode exactly. The run used the full
28-layer Llama-3.2-3B-Instruct checkpoint, float16, MPS, seed 0 and the L1 rewrite
fixture. No cap, model precision or decode policy was changed to obtain a match.

Receipts:

- [Readout measurements](../results/source_clause_diagnostic_llama_20261021/source_clause_readout.json)
- [Historical readout comparison](../results/source_clause_diagnostic_llama_20261021/readout_audit.json)
- [Source and receipt hashes](../results/source_clause_diagnostic_llama_20261021/sha256.txt)

| Generated tokens | Prefix ending | Greedy next piece | Comma rank | Comma gap to top-1 | ` while` rank | ` while` gap |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 7 | `to I` | `vo` | 62 | 17.492188 | 3025 | 24.296875 |
| 8 | `to Ivo` | ` before` | 2 | 1.125000 | 7 | 4.218750 |
| 9 | `to Ivo before` | ` sending` | 75 | 8.265625 | 787 | 11.511719 |

At prefix 8, the comma and ` while` ranks and gaps exactly reproduce the stored
survey's pre-intervention measurements. Those are the two overlapping
source-bound pieces at the surveyed position; this is not a claim that every
historical result was reproduced. The current source also contributes
` before` and `.\n`. It does not contribute the survey's unseen `;` or ` and`.

All three observations preserved tokens, cache, runtime step and active-edit
state. Each used one diagnostic slot: 12 -> 11 -> 10 -> 9. There were three
unedited physical replays, zero edited replays and zero provider calls.
`production_apply_allowed`, `certified_for_apply` and `policy_candidate_ready`
remained false. The harness selected the cards; Luna did not select them.

The narrow observation is a sharply position-dependent structural alternative
in this single trajectory: comma is much nearer the greedy readout after the
complete recipient name than immediately before or after that point. It does
not establish a safe operator, a causal rescue, a general clause-boundary rule
or task-score improvement. Prefix 7 is still inside the tokenized name `Ivo`.
The replay stops at prefix 9 rather than evaluating a new completed answer.

Resource note: available disk space started near 19 GiB, briefly fell to about
8 GiB during loading, and was about 11 GiB after process exit. No other local
experiment was stopped and no saved output was removed. MPS emitted its backend
correctness warning; the checkpoint/prefix/readout matches above are narrow
checks, not a validation of all MPS computations.

Post-run checks: `test_clause_structure.py` and `test_operation_cards.py`
completed with 36 passed and 14 dependency deprecation warnings.
`git diff --check` passed. Package versions and the dirty-source commit/branch
are recorded in
[environment.json](../results/source_clause_diagnostic_llama_20261021/environment.json).

## Next experiment

The three-prefix observation is now complete. The frozen choice protocol below
tests interface use before any new operator experiment.

Only after that interface test should an independently measured, normal-cap
structure-target operator be offered. Source attention/KV routes and
norm-relative cap policy remain separate research/design decisions. Visibility
alone is not a reason to authorize a punctuation-writing intervention.

## Frozen Luna choice protocol

The comparison is bounded to the retrospectively selected prefix 8. Selection
of this prefix uses the earlier observation; it is not a blind search for a
boundary. The worker reconstructs the prefix once without the historical
diagnostic ledger and exports three inputs:

- `off`: existing packet and offers, with only the source-clause offer and its
  explanatory hint removed.
- `offered`: the same packet with the additional optional observation.
- `observed`: the same generation position after the harness measured that
  observation, including the result and one spent diagnostic slot.

The old requests, order, costs and availability must match in `off` and
`offered`. Operation IDs bind whole packets, so IDs necessarily change across
conditions; the test compares their resolved request semantics, not ID bytes.
No forced-rescue continuation, offline role verdict, winning dose or preferred
separator is included. Pre-observation inputs must not contain measured
structure-readout rows.

`gpt-5.6-luna` uses the unchanged operation-card prompt and compact projection.
Each call is independent, with no conversational carry-over. The fixed order
is `off, offered, offered, off, observed, observed`: two observations per
condition, not enough to estimate a general selection effect. The result-shown
condition is **harness-supplied evidence**, not a claim that Luna first chose
the probe and then received its result in an interactive rollout.

The comparator loads no local model and dispatches none of Luna's selected
commands. It retains raw first replies, parse failures, input hashes and usage.
There is one attempt per call, no SDK retry, a 180-second request timeout and
at most six requests. API failures stop the batch and preserve the partial
receipt. A dry run validates and saves inputs without provider calls.

```sh
python3 -m SpiralInterventionLab.examples.replay_source_clause_diagnostic \
  --source-jsonl "$SRC/live/c1.jsonl" \
  --source-manifest "$SRC/manifest.json" \
  --worker-model-path "$LLAMA_MODEL_PATH" \
  --positions 8 --export-controller-packets \
  --output results/<new_dir>/frozen_prefix8.json

python3 -m SpiralInterventionLab.examples.compare_source_clause_choices \
  --source-receipt results/<new_dir>/frozen_prefix8.json \
  --controller-model gpt-5.6-luna --dry-run \
  --output results/<new_dir>/choice_preflight.json

python3 -m SpiralInterventionLab.examples.compare_source_clause_choices \
  --source-receipt results/<new_dir>/frozen_prefix8.json \
  --controller-model gpt-5.6-luna \
  --output results/<new_dir>/luna_choices.json
```

## Luna observations on 2026-10-21

The fresh prefix-8 export again reproduced comma rank 2 / gap 1.125 and
` while` rank 7 / gap 4.21875. The dry run passed before the six provider calls.
All six first replies parsed without a repair attempt and all frozen packets
were unchanged after invocation.

| Condition | Calls | Source-clause selections | Other diagnostic selections | Apply commands |
| --- | ---: | ---: | ---: | ---: |
| off | 2 | 0 | 0 | 0 |
| offered | 2 | 1 | 0 | 0 |
| observed | 2 | 0 | 0 | 0 |

The selected source-clause operation resolved to the exact offered context ID
and requested `hold_prefix`. The other offered-condition reply declined
diagnostics, citing improving progress and lack of a certified intervention.
The result-shown replies did not re-request the consumed probe. Neither reply's
short rationale establishes that the structural measurements changed its
decision; both cited progress and unavailable certified actions.

One `off` reply requested `hold_prefix` without an accompanying evidence request.
It is valid ControllerCommand JSON, but the existing `prefix_control.evaluate`
function rejects that hold with `no_fresh_requested_result`. This was checked
statically against the recorded command, not dispatched or silently repaired.
Parser success is therefore not a claim that all six clock decisions could
execute as written.

Important limitations:

- The fresh reconstructed context offered **zero** old diagnostic cards.
  `offered` added one; this tests uptake of an optional tool, not prioritization
  among competing diagnostic cards.
- This harness retains the earlier replay's **legacy** clock annotation.
  It does not test the explicit inspect/commit-clock interface.
- The task feedback at this prefix was `progressing`, partial score 0.694444,
  with four of nine required terms present. The measurements do not prove
  that choosing advance was irrational, or that the controller anticipated the
  later clause omission.
- Two calls per condition at one retrospectively selected position are not an
  estimate of reliability. The mixed offered-condition decisions are retained.
- The six choices dispatched no diagnostics, applied no edit and generated no
  worker continuation. The separate packet-export replay made one unedited
  observation. There is no new task-score or operator-effect result.

The evidence narrows the interface question: the source-bound tool can be
selected by exact ID, but optional availability does not guarantee uptake.
Result visibility alone supplies neither an executable structural operator nor
permission. Do not tune the prompt to make the recorded comma win. An explicit
clock / genuinely competing-menu comparison would be a separate interface
experiment; a normal-cap structure-target operator replay would be a separate
actuation experiment.

Receipts:

- [Frozen inputs and readout](../results/source_clause_luna_choices_20261021/frozen_prefix8.json)
- [Preflight](../results/source_clause_luna_choices_20261021/choice_preflight.json)
- [First replies and usage](../results/source_clause_luna_choices_20261021/luna_choices.json)
- [Choice audit](../results/source_clause_luna_choices_20261021/choice_audit.json)
- [Static hold-contract audit](../results/source_clause_luna_choices_20261021/hold_contract_audit.json)

Provider usage across the six calls: 72,692 input tokens, including 36,337
cached input tokens, and 961 output tokens; 73,653 total. No extra interview,
repair or retry call was made.

Post-comparison validation: the full suite completed with 623 passed,
9 subtests passed and 14 dependency deprecation warnings. New tests cover
off/on invariance, exact request preservation, corrupt or leaking snapshots,
six-call bounds, retained invalid first replies and stopping on provider error.

## Normal-cap operator connection

`source_clause_activation_patch_probe` is a separate, opt-in instrument behind
`--clause-structure-operator-diagnostics`, which also requires
`--clause-structure-diagnostics`. Observational runs are unchanged by default.

The controller receives one optional operation card per measured SOURCE token.
The card resolves to a context-bound operator ID, not a free-form patch. It does
not displace the existing frontier, nominate the comma, or prefer candidates by
historical rescue results. A successful source-clause observation at the exact
current prefix is required. Stale requests, missing cache, lost observations,
active patches and exhausted budgets do not create an executable offer.

Materialization binds the actual source token ID and its **first SOURCE
occurrence** directly. It does not retokenize punctuation through the entity
resolver or fall back to the constraint header. For each target, the recipe uses
the last *already offered* layer of `resid_pre` and `mlp_out`, where present.
This is not necessarily the model's final layer. No new surface is opened.
The fixed recipe is `activation_patch/blend`, alpha at most 0.04, step size at
most 0.08, each further bounded by the existing surface caps, and TTL 1.
There is no policy override or cap release.

One request costs at most one diagnostic slot and **four physical one-token
replays for one site, six for two sites**: two no-edit controls, then two repeats
per edited site. Every simulation must restore the prefix, logits, cache and
hooks before the next call. Preflight failures are free and consumed at that
context; failed physical attempts are retained and charged once. Hook execution
must be observed, not inferred from registration. Caps are checked unchanged.

The report retains exact source tensor identities, edit payloads, fingerprints,
hook telemetry, paired deltas and repeat variation. It measures all observed
source structure tokens to distinguish intended-token response from cross-token
lift. Effect labels require agreement in both repeats above observed repeat
variation; the mass-lift reporting threshold is 1e-4, or a top-20 entry.
Gap/rank-only movement remains distinct. This threshold labels diagnostic
evidence; it does not certify an actuator. Ownership is limited to the observed
structure-token set, not semantic roles or causal path fidelity.

One-token repetition/entropy/margin changes are reported, but clause correctness
and semantic safety remain **unassessed**. Even a target-aligned positive result
does not mint a frozen apply card, trial offer or production permission. A
positive punctuation logit is not a rescued rewrite.

For a bounded local pass, request every offered token in SOURCE order at one
recorded prefix, with no provider calls:

```sh
python3 -m SpiralInterventionLab.examples.replay_source_clause_diagnostic \
  --source-jsonl "$SRC/live/c1.jsonl" \
  --source-manifest "$SRC/manifest.json" \
  --worker-model-path "$LLAMA_MODEL_PATH" \
  --positions 8 --measure-source-operators \
  --output results/<new_dir>/normal_cap_structure_patch.json
```

This is harness-selected measurement, not evidence of controller uptake.
It is deliberately separate from the frozen Luna visibility comparison.

### Local attempt on 2026-10-21

The first normal-cap Llama attempt was stopped by an external supervisor during
model initialization, before the replay receipt was created. The supervisor
terminated only its own experiment child when free disk space fell below the
predeclared 8 GiB threshold (7.67 GiB at the end, after 35 seconds).
No controller calls or operator measurements were made.

This is an infrastructure interruption, not a negative operator result. The
earlier source-clause observations and frozen Luna choices remain unchanged.
Retained artifacts are the
[interruption record](../results/source_clause_activation_patch_llama_20261021/interruption.json),
[runtime log](../results/source_clause_activation_patch_llama_20261021/runtime.stdout.log)
and pre-run source snapshot. A fresh output path and resource preflight are
required before rerunning; do not overwrite this interrupted attempt.

Implementation validation after the interruption:
`python3 -m pytest SpiralInterventionLab/tests -q` completed with **641 passed,
9 subtests passed and 14 dependency deprecation warnings**. The 18 new tests
cover exact source-position binding, optional cards, existing-cap materialization,
paired hook execution, repeat-noise reporting, cross-token lift, missing-hook
failure, free preflight failure, attempted-request consumption, stale contexts,
opt-in CLI validation, and immediate stop on prefix or cap drift. These are
instrument-contract tests, not evidence that a structural actuator was found.

### Completed normal-cap retry on 2026-10-21

After disk space was freed, the same bounded replay completed in 118 seconds
in a fresh directory. The predeclared 8 GiB disk stop remained in place; the
lowest **recorded** free space was 11.44 GiB and the final reading was 12.45 GiB.
The earlier interrupted attempt is retained separately.

The natural no-edit prefix was again `" Nora will take the sample to Ivo"`
at eight generated tokens. All four structural readout rows exactly matched
the prior observation. The harness, not Luna, requested every offered SOURCE
token in source order. The default catalog offered only `resid_pre L14` for
this diagnostic, so this is **one site, four source tokens**, not a late-layer
sweep or the earlier high-dose L25-L27 rescue.

All edits used `activation_patch/blend`, alpha 0.04, step size 0.08 and TTL 1,
without changing existing caps. The run made one observational replay and
16 operator replays (two no-edit plus two edited per target). Five diagnostic
slots were spent and seven remained. All eight edited repeats recorded one
hook invocation; all receipts passed state-restoration and unchanged-cap checks.
Within each repeated condition, maximum absolute logit variation was zero.

| Intended SOURCE token | Source position | Intended probability delta | Largest observed structure-token lift | Ownership | Effect |
| --- | ---: | ---: | --- | --- | --- |
| ` before` | 74 | -0.00131961 | comma: +0.00242949 | `cross_bound` | `bound_readout_regression` |
| `,` | 76 | +0.00394066 | comma: +0.00394066 | `target_aligned` | `bound_target_lift_candidate` |
| ` while` | 86 | +0.00018236 | comma: +0.00402286 | `cross_bound` | `bound_target_lift_candidate` |
| `.\n` | 96 | +0.00000413 | comma: +0.00155482 | `cross_bound` | `no_resolved_target_response` |

These are probability differences, not percentage changes. The comma's own
patch adds about **0.394 percentage points** of probability; its rank remains
2. Its logit gap to the greedy winner narrows from 1.125 to 1.09375.
No target changes rank or gains a top-20 entry, and **all eight edited repeats
still choose ` before` greedily**. No full continuation or task score was
evaluated.

This is not only rank visibility: a normal-cap patch produces a repeatable
bound-token probability response. However, all four sources lift the comma
most among the observed structural tokens. The ` while` source even produces
a slightly larger comma response than the comma source. Thus
`target_aligned` describes the within-row alignment of intended target and
largest measured response; it does **not** establish source-specific
superiority. Shared separator-directed movement is a hypothesis, not a
demonstrated semantic mechanism or a rescued clause.

All rows saturate the step-size cap, with logged blend norms
0.07997803-0.08002315 around the configured 0.08 (fp16 rounding). No measured
one-token repetition increase or collapse-sharpening signal appeared.
Nevertheless, semantic safety and clause correctness remain `unassessed`;
all certification and production-permission flags remain false. No provider
calls, controller choices or rollout edits occurred in this retry.

Numerical scope also matters: TransformerLens warned about possible incorrect
MPS results with the installed PyTorch 2.12.1 backend. Identical repeated MPS/fp16
outputs do not validate another device or precision. Probability deltas use
CPU float64 softmax arithmetic on these logits; displayed readout probabilities
use the existing readout path and can differ slightly by rounding. Neither
eliminates the need for an independent numerical control.

The next comparison should therefore test **source specificity at fixed
prefix, site and cap**, rather than increase strength immediately: compare the
comma source with matched other-source controls and a norm-matched neutral
control, keeping the full observed-token response matrix. Separately validate
the small response on another supported numerical path before treating it as
robust. These are proposed experiments, not new permissions or completed tests.

Artifacts:

- [Raw replay receipts](../results/source_clause_activation_patch_llama_20261021_retry1/normal_cap_structure_patch.json)
- [Audit and response matrix](../results/source_clause_activation_patch_llama_20261021_retry1/audit.json)
- [Offline audit builder](../results/source_clause_activation_patch_llama_20261021_retry1/audit_run.py)
- [Runtime and supervisor log](../results/source_clause_activation_patch_llama_20261021_retry1/runtime.stdout.log)
- [Artifact checksums](../results/source_clause_activation_patch_llama_20261021_retry1/sha256.txt)

## Source-specificity controls

`--source-specificity-controls`, together with `--measure-source-operators`,
adds two offline controls at the same offered site, alpha, cap and prefix:

- `self_copy`: read the same activation position that the hook will write.
  The source-target distance and actual blend displacement must both be zero
  within 1e-6; otherwise the instrument fails closed.
- `radial_shrink`: blend toward a zero vector, so the displacement removes
  activation norm without importing a SOURCE token's direction. This is an
  active control, not a semantically neutral intervention. Actual displacement
  norms are compared to each SOURCE patch (2% reporting tolerance); matching
  total magnitude does not match its radial component or prove mediation.

All SOURCE targets are still measured in occurrence order. Comparison retains
each target's own-source response, strongest other-source response, radial
response, self-copy response, repeat floors and actual norms. These continuous
contrasts do not confer statistical significance or apply permission. Neither
control is registered as a controller operation card or added to its evidence
ledger.

The complete matrix must fit the diagnostic budget before SOURCE measurements
begin; two slots are reserved for the controls. The four-token, one-site pass
costs eight slots and 26 forwards: one cache-alignment forward, one structural
observation, four requests of four forwards each, and two controls of four
forwards each. With two offered sites, the upper bound is 38 forwards. No
production budget is enlarged.

### Invalid first control attempt retained

The first control run completed, but its `self_copy` moved the activation by
0.08000007 with source-target distance 12.78097248. This is not a valid null
control. Its original `status=complete` receipt is retained alongside an
explicit [post-hoc invalidation](../results/source_clause_specificity_llama_20261021/posthoc_validity.json);
it must not be interpreted as a source-specificity result.

The cause was temporal binding, not a semantic effect of copying an activation
to itself. A generation step appends its chosen token while retaining the cache
from the forward that predicted it. Thus a cache-relative `last` reference can
precede the current prefix's write position. The source-bound patches had used
explicit SOURCE positions; this new control incorrectly assumed `last` meant
the current forward's last position.

The corrected harness explicitly refreshes the cache with one charged,
no-edit current-prefix forward **before** taking the comparison snapshot.
It records old/new context fingerprints and unchanged tokens, step and caps.
Controls reject stale cache token identities, and verify actual null displacement
after hook execution. This is a local measurement preparation, not a global
change to runtime reference semantics or production authority.

### Aligned control result

The corrected run completed in 150 seconds, at the same natural prefix and
`resid_pre L14`, alpha 0.04, step size 0.08. It used 26 forwards and eight of
12 diagnostic slots; four remained. Every edited repeat executed one hook.
All state/cap checks passed. Self-copy's source-target distance, blend
displacement and observed-token probability deltas were exactly zero.
The four SOURCE responses reproduced the previous normal-cap measurements.

| Input condition | Comma probability delta |
| --- | ---: |
| Self-copy, aligned current position | 0.00000000 |
| Radial shrink, matched total displacement | -0.00046511 |
| ` before` SOURCE position | +0.00242949 |
| Comma SOURCE position | +0.00394066 |
| ` while` SOURCE position | +0.00402286 |
| `.\n` SOURCE position | +0.00155482 |

Thus, in this specific comparison, the positive comma response is not reproduced
by a pure norm reduction of the same total magnitude. Some directional input
matters. But the comma source does not beat the strongest other SOURCE input:
its advantage over the ` while` source is **-0.00008220**. It would be premature
to call this a comma-specific actuator.

There is a smaller, separate sign of input specificity: ` while`'s own-source
probability response is +0.00018236 versus at most +0.00004802 from another
SOURCE input, an advantage of +0.00013434. Yet that same while patch lifts comma
much more than while. **Between-source specificity and within-row ownership are
different axes**, and neither implies a greedy decision change.

All conditions still choose ` before`; ranks and top-20 membership remain
unchanged. There is no full-continuation or task-success result, no controller
call, and no production apply. Repeated MPS/fp16 outputs remain subject to the
numerical caveat above. These are deterministic local contrasts, not statistical
confidence estimates or cross-backend validation.

Artifacts:

- [Aligned raw receipts](../results/source_clause_specificity_llama_20261021_aligned/specificity.json)
- [Audit and six-condition response matrix](../results/source_clause_specificity_llama_20261021_aligned/audit.json)
- [Command and predeclared bounds](../results/source_clause_specificity_llama_20261021_aligned/manifest.json)
- [Run status](../results/source_clause_specificity_llama_20261021_aligned/run_status.json)
- [Test output](../results/source_clause_specificity_llama_20261021_aligned/tests.txt)

Post-control validation: `python3 -m pytest SpiralInterventionLab/tests -q`
completed with **650 passed, 9 subtests passed and 14 dependency deprecation
warnings**. Regression coverage includes stale-cache rejection, charged cache
alignment, zero self-copy displacement, repeat accounting, missing controls,
cross-context rejection, and reserving the complete comparison's budget.

## Wider interpretation

1. **Research:** the current evidence supports local structural-token
   sensitivity, not a successful rewrite controller. Separate source identity,
   displaced activation magnitude, output ownership and eventual clause
   completion. The same positive delta cannot certify all four.
2. **Engineering:** a token selector such as `last` needs a capture-time
   contract. Current write position and cached read position are not
   interchangeable. The new null control caught a measurement-construction
   error that repeatability and state-restoration checks alone did not catch.
   Explicit historical reads can still be legitimate; do not globally change
   their semantics to hide this distinction.
3. **Next bounded experiment:** validate the instrument on another numerical
   path, then compare a few source-direction controls at the same site/cap
   before increasing dose or opening new production permissions. Only after a
   source-selective, target-owned effect survives those controls should a
   separate continuation experiment ask whether it changes the clause.
