# Explicit inspection and generation clocks

## Intent

An edit `noop` must not silently mean that a diagnostic question has been
answered, or that the next greedy token is already accepted. The legacy loop
generates its first token before consulting the controller, and same-prefix
follow-ups depend on usable evidence or a recognized diagnostic transition.

The opt-in `--generation-clock-mode explicit` separates those clocks. It uses
the same runtime dispatcher, compiler, physical guardrails, diagnostic ledger,
trial validator and token-based TTL accounting. `legacy` remains the default.
This is not a new operator or a grant of production apply permission.

## Contract

The controller observes the initial answer boundary before the first token.
The hooked worker performs one initial prefill to populate activation cache,
readout metrics and task feedback without sampling or emitting a token. This is
observation work, not a new diagnostic or edit allowance. It is idempotent within
the initial boundary, has no TTL/status-progress tick, and resets per episode.
Explicit mode rejects runtimes without this preparation capability before reset;
the legacy clock remains usable for those backends. No cold-packet fallback is
allowed.
Every command includes a `meta.generation_action`:

- `inspect_prefix` requires edit decision `noop`. Diagnostics and existing tools
  may run, or the controller may use a bounded reasoning round without a tool.
  Generation and edit TTL do not advance. Failed, empty and duplicate diagnoses
  still return control at the same prefix, without being counted as new evidence.
- `commit_token` advances exactly one greedy token. It contains no diagnostic,
  tool or observer request. Its edit decision is independently `noop`, `apply`
  or `rollback`, subject to all existing authorization and physical checks.

The existing candidate-handoff allowance supplies at most two inspection rounds
per prefix, followed by a commit decision. Explicit mode does not stack legacy
rethink or positive-handoff rounds on top. Diagnostics remain capped at twelve
in the paired profiles; an inspection round is not a free physical replay.
At zero rounds, operation cards are unavailable for inspection. Trial edit
offers remain separate and can still be explicitly accepted or declined.

Unknown or contradictory time actions do not dispatch diagnostics or apply
edits. The provider uses its existing bounded JSON-repair path. Direct clients
get only the remaining bounded correction rounds; if those are exhausted, the
runtime records a `research_budget_fallback` commit with no new edit rather than
pretending the controller requested it. Provider errors still stop the run.
Legacy `hold_prefix`/`advance` are accepted as aliases, not preferred spellings.

Generation-prefix mutation during inspection, or explicit evidence that replay
failed to restore state, stops the episode before committing another token.
Hooks applied at commit are observed and expire only after actual generation.
No rollback, resampling, beam search or decoder masking is introduced.

## Observable clocks

- `generation_control` includes the generated token count, controller round,
  remaining inspection rounds and the last time-action report.
- `initial_observation_prepared` records the initial forward separately from
  generated-token, diagnostic and edit accounting.
- `controller_clock_decision` records requested versus effective phase and any
  rejected action, independently of edit selection.
- `prefix_inspection_complete` records results, zero generated tokens, and the
  remaining round budget even when there is no new usable evidence.
- `generation_commit` records who authorized advancement and the final edit
  decision; `generation_token_committed` confirms the actual token and TTL tick.

The paired-run audit checks the event order, contiguous token counts, at most
three controller decisions per prefix, and a first decision at token count zero.
It separately records diagnostics, measured cards, trial offers, applied edits
and task outcomes. A longer inspection sequence is not evidence of success.

## Reproduction

```sh
python3 -m SpiralInterventionLab.examples.run_iteration_pair \
  --profile gpt2_control --worker-model-path "$GPT2_PATH" \
  --controller-model gpt-5.6-luna --controller-action-view cards \
  --generation-clock-mode explicit --candidate-handoff-mode soft \
  --output-dir "$OUTPUT_GPT2"

python3 -m SpiralInterventionLab.examples.run_iteration_pair \
  --profile llama_l1 --worker-model-path "$LLAMA_PATH" \
  --controller-model gpt-5.6-luna --controller-action-view cards \
  --generation-clock-mode explicit --candidate-handoff-mode soft \
  --output-dir "$OUTPUT_LLAMA"
```

Each profile runs a fresh unedited B0 followed by C1, without a prompt baseline
or injected historical debrief. Profiles have different tasks, depths and dtypes;
their scores must not be pooled as a cross-model efficacy comparison. Turning
on cards and the clock together is a combined architecture experiment, not a
causal isolation of either change. Local model paths remain CLI-supplied.

## Checkpoint restoration

The GPT-2 directory contained its configuration and tokenizer but lacked weights.
With explicit user approval, only `model.safetensors` was fetched from
`openai-community/gpt2`, revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
Its SHA256 is
`248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`.
Existing configuration/tokenizer files were not replaced. Later model loading
uses offline mode; no weights or machine-specific paths are committed here.

## Validation scope

Tests cover first-token access, same-prefix inspection after failure, duplicate
request handling, bounded fallback, no inspection-time apply, mutually exclusive
commit/diagnostic requests, one-token TTL, failed restoration, prefix mutation,
parser repair, exhausted menus and unchanged legacy behavior. A synthetic valid
edit verifies the clock/TTL path; it does not certify a live operator.
The final local suite passed 526 tests and nine subtests (15 dependency warnings).
Initial-prefill tests check cache/readout availability, no generation or budget
charge, per-episode reset and unsupported-backend rejection before mutation.

## First pass before initial prefill, 2026-09-28

The first [GPT-2 paired receipt](../results/explicit_clock_gpt2_20260928/receipt.json)
records 23 controller commands: 12 inspection rounds and 11 explicit token
commits, with no fallback commits or parser retries. The first decision saw
zero generated tokens and chose to commit. Follow-up inspection found that this
initial packet had cold readout/task feedback: scheduling a decision before the
first token did not by itself make the boundary observable. This pass is retained
as evidence for that implementation gap, not an informed first-boundary choice.

A frozen measured card appeared at generated-token count 4 after eight diagnostic
slots. Luna subsequently selected one `measure_normal_cap_current_prefix` and
one `remeasure_current_prefix`, each executing four physical replays. All twelve
diagnostic slots were used. There were no trial offers or applied rollout edits.
B0 and C1 both produced ` In the case of a rewrite, the budget draft.` with score
0.5875 and task completion false. Unedited output identity is a useful integrity
check here, not proof that all diagnostic side effects are impossible.

The post-run interview described weak bound-target lift, rank-only movement and
insufficient current-context evidence. This is a qualitative pointer for later
work, not an independent finding or permission to relax certification. The new
clock is exercised; improved task performance is not demonstrated. Earlier
GPT-2 runs used different recorded contexts, so their scores are not a matched
before/after control for this architecture change.

The corresponding [Llama L1 first pass](../results/explicit_clock_llama_20260928/receipt.json) likewise emitted no rollout edits or trial
offers. Its first measured card arrived after six diagnostic requests at token
count 3; two current-prefix remeasurements ran eight physical replays total.
B0/C1 both scored 0.938889 and emitted ` Nora will take the sample to Ivo before
sending the report to Yara before dusk.` The task still failed. Ten inspection
rounds and eighteen explicit commits completed, without fallback. The runs below
must recheck the clock after initial prefill, rather than pooling these cold
boundary observations as corrected-run evidence.

## Prefill-corrected observation, 2026-09-28

The [corrected GPT-2 receipt](../results/explicit_clock_gpt2_prefill_20260928/receipt.json)
records a real initial prefill, followed by two inspections at generated-token
count zero and an explicit commit. A frozen measured card appeared at token
count 1 after four diagnostic slots. This is earlier than the first pass
(count 4 / eight slots), but these are two controller trajectories, not a
controlled estimate of the prefill change's average effect.

There were still no trial offers or rollout edits, and B0/C1 output and score
remained identical (0.5875, task incomplete). The controller used all twelve
diagnostic slots but selected no candidate-action remeasurement; the earlier
first pass selected two. Thus earlier visibility did **not** establish better
choice or actuation. Input usage was 520,944 tokens across 23 decisions, excluding
the debrief, versus 442,881 in the first pass. This is not a prompt-cost win.

The model's post-run account points to small local responses, missing bound-target
lift and binding constraints. Treat that as a qualitative lead, not evidence that
relaxing a gate would help. The next controlled question is which valid
current-prefix action was offered after a card appeared, and what the controller
actually selected. Do not conflate an absent trial offer with refusal to use one.
The recorded handoff rows themselves report `bound_target_lift` blockers and,
for alternate bindings, `canonical_binding`; this part is log evidence rather
than only interview interpretation.

The [first prefill-corrected Llama attempt](../results/explicit_clock_llama_prefill_20260928/receipt.json)
stopped after five committed tokens: the provider mistyped a packet-bound
operation ID on three attempts. Exact resolution rejected all three; there was
no fallback execution and no completed C1 score. This failed attempt is retained,
not hidden by the later retry. ID-error feedback now repeats available current
IDs in existing menu order (up to 32), with the option to decline. It neither
autocorrects IDs nor adds a retry, operation, diagnostic allowance or preference.

The [Llama retry receipt](../results/explicit_clock_llama_prefill_retry_20260928/receipt.json)
completed with 33 first-attempt valid commands, 15 inspection rounds, 18 explicit
commits and no fallback. All twelve diagnostic slots were spent. No frozen
measured card, trial offer or rollout edit appeared. B0/C1 again shared the same
output and score 0.938889, with task completion false. No ID repair was needed in
this retry, so it cannot quantify the new retry note's effectiveness.

At token counts 5, 6 and 7, `entity_insertion_operator_candidate_review` returned
`no_candidate_blueprints` and charged one diagnostic slot each. These were
different prefixes, not established duplicate executions at one prefix; the
rows do not report a physical replay count. This is a concrete next audit target
for offer availability, objective lifetime and accounting, not proof that all
three charges were invalid. Input usage was 587,152 tokens, including 196,638
cached tokens; the interview is excluded. Its heavily truncated view is not
used to reconstruct the complete event sequence.

## Reading and next experiment

| Corrected run | First measured card | Inspection / commit | Trial / rollout edits | B0 / C1 score |
| --- | --- | --- | --- | --- |
| GPT-2 | token 1, four diagnostic slots | 12 / 11 | 0 / 0 | 0.5875 / 0.5875 |
| Llama L1 retry | none | 15 / 18 | 0 / 0 | 0.938889 / 0.938889 |

The time contract is exercised in both models: inspect without advancing, then
explicitly commit. It does not establish general handoff improvement: GPT-2
exposed a card earlier while the completed Llama retry did not expose one at all.
The score scales belong to different tasks and are not compared across models.
The GPT-2 receipt predates the unsupported-backend/ID-repair follow-up; it used a
prefill-capable worker and had no failed provider replies. The Llama retry uses
the final code. Raw traces and sealed manifests remain local; committed receipts
carry their checksums rather than embedding checkpoint paths or large payloads.

Next, audit no-blueprint review offers at a fixed prefix, then compare current
measurement versus historical review on the same offered candidate and budget.
Measure physical work, first card, trial offer, controller choice and committed
effect separately. Do not add operators or loosen certification just to make
the applied-edit counter nonzero. A later legacy/explicit timing ablation should
hold the interface and prefix fixed rather than attributing these combined runs
to one change.

## Follow-up: recorded-evidence review accounting (2026-10-10)

The no-blueprint audit above is now resolved at the review-contract level.
The three reviews at token counts 5, 6 and 7 all read the same step-zero probe:
Nora, take and sample had become satisfied, while Ivo was already top-20 in
that recorded probe. The review itself did not execute a model forward or
operator replay. It combined historical readout rows with current missing-term
feedback. Prefix advancement alone did not make its readout evidence fresh.

Two runtime defects compounded this mismatch: `recorded_step=0` became `-1`
through a truthiness fallback, leaving a completed review on the frontier;
and empty reviews inherited the default one-slot diagnostic charge. An empty
missing-term list also incorrectly meant "missing status unknown" rather than
"all terms satisfied" in this review.

The repair binds a review to its recorded probe and the relevant current
missing-term set. Unchanged inputs are not offered again; changed evidence or
relevant feedback can reopen the review. A no-blueprint preflight is listed as
blocked rather than executable. Direct empty/reused requests cost zero, while
a new successful blueprint review retains its existing one-slot cost. Empty
receipts remain in diagnostic history to invalidate older positive blueprints;
budget accounting sums their explicit costs, not the number of retained rows.
Operation cards and compact results explicitly distinguish recorded-evidence
review from current-prefix measurement. No apply gate or inspection cap changed.

[The fixed-evidence audit](../results/entity_candidate_review_audit_20261010.json)
reuses the original trace and its 12-row/12-slot settings. Each review starts
from its historical remaining budget; all other diagnostic receipts are held
fixed, not rerun or rescored.

| Token count | Review outcome before / after | Old charge | New charge | New offer |
| --- | --- | --- | --- | --- |
| 0 | candidate blueprints ready / unchanged | 1 | 1 | yes |
| 4 | candidate blueprints ready / unchanged | 1 | 1 | yes |
| 5 | no candidate blueprints / unchanged | 1 | 0 | no |
| 6 | no candidate blueprints / unchanged | 1 | 0 | no |
| 7 | no candidate blueprints / unchanged | 1 | 0 | no |

All five blueprint lists and review matrices match the original results. The
three empty charges disappear without changing candidate eligibility. This is
an offline accounting/availability result, not a new Luna/Llama rollout, a
saved-token estimate, or evidence of improved task score. The original raw
trace and its failed task outcome remain unchanged. Regression coverage lives
in `SpiralInterventionLab/tests/test_entity_candidate_review.py`; the next live
comparison should still hold prefix, 12-slot budget and apply gates fixed.

## Live follow-up: cards reached, discovery stops early (2026-10-10)

After local capacity recovery, the Luna/Llama L1 pair completed in about
382 seconds. The earlier interrupted attempt never reached baseline completion
and is not counted as an efficacy result. The new sealed run is
`results/entity_review_accounting_llama_20261010_081550/`; its
[comparison receipt](../results/entity_review_accounting_llama_20261010_081550/comparison.json)
checks prompt, checkpoint hashes, dependencies, seed, policy arguments and all
18 pre-commit prefixes against the completed September 28 run. Fresh B0 also
matched the reference exactly. No runtime code changed during this pair.

| Observation | Previous Llama run | Accounting-repair rerun |
| --- | --- | --- |
| Charged diagnostic slots | 12 / 12 | 8 / 12 |
| Empty entity-review calls / charges | 3 / 3 | 0 / 0 |
| First measured card | none | token 3, after 7 slots |
| Distinct piece-bound frozen cards | 0 | 2 |
| Explicit current-prefix card remeasurements | 0 | 1, at token 4 |
| Inspect rounds / generation commits | 15 / 18 | 8 / 18 |
| Trial offers / rollout edits | 0 / 0 | 0 / 0 |
| B0 / C1 score | 0.938889 / 0.938889 | 0.938889 / 0.938889 |

The diagnostic path first diverged at token 3, before the old empty-review
steps. Instead of another generic operator replay, Luna selected the offered
source-localization comparison at ` Nora will take`. It measured two source
constructions with six physical replays, four piece-bound observables and
restored state. Two cards were frozen for ` sample` and ` Sample`; these share
one source construction and are not two independent successful actuators.
The centered-source rows were not carded because v1 requires an uncontrasted
recorded term-token source. These are separate availability and efficacy facts.

At token 4 (` Nora will take the`), Luna explicitly remeasured the canonical
card with four physical replays. The hook ran once, state was restored, and
both baseline and edited continuations were ` sample`. Target probability
changed by -0.0001308396, target logit by -0.015625, and rank stayed at 1.
Neither trial permission nor a rollout edit followed. The matched probe and
this remeasurement explicitly report three new measurements and ten physical
replays in total; other generic diagnostic work is not included in that count.

At tokens 5-17, four diagnostic slots remained. Empty entity review was blocked
and not reoffered, but every executable menu item concerned the existing
`sample` cards. The output already contained `sample` from token 5 onward.
The final missing term was `Selim`, not `sample`; task completion remained false.
The early-probe offer currently requires `steps <= 4` and no recorded entity
probe. Thus remaining budget does not imply an available fresh discovery action.
The next design target is bounded, objective-aware probe refresh after recorded
evidence is exhausted, not weaker apply gates or a forced intervention.

This is one controller trajectory, not a causal estimate of the repair. Card
arrival preceded the historical empty-review charges, so it cannot be explained
solely as spending those three recovered slots. API input tokens decreased from
587,152 to 545,223, but cached tokens changed from 196,638 to zero; this is not
evidence of lower API cost. MPS numerics have not been independently qualified
against CPU.

The post-run interview remains supplementary. Its "within repeat variation"
description should not imply noisy controls: recorded repeat/no-edit variation
was zero, while card-target logit movement was zero or negative. Its request for
a task-completion reason concerns the interview view; runtime task feedback
explicitly reported the missing `Selim`. Preserve raw trace evidence over these
free-text interpretations.

## Bounded missing-term probe refresh (2026-10-14)

The next change adds an offer, not a forced controller action. After an initial
entity probe, if its recorded rows no longer produce blueprints for currently
missing terms, `entity_probe_refresh` may expose a fresh
`target_entity_insertion_probe` operation card. The card names up to four
currently missing terms absent from episode probe coverage. Term spelling and
order come from task feedback; no entity-specific prior is introduced.

The request binds a `probe_refresh_id` and exact terms through operation-card
resolution and dispatch. Availability checks are pure. A selected refresh runs
one readout forward, costs one diagnostic slot, restores the readout cache and
does not advance generation. It reports one new readout measurement and zero
edited physical replays. A subsequent candidate review consumes the newly
recorded probe, not the initial probe. Existing measured cards remain available.

Coverage survives diagnostic-history eviction and clears on episode reset.
Stale IDs, altered terms and duplicate requests in a batch are rejected without
measurement or charge. An attempted forward that fails is charged as work but
is not positive evidence; its terms remain attempted to prevent automatic
retry loops. Unknown task feedback and exhausted budget expose no executable
refresh. Prefix advancement alone does not reset coverage.

This is deliberately narrower than general retargeting: previously probed
terms are not automatically remeasured, and pending recorded blueprints take
precedence over offering untouched terms. Review, replay, frozen-card creation,
confirmation and production trial remain separate stages. The change grants no
new apply authority and does not retire existing cards. Regression validation:
551 tests passed, plus 9 subtests (15 dependency deprecation warnings).

### Completed Luna/Llama retry

`results/entity_probe_refresh_llama_20261014_retry1/` completed in about
303 seconds. The earlier `entity_probe_refresh_llama_20261014_170447/` attempt
was interrupted after 15 generated tokens without an `episode_end`; its raw
trace is preserved and excluded from completed-outcome comparisons.
The [comparison receipt](../results/entity_probe_refresh_llama_20261014_retry1/comparison.json)
verifies unchanged checkpoint, fixture, versions, CLI policy settings and all
18 pre-commit prefixes against the completed accounting-repair run. Fresh B0
matched its reference, and the source hashes remained unchanged during the run.

| Observation | Accounting-repair run | Probe-refresh retry |
| --- | --- | --- |
| Charged diagnostic slots | 8 / 12 | 12 / 12 |
| Missing-term refresh calls | 0 | 1, at token 5 |
| New-term blueprints after refresh | 0 | 3, at token 6 |
| First measured card | token 3, after 7 slots | token 3, after 7 slots |
| Distinct piece-bound frozen cards | 2 | 2 |
| Trial offers / committed edits | 0 / 0 | 0 / 0 |
| B0 / C1 score | 0.938889 / 0.938889 | 0.938889 / 0.938889 |

At token 5 (` Nora will take the sample`), Luna selected the bounded refresh for
`Selim`, `send`, `report` and `Yara`. This was one readout forward with restored
cache, not an edited replay. The observed first pieces ranked as follows:
` Y` (Yara) 19, ` report` 41, ` send` 64 and ` Sel` (Selim) 119. These are
first-piece ranks, not evidence that either multi-piece name was completed.
At token 6, candidate review used that step-5 probe and emitted blueprints for
`report`, `send` and `Selim`. The log correctly reports
`source_probe_is_current_prefix=false` and `new_measurement_count=0` for this
review. Production and certification permissions stayed false.

The last four slots were spent on a second `sample` review at token 4, a
`sample` replay at token 5, the fresh missing-term probe, and its candidate
review. Budget then reached zero: no replay measured the three new candidates,
and the only frozen cards were still the two earlier `sample` bindings.
All 30 provider commands parsed, with 12 inspection rounds, 18 generation
commits and no fallback commit. The final output was unchanged and still missed
`Selim`; this is a discovery/blueprint handoff result, not an efficacy gain.

The next accounting boundary is now directly observable. The token-0 and
token-4 `sample` blueprint lists are exactly equal, but their review input IDs
differ and the latter review consumes one slot. Review-input novelty therefore
does not establish candidate novelty. Its follow-up replay runs at a new
prefix, so it is not an identical-measurement duplicate; nevertheless, the
generic offered request still resolves the old `sample` blueprint after
`sample` is already present. A follow-up should distinguish unchanged blueprint
review from explicit current-prefix remeasurement and bind replay offers to
their evidence/candidate generation. It should not silently force a new term
or weaken apply gates. This run does not establish that saving those slots
would produce a successful actuator.

Controller usage excluding the interview was 621,478 input and 3,960 output
tokens, with zero cached input tokens. The post-run interview omitted 344 trace
lines and did not discuss the new refresh-to-blueprint chain; use the raw
diagnostic receipts for that conclusion, not the interview's selected view.
This remains one completed trajectory per revision, with no independent CPU
qualification of MPS numerics.

### Interview coverage and blueprint novelty repair

The follow-up separates review-input identity from candidate identity. Blueprint
novelty uses the objective term, target piece/token, source provenance/span and
recommended operator families. Rank-only changes and a shorter missing-term
list do not create a new candidate. Episode-scoped known IDs survive receipt
eviction and reset with the episode; only a review with at least one structurally
new blueprint consumes a diagnostic slot. Fresh probes or changed candidate sets
can still refresh a review for free. Unchanged outputs from the same probe are
not offered again or retained as a fresh candidate generation for replay routing.
Current-prefix physical remeasurement and its accounting are unchanged, as are
all production-apply gates. This is not a refund of historical run budgets.

The interview sampler now keeps first/last missing-term refresh and candidate
review anchors, the last operator replay, the first repeated-blueprint review,
budget exhaustion, and the last recorded task feedback. Related context is the
nearest preceding same-step command/request context, not an asserted transaction
identity. Observation-ledger copies are not relabeled as fresh measurements.
Aggregate term-set effects, signed bound-piece effects and repeat/no-edit controls
remain distinct. The prompt explicitly rejects interpreting a false positive-lift
gate as proof of measurement noise. B0/C1 termination logs now include the existing
task evaluator's final-output feedback; historical logs without it are labeled as
last-observed-prefix feedback rather than silently upgraded to final checks.

An offline rebuild of the completed retry is saved in
`results/debrief_transition_audit_20261014/`. The digest changes from 17,078 to
19,470 serialized characters, within the unchanged 20,000-character ceiling.
It exposes `c1.jsonl:L195` (missing-term refresh), `L219` (three new blueprints
and exhausted budget), `L348` (last observed missing `Selim` feedback), and
`L158` (one charged review with no structurally new blueprint). Detailed related
commands are trimmed before transition anchors, with omissions recorded.

The original input, JSONL hashes and interview are preserved. The comparison is
deterministic and uses zero provider calls and zero worker runs. It establishes
better evidence coverage and identifies one redundant blueprint-review charge;
it does not establish that the saved slot produces a useful actuator, a trial,
or task improvement. A fresh fixed-prefix, 12-slot live comparison remains needed.

Validation: `python3 -m pytest SpiralInterventionLab/tests -q` passed 562 tests
plus 9 subtests, with 15 dependency deprecation warnings. `git diff --check`
passed. Tests cover feedback-only input changes, probe updates without blueprint
novelty, binding/span/family changes, history eviction/reset, explicit-prefix
measurement separation, bounded transition coverage and immutable offline rebuilds.

### Receipt revisions and refresh pending repair (2026-10-15)

The receipts above were produced by intermediate revisions, not by the merged
accounting. Both live runs record `entity_candidate_review.py` sha `0d3e996e…`
(merged: `ebff926f…`) and different `worker.py` hashes. At that revision a
review with changed inputs and any blueprints cost one slot. The 10-10 audit
table and the retry's token-4 review therefore show a one-slot charge for an
unchanged `sample` blueprint (`c1.jsonl:L158` repeats the token-0 blueprint).
Under the merged novelty rule that review costs zero and is not offered, which
would have left one slot after the token-6 review. No live receipt exercises
the merged accounting yet; the next fixed-prefix comparison should.

A follow-up repair concerns refresh availability. Pending recorded candidates
previously counted every blueprint from the latest probe, including ones
already reviewed. A failed but reviewed `Selim` blueprint could then keep the
refresh blocked as `recorded_candidates_pending` while other missing terms
were never probed. Pending now means blueprints not yet reviewed in the
episode. Regression: `test_reviewed_blueprints_no_longer_block_refresh_for_unprobed_terms`.
