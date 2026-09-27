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

The [first prefill-corrected Llama attempt](../results/explicit_clock_llama_prefill_20260928/receipt.json)
stopped after five committed tokens: the provider mistyped a packet-bound
operation ID on three attempts. Exact resolution rejected all three; there was
no fallback execution and no completed C1 score. This failed attempt is retained,
not hidden by the later retry. ID-error feedback now repeats available current
IDs in existing menu order (up to 32), with the option to decline. It neither
autocorrects IDs nor adds a retry, operation, diagnostic allowance or preference.
