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
