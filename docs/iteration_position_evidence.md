# Position evidence and two-worker iteration checks

## Scope

The frozen-candidate action contract now exposes **where evidence was measured**,
not where an intervention should win. `current_position` contains the generated
token count, a bounded visible prefix tail, last decoded piece, and trailing
surface category (empty, whitespace, punctuation, open text). A visible suffix
is not a certified word boundary: word completion remains unknown without
lookahead. No future token, reference completion, or target-specific boundary
list is consulted.

Each candidate card carries its last two completed measurement contexts, raw
bound-token metrics, restoration status, and output-token distance from the
current prefix. Other sources, token pieces and doses never share this history.
Prefix divergence yields an unknown distance rather than a misleading count.
History is chronological, not filtered to favorable outcomes. Omitted older
contexts are counted; a `hold`, review, failed attempt or cache read cannot
manufacture a new measurement.

The controller still chooses `review_existing`, `remeasure_current_prefix`, or
`hold`. Position facts neither auto-request diagnostics nor block any formerly
available choice. A fresh measurement still costs one diagnostic slot and four
paired one-token replays. All apply gates remain unchanged.

## Interview coverage

The post-run digest now includes first and last current-prefix measurement
anchors, explicit review/hold/blocked anchors, and aggregate counts of executed
actions and physical replays. Each measurement anchor retains its candidate,
context, position, controls and restoration result. This repairs a blind spot
in the earlier interview input; it does not retroactively rewrite old memos or
make free-text explanations evidence of internal reasoning or task success.

## Repeating the checks

Use two independent, fresh B0/C1 pairs after a substantive implementation. These
are model-local regression profiles, **not a controlled cross-scale comparison**:

| Profile | Worker | Task / seed | Precision | Output-token ceiling |
| --- | --- | --- | --- | --- |
| `gpt2_control` | Full 12-layer GPT-2 | Legacy constrained rewrite / 7 | FP32 | 18 |
| `llama_l1` | Full 28-layer Llama 3.2 3B Instruct | Rewrite L1 / 0 | FP16 | 64 |

Both use compact Luna control, the existing activation-patch surface profile,
12 diagnostic slots, fresh B0 before C1, and no added decoder control or rescue
budget. `rerank-mode=apply` retains the existing selector setting; it is not a
new production-apply grant. API usage remains independently metered.

```sh
python3 -m SpiralInterventionLab.examples.run_iteration_pair \
  --profile gpt2_control --worker-model-path "$GPT2_MODEL_PATH" \
  --output-dir results/<new-iteration>/gpt2

python3 -m SpiralInterventionLab.examples.run_iteration_pair \
  --profile llama_l1 --worker-model-path "$LLAMA_MODEL_PATH" \
  --baseline-reference results/llama_l1_luna_actions_20260922_r2/live/experiment_summary.json \
  --output-dir results/<new-iteration>/llama
```

The paths are supplied locally, never embedded as defaults. The optional
baseline reference stops before controller calls if the fresh B0 differs.
Existing output directories are rejected. Manifests hash checkpoint/source
files; a source archive, raw JSONL, controls, cost counts and a model-local
audit are retained. Source changes during a run invalidate its sealed audit.

On a shared-memory local machine, run these processes sequentially rather than
loading both workers onto MPS at once. This still provides both iteration
checks without conflating memory pressure with operator effects. Neither the
previous run nor its interview is added to the next controller context.

Acceptance is separate at each level: offered action -> requested action ->
actual replay -> restored state -> measured readout -> task result. No-action
runs are marked `not_exercised`, not successful action-contract validations.

## 2026-09-22 GPT-2 observation

The first `gpt2_control` pair completed with Luna. B0 and C1 both produced
` In the case of a rewrite, the budget draft.` at score 0.5875, `task_done=false`.
There were 11 parsed controller calls, 11 charged diagnostics, zero rollout
edits, and one explicit current-prefix measurement (four physical replays).
No `hold` or `review_existing` action was selected in this trajectory.

The selected candidate was `resid_pre L11`, alpha 0.04, cap 0.16, with frozen
source and target piece ` Mir` for objective Mira. At worker step 8, after
` In the case of a rewrite, the`:

| Observable | Paired result |
| --- | --- |
| Target piece logit delta | +0.0032253265 |
| Top-20 threshold logit delta | -0.0035877228 |
| Target-to-top-20 gap delta | -0.0068130493 |
| Target piece probability delta | +0.0000076963 |
| Target rank | 104 -> 104 |
| Top-20 hit delta | 0 |
| Baseline / edited next token | ` budget` / ` budget` |
| Null / repeat maximum logit difference | 0 / 0 |
| State restored | true |

Part of the gap reduction is a falling top-20 threshold, not just a rising
target logit. The measurement is repeatable at the recorded FP32 precision;
it does not demonstrate target-owned generation or task improvement. The
post-run interview input now contains this current-prefix measurement and its
controls; the qualitative memo accurately acknowledges that it occurred.

Evidence: [`gpt2/audit.json`](../results/iteration_positions_20260922/gpt2/audit.json),
[`gpt2/live/c1.jsonl`](../results/iteration_positions_20260922/gpt2/live/c1.jsonl),
and [`gpt2/live/post_run_debrief_input.json`](../results/iteration_positions_20260922/gpt2/live/post_run_debrief_input.json).
Source/checkpoint hashes and an executable source snapshot are retained beside
the logs. No cross-model score or runtime-speed claim follows from this pair.

Validation before both runs: 381 tests and two subtests passed; `git diff --check`
passed. The new tests cover exact-candidate history isolation, bounded compact
serialization with numeric types intact, prefix divergence, unchanged action
availability, and interview coverage of measurement versus review/hold.

## Initial Llama attempt: incomplete, not a second completed result

The same source revision was used for `llama_l1`. Its fresh B0 matched the
previous prompt/output/score/step count exactly (score 0.938889). Four frozen
candidates were registered. At controller step 5 / worker step 6, Luna chose
the offered `remeasure_current_prefix` action for canonical ` Sel`, `resid_pre
L27`, alpha 0.04, cap 0.16, after ` Nora will take the sample to`.

No result for that action was logged before the attempt was stopped. After
more than 26 minutes, host samples showed memory pressure and GPU synchronization
waits. Only this experiment's verified process was terminated; other local work
was left untouched. The worker was not truncated, precision was not changed,
and no extra apply authority was granted. Samples cannot prove which concurrent
process caused the slowdown.

This is evidence of visible choice, **not** completed measurement, restored
state, C1 task performance, or full Llama regression success. There is no C1
`episode_end`, no post-run interview, and no automatic retry. Rerun the same
profile in a fresh output directory when local resources are available.

The partial JSONL, source archive, manifest, matching B0 and explicit interrupted
status are preserved in [`llama/partial_audit.json`](../results/iteration_positions_20260922/llama/partial_audit.json).
The two profiles are a repeatable validation workflow, not a demonstrated speed
improvement; this attempt specifically shows why heavy workers should not be
overlapped with other memory-intensive local experiments.

## 2026-09-22 Llama rerun: completed under unchanged conditions

After the user explicitly requested a retry with resources available, a new
`llama_l1` pair completed in 492.1 seconds. Code, checkpoint, library versions,
fixture, precision and controller settings matched the interrupted attempt.
The prior attempt remains intact; its unfinished measurement is not pooled
with this result. No old trace or interview was supplied to Luna, and no
production gate was changed. This is not a controlled memory-performance study.

Fresh B0 matched the reference exactly. B0 and C1 both produced
` Nora will take the sample to Ivo before sending the report to Yara before dusk.`
at score 0.938889 in 18 steps, with `task_done=false`. All 18 controller calls
parsed, all rollout commands were noop, and zero rollout edits were compiled.
There were 13 diagnostic results but only 12 charges: one candidate review
returned `review_reused` without replay or additional cost. The six explicit
current-prefix measurements account for 24 physical one-token replays, not
for all diagnostic work in the episode. Once budget reached zero, the
remaining controller steps issued no diagnostic requests.

All six measurements used the same frozen candidate
`frozen_candidate:110296027c25eea11f40e151`: ` Sel` (token 24082), `resid_pre L27`,
alpha 0.04, step-size cap 0.16, with an unchanged source-tensor hash. Worker
steps below are measurement positions, not controller request indices.

| Worker step / observed prefix suffix | Target rank before -> after | Target logit delta | Top-20 gap delta | Baseline and edited next piece |
| --- | --- | --- | --- | --- |
| 7 / `sample to I` | 3211 -> 3195 | +0.00390625 | +0.00390625 | `vo` |
| 8 / `sample to Ivo` | 66 -> 65 | +0.00781250 | -0.00781250 | ` before` |
| 9 / `Ivo before` | 5 -> 5 | +0.00781250 | -0.00781250 | ` sending` |
| 11 / `sending the` | 131 -> 129 | +0.00781250 | -0.00781250 | ` report` |
| 12 / `sending the report` | 169 -> 169 | +0.00781250 | -0.00781250 | ` to` |
| 13 / `report to` | 2 -> 2 | +0.01562500 | -0.00781250 | ` Y` |

For every row, the top-20 hit delta was zero, null/repeat control maximum logit
differences were zero, state was restored, and continuation was unchanged.
Each edit hook fired once and was cap-saturated, with effective displacement
approximately 0.16. Bound-piece probability gains were positive but small;
the largest was +0.0001813 at worker step 13. These are FP16/MPS observations,
not evidence of higher-precision accuracy or a completed `Selim` insertion.

The important distinction is position-dependent baseline reachability versus
intervention efficacy. ` Sel` was already top-20 at steps 9 and 13; the patch
did not create either entry. At step 7 rank improved while the gap worsened
because the threshold rose faster than the target. A high piece rank after
`report to` also does not certify the intended subject/recipient relationship.
No target-specific preferred boundary or automatic scheduler follows from
this one controller-selected trajectory.

Telemetry caveat: the inherited `operator_recipe_id` string still says
`step_size=0.0800`, while the frozen action descriptor specifies 0.16 and the
hook reports approximately 0.16 displacement. Do not use that recipe string
alone as dose identity. The mismatch was preserved and documented rather than
changing runtime behavior partway through the comparison.

The post-run interview now acknowledges the six current-prefix measurements,
unchanged continuation and restoration controls. Its requests for fuller
review/observer context remain qualitative developer feedback, not runtime
evidence or permission. No explicit `review_existing` or `hold` action was
chosen, so this run does not validate live discrimination among all three
action choices.

Evidence: [`llama_rerun_1447/audit.json`](../results/iteration_positions_20260922/llama_rerun_1447/audit.json),
[`live/c1.jsonl`](../results/iteration_positions_20260922/llama_rerun_1447/live/c1.jsonl),
and [`post_run_debrief.md`](../results/iteration_positions_20260922/llama_rerun_1447/live/post_run_debrief.md).
Source and artifact hashes were verified after completion. No implementation
changed for this retry; the earlier 381-test result remains the implementation
validation, not a newly rerun suite. GPT-2 and Llama now each have a completed
model-local pair, both without task-score uplift; their different fixtures
must not be pooled as an intervention-efficacy comparison.
