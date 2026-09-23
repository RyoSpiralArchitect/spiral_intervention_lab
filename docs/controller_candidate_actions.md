# Historical Review vs Current-Prefix Measurement

## Contract

The controller chooses between reading an existing measurement, measuring the
same candidate at the current prefix, a bounded normal-cap investigation, and
holding. Merely changing review prose
to say "current prefix" does not request a new physical replay.

`runtime/candidate_actions.py` owns episode-local, bounded data and mechanical
resolution, not selection policy. A successful, restored matched-response matrix
can register up to eight frozen candidates. The controller sees at most four
cards in insertion order, not sorted by desired target or favorable effect.

V1 captures recorded `source_term_token`, non-contrastive activation blends at
the original measurement anchor. The source tensor is cloned; its digest must
match the measured row. Target token ID/text, target site/layer, alpha, step-size
and one-token TTL remain fixed. Unsupported rows return an unavailable receipt,
not a newly invented source or fallback candidate. Episode reset clears the
registry, issued requests and measurement cache.

Each card offers:

| Action | Execution | Accounting |
| --- | --- | --- |
| `review_existing` | Read original evidence | No physical replay or diagnostic charge |
| `remeasure_current_prefix` | Same candidate, current prefix, fresh paired controls | One diagnostic charge and four one-token replays when uncached |
| `measure_normal_cap_current_prefix` | New lower-dose identity under normal caps; same source/binding/site/alpha | One diagnostic charge and four replays when uncached; no inherited measurements |
| `investigate_normal_cap_current_prefix` | Measure the normal-cap child and conditionally confirm that exact child at the same prefix | Reserve two existing diagnostic slots/up to eight replays when fresh, one/four if cached; charge actual work; require a final decision round |
| `hold` | No measurement; does not pause generation | No diagnostic charge |

The controller returns only the offered ID:

```json
{"decision":"noop","meta":{"diagnostic_request":{"diagnostic":"candidate_action","action_id":"<offered ID>"}}}
```

This request is exclusive: stale `next_action` or controller-memory labels do
not append another diagnostic. An ID binds the action, candidate and current
context. A stale or unknown ID, mutation, active edit, terminal prefix, previous
restoration failure or exhausted diagnostic budget blocks new measurement.
There is no automatic substitute and no automatic action on prefix change.

At an already measured context, the exact candidate/context measurement is
reused without replay. A fresh measurement uses no-edit/null and edit/repeat
controls, restores runtime state, and compares against its own baseline. It
does not compare an edited new-prefix value with an old-prefix baseline.

All five actions keep `production_apply_allowed=false`, `certified_for_apply=false`
and `policy_candidate_ready=false`. The investigation may return an exact,
single-use `candidate_trial_offer` only after the existing physical confirmation
gate passes. That offer still requires a separate controller apply/decline
decision; the investigation itself never applies. The optional offline
conditional-name probe is not called by this interface.

To inspect a returned diagnostic before another greedy token, use the separate
bounded `meta.generation_action="hold_prefix"` loop choice. It shares existing
handoff rounds, not apply authority. See [explicit prefix investigation](explicit_prefix_investigation.md).
Historical live observations below predate these normal-cap and explicit-hold additions.

## Observability

The selection log records exposed choices. Results record `requested_action`,
`executed_action`, `candidate_id`, current/measurement context IDs, replay count,
new-measurement count, budget before/after and blocked reason. Compact prompts
preserve nested executable IDs, controls, metrics and restoration status.

Legacy closed-review reuse also normalizes a small set of observed informational
review labels. Changed dose, candidate ID, comparison axis or execution evidence
still invalidates that reuse key. Unknown modes are not collapsed together.

## Validation Boundary

Fixed-context tests cover read/reuse/hold, a changed-prefix physical measurement,
stale IDs, reset, zero budget, source/binding provenance, mutation, restoration
failure, exclusive request extraction and compact-packet preservation. These
test the interpreter contract, not whether an LLM always makes the best choice.

The live experiment uses the same full-depth Llama worker, task, seed, decoder,
12 diagnostic calls and apply limits as the previous Luna pair. Only the
diagnostic interface/prompt changes. Model paths remain CLI inputs. A fresh B0
must reproduce the frozen prompt/output/score before controller calls begin.

Live evidence: [first run](../results/llama_l1_luna_actions_20260922/)
and [corrected rerun](../results/llama_l1_luna_actions_20260922_r2/), kept separately.

## First Live Integration Failure

The first pair completed with identical B0/C1 output and score `0.938889`, no
rollout edits, and Selim still missing. All 18 controller calls parsed, but no
frozen candidate could be offered. The matched matrix had completed and restored
state; the legacy public-row projection dropped `measurement_complete`,
`state_restored` and the no-edit control delta before registration checked them.
The initial combined reason code was `measurement_not_at_current_anchor`, so it
did not distinguish missing proof from actual context drift.

The fix preserves these fields and separates the unavailable reasons; it does
not infer missing completion or weaken the registration gate. A worker-path
regression test now verifies the projection, in addition to registry unit tests.
The first run, audit and pre-fix source snapshot remain in its own directory.
Its zero action executions do not establish live selection success.

The corrected rerun is stored separately at
[`llama_l1_luna_actions_20260922_r2`](../results/llama_l1_luna_actions_20260922_r2/).

## Corrected Live Result

The paired run completed in **400.753 seconds** (excluding checkpoint prechecks).
All **18 Luna calls parsed**, all rollout commands were `noop`, and no rollout
edit was compiled. B0 and C1 both emitted:

> Nora will take the sample to Ivo before sending the report to Yara before dusk.

Both scored **0.938889**, with Selim missing, 15 words within the 18-word budget,
and no forbidden terms. There is **no task-score uplift**.

The interface result is different from the task result:

- Four fixed candidates were registered and exposed from controller step 5.
- Luna explicitly selected `remeasure_current_prefix` seven times: two candidate
  identities across seven contexts, **28 one-token physical replays**.
- Every selected ID matched an exposed offer and every executed action matched
  its request. All measurements restored state; no-edit and repeat controls had
  zero observed logit variation at the configured precision.
- Twelve diagnostics were charged: five initial diagnostics plus seven
  remeasurements. Two later `inspect_evidence` calls used the separate inspection
  allowance. There were no fresh remeasurement requests after exhaustion.
- Some commands still contained a legacy `next_action` alongside the explicit
  ID. The exclusive request contract executed only the ID, not an extra replay.
- `review_existing` and `hold` were offered but not selected through this new
  interface in this run. Their execution and cache semantics are covered by the
  fixed-packet tests, not established as live LLM choice behavior.

All rows below use the same recorded source activation and alpha `0.04`, at
`resid_pre L27`. The lowercase piece rows share the same frozen candidate ID and
cap `0.16`; the uppercase row is a different candidate with cap `0.04`.
Worker steps are prefix lengths; controller selection steps are one lower.

| Worker Step | Prefix Ending | Piece | Cap | Rank Before -> After | Gap Delta |
| --- | --- | --- | --- | --- | --- |
| 6 | `sample to` | ` sel` | 0.16 | 31 -> 30 | -0.015625 |
| 7 | `sample to I` | ` Sel` | 0.04 | 3211 -> 3209 | +0.0078125 |
| 8 | `to Ivo` | ` sel` | 0.16 | 398 -> 395 | -0.015625 |
| 9 | `Ivo before` | ` sel` | 0.16 | 103 -> 103 | -0.0078125 |
| 10 | `before sending` | ` sel` | 0.16 | 97 -> 96 | -0.015625 |
| 11 | `sending the` | ` sel` | 0.16 | 1332 -> 1334 | 0 |
| 12 | `the report` | ` sel` | 0.16 | 1609 -> 1600 | -0.01953125 |

All seven top20-hit deltas were zero. The largest positive bound-piece probability
delta was about `1.3284e-6` at step 9, not certified target actuation. Step 7
improved rank without lifting the target logit and with a worse top20 gap;
step 11 slightly increased probability despite a worse rank. These readouts
should not be collapsed into one generic "positive" score. Nor should the
uppercase and lowercase/cap rows be pooled as one frozen-candidate experiment.

Controller input usage excluding the debrief was **358,463 tokens** versus
351,988 in the previous reuse run. No token-saving or optimal-position claim is
made. This is one independent trajectory, not a matched request-sequence ablation.

## Remaining Boundaries

The bridge from a selected ID to a physical current-prefix measurement now works
in live use. Choosing informative positions, allocating the remaining diagnostic
budget, and demonstrating a safe task-changing actuator remain separate work.
Seven successful tool calls do not prove the controller always chooses correctly.

The qualitative debrief incorrectly says historical candidates were not
revalidated in the current context. Its bounded input contains no
`candidate_action`/`remeasure_current_prefix` records: the older debrief digest
does not yet expose this event contract. Preserve the memo as-is, but do not use
that statement to contradict the seven verified JSONL measurements. Updating
the debrief digest is follow-up observability work, not apply authorization.

Artifacts: [audited results](../results/llama_l1_luna_actions_20260922_r2/audit.json),
[C1 JSONL](../results/llama_l1_luna_actions_20260922_r2/live/c1.jsonl),
[B0 JSONL](../results/llama_l1_luna_actions_20260922_r2/live/b0.jsonl),
[manifest](../results/llama_l1_luna_actions_20260922_r2/manifest.json),
[qualitative debrief](../results/llama_l1_luna_actions_20260922_r2/live/post_run_debrief.md).

## Reproduction and Tests

`python3 -m pytest SpiralInterventionLab/tests -q`: **372 passed, 2 subtests passed,
16 warnings** after retaining the measurement proofs. `git diff --check` passed.
No dependency or production policy change was needed.

The paired runners accept the same CLI settings as the prior sealed Luna run;
only `--worker-model-path` and the fresh `--log-dir` may change. For another local
checkpoint location, supply it through the CLI rather than editing code. The
checkpoint contents, task prompt, decoder budget and reference B0 are checked
before API calls. API credentials remain in the environment and are never logged.

The interface checks are deterministic. The live controller is not: one successful
choice sequence does not establish that it will always choose the most informative
position, nor that four readings of a candidate constitute independent task gains.
