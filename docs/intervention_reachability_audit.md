# Intervention Reachability Audit

## Question and Evidence Boundary

Does the controller choose not to intervene, or does the available interface
make a permitted intervention unreachable? These are different observations.
The audit separates the following stages rather than optimizing edit count:

`observable -> measured support -> same normal-budget executable -> physical
confirmation -> ephemeral trial offer -> controller choice -> generated-token
exposure -> task outcome`

The Llama handoff pair and the preceding GPT-2 pair both had zero rollout edits
and unchanged task scores. Neither exposed a valid trial offer. Therefore these
runs do **not** test whether Luna would decline an eligible trial. They also do
not show that all possible safe operators are ineffective.

The [read-only JSONL audit](../SpiralInterventionLab/examples/audit_trial_reachability_jsonl.py)
records each admission stage, source hashes, visible candidates and measurement
contexts. Missing older instrumentation is `null`, not a negative gate result.
It has no runtime authority and never reclassifies a diagnostic as permission.
The [comparison artifact](../results/iteration_handoff_20260922/reachability_comparison.json)
covers the old Llama trace, the new Llama handoff trace and the GPT-2 handoff trace.

## Confirmed Implementation Faults

### Ordinary One-Token Edits Expired Before Generation

The loop generated a token, installed an edit, and immediately decremented TTL.
For ordinary `ttl_steps=1` edits this removed the hook before any subsequent
generated token. Production trials had a special exemption. A minimal replay
reproduced one compiled edit with no token exposure at TTL=1, versus one token
exposure at TTL=2. This was a real execution bug, not a conservative decision.

The loop now observes effects and expires edits **after generated tokens**, before
the next controller packet. Ordinary TTL=1 reaches one token; TTL=2 reaches two.
The compiled-edit registry expires consistently with the runtime hooks. Trial
expiry remains one token and occurs before the next diagnostic round. No edit
allowlist, cap, target-lift threshold or authorization check was loosened.
The loop also rejects apply at a terminal prefix: installing a hook when no next
token exists must not consume budget or masquerade as a generation intervention.

Existing tiny-model integration tests had encoded the off-by-one lifetime:
TTL=2 produced `aca`. They now correctly expect `acc`; this is a fixture-level
lifetime correction, not a new live model-success claim. New tests cover TTL=1
resid_add/activation_patch and TTL=2 resid_add.

This fault cannot explain the new Llama run's zero **compiled** edits: the
controller emitted no apply command there. It would have mattered after an
ordinary edit was selected. Its correction must not be confused with fixing the
earlier admission failure.

### Rejected Trial and Debrief Information Loss

A trial rejected by the exact authorization validator was logged as noop but
could leave the latest controller-memory decision as apply. The loop now records
the guarded noop and rejection reason in memory as well; the earlier intention
remains in the event trace, not relabeled as execution.

The debrief projection retained positive token deltas but omitted the handoff's
cap-mismatch reason. The completed Llama interview emphasized weak effects;
its compact evidence did not include that admission failure. The projection now
includes bounded handoff checks/reasons. The old interview is preserved unchanged
and remains qualitative, not an independent causal diagnosis.

## Remaining Design Restrictions

This table describes the audited pre-change state. The subsequent
[explicit prefix investigation change](explicit_prefix_investigation.md) adds
normal-cap variants, removes frontier preference from exact physical context
verification, and adds bounded explicit hold-prefix rounds. Historical run
results below are not reinterpreted as tests of that newer implementation.

| Restriction | Verified observation | Interpretation |
| --- | --- | --- |
| Diagnostic versus normal cap | Current frozen high-dose candidate uses 0.16; normal resid_pre L27 cap is 0.12. Exact candidate identity blocks substitution. | That candidate cannot be trialed under current policy, regardless of its readout improvement. Correct guard, incomplete diagnostic-to-normal-budget path. |
| Restricted dose grid | Matched response and frozen-prefix v1 support only 0.04 and 0.16. | The normal 0.12 cap cannot be requested through these tools. The weaker 0.04 candidate is available, but was not remeasured at the three chosen live prefixes. This is a coverage gap, not evidence that 0.12 works or fails. |
| Frontier equality as context | A synthetic exact-context positive passes both physical confirmations; changing only `diagnostic_frontier_bundle_key` blocks shadow certification. | Legacy context equality also imposes a helper-selection preference. This conflates state identity with policy preference. The audit test documents it; the gate is not relaxed here. |
| Generation before first decision | `worker.step()` precedes the first controller invocation; the first live prefix is already ` Nora`. | The controller cannot intervene on the first answer token through this loop. Claims about first-answer-token control need a separate pre-generation contract. |
| Noop plus diagnostic advances time | Same-prefix follow-ups require a qualifying handoff/offer; otherwise the next round generates a token. | Noop is not a neutral pause: it commits another greedy token. Multi-stage investigation can consume an opportunity before it becomes executable. |
| Absolute raw-logit check | Positive bound probability/top-20 evidence also needs positive target-logit delta above controls. | Relative redistribution can improve probability without increasing the raw target logit. This is a stricter policy choice, not a mathematical requirement for probability lift. It remains unchanged. |
| Decoder `off` retains guards | Forbidden-phrase and word-budget masks run before the `off` return. GPT-2's late ` Mir` target was masked. | B0 is a matched no-controller baseline with the same guards, not unmodified vanilla generation. Masked readouts cannot be called dead actuators. |
| Prompt example imbalance | The compact prompt's five micro examples all choose noop. | A possible framing bias, not a demonstrated cause. A balanced abstract-contract A/B would be needed; no answer-specific hints should be added. |

These restrictions do not prove universal deadlock. The synthetic positive
fixture still traverses confirmation, promotion and one-token trial authorization
when identity, frontier, effect and budget conditions all agree. Conversely, that
fixture alone does not prove the candidate generator ever supplies such a row
in a live run.

In particular, `runtime_guardrail` currently names both physical invariants and
task-specific lexical/length masks. The latter are decoder policy, not tensor
physics. Their owner and enabled state should be explicit in any later design
comparison. This audit does not disable them or present `decoder_control=off`
as an unguarded baseline.

## Semantic Ownership Is Not Sentence Correctness

The live readout-qualified position was `...take the sample to`, with ` Sel`
as target and ` I` as the actual next token. Increasing Selim's piece probability
there is not necessarily useful: the source says the sample goes to Ivo and
Selim sends the report. No changed continuation was observed, and no sentence
role-preservation claim follows from the token delta. Target-piece ownership,
whole-name completion and predicate/argument faithfulness remain separate tests.

Similarly, a self-actuator label is not proof of a mechanistic causal path.
The current classification measures ownership of observable readout changes,
not a complete causal-mediation account inside the model.

## Next Bounded Experiment

1. Expose each candidate's measured dose, normal-policy cap and exact-execution
   compatibility as facts, not a preferred action. Allow an explicitly requested
   normal-cap candidate to be measured at the same prefix under a new identity.
   Do not clip a 0.16 candidate to 0.12 and reuse its evidence.
2. Compare normal-budget candidates with fixed source, binding, site and context.
   Keep target, ownership, safety and variation gates unchanged. This tests the
   missing actuator evidence without opening apply permission.
3. Separate exact state/source identity from frontier preference in a controlled
   design change. The controller should own objective selection; mismatched
   tensor/context/target identity must still veto execution.
4. Only then test a bounded pre-generation or hold-prefix investigation round,
   and a balanced prompt contract. Log generated-token time independently of
   controller/diagnostic rounds so noop does not silently mean wait.

Live scope is one model-local seed on MPS, not a multi-seed controller comparison.
The loader's MPS numerical-correctness warning is retained. Controls were stable
within the measured replays, but this is not a cross-device precision validation.
All live source snapshots precede the subsequent audit fixes unless explicitly
identified otherwise. Do not overwrite or retroactively relabel those artifacts.

## Post-Fix Regression and Recovery

The GPT-2 token-TTL pair used the corrected lifetime/memory code, full 12-layer
FP32 worker and Luna with the existing seed-7 fixture. Fresh B0 matched its
reference. B0/C1 both completed with
` In the case of a rewrite, the budget draft.`, score 0.5875, 11 tokens,
`task_done=false`, and zero rollout edits. The controller made 11 parsed noop
decisions and used ten diagnostics. Five requests concerned shadow/trial review;
no evidence-ID physical confirmation or valid trial offer was reached. This is
not evidence of unwillingness to use an offered trial.

Generation completed, but the newly added debrief projection raised a TypeError
on a null optional handoff list. A regression test now covers that case. Only
postprocessing was rerun, not either worker trajectory. The original failed
`run_status.json`, generation files and source snapshot remain intact. The
[recovery audit](../results/iteration_handoff_20260922/gpt2_token_ttl/recovery_audit.json)
checks the archived code against its manifest and seals unchanged generation
hashes alongside separately generated debrief artifacts. The terminal-prefix
apply guard was added after this pair and has test coverage, not a further live
pair. The episode had no apply attempts, so this guard was not exercised live.

See also [the token-TTL reachability report](../results/iteration_handoff_20260922/gpt2_token_ttl_reachability.json).
Final verification: `python3 -m pytest SpiralInterventionLab/tests -q` passed
423 tests and two subtests, with 15 existing dependency warnings;
`git diff --check` passed. Positive scripted/tiny-model tests establish execution
reachability, not an improvement on either live language-model benchmark.
