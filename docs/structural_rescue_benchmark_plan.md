# Structural Rescue: From Bound Operation to Controller Value

Status: design after PR #37, not a new efficacy result or permission grant.

## Evidence and Open Questions

The existing [survey](clause_structure_survey.md) found forced-connective rescues
for three of six failed ladder cases. That establishes reachability under token
forcing, not normal-cap actuation. Large source-patch rescues were measured at
other sites/backend/doses; their success is not transferable to the small CPU32
component intervention.

The completed [radial comparison](source_common_radial.md) narrows the local
mechanism: on the measured Llama prefix, parallel shrink lowers comma probability
while orthogonal raises it. No greedy winner changes. GPT-2's different response
sign is not a model-only ablation: tokenizer, site, scale, baseline and prefix
history differ. Neither finding measures controller added value.

## Sequence and Stop Conditions

1. **Bind the operation.** The implemented first-three-source pair catalog binds
   exact occurrences, raw vectors, prefix/context, component, sites and caps.
   Offline and controller replay must agree on IDs and materialized expressions.
   Displaying a pair, observing a gap, and authorizing apply remain separate.
2. **Fix a structural-rescue evaluation set before new measurements.** Keep the
   previously explored ladder cases as calibration, including baseline successes.
   Freeze unused name/role/order/paraphrase variants as a confirmation split;
   publish the manifest and its hash before running it. Do not select only cases
   already known to respond to a forced comma. L2 compression remains a separate
   series. This document does not claim that a held-out set has already been made.
3. **Measure the bounded actuator first.** Freeze a small pair/component/site
   matrix at predeclared boundaries and unchanged caps. Compare no-edit, common,
   parallel and orthogonal with repeated baselines. If calibration shows no
   normal-cap branch change, report that null result; do not interpret a later
   controller failure as evidence against its decision quality.
4. **Then compare selection policies.** B0, a predeclared fixed rule, seeded
   random selection and the controller receive the same eligible operation
   catalog and observation access. The controller receives no hidden role answer
   or forced-rescue label. Freeze seeds/rules and maximum actual-forward budget
   before running. Different actual consumption must remain visible even under
   equal ceilings; setup, packet canaries and provider tokens are separate costs.

## Evaluation Contract

Keep the current lexical score unchanged for historical comparability. Add a
separate report of full constraints and agent/action/theme/recipient bindings.
The current role checker is a heuristic, not ground truth. For a small fixed
benchmark, freeze explicit expected role tuples and audit the scorer on adversarial
role swaps, shared-subject gerunds and missing arguments. Retain `unknown` and raw
outputs for ambiguous cases rather than awarding a confident semantic pass.
These evaluation-only annotations do not enter worker prompts or controller hints.

Each condition should report:

| Axis | Fields / interpretation |
| --- | --- |
| Identity | checkpoint, tokenizer, backend/dtype, prefix provenance and token hash, pair/recipe ID, exact sites and caps |
| Readout | full-vocabulary fixed-baseline-winner gap before/after, gap closure fraction when baseline gap is positive, target probability and rank |
| Actuation | actual written norm, relative dose, hook execution, greedy branch change |
| Continuation | raw full output, constraint violations, role errors/unknowns, joint success |
| Accounting | actual forward attempts/completions, setup/canary/replay/generation counts, diagnostic slots, provider calls/tokens |

Gap closure is `(baseline_gap - edited_gap) / baseline_gap`, with the winner
token fixed from the baseline. A nonpositive baseline gap makes the fraction
not applicable. Unmeasured continuation or role correctness is `unmeasured`, not
success or failure. Only a role-correct, constraint-satisfying full continuation
counts as joint success; a token probability increase alone does not.

Forced tokens and released-cap source patches remain a separately labeled
reachability ceiling. They never count as normal-budget wins. Production/trial
gates, DSL limits and the controller's policy ownership are unchanged.

## Immediate Handoff

The next live question is narrow: can a controller select the exact comma/while
component card, and does its current-prefix receipt match the offline operation?
That is a plumbing/identity check before an efficacy comparison. No new local
model run or provider call is included in this implementation's test results.
