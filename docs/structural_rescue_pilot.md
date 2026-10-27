# Structural Rescue Pilot

This experiment connects the completed radial controls to full-continuation
evaluation and a bounded policy comparison. It uses the existing normal-cap
activation-patch compiler in restored diagnostic branches. It does not change
production or trial admission.

## Existing Radial Result

The completed CPU32 retry and its original interrupted attempt remain in
[source_common_radial.md](source_common_radial.md). At the fixed Llama boundary,
comma probability changes by +0.001517545 for common, -0.000492807 for the
parallel component and +0.002361577 for the orthogonal component. The greedy
winner remains `before`. The GPT-2 diagnostic-prefill result has a different
sign and baseline winner; it is not an isolated model ablation.

These results answer the radial question locally. They do not establish a
useful actuator, additive component attribution or a general semantic direction.
The pilot below uses them to define a fixed comparison, without another
direction/site/dose search.

## Frozen Cases and Evaluation

`tasks/structural_rescue.py` defines three calibration cases from the old L1
ladder, including its natural successes, and six predeclared confirmation
variants. The variants change names, clause order and source phrasing. They are
related cases, not six independent population samples. Their baseline success
or failure is not a selection criterion. L2 compression is outside this pilot.

The manifest includes the complete case set, evaluation-only role tuples,
prompt hashes, rules, seeds and budgets. `--prepare` saves it before loading a
model or making a provider call. `--run` requires its exact SHA-256 hash and
verifies the local checkpoint against the existing recorded manifest.

The old lexical score remains unchanged. A separate evaluator reports:

- lexical constraints and a single completed sentence;
- agent/action/theme/recipient bindings;
- `pass`, `fail` or `unknown` role status and raw output;
- joint success only when constraints and the role check pass.

The evaluator rejects known role swaps and shared-subject gerunds with a
possessive name. Missing bindings, repeated actions and unsupported relative or
passive constructions retain uncertainty. This is conservative surface analysis,
not a general semantic or grammar oracle. Expected role tuples and these
evaluation reports never enter controller packets.

## Conditions

All conditions share the first natural prefix ending in `to <first SOURCE
recipient>`. The timing rule is externally fixed from the source case; this
pilot does not measure the controller's ability to discover an intervention
time. No prefix token is forced. Cases without that boundary are retained with
an explicit unavailable decision and baseline continuation.

At the boundary, alignment and one structural observation precede a frozen
catalog of common, parallel and orthogonal recipes for pairs among the first
three exact SOURCE structural tokens. The site is the standard middle-layer
`resid_pre:last` surface. Binding, expression and actual source tensor identity
must match before execution.

| Policy | Choice |
| --- | --- |
| B0 | No edit |
| Fixed | Orthogonal component of the first source pair |
| Random | SHA-256 of case ID and seed 17 modulo catalog size |
| Controller | One JSON choice of an offered recipe ID, or noop |

The provider receives the generated prefix, structural readout, recipe formulas,
geometry, identities and caps. It receives no expected roles, reference output,
forced-rescue label, or measured candidate continuation. Candidate outcomes are
not fed back into subsequent choices. This measures one-shot recipe selection;
the original multi-step controller hypothesis remains a later evaluation.

Each chosen recipe uses alpha 0.04, step cap 0.08, norm cap 1 and TTL 1, bounded
further by the standard offered surface. Two no-edit repetitions precede each
branch. The edit expires after one generated token, then the worker completes
the sentence. The harness checks one hook call and restoration of state/caps,
and records actual and relative written norms. No certificate is minted.

Calibration additionally completes the first pair's common/parallel/orthogonal
branches under the same conditions. This matrix is not supplied to the
controller and is not used to change the confirmation protocol.

## Work and Readout

Each policy has the same ceiling of 24 diagnostic and 64 continuation forwards;
unused work is not padded. The counter wraps actual `run_with_cache` calls,
including packet preparation and failed attempts. Shared natural reconstruction,
alignment/observation and the calibration matrix are reported separately.
Provider requests, tokens and latency are separate costs. There is at most one
provider request per eligible case and no automatic retry or response repair.

Raw readout compares each observed structural token to the fixed baseline
full-vocabulary winner. Gap closure is `(gap_before - gap_after) / gap_before`
only when the original gap is positive. Post-constraint token choice and full
continuation are retained separately. Deterministic repetition checks drift;
it does not increase the statistical sample size.

The pilot stops its own model calls after the wall-clock limit, below 10 GiB
free disk, or at a forward ceiling. The MPS memory fraction is an explicit run
setting bounded at 0.85. The initial attempt with 0.7 failed on its first
forward, before any controller request; its receipt is retained. A retry uses
the previously used Llama value 0.85 without changing any intervention cap.

## Reproduce

```sh
python3 -m SpiralInterventionLab.examples.structural_rescue_benchmark \
  --prepare --output-dir results/<fresh-directory>

python3 -m SpiralInterventionLab.examples.structural_rescue_benchmark \
  --run --output-dir results/<fresh-directory> \
  --manifest-sha256 <printed-hash> \
  --worker-model-path "$LLAMA_MODEL_PATH" \
  --checkpoint-manifest results/entity_probe_refresh_llama_20261014_retry1/manifest.json \
  --controller-model gpt-5.6-luna --device mps --dtype float16 \
  --mps-memory-fraction 0.85 --wall-seconds 3600

python3 -m SpiralInterventionLab.examples.summarize_structural_rescue \
  results/<fresh-directory>
```

Existing result directories cannot be overwritten. The source snapshot,
execution plan and environment versions accompany the live receipt. `analysis.json`
and `report.md` can be rebuilt from `results.json` with no model or provider calls.
The runner now creates the source archive and environment receipt itself before
loading a model or provider. New plans hash every package Python file and
`pyproject.toml`; the original live pilot manually recorded only four source hashes.
The v1 pilot supports OpenAI only, with retries explicitly disabled; other provider
clients elsewhere in the repository do not inherit this pilot's one-attempt contract.
Provider failures are excluded from efficacy pairs, with their counts reported;
they are never recoded as evidence of ineffective control.
Unavailable-boundary cases remain in overall policy outcomes, but are identified
separately from eligible intervention pairs. Their shared fallback decode is charged
once, not once per policy. Interrupted branches retain attempted-forward counts.

## Observed Pilot: 2026-10-27

The MPS/float16 retry completed all nine cases and 36 policy conditions in
848.203 seconds. All boundaries were found naturally. The complete manifest
hash is `81326474db5ae819ba997612019c50f77b52a7ad057fc4c4567d786bce572324`.
The analyzer verifies that hash and the executed code hashes against the saved
source snapshot before producing a report.

| Split | B0 | Fixed | Random | Controller |
| --- | ---: | ---: | ---: | ---: |
| Calibration joint success | 2/3 | 2/3 | 2/3 | 2/3 |
| Confirmation joint success | 6/6 | 6/6 | 6/6 | 6/6 |
| Changed full output vs B0 | 0 | 0 | 0 | 0 |

The review failure remains `Nora ... before sending the report ...` with Selim
missing and the sending action attributed to Nora. Its legacy score remains
0.938889 and its role-aware joint success is false. All six confirmation variants
are baseline successes; they give no headroom for a success-rate improvement.
They are retained rather than replaced by cases selected after observing failures.

The real controller made nine requests: eight selected a current recipe and one
chose noop. Selected components and source pairs varied across cases. All eight
selections reached normal-cap physical execution; no selection/provider error or
admission failure explains the unchanged outputs. This is a bounded one-choice
selection result, not a test of adaptive multi-step control.

The nine additional calibration component branches also preserve the raw greedy
winner and complete output. Across all 45 completed branches, restoration and
cap checks pass and maximum repeated-baseline raw-logit drift is zero. There
were 26 edited policy branches plus nine edited calibration branches, each with
one confirmed hook call and TTL 1. Task gain per executed policy edit is zero.

| Measured work | Actual forward attempts/completions |
| --- | ---: |
| Shared reconstruction/alignment/observation | 85/85 |
| B0 | 126/126 |
| Fixed | 144/144 |
| Random | 144/144 |
| Controller | 142/142 |
| Separate calibration matrix | 144/144 |
| Total | 785/785 |

Provider usage is 52,849 input tokens and 1,846 output tokens (54,695 total),
separate from local work. These actual consumption differences remain visible
under equal ceilings. The controller's noop saves two local forwards relative
to always selecting a recipe; this does not offset or hide the API cost.

The result supports stopping this exact normal-cap/site/recipe pilot. It provides
no evidence of task uplift. The measured recipes do not change the known failed
calibration branch, and the confirmation set is at baseline ceiling. More
controller calls alone cannot create success-rate headroom in that confirmation set.
Further work needs an independently specified task set with baseline headroom
and an actuator test that can demonstrate branch changes under its stated budget.
No population efficacy, semantic circuit, production safety or new permission is
inferred from this pilot. Its small MPS readout differences are not promoted to
CPU32 numerical replication claims.

Validation: the full suite passed 796 tests and nine subtests. Four report tests
then passed after adding manifest/source-snapshot integrity checks. These include
refusal to publish an analysis when the manifest or archived executed code is
tampered with. `git diff --check` passes.

## PR Review Corrections

Review found and corrected missing automatic source capture, duplicated fallback
costs, dropped failed-branch work and acceptance of unfinished text after a sentence.
Model-load/checkpoint failures now also preserve an error receipt. Regression tests
cover these paths without model downloads or provider calls. The original live
receipts and archived code remain unchanged; these are post-pilot code corrections,
not a new live validation or a retroactive change to the frozen cases.
The fixed policy also retains the predeclared first-pair identity when availability
changes; it reports an unavailable/no-edit choice rather than switching pairs.
See the [review findings and residual scope](structural_rescue_pr_review.md).

Artifacts (public copies, original hashes and exact publication scope in the
[publication notes](../results/published_intervention_20261027/README.md)):

- [Frozen manifest](../results/published_intervention_20261027/pilot/manifest.json)
- [Full results](../results/published_intervention_20261027/pilot/results.json)
- [Paired outcome and cost report](../results/published_intervention_20261027/pilot/report.md)
- [Verified analysis](../results/published_intervention_20261027/pilot/analysis.json)
- [Original memory-limited attempt](../results/published_intervention_20261027/initial_attempt/results.json)
