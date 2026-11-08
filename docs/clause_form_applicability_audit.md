# Clause-form applicability audit

2026-11-09. Development audit of PRs [#49](https://github.com/RyoSpiralArchitect/spiral_intervention_lab/pull/49)
and [#50](https://github.com/RyoSpiralArchitect/spiral_intervention_lab/pull/50),
starting from main `768fa69fda0042a004996e393aacf5b36c9b14a4`.

**The PR50 flag does not isolate the observed failure.** Its source-wide
`with|who` search is true for 11 of the 12 fixed cases. It is false for
`gen_017`, where both comma and starter succeeded, and true for `gen_059`,
where the starter replaced the successful comma choice. The old flag is
advisory: an offered recipe ID passes the controller response validator even
when `applies_to_this_case` is false.

This change adds an opt-in, bounded syntax rule and a controller selection
gate. It also preserves a reproducible audit of the existing receipts. There
are **zero new worker/controller experiment calls** and no new effectiveness measurement. Historical
results, manifests, evaluation code, and source archives remain unchanged.

## Reproduce the audit without model dependencies

From the repository root:

```bash
python -S scripts/audit_clause_form_applicability.py > /tmp/clause_form_audit.json
```

The script uses only the standard library, loading the pure syntax helper
directly with `runpy` to avoid package initialization and model imports. It
reads all twelve sources, boundary prefixes, catalogs, controller responses,
and outcome receipts. It verifies:

- Frozen manifest hashes and receipt/execution-plan bindings.
- The same twelve unique cases in the same order.
- Prefix/observation agreement, catalog IDs, selected recipe/command targets,
  controller packet hashes, and controller system-prompt hashes.
- Recorded aggregate counts, restoration, and zero repeated-baseline drift.
- Identical inputs, prefixes, B0 outputs, and fixed-comma outputs across the
  three historical runs below.

JSON output contains evidence-file hashes, the new helper's hash, original
outputs and scores, recipe targets and observed gaps, and new syntax
assessments in separate fields. It copies historical scores verbatim; it
does not evaluate outputs again. The syntax helper receives only the source
text and generated tail, never a case ID, role annotation, or outcome.

| Historical run | Frozen manifest SHA-256 | Controller successes |
|---|---|---:|
| [Readout datasheet](../results/structural_rescue_cap012_v3_20261107/manifest.json) | `a0c537479e8249fcbfcd19a55ef3ce29f4c9610dce811b6a970b4ec04c9b39b7` | 2/12 |
| [Starter offered, before form hint](../results/structural_rescue_clauseform_v3_20261108/manifest.json) | `4ef1be4aecc7a2436b94aef4bdf131fd96101602088232a9c862b45c82d8fa0c` | 2/12 |
| [PR49 form hint](../results/structural_rescue_clauseform_ds_v3_20261108/manifest.json) | `76fc0dd072500512e2add5ac678f0450b17ec35329c64963b396eb00246cd252` | 6/12 |

## The twelve cases

Full sources are in the linked manifests and audit JSON. These are the
pending second-clause constructions, observed first-clause tails, and saved
outcomes from the [PR49 receipts](../results/structural_rescue_clauseform_ds_v3_20261108/results.json).
All tails stop at `to <recipient>`; none already contains the second clause's
`with`. Consequently, first-clause completeness alone does not distinguish
the three source families.

`C` means the fixed comma readout; `W` means the fixed while starter. Pass and
fail reproduce the saved `joint_success` values, with the evaluator caveat
below. The last column is a new **eligibility assessment**, not a new outcome.

| Case | Pending source construction | Generated tail | PR50 flag | Saved C / W | Saved controller | syntax_v1 applies |
|---|---|---|---:|---|---|---:|
| `gen_008` | `with Wanda giving …` | `Henrik will take the receipt to Ilse` | true | fail / fail | W, target not reached | true |
| `gen_017` | `while Yusuf should bring …` | `Alma takes the manual to Vesna` | false | pass / pass | W, pass | true |
| `gen_018` | `with Stefan giving …` | `Olga brings the drawing to Nikos` | true | fail / pass | W, pass | true |
| `gen_028` | `with Rosa taking …` | `Boris gives the proposal to Alma` | true | fail / fail | W, target not reached | true |
| `gen_039` | `Boris, who should take …, also finishes` | `Pablo sends receipt to Nikos` | true | fail / fail | W, action omitted | false |
| `gen_058` | `with Celia taking …` | `Stefan brings the ticket to Edda` | true | fail / pass | W, pass | true |
| `gen_059` | `Celia, who should give …, also finishes` | `Stefan brings the manual to Pablo` | true | pass / fail | W, action omitted | false |
| `gen_068` | `with Marta taking …` | `Stefan sends the blueprint to Greta` | true | fail / pass | W, pass | true |
| `gen_073` | `with Marta taking …` | `Pablo gives the proposal to Ugo` | true | fail / pass | W, pass | true |
| `gen_074` | `Boris, who should bring …, also finishes` | `Edda sends the manual to Celia` | true | fail / fail | W, action omitted | false |
| `gen_078` | `with Kira taking …` | `Marta sends the blueprint to Olga` | true | fail / pass | C, fail | true |
| `gen_088` | `with Henrik taking …` | `Celia sends the voucher to Stefan` | true | fail / pass | W, pass | true |

### gen_017: retain a finite-clause option

Source:

> Alma should take the manual to Vesna, while Yusuf should bring the voucher to Nikos, in order to close before midnight.

At `Alma takes the manual to Vesna`, B0 continues `before midnight.` and
drops the second action. The fixed comma gives:

> Alma takes the manual to Vesna, Yusuf brings the voucher to Nikos before midnight.

The fixed starter and PR49 controller give:

> Alma takes the manual to Vesna while Yusuf brings the voucher to Nikos before midnight.

Both pass the preserved evaluator. This case was already successful under
the earlier controller's comma choice. The `with|who` test is therefore not
an appropriate test for whether while can continue this boundary.

Its [saved packet](../results/structural_rescue_clauseform_ds_v3_20261108/gen_017_observation.json)
also contains two while recipes: a source-bound `readout` and a `clause_form`
starter. Both target token `1418` at the final layer with cap `0.12`; their
provenance and recipe IDs differ. Source presence changes provenance, not
the readout push's mathematical target.

### gen_059: distinguish a relative action from a participle

Source:

> Before lunch, Stefan should bring the manual to Pablo; meanwhile, Celia, who should give the drawing to Tilda, also finishes.

At `Stefan brings the manual to Pablo`, the saved fixed comma gives:

> Stefan brings the manual to Pablo, while Celia gives the drawing to Tilda before lunch.

The starter reaches while, but its continuation is:

> Stefan brings the manual to Pablo while Celia finishes her drawing for Tilda before lunch.

The relative clause already has finite `should give`. Its required delivery
action competes with the matrix predicate `also finishes`. A token that
encourages a finite clause does not establish which action that clause will
express. In this receipt `give` is lost.

The other two relative sources, `gen_039` and `gen_074`, fail under both
comma and starter. They are counterexamples to treating relative syntax as
a comma-success rule. Excluding an unsupported starter does not force the
controller to select comma and does not prove a rescued outcome.

## Selection, target reach, and content are different observations

The PR49 prose says “picked the starter on 9/12.” Its saved controller
responses actually select the starter on **11/12**. The first token becomes
while on **9/12**; `gen_008` and `gen_028` remain at before despite a starter
selection. Only `gen_078` selects comma. The saved success total **6/12** is
consistent with the receipts.

Relative to the previous 2/12 controller run, PR49 retains `gen_017`, gains
`gen_018`, `gen_058`, `gen_068`, `gen_073`, and `gen_088`, and loses `gen_059`.
The six successful cases therefore comprise one retained and five new wins.
This audit corrects the interpretation while preserving the earlier files.

The packets provide a calibrated reach estimate for while, but its current
`gap_to_top1` is absent on the eleven cases where while is absent from the
source. Only `gen_017` includes its while gap. The write-policy allowlist's
extra token ID does not add a diagnostic readout row. A syntax eligibility
rule cannot repair this missing measurement; the two reach failures remain
eligible under syntax_v1. No additional forwards are introduced here.

## Opt-in syntax rule and enforcement

`--clause-form-gate syntax_v1` uses the pure
[`clause_form_gate.assess`](../SpiralInterventionLab/runtime/clause_form_gate.py)
helper. It recognizes a bounded subset of the benchmark's delivery syntax:

- The generated tail must match a complete finite first delivery clause
  extracted from source text.
- A directly expressed finite or participial second delivery clause is
  supported.
- A relative embedding with a competing matrix predicate is unsupported.
- Unrecognized or ambiguous syntax remains unsupported; it is not inferred
  from an incidental `with` or `who` elsewhere in the source.

The assessment records the source construction, prefix match, decision,
reason, and rule version. It contains no case-specific success table and
does not use `frames_evaluation_only`. Names and objects are matched from
the source rather than enumerated as successful examples. This is a narrow
grammar recognizer, not an unrestricted English parser.

The controller's offer excludes unsupported starters and equivalent
readout aliases of the same target. This matters because the generator can
also create a relative source with `, while` as its connective: suppressing
only the starter's kind would leave the same while push under `readout`.
Controller selection is validated against the filtered offer. An excluded
recipe is rejected; there is no forced-comma fallback.

Fixed policies and random policies keep their complete catalogs so their
existing comparisons retain their meaning. Omitting the new option keeps
the legacy prompt, packet, and manifest behavior. Opting in records the new
condition in a distinct manifest. The regression checks cover enforcement,
renaming invariance, prefix mismatch, incidental keywords, and equivalent
readout paths. They establish the selection contract without predicting
new model outputs.

## Preserved evaluator caveat

The eight participial comma outputs report `role_status=pass` and
`constraint_violations=[]`, yet `constraints_ok=false`. Their receipts also
record `required_action_covered_by_inflection`.

In [`StructuralRescueEnv.evaluate`](../SpiralInterventionLab/tasks/structural_rescue.py),
recognized inflections remove a missing-term violation from local feedback.
However, `constraints_ok` still calls inherited `done()`, whose required-term
check uses literal substring coverage: `take` is absent from `taking`, and
`give` from `giving`. The reported finite-clause gains partly track this
lexical evaluation behavior. They are not a separate grammaticality score.
The `gen_059` omission of the give action remains visible independently.

This change preserves the evaluator and all historical scores. Any future
evaluator repair requires a separately versioned analysis, with the
original evaluation retained for comparison.

## Development and independent verification protocol

**Current status: model-free development only.** The twelve audited cases
are development/regression evidence because their outcomes informed this
rule. The 120 candidates from the original screen also have published B0
outcomes. Reusing them does not create an untouched validation set.

The prospective specification is frozen in
[`clause_form_validation_protocol.json`](clause_form_validation_protocol.json),
with source, evaluator, prompt and checkpoint hashes. Its independent inputs
are fixed by generator version, seed, ordered IDs and the candidate-row hash
`07b7bedf62166cd2545a235ab34d3c1589a648f4015cbdb3645dbfbcc6aa5343`.
Generating those inputs required no model. All 120 are distinct and have no
exact source overlap with the 250 distinct inputs in existing manifests.
This is new input combinations within the same generator, not a test of
arbitrary English or a different model population.

No run is launched. The existing CLI loads outcome-screened selections; the
future all-case execution adapter must bind these fixed inputs without
screening or relabeling them after outcomes are observed. Both execution
manifests and the complete source archive must be saved before that run.

| Condition | Development replay | Independent verification |
|---|---|---|
| Inputs | Existing twelve cases, manifest `76fc0dd0…` | Generator v3, seed `2026110901`, 120 candidates, existing family rotation; ordered source/prompt rows bound by the hash above |
| Selection | All twelve, including failures and regressions | All 120, without outcome-based screening; report unavailable boundaries rather than dropping them |
| Overlap check | Already inspected | Check source hashes against all prior screens before running; any exact overlap blocks preparation and requires a newly documented input manifest |
| Controller comparison | Legacy PR50 mode versus syntax_v1 | The same two frozen modes, one call per available case per mode; no retries chosen by outcome |
| Shared controls | B0, fixed patch, fixed comma, fixed starter, random | Same policies and complete fixed/random catalogs |
| Worker and budget | Preserve checkpoint, device/dtype, greedy decoding, boundary rule, cap `0.12`, one edit/token, and existing 64-token/forward limits | Identical to development, pinned in the execution plan; disclose any required deviation before running |
| Resources | One ordered shard of twelve; 7,200 seconds, 6 GiB disk floor, MPS memory fraction 0.85 | Ten consecutive shards of twelve with the same limits; retain incomplete attempts, no outcome-based reordering |
| Controller configuration | Pin model identifier, exact prompt/packet schema, provider settings, call and output-token limits | Same frozen configuration; record provider revision/response metadata when available |
| Primary score | Original evaluator and recorded `joint_success` definition | Same version, with the inflection caveat retained; failures and harms reported by source construction |
| Mechanism measures | Selected recipe, rejection, target reach, action preservation | Same measurements; a reached token is not counted as content success |

The specification pins the helper, benchmark/evaluator sources, controller
prompts, requested policies, model checkpoint and historical model-run
dependency versions. Preserve those bindings in both execution plans.
Do not retune the grammar or change the held-out inputs after observing
validation responses. Any such revision becomes a new development version
and needs a new independent set. The comparison measures the complete
syntax_v1 change; separating the effects of its evidence and its filter
would require an additional predeclared arm.

No new success rate is inferred from filtering these existing receipts.
The current deliverable establishes that eligibility depends on observable
syntax and that the controller cannot select a recipe excluded by that
condition.
