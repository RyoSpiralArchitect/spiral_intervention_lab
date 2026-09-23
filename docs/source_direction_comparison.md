# Source Construction, Binding, And Staged Trial Review

Date: 2026-09-14. Routine controller: `gpt-5.6-luna`.

## Experiment

The preceding equal-dose measurement found that direct/observed seed labels
materialized the same edit. This experiment changes **source construction**,
not the seed label: `source_term_token` versus `source_centered_pm1`, using one
recorded anchor. Centered means term-span mean minus adjacent-token mean. It is
not necessarily a pure angular rotation, and it is not a new runtime operator.

Fixed conditions: GPT-2, seed 7, controller round 4 / worker step 5, prefix
` In the case of a`, Mira objective, `resid_pre` layer 11, blend alpha 0.04,
step-size caps 0.04 and 0.16, TTL 1, and pre-edit canonical/alternate token
bindings (` Mir` / ` mir`). The baseline is shared. Each unique edit is repeated;
both bindings are read from the same edited logits, not independent executions.

The source tensor hashes differ. Four unique executions produce eight
observables using ten one-token simulations (two no-edit, four edited, four
identical-repeat). No-edit/repeat maximum finite-logit differences are zero;
the four masked tokens remain identically masked. State restoration passes and
the rebuilt context matches the original live context. The two term-source
execution IDs match the preceding run; the two centered-source IDs are new.
A second local replay from the packaged compressed episode reproduced all eight
rows, comparisons and rejection checks exactly (ten additional simulations).

## Observations

Raw logit deltas and top20 threshold gaps, relative to the shared no-edit
baseline. More negative gap delta means a smaller gap, not necessarily an
increased target logit or a new top20 hit.

| Source | Dose | ` Mir` logit delta | ` mir` logit delta | ` Mir` gap delta | ` mir` gap delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| term token | 0.04 | +0.000392914 | -0.000257492 | -0.000808716 | -0.000158310 |
| centered pm1 | 0.04 | +0.000148773 | +0.000146866 | -0.000396729 | -0.000394821 |
| term token | 0.16 | +0.001549721 | -0.001041412 | -0.003194809 | -0.000603676 |
| centered pm1 | 0.16 | +0.000593185 | +0.000605106 | -0.001549721 | -0.001561642 |

Both sources saturate the step-size cap with matching effective delta norms
(approximately 0.04 / 0.16). Their raw source norms differ (227.40 / 115.21).
All bound rank and top20-hit deltas remain zero. Canonical token probability
changes are approximately 0.46e-6 to 4.77e-6, below the promotion floor 1e-4.

The centered source flips the lowercase logit response to positive, but weakens
the canonical response at both doses. Even with a falling lowercase logit, its
probability can rise because the denominator also changes. This is a controlled
source-by-observable difference, **not task success or a target actuator**.

Ownership uses the existing term-readout composite score, which includes rank
movement. At dose 0.16, centered changes alignment from -0.006 to +0.032 and
receives the legacy `self_actuator` ownership label, but its `actual_delta_class`
is still `rank_carrier`. Neither that label nor an improved alignment certifies
target mass/top20 lift. The new review checks reject every measured observable.

Scope: one frozen prefix, one worker, one seed, two source constructions. Repeat
controls measure local reproducibility, not a statistical confidence interval.
No controller was called during this local comparison; this is not another full
live run. The source Luna episode scored 0.5875, remained incomplete, and made
zero production applies. No apply was performed by the comparison either.

## Controller Tools And Authority

Use the existing `matched_response_probe` diagnostic with
`comparison_axis=source_localization`, an objective, optional recorded recipe IDs
(at most two), and doses from `[0.04, 0.16]`. It constructs both variants from one
anchor; no observed-positive label is inherited by a new source construction.
Recipe, execution, observable, and source-tensor identities remain distinct.
Aliases, missing seeds, changed materialized doses, and invalid controls are
explicit outcomes. Evidence inspection remains read-only and separately capped.

The controller can explicitly name an `evidence_id` in these existing requests:

1. `activation_patch_promotion_gate_review`
2. `activation_patch_production_shadow_replay`
3. `activation_patch_production_trial_gate_review`

The runtime resolves the ID from its own ledger, not scores supplied by the
controller. Before entering the existing ladder, the evidence must be complete,
restored, current-context, canonical-bound, finite, nonregressing, and target-owned.
The bound token must gain at least 1e-4 probability or a top20 hit, with a positive
logit response above repeat variation. Gap-only evidence cannot qualify.

The candidate must rematerialize to the **identical normal-budget edit**, including
source, target, alpha and dose; a diagnostic cap cannot silently become trial
authority at a different strength. Qualified evidence then receives an actual
two-token normal-budget replay plus no-edit control (three simulations), not just
a summary relabeled as a replay. Failure, drift, or negative confirmation blocks
review. The original promotion, shadow, harmful-memory, TTL, rollback, and trial
budget checks still run. Only an explicit trial review can return a bounded
trial candidate. The controller must still select that trial; general
`production_apply_allowed` remains false.

This expands access to the existing staged trial path, not global apply powers,
alpha caps, or edit budgets. Synthetic positive fixtures exercise the complete
path to `production_trial_allowed=true`; real evidence here does not qualify.
Different prefixes require a new matched measurement, not historical permission.

## Reproduce And Audit

Validation: `python3 -m pytest SpiralInterventionLab/tests -q` reports
**317 passed**, with 17 dependency/runtime warnings. `git diff --check` passes.
Regression fixtures cover execution aliases, source contrasts, invalid doses,
stale contexts, binding drift, non-finite controls, failed confirmation, normal
cap mismatch, exhausted trial budgets, and a synthetic positive bounded trial.

The [packaged evidence](../results/source_direction_comparison_20260914/manifest.json)
includes a lossless compressed source episode, the local response report, and
an audit. The decompressed source hash is
`cdf22094fb565d26cfb875b79faa9ea38b9d03df8030bb8f6862559b01b2f4ae`.
The [raw measured report](../results/source_direction_comparison_20260914/replay.json)
retains positive and negative binding responses, controls, ownership fields,
materialized edits, and explicit review rejection reasons.

```sh
python3 -m SpiralInterventionLab.examples.replay_matched_response_probe \
  --source-jsonl results/source_direction_comparison_20260914/source_episode.jsonl.gz \
  --worker-model-path "$WORKER_MODEL_PATH" \
  --comparison-axis source_localization \
  --output /tmp/source_direction_replay.json
python3 -m pytest SpiralInterventionLab/tests -q
```

The CLI takes the model path explicitly and uses offline loading, with no
provider calls. No-edit greedy reconstruction is restricted to the supported
constrained-rewrite trajectory and verifies the prefix and context; it is not a
general state-restoration facility for trajectories with active edits.

`build_audit.py` in the evidence directory can rebuild hashes and checks from
`--source-jsonl`, `--worker-model-path`, and optionally `--verify-replay` pointing
to a newly generated report. Code hashes are labeled as packaging-time hashes,
not the source episode's original code snapshot.

## Next Decision

Keep the canonical binding fixed. Before increasing strength or granting more
permission, repeat this paired source comparison at another prespecified prefix
or site, changing only that axis. Ask whether canonical probability/top20 lift
and ownership improve together, rather than optimizing the lowercase observable
after seeing its favorable response. Route only genuinely qualified evidence to
the new physical-confirmation path. If none qualifies, preserve the negative
result; the next operator hypothesis is not an excuse to weaken the gate.
