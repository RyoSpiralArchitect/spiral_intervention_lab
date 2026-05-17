# Readout Escape Research Observations

Status date: 2026-05-17

This note summarizes the current research state of the readout-escape line. It is intentionally about observations and interpretation, not just implementation history.

## Core Question

Can a controller improve a worker model through bounded internal interventions while preserving the worker as the actor?

For the current phase, that question has narrowed:

> Can the system find a bounded, auditable operator that improves first-token answer readout, and can the resulting lift stay owned by the intended bundle?

The latest experiments are mostly on GPT-2. They should be read as measurement-system development, not as a claim that a specific GPT-2 recipe will transfer to larger models.

## Current Architecture Contract

The project is converging on a strict control split:

| Component | Role | Not allowed to do |
| --- | --- | --- |
| Controller | Own policy, select actions, request diagnostics, decide noop/apply/trial | Delegate final policy to helpers |
| Analyzer / sidecar | Emit bounded evidence and feature hints | Override, promote, veto, or select |
| Gate | Evaluate controller-provided rules and expose reasons | Become a hidden selector |
| Operator replay | Produce certification evidence | Grant production apply by itself |
| Runtime guardrail | Enforce physical invariants: budget, dtype, shape, TTL, compile validity | Make semantic policy choices |

The practical rule is:

> If an action changes the worker, the controller log must be able to explain why in structured fields without relying on helper mystique.

## What Is No Longer the Main Bottleneck

The system is no longer primarily blocked on seeing the frontier.

Recent direct-scan readout-escape packets can expose a plausible `budget` frontier with:

- source-body provenance
- reachable/readout-local signals
- attention-head sensitivity
- feature support from the readout analyzer path
- operator diagnostic rows
- per-bundle diagnostic status

This means the stack can now often say:

- what term is relevant
- where it appears in the source
- which diagnostic families are non-dead
- why a candidate is still not production-ready

That is real progress. Earlier failures mixed visibility, candidate generation, and unsafe application into one foggy failure mode. They are now separated.

## Main Current Bottleneck

The bottleneck is operator quality at the answer boundary.

More precisely:

1. The selector can nominate plausible challengers.
2. The diagnostics can find support for those challengers.
3. But the available runtime operators often fail to make the intended bundle win.
4. Some operators move the distribution while still moving it in the wrong way.

The important distinction is now:

| Question | Current answer |
| --- | --- |
| Can the system see `budget`? | Often yes |
| Can it nominate `budget`? | Often yes |
| Can it compile a bounded candidate? | Often yes |
| Can it certify a safe self-actuator? | Usually no |
| Can it avoid applying unsafe candidates? | Increasingly yes |

## Ownership Findings

A major observation is that "the logits moved" is not enough.

Ownership-aware replay distinguishes:

- `self_actuator`: intended bundle receives the lift
- `bridge_actuator`: another bundle can act as a useful bridge for the objective
- `cross_bound`: some lift exists, but another bundle steals more of it
- `dead_actuator`: no meaningful movement
- `collapse_sharpener`: intervention sharpens a bad basin
- `noisy_or_harmful`: movement is not interpretable as safe progress

This has changed the research target. The goal is not raw activation movement; it is target-owned, phase-appropriate movement.

## Production Trial Ladder

The current activation-patch ladder is:

```text
activation_patch_candidate_review
-> activation_patch_runtime_support_probe
-> activation_patch_promotion_gate_review
-> activation_patch_production_shadow_replay
-> activation_patch_production_trial_gate_review
-> production_trial
-> production_trial_outcome_ledger
-> alternate evidence / confirmation / veto
```

The important control invariant:

`production_trial` is not production apply.

It has:

- `ttl_steps=1`
- explicit rollback/norm limits
- separate budget accounting
- `production_apply_allowed=false`
- `certified_for_apply=false`
- ledgered outcome evidence

This lets the controller test one bounded step without turning a diagnostic into a general permission slip.

## GPT-2 Observation: Follow-Up Trial Ladder

The recent direct-scan replay established a two-stage trial comparison.

### Primary Trial

Recipe:

```text
readout_escape|activation_patch|resid_pre|L6|source_term_token_to_last|blend|a0.050
```

Outcome:

```text
production_trial_budget_class = primary
trial_effect_class = regressing
verdict = harmful
signal_profile = regressing
source_localization = source_term_token
contrast_mode = none
```

Interpretation:

The source-local term-token recipe is not safe as a first bounded trial. It still moves the worker toward a bad basin rather than task progress.

### Follow-Up Trial

Recipe:

```text
readout_escape|activation_patch|resid_pre|L6|source_centered_pm1_minus_stealer_l025_to_last|blend|a0.050
```

Outcome:

```text
production_trial_budget_class = alternate_followup
production_trial_followup_allowed = true
trial_effect_class = neutral
verdict = neutral
signal_profile = flat
source_localization = source_centered_pm1_minus_stealer_l025
contrast_mode = minus_stealer
stealer_term = send
```

Interpretation:

This is not task success. But it is a meaningful negative-to-neutral shift:

- the primary source-local recipe regressed
- the contrastive follow-up did not immediately regress
- the follow-up used a separate trial budget, not a widened primary budget
- production apply remained closed

That makes the comparison cleaner. The system can now say:

> The first source-local actuator was harmful. A contrastive source-local alternate was bounded and neutral. Continue searching around contrastive/source-centered recipes, but do not treat this as production permission.

## Why This Matters

The latest win is not better task output. It is better research hygiene.

Before this split, a failed trial could easily cause one of three bad interpretations:

- "the controller should try harder"
- "the whole surface family is bad"
- "the helper should get more authority"

The current logs support a better interpretation:

- a specific recipe was harmful
- the controller requested alternate evidence
- the alternate recipe was reviewed, shadowed, gated, and trialed
- the alternate was neutral, not certified
- no production permission was granted

That is a clean measurement loop.

## Working Hypotheses

### 1. Visibility Is Ahead Of Actuation

The system can gather more evidence than it can safely execute. This is healthy as long as evidence does not become hidden policy.

### 2. Broad Span Means Are Too Shared-Context Heavy

`source_span_mean` tends to carry too much shared sentence context. It can move the distribution without producing target-owned lift.

Source-local and contrastive variants are more promising because they try to preserve carrier structure while reducing stealer overlap.

### 3. Contrastive Recipes Are Not Dead

The contrastive follow-up moving from harmful/regressing to neutral/flat suggests that subtracting a stealer direction can reduce collapse risk. It does not yet prove target lift.

### 4. Attention Head Ablation Is Diagnostic First

Attention diagnostics can show rank/readout carrier sensitivity, but the current evidence does not justify treating attention ablation as a production actuator.

### 5. SAE / Readout Analyzer Belongs On The Evidence Plane

SAE-style analysis is promising as a feature-emitter backend. It should stay on the analyzer side of the constitution:

- emit feature hints
- expose support and risk
- improve the controller's map
- avoid becoming a second generator or selector

## What Not To Claim Yet

Do not claim:

- GPT-2 recipes will transfer unchanged to larger models
- activation patch is currently a solved actuator
- neutral follow-up is success
- SAE-sidecar evidence is production certification
- attention head sensitivity means attention editing is safe
- the controller should get broader apply authority

The current claim is narrower:

> The measurement and control-plane contracts can now distinguish visibility, nomination, diagnostic support, bounded trial, harmful effect, and neutral alternate follow-up without collapsing those into production apply.

## Current Best Research Reading

The project has moved from "can we see anything?" to:

> Can a rank/readout carrier be converted into a target-owned, safe runtime actuator?

The latest neutral follow-up is encouraging because it shows that recipe structure matters. It suggests there may be a gradient from:

```text
harmful span/term patch
-> neutral contrastive source-local patch
-> maybe target-lifting contrastive/source-centered patch
```

But the last arrow is not yet established.

## Latest GPT-2 Observation: Rank Carrier Split

The newest direct-scan replay sharpens that last arrow.

The previous activation-patch diagnostics could label a recipe as a
`self_actuator` when it produced large ownership/rank movement:

```text
self_delta > 0
alignment_margin >= 0
focus_rank_delta > 0
```

That was too coarse. A recipe can move the intended bundle's rank without
moving the target token mass or creating a top-20 target hit. In the current
readout-escape setting, that is not yet a safe actuator.

The runtime now separates:

| Class | Meaning |
| --- | --- |
| `self_target_actuator` | Intended bundle owns lift and target mass/top20 improves |
| `self_rank_carrier` | Intended bundle owns rank/readout movement, but target mass/top20 does not improve |
| `wrong_direction` | Another bundle receives more useful lift |
| `collapse_sharpener` | Intervention sharpens a bad/repetitive basin |
| `dead` | No meaningful movement |

This split changed the live replay behavior. The earlier ladder could promote a
rank-moving activation patch into a bounded production trial; the latest ladder
blocks that path and requests more evidence instead.

Observed diagnostic sequence in local GPT-2 direct-scan replay:

```text
activation_patch_candidate_review
-> compare_extra_operator_diagnostics
-> cross_bundle_bridge_search
-> compare_extra_operator_diagnostics(post_bridge_exhaustion)
```

The important part is what did not happen:

```text
no production_trial
no production apply
no repeat of the same compare loop
```

The post-bridge expansion produced:

```json
{
  "status": "rank_carrier_family_found",
  "failure_mode_counts": {
    "self_rank_carrier": 3,
    "wrong_direction": 9
  },
  "best_target_actuator_recipe_family": null,
  "best_rank_carrier_recipe_family": "resid_pre|source_term_token|blend",
  "recommended_next_family": "convert_rank_carrier_to_target:resid_pre|source_term_token|blend"
}
```

Interpretation:

- `budget` is no longer merely invisible.
- Some recipes can move `budget` in rank/readout space.
- Those recipes still do not lift `budget` as a first-token target.
- Many expanded recipes are still stolen by the neighboring `send` bundle.
- The next operator problem is conversion, not broader controller authority.

This is a useful negative result. The system can now say:

> I found a rank/readout carrier for the objective, but not a target actuator.
> Production trial remains closed. Search for a recipe that converts the carrier
> into target mass/top20 lift.

That is a much better failure mode than either applying the carrier directly or
calling the entire family dead.

## Near-Term Experimental Plan

Recommended next experiments:

1. Convert the best rank carrier into a target actuator.
   The current best family is `resid_pre|source_term_token|blend`. The next
   sweep should ask whether that rank carrier can be turned into target
   mass/top20 lift without increasing collapse risk.

2. Continue the contrastive source-local recipe sweep.
   Try small variations around `source_centered_pm1_minus_stealer_l025` and
   `source_term_token_to_last` before opening larger search.

3. Add confirmation replay for neutral follow-ups.
   Neutral is not success, but it is a safer local neighborhood than harmful/regressing.

4. Compare centered and fused seeds.
   Keep the matrix small:
   `source_centered_pm1`, `source_centered_pm1_minus_stealer_l025`, `source_term_fused`, and one orthogonal variant.

5. Keep attention ablation diagnostic-only.
   Use it to identify carrier heads and rank sensitivity. Do not promote it to apply until it has ownership-style certification.

6. Strengthen the readout analyzer as a feature emitter.
   Improve evidence quality, not authority. Prefer backend-swappable feature vectors over controller-visible answer hints.

7. Preserve controller-owned explanations.
   Every noop, trial, veto, or follow-up should leave a one-sentence reason reconstructible from structured fields.

8. Treat GPT-2 as a measurement rig.
   The next larger-model step should test whether the classification axes transfer, not whether a specific recipe transfers.

## Open Questions

- Can the current `resid_pre|source_term_token|blend` rank carrier be converted into positive target lift?
- Can contrastive source-local activation patches produce positive target lift, or only avoid harm?
- Is `budget` inherently a payload term at this decode point, requiring a bridge actuator such as `send`?
- Which evidence type best predicts safe trial outcome: ownership replay, attention carrier score, SAE feature support, or first-piece reachability?
- Does neutral follow-up become helpful when combined with a later composition phase?
- How much loop slack should the controller get before it starts repeating stale diagnostic ladders?
- When scaling up, do `self_actuator / bridge_actuator / cross_bound / collapse_sharpener` remain useful categories?

## Current Summary

The system has not solved readout escape yet.

But it has made the failure much sharper:

- candidate visibility is much better
- helper authority is better contained
- unsafe apply is better blocked
- operator failures are now recipe-local rather than mystical
- a harmful primary trial can now lead to a bounded neutral alternate trial
- a rank-moving recipe can now be blocked as `self_rank_carrier` instead of mistaken for a production-safe self-actuator

The next breakthrough likely requires converting non-dead diagnostic carriers into target-owned actuators, not giving the controller broader apply power.

## Positive Operator Memory

The latest loop adds a small positive-memory layer for operator diagnostics.

This is intentionally not a permission system. It records bounded positive
traits such as:

- `ownership_preserving`
- `rank_carrier`
- `target_reachable`
- `top20_gap_measured`
- `top20_gap_closer_candidate`
- `top20_gap_closer_certified`
- `anti_collapse`
- `neutral_stable`
- `rank_to_mass_convertible`

The purpose is to keep the controller from learning only negative governance.
The harmful ledger, veto memory, bridge exhaustion, and collapse-sharpener
classes are good at saying "do not apply this." Positive operator memory adds a
separate, still diagnostic-only signal:

> This family is not certified, but it produced a non-harmful partial movement
> worth deepening locally.

Positive memory is deliberately scoped and short-lived. It carries the current
`objective_bundle_key`, objective term, target piece, recipe family,
`operator_recipe_id`, `ttl_steps`, and `stale_after_context_change=true` so a
local hint cannot quietly become a cross-prompt superstition.

The controller now receives a `positive_operator_deepening_plan` derived from
that memory. The plan is structured as diagnostic guidance:

```json
{
  "kind": "positive_operator_deepening_plan",
  "permission": "diagnostic_only",
  "production_apply_allowed": false,
  "ttl_steps": 2,
  "stale_after_context_change": true,
  "scope": {
    "objective_bundle_key": "kv_pair:budget:source_body:72:73",
    "target_piece": " budget",
    "operator_recipe_id": "post_bridge_target_readout_patch_l005_a060_gap"
  },
  "recipe_family": "readout_steering",
  "next_action": "deepen_local_gap_closer",
  "deepening_axis": "target_top20_gap_closing",
  "reason_code": "positive_memory_local_gap_closer"
}
```

This preserves the constitution:

- memory records promising partial effects
- the plan explains what to inspect next
- the controller decides whether to request that diagnostic
- production apply remains closed until a separate certification path passes

In the current GPT-2 replay, readout steering remains a `self_rank_carrier`
family rather than a target actuator. The useful progress is that the loop can
now say:

> Readout steering moved the target-owned readout signal without collapse, but
> the target top-20 gap remains large. Deepen local gap-closing diagnostics;
> do not treat this as production permission.

The next implementation step turns that plan into a small, auditable local
sweep. The readout-steering deepening path now marks gap-probe recipes with
`readout_gap_closer_recipe=true` for backwards-compatible visibility, but the
evidence is split into candidate vs. certified levels. A recipe is only a
certified closer when its top-20 threshold gap shrinks against the no-edit
baseline without collapse-sharpening.

```json
{
  "readout_gap_closer_recipe_count": 3,
  "readout_gap_probe_recipe_count": 3,
  "readout_gap_closer_candidate_count": 3,
  "readout_gap_closer_certified_count": 0,
  "best_readout_gap_closer_recipe_name": "target_readout_patch_l005_a060_gap",
  "best_readout_gap_closer_target_top20_threshold_gap": 6.09,
  "best_readout_gap_closer_target_top20_threshold_gap_delta": null
}
```

The key contract is:

- `top20_gap_measured`: the gap was observed
- `top20_gap_closer_candidate`: the target piece moved in a plausible direction, but baseline gap delta is not yet sufficient
- `top20_gap_closer_certified`: the gap shrank relative to baseline while staying collapse-safe

The first gap-closer sweep stays deliberately narrow:

- pure target readout at the current alpha cap
- target readout with very light attractor subtraction
- contrastive readout with a low competitor subtraction

The research question is not "did this solve the task?" yet. It is:

> Did any bounded readout steering variant reduce the target top-20 gap without
> becoming a collapse sharpener or a wrong-direction bridge?

Latest GPT-2 direct-scan replay with this visible gap-closer ledger produced:

```text
gap_count = 3
best_gap_recipe = post_bridge_target_readout_patch_l005_a060_gap
best_gap = 6.099704
best_target_piece_logit_delta = 0.002202
```

The output still fell into the repetition basin (`the the the the`), so this is
not an escape success. The useful result is narrower: the controller can now
compare several bounded readout gap closers and carry the best one forward in
the positive deepening plan without treating it as production permission.
