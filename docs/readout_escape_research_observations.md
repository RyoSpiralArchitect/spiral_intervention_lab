# Readout Escape Research Observations

Status date: 2026-05-15

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

## Latest GPT-2 Observation

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

## Near-Term Experimental Plan

Recommended next experiments:

1. Continue the contrastive source-local recipe sweep.
   Try small variations around `source_centered_pm1_minus_stealer_l025` before opening larger search.

2. Add confirmation replay for neutral follow-ups.
   Neutral is not success, but it is a safer local neighborhood than harmful/regressing.

3. Compare centered and fused seeds.
   Keep the matrix small:
   `source_centered_pm1`, `source_centered_pm1_minus_stealer_l025`, `source_term_fused`, and one orthogonal variant.

4. Keep attention ablation diagnostic-only.
   Use it to identify carrier heads and rank sensitivity. Do not promote it to apply until it has ownership-style certification.

5. Strengthen the readout analyzer as a feature emitter.
   Improve evidence quality, not authority. Prefer backend-swappable feature vectors over controller-visible answer hints.

6. Preserve controller-owned explanations.
   Every noop, trial, veto, or follow-up should leave a one-sentence reason reconstructible from structured fields.

7. Treat GPT-2 as a measurement rig.
   The next larger-model step should test whether the classification axes transfer, not whether a specific recipe transfers.

## Open Questions

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

The next breakthrough likely requires converting non-dead diagnostic carriers into target-owned actuators, not giving the controller broader apply power.
