# Readout Escape Insight Taxonomy

Status date: 2026-05-17

This note organizes the current readout-escape findings into three buckets:

1. mechanistic-interpretability research value
2. engineering / practical runtime value
3. ideas that are both

The short version:

> The project has not solved readout escape yet, but it now has a much cleaner
> measurement language for why it has not solved it.

The useful boundary is:

```text
evidence != permission
rank/readout movement != target-owned actuator
diagnostic support != production apply
```

## 1. Mechanistic-Interpretability Value

These findings are academically interesting even if the current GPT-2 recipes do
not transfer unchanged to larger models.

### Readout Escape Is An Attractor-Basin Problem

The failure mode is not just "the model forgot a term."

In the constrained rewrite runs, the worker can expose a plausible `budget`
frontier while the answer boundary still collapses into repeated or junky first
tokens such as `the the the the`. That suggests the answer-start state has
already entered a local basin where broad semantic support is not enough.

Useful framing:

- semantic visibility can exist without readout competitiveness
- first-token readout needs its own diagnostics
- entropy reduction can be harmful if it sharpens the wrong basin
- "stability" is not the same as progress

### Ownership Is A Causal-Faithfulness Problem

The most important interpretability distinction is ownership:

```text
did anything move?
did the intended bundle move?
did the intended bundle own the useful lift?
did the movement improve target readout?
```

The current taxonomy is:

- `self_actuator`: the intended bundle owns useful lift
- `bridge_actuator`: another bundle can serve as a useful route for the objective
- `cross_bound`: lift exists, but another bundle steals more of it
- `dead_actuator`: no meaningful movement
- `collapse_sharpener`: the intervention sharpens a bad basin
- `self_rank_carrier`: the intended bundle owns rank/readout movement, but not target mass/top-20 lift
- `self_target_actuator`: intended bundle ownership plus direct target readout lift

This is close to a causal-faithfulness question for activation steering:

> Did the intervention work through the path we think it worked through?

### Rank Carriers Are Not Target Actuators

The current readout steering and activation-patch diagnostics often find
`self_rank_carrier` behavior:

- focus rank moves
- ownership can stay with the intended bundle
- collapse may not worsen
- target mass/top-20 still does not improve enough

This is a useful negative result. A rank/readout carrier is evidence that a
direction touches the right neighborhood, but it is not yet an actuator.

Mechanistic hypothesis:

> Some directions can make the target representation more visible to probes
> without moving the answer-boundary logit geometry enough to compete.

### Stealers Are Operator-Time Competitors

The `send` / `budget` case made a second distinction visible:

- selector-time competitor: the bundle currently winning or nominated
- operator-time stealer: the bundle that captures lift when a recipe is applied

Those are not guaranteed to be the same. This matters for contrastive recipes:

```text
target - base_winner
```

is not always the right direction. Sometimes the right diagnostic question is:

```text
target - observed_stealer(target)
```

### Attention Heads Are Carrier Diagnostics First

Attention-head ablation and scaling can reveal rank/readout sensitivity. That
does not yet justify attention editing as an actuator.

Current interpretation:

- attention heads can identify carrier routes
- partial scaling can move rank-like signals
- target mass/top-20 certification is still required
- attention edits should remain diagnostic-only until ownership-style replay
  shows target lift

### SAE / Readout Analyzer Evidence Is A Feature-Emitter Track

SAE-style analysis is promising as a way to improve the map of the local basin.
But the current design intentionally accepts information loss in exchange for
policy cleanliness:

- analyzer emits feature hints and risk signals
- controller owns policy
- runtime owns physical guardrails
- feature evidence does not become an answer channel

Academic claim to test later:

> Some failures reported as "steering does not work" may be ownership failures:
> the representation moves, but the useful lift is stolen or remains a rank
> carrier.

## 2. Engineering / Practical Runtime Value

These findings matter because they make the system safer and more debuggable,
even before task success improves.

### Typed Intervention DSL And Runtime Budgets

The runtime now treats interventions as bounded edits:

- typed operations such as `resid_add`, `kv_mix`, `rank1_patch`, and readout-direction patches
- TTL-scoped edits
- norm and step-size controls
- compile validation
- rollback accounting

This prevents the controller from turning a vague idea into an unbounded edit.

### Control-Plane Constitution

The project now has a stable responsibility split:

| Plane | Role |
| --- | --- |
| Controller | policy owner; asks for diagnostics; decides noop/trial/apply |
| Analyzer / sidecar | evidence emitter only |
| Gate | pure evaluation of controller-owned rules |
| Operator replay | certification evidence |
| Runtime guardrail | physical invariants: dtype, shape, budget, TTL, rollback |

This prevents the common failure where a helper quietly becomes a second
controller.

### Candidate Compiler Improvements

The candidate compiler is now less noisy:

- source provenance is separated
- `source_body` is prioritized over constraint headers
- same-term / same-family duplicates are pruned
- bundles are formed after dedupe
- candidate logs show before/after prune counts and provenance

This turns "seen by observer" into "executable candidate" more honestly.

### Diagnostic Evidence Ledger

The ledger is one of the most practical pieces of infrastructure.

It makes each frontier bundle readable as:

- `readout_reachable`
- `head_sensitive`
- `feature_supported`
- `rank_readout_carrier`
- `diagnostic_operator_supported`
- `policy_candidate_ready`
- `production_operator_certified`
- `blocked_by`
- `next_evidence_needed`

The ledger also keeps gap-closer rows visible before truncation, so important
diagnostic rows do not vanish from the controller's view.

### Production Trial Ladder

The system now has a ladder between "diagnostic evidence" and "production apply":

```text
candidate review
-> runtime support probe
-> promotion gate
-> production shadow replay
-> production trial gate
-> bounded production trial
-> outcome ledger
```

The key engineering value is that `production_trial` is explicitly not
production apply. It is a bounded, TTL=1 experiment whose outcome becomes
evidence.

### Positive Operator Memory

Earlier loops mostly learned what to avoid:

- harmful ledger
- collapse-sharpener veto
- dead-actuator blocks
- bridge exhaustion
- rank-carrier-not-target blocks

Positive operator memory adds bounded curiosity:

- `ownership_preserving`
- `rank_carrier`
- `target_reachable`
- `top20_gap_measured`
- `top20_gap_closer_candidate`
- `top20_gap_closer_certified`
- `anti_collapse`
- `neutral_stable`
- `rank_to_mass_convertible`

This is not a permission system. It is a way to say:

> This family is still uncertified, but it is locally worth deepening.

It is scoped by objective bundle, target piece, recipe family, and
`operator_recipe_id`, and it carries a short TTL plus
`stale_after_context_change=true`. That keeps positive memory as bounded
curiosity rather than long-lived superstition.

### Readout Gap-Closer Sweep

The latest readout-steering deepening path adds narrow gap-probe recipes and
summarizes absolute gap plus baseline-relative movement:

- `readout_gap_closer_recipe_count`
- `readout_gap_probe_recipe_count`
- `readout_gap_closer_candidate_count`
- `readout_gap_closer_certified_count`
- `best_readout_gap_closer_recipe_name`
- `best_readout_gap_closer_target_top20_threshold_gap`
- `best_readout_gap_closer_target_top20_threshold_gap_delta`
- `best_readout_gap_closer_target_piece_logit_delta`

Latest GPT-2 replay:

```text
gap_count = 3
best_gap_recipe = post_bridge_target_readout_patch_l005_a060_gap
best_gap = 6.099704
best_target_piece_logit_delta = 0.002202
best_gap_delta_vs_baseline = not yet certified in this run
episode_output = the the the the
```

This is not a task win. It is a better local measurement. The term
`gap_closer_certified` stays reserved for baseline-relative gap reduction that
is also collapse-safe; otherwise the observation is only a measured/candidate
signal.

## 3. Both: Research And Engineering Value

These ideas are scientifically interesting and also improve the runtime.

### Evidence Is Not Permission

This is both a research hygiene rule and a software safety rule.

Evidence types include:

- analyzer support
- attention sensitivity
- readout-logit adjacent probes
- rank-carrier movement
- positive operator memory
- shadow replay

None of these alone grants production apply.

### Objective And Actuator Can Diverge

The system now distinguishes:

```text
objective_bundle_key
actuator_bundle_key
realized_lift_bundle_key
```

This is essential for bridge plans. It lets the controller express:

> The objective is `budget`, but the current step may need a `send` bridge
> actuator, if and only if the bridge is certified.

It is also a mechanistic statement about distributed computation: the path that
helps a term emerge may not be the term's own bundle.

### Diagnostics Are Active Measurement, Not Passive Logging

The controller can now ask for:

- activation-patch candidate review
- cross-bundle bridge search
- extra operator diagnostics
- attention head diagnostics
- readout steering deepening
- production-trial follow-up review

This turns the loop into a controlled experimental system. The controller gets
more curiosity without getting broader physical authority.

### Failure Modes Are Now Typed

Typed failure modes make research and engineering move together:

| Failure | Research reading | Runtime consequence |
| --- | --- | --- |
| `dead_actuator` | no causal handle found | do not deepen that exact recipe |
| `collapse_sharpener` | wrong basin sharpened | veto family temporarily |
| `cross_bound` | ownership leak / stealer | inspect bridge or contrastive route |
| `self_rank_carrier` | right neighborhood, weak readout | deepen gap-closing |
| `self_target_actuator` | possible causal actuator | send through shadow/trial ladder |

## Current Integrated Reading

The system has progressed through four stages:

1. detect collapse
2. nominate a plausible frontier
3. distinguish rank/readout carriers from target actuators
4. locally deepen the most promising gap closers

The current state is:

- visibility is much better than actuation
- `budget` can be seen and locally touched
- readout steering can move nearby logits but still leaves a large top-20 gap
- production apply remains correctly closed
- the controller now has a positive local search signal instead of only vetoes

This is a healthy plateau, not a dead end.

## Next Strategy

### Immediate: Gap-Closer Delta Accounting

The next run should compare gap closers against a no-edit readout baseline:

- absolute target top-20 gap
- gap delta versus baseline
- target-piece logit delta
- target-piece probability delta
- target rank after edit
- repeat/collapse deltas

This will tell us whether the best gap closer is actually closing the gap or
merely being selected as the least bad row.

### Short Term: Localize Around The Best Gap Recipe

The current best recipe is:

```text
post_bridge_target_readout_patch_l005_a060_gap
```

Suggested bounded sweep:

- `negative_scale`: `0.0`, `0.025`, `0.05`, `0.075`, `0.10`
- `alpha`: capped values around `0.05`, `0.06`
- same target token set
- same bad-attractor set
- no production apply

Goal:

> Improve target top-20 gap without collapse sharpening.

### Medium Term: Carrier-To-Actuator Conversion

If gap closers keep improving logit deltas without top-20 hits, test whether a
two-stage diagnostic can convert carrier movement:

```text
stage 1: anti-collapse or low-attractor target readout
stage 2: target readout / contrastive target readout
```

Keep this shadow-only until both stages are non-harmful.

### Medium Term: Bridge Plan As A First-Class Research Object

If `budget` remains hard to lift directly, treat it as a possible payload term:

```text
objective = budget
actuator = certified bridge bundle
reason = budget self-actuator unavailable at this step
```

This should stay explicit in logs, not hidden inside selection.

### Scale-Up Plan

GPT-2 should remain the measurement rig. The first scale-up claim should be:

> Do the classification axes transfer?

not:

> Does the exact GPT-2 recipe transfer?

The categories to test on larger local models:

- `self_rank_carrier`
- `self_target_actuator`
- `cross_bound`
- `bridge_actuator`
- `collapse_sharpener`
- `positive_operator_deepening_plan`

## PR Scope Recommendation

This PR should be framed as a control-plane and measurement PR, not a task-score
PR.

Recommended title:

```text
[codex] add readout escape diagnostic contracts and positive operator memory
```

Recommended claim:

> This adds diagnostic contracts that separate visibility, rank-carrier support,
> target-owned actuation, bridge search, bounded production trial, and positive
> local deepening without granting production apply.

Recommended non-claim:

> This does not solve the constrained rewrite task yet.
