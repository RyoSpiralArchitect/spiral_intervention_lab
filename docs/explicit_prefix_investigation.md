# Explicit Prefix Investigation

## Contract

The controller can investigate a chosen objective without either inheriting
permission from a diagnostic dose or spending the next greedy token merely to
read a result. This repairs reachability, not efficacy. Zero rollout edits remain
a valid outcome when current evidence fails the existing gates.

### Normal-Cap Candidate

`measure_normal_cap_current_prefix` is an explicitly offered `candidate_action`.
It derives exactly one lower dose using the current surface's normal policy cap.
Source expression and tensor hash, target piece, site, layer and alpha must remain
identical. The dose change creates a new candidate/execution identity. No positive
metrics, ownership or safety claims transfer from its parent.

Fresh measurement costs one diagnostic slot and four one-token physical replays
(baseline, null, edit, repeat). Same-child/same-context cached results may be read
without recharging. Parent/child IDs and `inherits_measurement=false` are logged;
the pair auditor checks this lineage rather than requiring identical IDs.
Changed source, stale offer, exhausted budget and candidate capacity fail closed.
The original matched-probe grid remains 0.04/0.16; it is not a new arbitrary sweep.

### Objective Versus Frontier

`diagnostic_frontier_bundle_key` is helper advice, not measured context identity.
The evidence-ID physical-confirmation path can certify an explicitly chosen
objective even when that advice names another objective. It must still pass
current source/execution identity, requested objective/actuator equality, canonical
target binding, two repeated ownership/safety checks and normal trial budgets.

The physical confirmation is passed internally, not accepted as a controller
request flag. Shadow logs retain both `frontier_matches_objective` and
`physical_context_verified`. Legacy packet-only review is still not permission.
Any eventual trial authorization remains exact, current-prefix and single-use.

### Explicit Hold

Use `decision=noop`, an executable diagnostic request, and
`meta.generation_action="hold_prefix"` to read the result before another token.
`strategy_hints.generation_control` exposes availability and remaining rounds.
The loop logs `controller_prefix_hold` acceptance/denial separately from the
command's intent. During an accepted hold no token is generated, no TTL ticks,
and no generation budget or diagnostic budget is replenished.

Holds and automatic qualified candidate/trial handoffs share the existing
`--candidate-handoff-rounds {0,1,2}` pool, rather than adding two more rounds.
Existing readout diagnostic-rethink limits are unchanged and separately counted.
Terminal, blocked, empty, duplicate or failed-restoration requests cannot hold.
The old candidate action `hold` only declines measurement: it is not a pause.
Ordinary noop/omitted `generation_action` retains the old advance-after-followups
semantics. No hold, source reduction or frontier change grants apply authority.

## Validation and Limits

Synthetic tests exercise normal-cap derivation without evidence transfer,
identity drift and capacity rejection, a non-frontier objective reaching exact
trial authorization, mismatched objective/actuator/state/source/binding rejection,
and explicit prefix holds with unchanged token/TTL clocks and bounded calls.
Negative/no-action results are not tuned away to obtain nonzero intervention count.

This does not add a pre-first-token controller call: the first invocation still
follows one worker token. Raw-logit, target-lift, semantic/safety and decoder-mask
rules remain unchanged. Name-piece probability gain is not sentence-role
correctness or task success. The preceding [reachability audit](intervention_reachability_audit.md)
records the older live observations; those immutable traces predate this change.

## Live Paired Observation (2026-09-22)

Both Luna runs used the existing model-local seeds, full model depth, MPS and
twelve diagnostic slots, sequentially. Fresh B0 matched each prior reference.
Package source snapshots match between the two runs and all nine sealed artifact
hashes per run verify. No candidate, diagnostic, hold or apply was forced.

| Profile | B0/C1 score | Generated tokens | Controller commands | Accepted holds | Normal-cap measurements | Rollout edits |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2, 12 layers, FP32, seed 7 | 0.5875 / 0.5875 | 11 | 14 | 3 | 0 | 0 |
| Llama-3.2-3B, 28 layers, FP16, seed 0 | 0.938889 / 0.938889 | 18 | 21 | 3 | 1 | 0 |

Both consumed twelve diagnostic slots; both outputs matched B0 and remained
`task_done=false`. Holding a prefix costs controller calls even though it does
not consume generated-token time or add diagnostic budget. The GPT-2 run did
not expose frozen candidate actions. The Llama run did, and Luna selected two
remeasurements plus one new normal-cap measurement: twelve physical replays,
all restored. Elapsed pair times were 202.5s and 714.4s respectively; these are
shared-device/API observations, not a performance comparison.

The canonical ` Sel` / resid_pre L27 / alpha 0.04 observations were:

| Observed prefix suffix | Step size | Bound probability delta | Top-20 entry delta | Handoff block |
| --- | --- | --- | --- | --- |
| `sample to` | 0.16 | +0.0002275676 | 0 | normal trial differs from measured diagnostic |
| `sample to I` | 0.16 | +1.002647e-13 | 0 | bound target lift |
| `sample to Ivo` | 0.12 | +6.624885e-7 | 0 | bound target lift |

The 0.12 row is a new child candidate, not clipped 0.16 evidence. It produced
logit delta +0.0078125, gap delta -0.0078125, rank 66 -> 66 and no changed
continuation. These rows have different prefixes and are **not** a dose-response
comparison. The earlier 0.16 response does not establish that 0.12 would pass
at that earlier position. Nor would inserting Selim in Ivo's recipient position
establish sentence-role preservation.

The run reached neither evidence-ID physical confirmation nor a trial grant.
Thus frontier-independent *authorization* has synthetic coverage, not a live
success claim. The remaining experimental question is which same-prefix
measurement the controller chooses after a cap mismatch, not whether the
interface can execute a normal-cap measurement at all.

Evidence: [GPT-2 audit](../results/iteration_prefix_control_20260922/gpt2/audit.json),
[Llama audit](../results/iteration_prefix_control_20260922/llama/audit.json),
[admission-path comparison](../results/iteration_prefix_control_20260922/reachability_comparison.json).
Interviews remain qualitative and partially projected, not independent outcome
evidence. The MPS numerical-correctness warning remains unsuppressed.

## Final Hardening Boundary

After sealing these runs, two small fixes were made: the normal-cap diagnostic
edit now explicitly clears the normal materializer's internal trial metadata,
and legacy rethink remaining-round logs subtract legacy rounds rather than all
controller rounds. Public diagnostic permission was false in the live run;
the Llama trial budget stayed at one edit / alpha 0.15 / cost 0.15 throughout,
with no grant or compiled rollout edit. The internal metadata still needed to
be made consistent rather than left to imply permission.

These final fixes have regression coverage, not another live rerun. The sealed
traces and snapshots are preserved unchanged. Final full-suite validation:
`450 passed, 15 warnings, 2 subtests passed`; `git diff --check` passed.
