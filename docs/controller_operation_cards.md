# Controller operation cards

## Problem and scope

The early blueprint comparison exposed a choice-interface limitation: a
canonical replay and an early activation-patch review were both available, but
their costs, possible evidence and prefix timing were described differently.
Frozen candidate actions already had explicit IDs and costs. The new opt-in
interface extends that description to the diagnostic entry points.

`--controller-action-view cards` projects existing runtime offers into a common
menu and uses `controller_operations_v1.txt` for the standard prompt profiles.
The default remains `legacy`. Explicit custom prompt assets/system prompts are
honored; they must implement the operation-ID contract themselves. Both packet
views, normal paired experiments, C1-only experiments and seed sweeps accept the
new option. Checkpoint paths remain CLI arguments.

## Contract

Every card describes the operation's purpose, prerequisites, availability,
possible result, diagnostic-slot and physical-replay cost, and prefix timing.
Diagnostic cost is an upper bound where the existing dispatcher provides one;
actual work is charged by the existing dispatcher. A null physical replay bound
means unknown before dispatch. In particular, a blueprint review costing one
diagnostic slot must not be represented as one physical replay.

The menu preserves the existing offer order. It does not add a ranking, prefer
a target term or assert that an operation will succeed. It covers the offered
diagnostics, frozen-candidate actions and exact evidence-ID confirmations.
Unmeasured blueprint review, historical inspection, new measurement and
conditional confirmation retain distinct possible results. Existing trial-edit
offers and physical authorization checks remain separate and unchanged.

The controller returns `decision=noop` and `meta.operation_id`. An ID binds the
complete input packet and exact original request. The runtime resolves it
without copying fields from controller prose or the canonical frontier. The
choice is exclusive: stale diagnostic aliases, memory requests and simultaneous
tool/observer fields cannot append another operation. Unknown, blocked or stale
IDs return no diagnostic and record a rejection reason; they never fall back to
the canonical request. The provider can retry invalid card commands through its
existing bounded parser-repair mechanism.

Cards do not grant apply authority, pause generation, reserve new budget or
alter the current loop. Timing describes the existing explicit `hold_prefix`
option and its remaining rounds. The controller must still request that hold,
and it is conditional on a usable result. A normal-cap investigation or exact
confirmation also needs a remaining decision round. Their runtime checks run
again when dispatched. A candidate action named `hold` declines measurement;
it does not hold the prefix.

## Implementation and logging

`runtime/operation_cards.py` is a pure projection and resolver. It uses the
existing `candidate_actions` and `candidate_trial` offers, not a second registry
of candidates or permissions. `ProviderControllerClient` builds the menu from
the full packet before compacting the evidence. The loop resolves the selected
ID against that same full packet. Compact formatting cannot truncate or rewrite
the executable request.

The card view replaces duplicated navigation fields, while retaining diagnostic
evidence, budgets, safety checks and trial offers. Its prompt explains operation
selection generally instead of prescribing the historical sequence of diagnostic
families. Therefore a legacy/cards comparison is an **interface-and-prompt
ablation**, not an isolated test of one label or of priority alone.

`controller_operation_menu` records the actual cards shown. Selection/command
events record `controller_operation_id`, the resolved request and any blocked
reason. Provider traces retain raw replies, normalization and parser attempts.
Worker results retain physical work, accounting and restoration evidence.

## Fixed-prefix comparison

The harness `examples/replay_controller_operation_cards.py` reconstructs the
same early context and two recorded predecessor diagnostics used by the earlier
seed-discovery comparison. Both views start from one full packet and the same
available operations. The legacy/cards call order is counterbalanced. No
generation advances after the chosen prefix and no rollout edit is applied.

Optional `--dispatch-selected` then executes each distinct selected diagnostic
once after independently restoring the early context. Identical repeated
choices share that execution receipt; they are not independent replications of
operator efficacy. The harness counts actual `_simulate_decode` calls rather
than estimating physical work from summary row counts. That counter is not a
model-forward count or a wall-time speed benchmark.

```sh
python3 -m SpiralInterventionLab.examples.replay_controller_operation_cards \
  --profile llama_l1 --source-jsonl "$SOURCE_JSONL" \
  --worker-model-path "$MODEL_PATH" --prefix ' Nora' \
  --objective entity_insert:sample:source_body:near_reachable \
  --output "$OUTPUT" --dispatch-selected
```

Compare the selected operation, parser attempts, new frozen cards, physical
simulation calls and remaining diagnostics. A changed choice or an earlier card
is an interface result; target response and final task success require separate
measurements. This reconstruction is not a full historical worker snapshot.

### Llama observation, 2026-09-27

The first live comparison used Llama-3.2-3B, Luna (`gpt-5.6-luna`), the fixed
one-token prefix ` Nora`, and ten remaining diagnostic slots after two recorded
predecessors. Both call orders selected `operator_diagnostic_replay`: legacy
2/2 and cards 2/2. All four responses parsed on the first attempt. The card
selections resolved the offered ID without canonical-frontier field merging.

Each distinct selected request was then executed once from the restored context.
Both used four `_simulate_decode` calls and one diagnostic slot, produced zero
frozen candidate cards, and left the generation context unchanged. No rollout
edit was applied. The two request identities differ because legacy extraction
adds reason/canonical metadata; this is not evidence of different recipes.

Measured API input tokens were 12,319 per legacy call and 10,353 per card call,
about 16% fewer for this packet. The reduction came from the shorter system
prompt (15,219 to 3,299 characters); the payload itself grew from 29,076 to
30,449 characters. This is not a general cost or latency benchmark. The second
call to each interface also used provider-side input caching.

The [compact receipt](../results/controller_operation_cards_llama_20260927/receipt.json)
records hashes, call counts and execution accounting. This validates ID delivery
and a lighter combined input, not better selection, earlier frozen cards, or
task success. The MPS/TransformerLens compatibility warning remains unresolved;
no cross-device numerical claim is made. GPT-2 was not rerun because its local
checkpoint weights were unavailable. Neither a download nor a model substitution
was performed.

## Validation

Focused tests cover exact resolution, unchanged evidence and operation order,
unknown physical costs, blocked/stale IDs, exclusive dispatch, exhausted prefix
rounds, legacy compatibility, and a real worker request path with four synthetic
controlled replays and one diagnostic charge. They do not establish live worker
efficacy.

Final full-suite validation on 2026-09-27:
`python3 -m pytest SpiralInterventionLab/tests -q` completed with 509 passed,
9 subtests passed and 15 dependency deprecation warnings. `git diff --check`
passed. The compact receipt was checked against the full comparison artifact.
