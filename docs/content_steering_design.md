# Content-level steering: design note

Status: P1 complete — verb_slot recipe fired on both substitution cases
(`results/structural_rescue_verbslot_ds_v3_20261109_r2/`); P2 unnecessary
since single-slot pushes sufficed on this split. Design rationale below.
Follows the clause-form gate result
(`results/structural_rescue_clauseform_withonly_ds_v3_20261109/`, controller
8/12 = union of evaluated policies).

## The remaining failure

Four rescue_v3 cases had no win under any evaluated policy. Two families:

- **Verb substitution** (`gen_039`, `gen_074`; `gen_059` before it was saved
  by a lucky re-roll): the second clause's required verb slot is occupied by a
  generic verb — `Boris finishes roster for Edda` where the frame requires
  `take`. The failure is *which token* occupies a mid-sentence position, not
  the clause's form. Receipts already name it:
  `action_substituted:finishes`, `incomplete_frame:take`.
- **Unflippable participles** (`gen_008`, `gen_028`): the starter recipe is
  offered and selected but the first token does not change — reach, not
  content. Out of scope here; it wants a cap/reach study, not a new recipe
  kind.

## Why the current contract cannot express the fix

- `policy.py` requires `ttl_steps=1` and the boundary token mode is `last` —
  every recipe acts on exactly one forward, the first continuation token.
  The substituted verb is 3-5 tokens later, outside any edit's reach.
- The write-contract allowlist (`readout_token_ids`) covers only source
  structural lexemes plus the ` while` starter. A required verb is a content
  token; pushing it is a contract expansion that must be pre-registered.

## Design: a late boundary, same contract shape

Keep `ttl_steps=1`, `resid_pre`, `cap`, guard, and restore-verification.
Change only *where the boundary is* and *what the push targets*.

**Boundary rule v2 (verb slot).** The rescue continuation already runs b0
first. Locate the substituted verb's position in the b0 output — the token
the receipt names (`action_substituted:<verb>`) or, failing that, the first
verb-slot position after the second frame's agent. The late boundary is the
prefix up to that position. Deterministic, derived from the observed failure,
not from a new learned component.

**Recipe kind `verb_slot`.** A readout push toward the required action's
*inflected* token — ` takes`, ` gives`, ` brings`, ` sends` are each single
Llama-3.2-3B tokens (verified). Same `resid_add` at `cap * ||h||` on
`resid_pre` last layer. `applies_to_this_case` = the substitution receipt
family (b0 role-unknown with `action_substituted` or `incomplete_frame` on
frame 2); for the frozen split this is the `who`-template cases.

**Applicability (as implemented).** The recipe is offered only when the
simulated b0 continuation contains a generic verb (the
`_COMMON_ENGLISH_VERBS` list) after the second frame's agent — a lexical
scan, not the substitution receipt. It does not consult role bindings, so it
can in principle fire on cases where the verb is legitimately generic;
on this split it fired exactly on the two receipt-family cases
(`gen_039`, `gen_074`). "Offered where the substitution is observed" is the
honest description — the gate is an observed-failure heuristic, not a
predictive one.

**Catalog phase.** `verb_slot` recipes are added to `catalog_for` behind a
`--verb-slot` flag (new manifest version suffix `_verbslot`), the inflected
token ids are appended to `policy_for(extra_readout_token_ids=...)`, and the
manifest records `verb_slot_rule` (boundary rule text + target inflection
rule) the way `clause_form_applicability_rule` now records the starter gate.

**Controller packet.** The row carries `target_piece` (e.g. ` takes`) and
`applies_to_this_case`. The prompt addendum is one honest sentence: the push
targets the token the b0 output substituted, and helps only where that
substitution is the failure.

## Phase plan

- **P0 — reach diagnostic (no contract change).** At the verb-slot position
  of the b0 continuation for `gen_039`/`gen_074`, record the logit gap
  between argmax (` finishes`) and the required verb token. If the gap is
  inside the measured closure band (~6 logits at cap 0.12), P1 is worth
  running; if it is far outside, the whole design needs a different lever
  before any benchmark run.
- **P1 — verb_slot pilot.** Add the recipe + late boundary, run the 12-case
  rescue_v3 split at cap 0.12 with the same six policies. Success criterion
  is `gen_039`/`gen_074` flipping without losing the eight wins — N=2 for
  the target family, so the claim stays "the mechanism can fire", nothing
  more.
- **P2 — multi-step (only if P1 shows reach but loses).** Chain two ttl=1
  pushes in one branch: connective flip at the first boundary, verb push at
  the verb slot. This needs a scheduled-edit mechanism (`step_offset` on the
  edit or an ordered (boundary, recipe) list) — the current command schema
  has no delay field, so P2 is a runtime change, not just a new recipe.

## Honest caveats

- Two substitution cases cannot support an efficacy claim; P1 at most shows
  the mechanism exists. A real claim needs a screen v4 that generates more
  substitution failures.
- The late boundary is derived from the b0 failure, so the recipe can never
  win on cases b0 already passes — by construction it only runs where it is
  needed, which also means its gate is fit to observed failures again.
- Inflection is resolved by template (`<name> <verb>s`), not grammar; a case
  needing a bare ` take` would need the token choice widened — record the
  rule in the manifest so the assumption is visible.
- `gen_059` showed this failure is sometimes saved by a connective re-roll;
  P1's contribution must be measured against the policy-union baseline, not
  against b0 alone.
