# Norm-Relative Structural Write Contract

Date: 2026-11-03. Follows [`structural_rescue_pilot.md`](structural_rescue_pilot.md)
(v1, null by construction) and [`operator_dose_response.md`](operator_dose_response.md)
(the dose at which structure moves).

## Why v1 could not move anything

The v1 pilot ran every recipe at alpha 0.04 with an absolute step cap of 0.08.
At the Llama boundary the edited residual norms are tens of units (about 43.5
at `blocks.27.hook_resid_pre`), so the normal caps write about 0.2% of the
state. The dose study found that the role-correct review rewrite needs a
`,` push of about 6% of that norm, while writing a name needs 12-24%, which is
token injection. "Bounded" was therefore defined in units that cannot express
the one structural move that works, nor rule out the one that would cheat.

## The contract

`norm_relative_structural_v1` is opt-in and off by default:

- **Write cap.** Each written position moves by at most `r` times its own
  pre-edit norm (`r = 0.08` here). The cap is enforced at hook time on the hook's
  output minus its input, so it holds for any adapter. Inside the contract it
  replaces the absolute alpha and step caps; norm-clip, TTL, revertibility,
  deny rules, op allowlists and expression limits still apply.
- **Structural targets only.** Readout pushes may target only source tokens
  whose lexeme is in `clause_structure.SOURCE_LEXEMES` (punctuation and
  connectives). Source refs may point only at those tokens' positions or the
  current last token. Lexemes that occur inside a required term (`before` in
  `before dusk`) are protected and excluded, as are names and content words,
  which are never in the lexicon. No negative readout targets.
- **One token, one edit.** TTL 1, exactly one edit per command, last-token
  `resid_pre`/`resid_post` only, `resid_add` or `activation_patch` blend only,
  never a production trial. Raw requests are bounded (blend alpha <= 1, add
  alpha <= 1024) so half-precision arithmetic stays finite before the guard.
- **Natural boundary, once per episode.** Enforced by the harness: the v2
  runner applies one command at the v1 boundary in a restored branch.

## Implementation

| Piece | Where | Default behaviour |
| --- | --- | --- |
| `WriteContract`, `HarnessPolicy.write_contract` | `runtime/policy.py` | `None`: any edit declaring the contract is rejected |
| Declaration `meta.write_contract`, `meta.relative_write_cap` | `runtime/edit_budget.py` | Partial or unknown declarations fail closed |
| Hook guard `clip_relative_write` | `runtime/edit_budget.py`, wrapped in `runtime/compiler.py` | Only contract edits are wrapped |
| Budget in relative units | policy usage, compiler registration | Contract edits charge `r`, never the raw request, and only to the main pool |
| Telemetry | `runtime/worker.py` collector | New keys appear only on contract rows |
| Per-prompt allowlist, recipes, commands | `runtime/structural_contract.py` | Not used unless a runner calls it |

No schema dataclass gained a field, so `asdict(caps)`-based recipe IDs and
`asdict(command)` comparisons are unchanged. The v1 manifest hash remains
`81326474db5ae819ba997612019c50f77b52a7ad057fc4c4567d786bce572324` (tested).

The guard clips in float32 and casts back. Telemetry reports the realized write
after the cast, which may exceed `r` by the activation dtype's rounding; the v2
runner refuses a branch whose realized relative write exceeds `r + 0.002`.
Contract rows report `relative_write_cap`, `guard_call_count`,
`written_positions`, `site_norm`, `write_limit_norm`, `raw_write_norm`,
`write_norm`, `relative_write_norm` and `relative_clip_saturated`. For patches,
the adapter's own `blend_*` keys describe the blend before the guard.

## v2 pilot

`examples/structural_rescue_benchmark_v2.py` reruns the frozen v1 cases, boundary
rule, restored branches, forward ceilings and repeated baselines under the
contract. Its manifest is separate (`structural_rescue_v2_norm_relative_contract`).

Catalog at each boundary, from the first three eligible source tokens:

- **Patch** recipes: the v1 `common`, `common_parallel` and `common_orthogonal`
  components for each pair, at the middle layer (`n_layers // 2`), blend alpha 1,
  so the write is the full displacement clipped to `r`. Pairs containing a
  protected lexeme are dropped, so the catalog is not identical to v1's.
- **Readout** recipes: one push per eligible token at the last layer
  (`n_layers - 1`), requesting exactly `r` times the boundary norm. This is a
  deliberate extension of v1's catalog, made because the dose study located the
  working move there.

| Policy | Choice |
| --- | --- |
| B0 | No edit |
| Fixed | Orthogonal patch of the first eligible pair (v1 rule) |
| Fixed readout | Readout push of the first eligible source token |
| Random | SHA-256 of case ID and seed 17 modulo catalog size |
| Controller | One JSON choice of an offered recipe ID, or noop |

The controller receives recipe kinds, source pieces, layers, requested and
expected relative writes and patch geometry; never expressions, tensors, doses,
role tuples, references or measured outcomes. Calibration cases additionally run
every catalog recipe; that matrix is not shown to the controller.

## Baseline-failing cases (pre-registered)

All six v1 confirmation cases succeed at baseline, so they can only show harm.
`tasks/structural_rescue_generator.py` draws two-clause delivery cases from one
seeded stream: three rotating source families (purpose first, deadline first,
purpose last), five connectives, six deadlines and name/object pools disjoint
from the v1 cases. Verbs are limited to forms the role check recognizes.

`examples/structural_rescue_screen.py --prepare` freezes the generator version,
seed 20261103, 72 candidates with prompt hashes, and the selection rule before
any model call: B0 joint failure with a natural boundary, first 12 in generation
order. `--run` decodes each candidate once from the boundary without edits,
exactly as the v2 B0 branch does. The v2 runner's `--screen-dir` rebuilds the
selected cases from the generator, refuses an incomplete or tampered screen, and
binds both screen hashes into its own manifest. The contract and policies stay
unchanged for that `rescue` split.

The first screen attempt (`screen`) ran out of MPS memory after 17 candidates:
each per-case worker kept a full activation cache on the device until garbage
collection. The screen and the v2 runner now drop that cache and empty the
allocator between cases; neither manifest changed. `screen_r2` reruns all 72
candidates from the start under the same pre-registered hash, so its first 17
rows also check determinism against the aborted attempt.

One property of the frozen rule became visible in those 17 rows. "B0 joint
failure" includes outputs whose roles the conservative check cannot establish:
`Yusuf gives Wanda the badge` is correct English, but the check binds recipients
only through `to <name>`, so its role status is `unknown`. The pre-registered rule
stays the primary analysis. Rescue outcomes will additionally be broken down by
failure type (constraint or role failure versus role `unknown`), as a secondary,
exploratory split that was not pre-registered.

## Protocol deviation: disk floor

v1 stops a run below 10 GiB of free disk. On this machine free disk is about
13 GiB at rest and falls to about 8 GiB while the 3B model is loaded, because
macOS grows its swap files. `precheck` ended with its launching session during
model load; `precheck_r2` completed nine `cal_delivery` branches and then stopped
on the floor; `precheck_r3` stopped on it during setup. All three receipts are
kept under `results/structural_rescue_contract_20261103/`; none called a
provider, and r2's branches are superseded by the complete rerun. At the user's direction the v2 pilot and
the screen use a 6 GiB floor. The v2 manifest records it as
`disk_floor_gib: 6` with a `protocol_deviations` entry, which changed its hash
from `f9cd0dd7…` to `92f1138a7015ed0cbbed701340193d78e6e39df9aa9ce0a561248e7d843f4e99`.
v1's `Meter` default is still 10 GiB.

## Results

### Calibration pre-check (no provider)

`precheck_r4`: manifest `92f1138a…`, B0/fixed/fixed-readout/random plus every
catalog recipe on the three calibration cases, 48 branches in 1228 s on
MPS/float16. Every branch restored state, confirmed one guarded write and stayed
within the cap (realized relative write 0.08 throughout). The eligible sources
were `,`, ` while` and `.` in every case; ` before` was protected.

| Recipe | First token changed | Outcome |
| --- | ---: | --- |
| Patch, middle layer, any component or pair (27 branches) | 0/27 | identical to B0 |
| Readout `,`, last layer | 3/3 | rescues `cal_review`; the two successes stay successes |
| Readout ` while`, last layer | 1/3 | `cal_workshop` only, still a success |
| Readout `.`, last layer | 0/3 | identical to B0 |

The known failure becomes the role-correct rewrite under the contract:

- B0: `Nora will take the sample to Ivo before sending the report to Yara before dusk.`
- Readout `,` at 8%: `Nora will take the sample to Ivo, while Selim sends the report to Yara before dusk.`

On the two successes the same push only moves the deadline phrase to the end
(`… to Omar, while Leena brings the contract to Priya before noon.`). Middle-layer
patches saturate the cap and still never change the first token, consistent with
the earlier estimate of about 0.2 logit at 8%. Fixed-readout is the pre-declared
"first eligible source token" rule; that it is also the winning recipe in all
three calibration cases is a property of these cases, which all put `,` first.

### Frozen nine cases, all policies

`pilot`: the same manifest, all five policies on all nine v1 cases plus the
calibration matrix, 1474 s, Luna (`gpt-5.6-luna`) as controller. The analyzer
verified the manifest and every executed source hash against the snapshot. All
branches restored state; repeated-baseline drift was zero; the 72 edited
branches wrote 0.07998-0.08002 of the residual norm.

| Split | B0 | Fixed | Fixed readout | Random | Controller |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration joint success | 2/3 | 2/3 | **3/3** | 2/3 | 2/3 |
| Confirmation joint success | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| Wins / losses vs B0 | - | 0 / 0 | 1 / 0 | 0 / 0 | 0 / 0 |
| Outputs changed vs B0 | - | 0 | 6 | 0 | 5 |

No policy lost a baseline success. The contract-bounded `,` push changed six
outputs, rescued the one failure, and left the other five correct.

The controller made nine requests and selected a recipe every time, always a
readout push: `,` in eight cases and `.` in `cal_review`, the only case where
the choice mattered. It therefore missed the rescue that the fixed rule found,
on the one case that discriminates between them. With one discriminating case
this is an observation, not a comparison. Provider usage was 32,020 input and
1,675 output tokens. Random selection drew a middle-layer patch in seven cases
and a readout push in two (`.` and ` while`), none of which changed a task outcome.

Actual forward attempts: B0 126, fixed 144, fixed readout 146, random 144,
controller 143; shared setup 85; calibration matrix 577.

### B0 screen

`screen_r2` (manifest `e018bbcc…`, 1254 s) decoded all 72 candidates. Its 17
rows that overlap the aborted attempt are identical. 59 candidates reached the
boundary, and 52 of those succeeded jointly; the 13 without a boundary all
failed. Seven met the frozen rule, short of the limit of 12:

| Case | B0 failure | B0 output |
| --- | --- | --- |
| `gen_002` | constraint | `Lars will send the blueprint to Pablo before lunch.` |
| `gen_009` | role unknown | `Rosa sends the blueprint to Celia before dawn, while Yusuf gives Wanda the badge.` |
| `gen_018` | role unknown | `Kira sends the roster to Olga before dawn; Tilda mails Boris the badge.` |
| `gen_045` | constraint | `Felix brings the permit to Nikos before noon.` |
| `gen_047` | constraint | `Lars will take the receipt to Nikos before dawn.` |
| `gen_058` | constraint | `Boris sends the roster to Tilda before midnight, with Edda giving Alma the permit afterwards.` |
| `gen_060` | constraint | `Ugo must bring the blueprint to Kira before midnight.` |

Four constraint failures drop the second clause outright; `gen_058` keeps it as
a participle (`giving` does not contain the required `give`). The two role-unknown
cases are correct double-object datives. The 13 candidates without a boundary
all open with that construction (`Wanda gives Lars the roster`), so no
`to <first recipient>` prefix exists. Verb choice in the generator (`give`,
`hand`, `mail`) interacts with the role check's `to <name>` binding; the next
screen should either extend the check to double-object datives or restrict the
verbs, decided before it runs.

### Rescue split (pre-registered primary analysis)

`rescue`: manifest `95f8b371…` (binds the screen's manifest and results hashes),
all five policies on the seven selected cases, 504 s. The analyzer verified the
manifest and executed sources. B0 failed again on all seven. The 28 edited
branches wrote 0.07998-0.08003 of the norm, restored state and showed zero drift.

| Case | B0 failure | Fixed | Fixed readout | Random | Controller |
| --- | --- | :---: | :---: | :---: | :---: |
| `gen_002` | constraint | - | ✓ | - | ✓ |
| `gen_009` | role unknown | - | - | - | - |
| `gen_018` | role unknown | - | ✓ | ✓ | ✓ |
| `gen_045` | constraint | - | - | - | - |
| `gen_047` | constraint | - | ✓ | - | ✓ |
| `gen_058` | constraint | - | ✓ | - | - |
| `gen_060` | constraint | - | - | - | - |
| **Joint successes** | | **0/7** | **4/7** | **1/7** | **3/7** |

B0 is 0/7 by construction. Rescued outputs:

- `gen_002`: `Lars will send the blueprint to Pablo, while Henrik will mail the badge to Ilse before lunch.`
- `gen_047`: `Lars will take the receipt to Nikos, while Marta sends the ledger to Yusuf before dawn.`
- `gen_058` (fixed readout): `Boris sends the roster to Tilda, Edda gives the permit to Alma before midnight.`
  This is a comma splice that the evaluator accepts.
- `gen_018`: `Kira sends the roster to Olga, and Tilda mails the badge to Boris before dawn.`
  The B0 output was already correct English, so this success only reflects the evaluator.

The controller made seven requests and again always chose a readout push: `,`
four times, ` and`, `;` and `.` once each. It matched the fixed rule on three
rescues and missed `gen_058` by choosing `.`, although its packet showed `,` at
rank 2 with `gap_to_top1` 1.52. Random drew two readout pushes (` and` on
`gen_018`, which rescued it, and `.` on `gen_045`) and five middle-layer patches,
which again changed nothing. Provider usage was 24,873 input and 1,089 output tokens. By failure type
(exploratory), fixed readout rescued 3/5 constraint failures and 1/2 role-unknown
cases, and the controller 2/5 and 1/2.

### Reach of an 8% push

Across all 16 cases of the pilot and the rescue split, the fixed `,` push closed
the `,`-to-winner logit gap by 1.53-2.08 (mean about 1.8). It flipped the first
token in every case whose baseline gap was at most 1.92 (10/10) and in none whose
gap was 2.0 or more (0/6). Every flip ended in joint success: five rescues and five
unchanged successes. The three rescue cases out of reach (`gen_009`, `gen_045`,
`gen_060`) are exactly the gaps of 2.14-2.42.

So under this contract the actuator is not the bottleneck once the structural
alternative is within about two logits: one bounded token is enough, and the
worker writes the missing clause itself. What the cap cannot buy is a larger
gap, and the one-shot controller's choice, not the actuator, separates it from
the fixed rule (5 of 8 failures rescued by fixed readout, 3 of 8 by the controller,
counting `cal_review`).

### Reach curve (cap sweep, calibration only)

`cap_sweep` (manifest `ebd542c5…`, 720 s): every eligible source token's
last-layer readout push at caps 0.04-0.12 on the three calibration cases, one
B0 branch per case, no provider. All 45 branches restored state, confirmed one
guarded write and showed zero repeated-baseline drift; realized writes matched
the cap (0.0400-0.1200). The environment matched the recorded pilot versions
(torch 2.12.1, transformers 5.13.0, transformer-lens 3.5.1).

Gap closure is approximately linear in the cap, slightly sublinear at the top
of the range. The absolute closure slope differs by target: about 0.21-0.25
logits per 1% of norm for `,`, 0.49-0.54 for ` while` and 0.36-0.50 for `.\n`.

| Source | Baseline gaps | Closure slope (logits per unit cap) | First-token flip |
| --- | --- | --- | --- |
| `,` | 1.13-1.92 | 21-25 | 3/3 cases; at 0.06 for gap 1.13, at 0.08 for gaps 1.45-1.92 |
| ` while` | 3.66-4.41 | 49-54 | at 0.08 for gap 3.66, at 0.10 for gaps 4.22-4.41 |
| `.\n` | 7.20-8.42 | 36-50 | none up to 0.12 |

Every first-token flip preserved joint success: the `,` flips rescued
`cal_review` at 0.06 already (B0 gap 1.125) and moved the deadline phrase to
the end on the two baseline successes, while ` while` flips at 0.10+ also
produced valid conjunctions. No flip produced a failure. Pushes that did not
flip the first token left the output byte-identical to B0.

Extrapolating the `,` slope to the rescue split: its three unreached gaps
(2.14-2.42 logits) would need roughly a 0.10-0.12 cap, inside the observed
linear region. The `.\n` pushes the controller picked on `cal_review` and
`gen_058` face gaps of about 7-8 logits; at the measured slope they would need
roughly a 0.17-0.20 cap — outside the contract's tested range and far beyond
what a modest raise of `r` could justify — consistent with those branches'
outcomes.

### Controller datasheet pilot

`datasheet` (manifest `5b6e9b91…`, 331 s): the same rescue split and contract,
with each readout recipe carrying the cap sweep's measured closure slope and its
estimated logit closure at this cap (`structural_rescue_v2_norm_relative_contract_datasheet`).
Three policies: b0, fixed readout, controller. All branches restored state,
zero drift, one guarded write per edit.

| Case | `,` gap | Estimated `,` reach | Controller choice | Fixed readout |
| --- | ---: | ---: | --- | --- |
| `gen_002` | 0.75 | ~1.8 | `,` ✓ | ✓ |
| `gen_009` | 2.14 | ~1.8 | noop | - |
| `gen_018` | 1.41 | ~1.8 | `,` ✓ | ✓ |
| `gen_045` | 2.27 | ~1.8 | noop | - |
| `gen_047` | 0.86 | ~1.8 | `,` ✓ | ✓ |
| `gen_058` | 1.52 | ~1.8 | `,` ✓ | ✓ |
| `gen_060` | 2.42 | ~1.8 | noop | - |

The datasheet-equipped controller matched the fixed rule's 4/7 joint successes
exactly — including `gen_058`, which it missed without the datasheet by picking
`.` — and it selected `,` on every case whose gap was inside the estimated
reach. On the three unreachable cases it chose noop rather than spend a write,
so it applied 4 edits for 4 wins where the fixed rule applied 7 for the same 4.
The measured boundary of its behaviour (does it pick `,` at a gap of exactly
1.8, or overshoot a 1.6-2.0 estimate) is untested: no screened case sits in
that band. Provider usage: 27,487 total tokens, one call per case.
Post-run, review tightened the datasheet prompt wording (unmeasured pieces carry
no estimate) and versioned the packet as `structural_rescue_choice_v2_datasheet`;
future datasheet manifests therefore hash differently from `5b6e9b91…`, which
remains the receipt of this run.

### Second screen and rescue_v2 split

`screen_v2` (manifest `789b6d68…`, 4157 s, 120 candidates, generator v2) applied
both pre-registered mitigations: the role check binds double-object datives
("gives Wanda the badge") and the generator draws only send/take/bring, whose
natural output keeps the `to` boundary. Effect: boundary-less candidates fell
from 13/72 to 2/120, and evaluator-only role-unknown candidates from 2/7 to 0 —
but joint B0 failures also fell, from 7/72 to 4/120. All four selected
failures are the same failure family: a dropped second clause scored
role-unknown (`incomplete_frame`), never a constraint failure alone.
`attempt1_disk` kept 38 rows before the 6 GiB floor; the rerun's first 38 rows
are identical.

`datasheet_v2` (manifest `0aad3f4e…`, 553 s): all five policies on the four
selected cases, controller with the datasheet. B0 0/4 by construction; fixed
patch and random 0/4; fixed readout 3/4; controller 3/4 with the same wins —
and on `gen_049` (`,` gap 2.83 vs ~1.8 estimate) it chose noop rather than a
doomed write. On `gen_011`/`gen_110` it picked ` while` (gaps 3.73/3.25 inside
the measured ` while` reach of ~4.1) instead of `,`, a choice the fixed
first-token rule cannot express — the controller used the slope table, not
just a threshold. Provider usage: 16,088 total tokens. All branches restored
state, zero drift, one guarded write per edit.

The four-case split is thin and single-family, so these remain descriptive
counts; but the datasheet controller has now matched the fixed rule on every
reachable case across both splits (7/7 reachable) while spending fewer writes.

### Cap 0.12 rerun (boundary-band verification)

`--cap` now takes the relative write cap into the manifest (separate `_cap*`
version), system prompt, catalog, packet, policy and branch check. Two datasheet
runs at 0.12 test the band the 8% cap could not reach:

- v1 rescue split (manifest `20aaa4d8…`, 1410 s): fixed readout and controller
  both 6/7. `gen_045` (`,` gap 2.27, previously unreachable) rescued; `gen_060`
  (gap 2.42) flipped to `, giving Yusuf the ticket` — a gerund that drops a
  required term, so still a failure; `gen_009`/`gen_018` now pass at B0 because
  the double-object evaluator binds their already-correct outputs.
- rescue_v2 (manifest `238c2e00…`, 1286 s): fixed readout and controller 4/4.
  `gen_049` (`,` gap 2.83 vs ~2.7 estimated `,` reach at 0.12) — the controller
  correctly judged `,` unreachable and chose ` while` (estimated ~6.1), the
  first observed case where its datasheet reasoning differed from a blind
  first-token rule and still won.

Realized writes were 0.11999-0.12004 throughout (cap plus float16 rounding,
inside the 0.002 tolerance); state restored, zero drift.
The `,`-push datasheet boundary is now covered on both sides: 1.52 won at 0.08,
2.83 declined at 0.12 where the estimate says unreachable.

### Screen v3 and rescue_v3 split

Screen v3 (manifest `f4e86bb1…`, `results/structural_rescue_screen_v3_20261107/`,
6104 s) adds participial-second-clause and agent-relative sources plus
`give`. 120 candidates, 6 boundary-less, **16 B0 failures** — the selection
limit 12 was reached for the first time. Two new failure families appear:
8 "with X -ing" participle outputs that keep both frames but drop required
terms or the deadline, and 4 verb-substitution/possessive role-unknowns.

rescue_v3 at cap 0.12 + datasheet (manifest `a0c53747…`, 1393 s): only 2/12
rescued, all by readout pushes. `gen_017` is the **first patch success** (the
common_orthogonal blend rescued it too). The 8 participle-family cases resisted
every flip: pushing `,`/` while` produced `, with X -ing` continuations that
still fail constraints — the intervention can flip the token but cannot steer
the clause into finite form. Controller matched fixed readout on both rescued
cases; state restored, zero drift.

Bottleneck ranking after this run: (1) clause-form steering is what participle
failures need, and a token-level readout push cannot express it; (2) verb
substitutions (`finishes her drawing`) need a richer action surface or a
content-preservation check beyond lexical terms.

### Clause-form starter (rescue_v3 rerun)

`--clause-form` adds one bounded readout push toward ` while`, provenance
`clause_form_starter`, offered even when the lexeme is absent from the source,
plus a `fixed_clause_form` policy. Rerun at cap 0.12 + datasheet (manifest
`4ef1be4a…`, `results/structural_rescue_clauseform_v3_20261108/`, 1000 s):

- **fixed_clause_form: 7/12** — every win is a participle-family case flipped
  from `with X -ing` to `while X <finite>` (`gen_018`, `gen_058`, `gen_068`,
  `gen_073`, `gen_078`, `gen_088`, plus `gen_017`). Clause-form steering works.
- Controller: 2/12, identical to the readout-only run. It never picked the
  starter — the datasheet reach estimate tells it `,` flips the first token
  (true), but nothing in the packet distinguishes "flips and keeps the
  clause non-finite" from "flips into a finite clause".
- Random hit the starter once by chance (`gen_088`); patch only `gen_017`.

So the next bottleneck is **controller information**: the starter's advantage
is about the *form* of the continuation, which reach estimates do not encode.
A `clause_form`-aware datasheet line (e.g., "starters bias toward finite
clauses where the generated tail already ends in `with`") is the smallest
honest probe; claiming a selection rule from 8 cases is not.

That probe ran: with `form_hint` on the starter row and a clause-form prompt
addendum (manifest `76fc0dd0…`,
`results/structural_rescue_clauseform_ds_v3_20261108/`, 985 s) the controller
picked the starter on 9/12 cases and scored **6/12** — wins `gen_017`,
`gen_018`, `gen_058`, `gen_068`, `gen_073`, `gen_088` — versus the
starter-less datasheet run's 2/12. It lost `gen_059` where a plain comma was
the winning recipe, and on `gen_078` picked `,` anyway. The form hint moved
selection, and the participle-family mechanism now goes through the
controller, not just the fixed rule.

### Applicability gate on the starter

`applies_to_this_case` was added to the starter row, meant to be true only for
participial-second-clause sources. A first rerun (manifest `787833c5…`,
`results/structural_rescue_clauseform_gated_ds_v3_20261109_r2/`, 1171 s)
scored 5/12 — and the packet shows why: the gate regex matched `with` *or*
`who`, so 11/12 cases read as applicable. `who` catches the relative-clause
substitution family (`gen_039`, `gen_059`, `gen_074`: "X, who should V, also
finishes"), where the model swaps the required verb for a common one and a
clause-form flip cannot restore it. The controller re-lost `gen_059` under a
gate that was supposed to exclude it.

Restricting the gate to `with` (manifest `6137df71…` — the manifests differ
only in `controller_system_prompt_sha256`, since the prompt text was updated;
the regex itself is not manifest-recorded),
`results/structural_rescue_clauseform_withonly_ds_v3_20261109/`, 1265 s): the
controller scored **8/12** — wins `gen_017` (readout), `gen_018`, `gen_058`,
`gen_068`, `gen_073`, `gen_078`, `gen_088` (starter), `gen_059` (readout).
That equals the union of the policies actually evaluated — though not all
catalog recipes were run (`calibration_matrix` is empty on the rescue split),
so this is a policy-union score, not a proven catalog ceiling.

Attribution needs care: the gate clearly caused `gen_059` (identical intent,
flag now false, readout selected and won). But `gen_068` and `gen_078` flipped
on *byte-identical packets* — only the system prompt hash differed — so those
gains are prompt-wording or controller sampling noise, not the gate. The
controller is nondeterministic and this is one run per variant; treat the +3
as "+1 attributable, +2 uncontrolled". A trivial hand-rule
("applies ? fixed_clause_form : fixed_readout") would also score 8/12 here, so
the honest claim is narrower: a discriminative applicability flag is
sufficient for the controller to avoid the observed over-selection, not that
controller selection is optimal. The gate was also fit to this split after
seeing `gen_059` fail — the same warning this doc raised about claiming
selection rules from few cases applies.

The four remaining cases had no win under any evaluated policy:

- `gen_008`, `gen_028`: participial sources, but even the starter leaves the
  output unchanged — a reach/strength problem, not a selection problem.
- `gen_039`, `gen_074`: the substitution family (`finishes` occupies the
  required-verb slot). `gen_059` shows the same failure is rescuable when the
  worker happens to emit the required verb under a comma push — content-level
  steering (multi-token or multi-step intervention) is the missing mechanism,
  not another single-token push.

Caveats on the gate itself: `\bwith\b` is a word match that identifies the
case-generator's participial template, not the grammar — a prepositional
"with" or a bare participle would mislabel it, and the flag is invisible to
the controller's own input (the packet omits the source). It also excludes
`gen_017`, where the starter happens to win anyway.

### Verb-slot pilot (content-level steering)

Design: `docs/content_steering_design.md`. A P0 reach check found the
required action is rank-2 at the substituted-verb position in all three
substitution-family continuations (gaps 0.08–0.53 logits — the substitution
is a coin flip, not a preference). `--verb-slot` adds a `verb_slot` recipe:
simulate the b0 continuation, take the first generic verb after the second
frame's agent as the slot, replay the b0 prefix (forced pre-roll), and apply
the same ttl=1 `resid_add` push at that step toward the inflected required
action (` takes`, ` brings`, …). The inflected token ids join the readout
allowlist; `verb_slot_rule` is recorded in the manifest.

Rerun at cap 0.12 + datasheet (manifest `d2bff070…`,
`results/structural_rescue_verbslot_ds_v3_20261109_r2/`, 1632 s):

- **fixed_verb_slot: 2/2 where offered** — `gen_039` ("Boris takes roster to
  Edda") and `gen_074` ("Boris brings the blueprint to Nikos") both flipped.
  The recipe is only offered where the simulated b0 shows a generic verb, so
  it noop'd elsewhere. First content-token intervention under the contract:
  `relative_write_norm` 0.11999 within cap, one guarded write, state restored.
- **controller: 10/12** — it picked `verb_slot` on both substitution cases
  unprompted and matched the previous wins. `unknown_roles` dropped to 0 for
  the controller. The two losses (`gen_008`, `gen_028`) are cases where even
  the starter cannot flip the first token — no evaluated policy wins there.
- `gen_059` got no `verb_slot` recipe: the agent locator matches the name in
  single token pieces and "Celia" tokenizes into two pieces. It was rescued
  by readout anyway; the locator is the known gap.
- Protocol note: the first attempt
  (`results/structural_rescue_verbslot_ds_v3_20261109/`) exposed a meter bug —
  `raw_first` was only captured at boundary-length forwards, so delayed
  edits had no logits. The capture condition is now "first forward matching
  the capture mode" (edited forward for edited branches); `edited_winner`
  5097 (` takes`) confirms the capture landed at the slot.

The substitution family is now reachable by a bounded one-token push — the
mechanism exists (N=2, no efficacy claim). `gen_008`/`gen_028` remain the
open reach problem.

### Scope

- One worker model, one boundary rule, greedy decoding.
- Twelve selected failures on rescue_v3, so all counts are descriptive.
- The role check is a conservative surface heuristic. `gen_018` shows a success that only reflects the evaluator, and `gen_058` a comma splice it accepts.
- Restored diagnostic branches only; no production permission, certificate or new authority.
- The cap (8% early, 12% on rescue_v3) and the last-layer readout site were chosen from the earlier dose study, not tuned on the rescue split.
- The controller (`gpt-5.6-luna`) samples nondeterministically; single-run score deltas of ±1-2 are within noise.

### Next

- ~~Pre-register a second screen~~ Done: [screen v2 and rescue_v2](#second-screen-and-rescue_v2-split), `results/structural_rescue_screen_v2_20261106/` + `results/structural_rescue_datasheet_v2_20261106/`. Both mitigations were applied; failures are now rare (4/120) and single-family, so a next screen needs a harder candidate generator, not more of the same.
- ~~Add a calibration-derived datasheet to each readout recipe~~ Done: [datasheet pilot](#controller-datasheet-pilot), `results/structural_rescue_datasheet_20261106/` (manifest `5b6e9b91…`). With reach estimates the controller matched fixed readout at 4/7 and noop'd on unreachable cases; the boundary band near the estimate (~1.6-2.0 logits) is untested.
- ~~Measure the reach curve directly with a cap sweep~~ Done: [reach curve](#reach-curve-cap-sweep-calibration-only),
  `results/structural_rescue_cap_sweep_20261106/` (manifest `ebd542c5…`).
