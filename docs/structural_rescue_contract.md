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
