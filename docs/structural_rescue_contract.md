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

## Results

Pending.
