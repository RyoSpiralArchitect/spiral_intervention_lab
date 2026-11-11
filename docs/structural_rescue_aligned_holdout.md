# Aligned readout and held-out policy comparison

The PR #53 receipt still demonstrates two full-text verb-slot rescues under
its role/constraint evaluator. Its delayed readout deltas are not physical
effect estimates: they compare the edited verb slot to the initial boundary.
Keep that receipt unchanged.

In the new harness, two unedited replays and the edited replay capture logits
at the branch's scheduled decode step. All three capture the same token
sequence (checked by tensor identity); a missing or mismatched capture fails
the branch. The readout includes the actual pushed token even if it was not in
the original structure readout. Decode-level joint success is separate from
this readout.

## Frozen comparison

- Use all 10 cases from v3 generator seed 20261111, two per source family.
  Do not screen on B0, omit unavailable boundaries, or replace cases.
- Compare B0, fixed routing, and Luna on the same worker checkpoint. Fixed
  routing selects the first applicable verb-slot recipe, then the first
  applicable clause-form starter, then the first structural readout recipe,
  or no edit. It sees the same packet as Luna and runs no extra search.
- One write per branch, one token of TTL, cap 0.12 of the residual norm.
  The 40 setup, 24 diagnostic, and 64 continuation forward ceilings are
  shared across policies; attribute all shared setup calls, including B0
  lookahead, to each arm for end-to-end cost comparison. Report provider calls
  and token usage separately.
- Report paired controller-only and fixed-only joint successes, regressions
  from B0, unknown roles, unavailable boundaries, and failed or missing pairs.
  Positive evidence for controller selection requires more controller-only
  than fixed-only wins without adding regressions on B0 successes. Even
  then, this is descriptive evidence on new cases from the same generator,
  not a population claim or a test of online timing. The evaluator is a
  surface role heuristic, and the B0 pre-roll is diagnostic only.

From the repository root:

```sh
RECEIPT=results/structural_rescue_verbslot_ds_v3_20261109_r2/results.json
RUN=results/structural_rescue_heldout_v1
python3 -m SpiralInterventionLab.examples.structural_rescue_benchmark_v2 \
  --prepare --heldout --checkpoint-manifest "$RECEIPT" --output-dir "$RUN"
DIGEST=$(shasum -a 256 "$RUN/manifest.json" | cut -d ' ' -f 1)
python3 -m SpiralInterventionLab.examples.structural_rescue_benchmark_v2 \
  --run --heldout --checkpoint-manifest "$RECEIPT" --output-dir "$RUN" \
  --manifest-sha256 "$DIGEST" --worker-model-path "$LLAMA_CHECKPOINT"
```

The run verifies checkpoint hashes before creating the controller provider.
The held-out profile fixes MPS/float16, a 10 GiB disk floor, and the same
selector prompt and candidate catalog as the latest verb-slot pilot.
Use a fresh output directory for any retry and retain incomplete results.
There is no production apply permission.

## 2026-11-11 run

The eight checkpoint files were restored byte-for-byte: weight shards came
from the Furnace cache, and configuration and tokenizer files from Meta's
Hugging Face revision `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
The first two attempts stopped at the MPS memory cap before any provider call.
Their receipts remain in `structural_rescue_heldout_v1` and
`structural_rescue_heldout_v1_retry1` under `results/`. The completed
[retry2 receipt](../results/structural_rescue_heldout_v1_retry2/results.json)
uses the same manifest, with unchanged caps, checkpoint, and precision.
Readout directions now gather before the FP32 cast, and the MPS harness
lays out the unembed weights once rather than allocating a copy per forward.
The first case's unedited output and baseline logits hashes match the first attempt.

B0 passed 10/10, fixed routing 9/10, and Luna 10/10 (nine provider calls).
One unavailable boundary was retained as a shared baseline outcome.
On `heldout_001`, fixed routing chose a comma and the role evaluator
attached the second action to Rosa rather than Felix; Luna chose `while`
and preserved the binding. B0 already passed, so this is one avoided
regression versus fixed routing, not a rescue versus B0. No delayed verb-slot
branch was exercised in this run.

## Selector readback and next comparison

The [offline report](../results/structural_rescue_heldout_v1_retry2/report.md)
separates the nine eligible selector pairs from the shared no-boundary
outcome. The two selectors chose the same recipe on six pairs and differed
on three. Luna's explicit noop on `heldout_002` preserved the B0 text.
On `heldout_001`, comma and `while` both became the raw winner under their
respective writes. Their B0 ranks were 3 and 6, respectively. Reach and
role preservation are therefore distinct measurements; neither picking the
higher-ranked token nor always preferring `while` is established as a rule.

The opt-in `--compare-b0-preview` profile adds a preview selector beside
the plain selector, fixed routing, and B0. Both Luna arms use the same
instruction, offers, and one-call budget. Only the preview arm sees the raw
unedited continuation already computed for verb-slot discovery, capped at
40 tokens. There is no extra worker forward for the preview, no evaluator
label or edited continuation in the packet, and no change to apply authority.
This is a restored diagnostic branch with lookahead, not an online controller.

For a new rescue study, first freeze a fresh B0 screen. The preview profile
requires the complete screen, reproduces its selection from the rows, binds
the same checkpoint, and runs every selected case without dropping B0
reproduction passes. The archived v3 screen can be used for development
only. Keep rescue outcomes separate from regressions on B0 successes.

```sh
python3 -m SpiralInterventionLab.examples.structural_rescue_benchmark_v2 \
  --prepare --compare-b0-preview --screen-dir "$SCREEN" \
  --checkpoint-manifest "$SCREEN/results.json" --output-dir "$RUN"
```

The run uses the same profile flags and adds `--run`, the printed manifest
digest, and the local worker path as above. It can make two provider calls
per eligible case, one per Luna arm. The fresh live follow-up is recorded below.

## Fresh v4 screen protocol

The follow-up freezes generator seed 2026111101 and 30 candidates (six per
existing source family), before any model sees them. The B0 screen retains
all rows, including successes and unavailable boundaries, then selects the
first at most 12 eligible failures in generation order. Zero selected
failures ends this attempt without automatically extending the screen.

The checkpoint, MPS/float16, memory fraction 0.7, 10 GiB disk floor, and
1800-second screen wall limit are frozen. Actual model forwards are counted
per case with the same setup/diagnostic/continuation ceilings. Screening uses
no provider and no intervention. The preview study runs every selected case
with four arms: B0, fixed routing, plain Luna and preview Luna; both Luna arms
share the same prompt and offers and each has one provider call.

The screen's denominator is 30 candidates; rescue efficacy is conditional
on screen selection, not an estimate of the full task distribution.

## Fresh v4 live result

The [completed screen](../results/structural_rescue_screen_v4_20261111_retry1/results.json)
ran all 30 frozen cases: 27 recorded B0 passes, five unavailable boundaries,
and three selected failures (`gen_008`, `gen_019`, `gen_024`).
The first attempt stopped at the MPS cap after 12 completed cases and remains
in `results/structural_rescue_screen_v4_20261111/`. Clearing the disposable
screen worker's prefix cache before an unedited continuation avoided cloning
it solely for restoration; the first 12 outputs and judgments reproduced
exactly on retry. Checkpoint, arithmetic, forward caps, precision, and manifest
were unchanged.

The [paired study](../results/structural_rescue_b0_preview_v4_20261111/report.md)
completed all 12 branches and six provider calls. Under the frozen evaluator,
B0 scored 0/3 and each of fixed routing, plain Luna, and preview Luna scored
3/3. All three selectors chose the same recipes and produced the same outputs:
there is no observed preview or controller advantage over this fixed rule.
All restored-state checks passed; production permission remains false.

- `gen_019`: a delayed `takes` push at decode offset 5 repaired
  `Felix finishes the proposal for Alma` to
  `Felix takes the proposal to Alma afterwards`. The same-prefix readout at
  109 input tokens moved `takes` from rank 2 to rank 1 and probability
  0.325108 to 0.983142, with realized relative write 0.115396 below cap 0.12.
  This is one new matched-position actuator observation, not a population claim.
- `gen_024`: no verb-slot recipe was offered. A boundary comma push instead
  changed `Ugo finishes the blueprint for Jonas` into
  `Ugo takes the blueprint to Jonas before midnight`. This is a structural
  reroute, not a delayed content intervention.
- `gen_008`: B0 already reported no violations and passing roles, but its
  `giving` form failed the inherited literal-term `done()` check.
  `evaluate()` accepts action inflections in feedback but still calls that
  inherited check for final constraint success. The report flags this
  contradiction without rescoring, excluding it, or counting it as established
  rescue evidence. A versioned evaluator repair is needed before the next
  efficacy study; the current 3/3 headline must not be treated as three
  demonstrated semantic rescues.
