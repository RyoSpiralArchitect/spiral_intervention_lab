# Reach Datasheet, Cap Sweep and a Second Rescue Split

Date: 2026-11-03. Follows [`structural_rescue_contract.md`](structural_rescue_contract.md)
(PR #39), whose three next steps this note carries out in order:

1. Give the controller the actuator's calibrated reach (a datasheet).
2. Measure reach as a function of the cap (4-12% of the residual norm).
3. Fix the evaluator and pre-register a second baseline-failing split.

Before any model run, six independent critics reviewed the design and five
reviewed the code; the design below is the revised one.

## Step 0: evaluator v2

`StructuralRescueEnv.evaluate_v2` adds three things to v1, which stays the
default for every earlier runner and reproduces all 236 stored verdicts:

- **Double-object datives.** A known name right after the action, followed by
  the theme, is the recipient (`Yusuf gives Wanda the badge`), unless a
  `to <name>` also follows the theme. A name right after `to` or after an action
  form is an object, never an agent. Curly apostrophes count as possessives, and
  `handed`, `mailed`, `gave` and `given` are actions.
- **Clause joins.** The text between the first clause's last role word and the
  second clause's agent, with the deadline phrase treated as a marker, must join
  two clauses without changing what the source says: `;`, `:`, a dash, or one of
  *and, while, whilst, whereas, but, yet, although, though, when, as* after the
  last remaining comma. A bare comma is a comma splice, nothing is a run-on, and
  *or, nor, so, because, since, after, before, until, once, unless, if* change
  the meaning (the source's two tasks are simultaneous). A subordinator before
  the first clause makes a bare comma correct and a second connective wrong, and
  a non-finite second clause (`with Tilda mailing Boris the badge`) is an adjunct,
  which passes. A preposition or article right before punctuation fails.
- **Dangling final word.** A sentence may not end on a preposition, article or
  connective (`... to Wanda before.`).

Articles are not checked: a bare theme (`sends drawing to Ilse`) is reported, not
failed. These decisions were fixed before any v2 run; forty hand-labelled
adversarial outputs from the evaluator review are a test.

On the stored PR #39 outputs v2 changes 26 verdicts: 12 datives now pass, and 10
comma splices, one dangling `before.` and the `gen_058` fixed-readout splice now
fail. That changes some PR #39 statements. **These are a post-hoc sensitivity
analysis; PR #39's pre-registered primary analysis stays v1.**

| PR #39 statement (v1) | Under v2 |
| --- | --- |
| Rescue split: B0 0/7, fixed 0/7, fixed readout 4/7, random 1/7, controller 3/7 | B0 2/7, fixed 2/7, fixed readout 4/7, random 2/7, controller 4/7 |
| Wins vs B0: fixed readout 4, controller 3, random 1 | fixed readout 2, controller 2, random 0 |
| Every flip of the `,` push ended in joint success (10/10) | 9/10; `gen_058` becomes a comma splice |
| The controller chose `.` twice where `,` would rescue | Once (`cal_review`) |
| Screen: 52 of 59 boundary candidates succeeded; the 13 without a boundary all failed; 7 met the rule | 43 of 59; 10 of the 13 pass; 16 meet the rule |
| Fixed readout rescued 5 of 8 failures, the controller 3 of 8 (with `cal_review`) | 3 of 6 and 2 of 6 |

The nine-case pilot's verdicts are unchanged.

## How the runs work

A TTL-1 edit is rolled back after the first decoded token, and every later step
recomputes the whole token sequence, so a restored branch's output is fixed by
its first token. In the PR #39 receipts 164 branches form 35 (case, first token)
groups with exactly one output each. So:

- `examples/structural_first_token_table.py` measures every catalog push at a
  boundary with one edited forward, then decodes one PR #39-path branch per
  distinct first token, and repeats that decode for a second push with the same
  first token on every sixth case. Mismatches are recorded, not raised. The
  worker can reuse a packet built for the same state, which removes an unedited
  readout forward per push.
- `examples/structural_controller_sampling.py` asks the controller arms k times
  per case on the packets the table stored. No model runs; each choice is scored
  by lookup.
- `examples/summarize_structural_table.py` verifies every part (manifest,
  executed sources, same code, environment, device, dtype, MPS fraction and
  checkpoint across parts), checks the table against earlier receipts bitwise,
  and scores the arms.

Arms: B0; fixed (L14 orthogonal patch, continuity); fixed readout (first eligible
source token); random (continuity) and random readout (chance over pushes);
datasheet rule; three controller arms; and the oracle (best single push,
headroom only, never shown to a policy).

- **Datasheet rule.** Readouts in source order; skip a target that ends the
  answer or is already the top token; push the first whose gap is below the
  maximum calibrated closure for its stratum (at least 5 calibration pushes);
  otherwise noop.
- **Controller arms.** `controller` is the PR #39 prompt and packet byte for byte
  (at 0.08). `controller_mechanics` adds actuator facts that hold by
  construction: the edit acts on the next token only, an unchanged next token
  means an identical continuation, pushing the current top token changes nothing,
  and sentence-final punctuation ends the answer. `controller_datasheet` also
  gets packet v3, whose recipes carry the datasheet: closure (n, min, median,
  max) for the target's stratum, a reach band (likely, borderline, unlikely,
  already top, uncalibrated) and whether the target ends the answer.

## Step 1: datasheet for the controller (development)

The 16 PR #39 cases at 0.08, all 12 recipes each. Datasheets are leave-one-out
over those 16. The table must reproduce PR #39 bitwise. Three controller arms,
three calls each. Choice-level measures are primary; outcome differences are
possible only where a choice changes the first token. Results: pending.

## Step 2: cap sweep 4-12%

Pool: the 3 v1 calibration cases and the 59 screen-1 candidates with a boundary.
v1 confirmation cases are kept out for the harm replay. Readouts at 0.04-0.12;
the orthogonal patch at 0.12 on every case and all patches on spot-check cases.
The cap rule for step 3, fixed before the sweep: among 0.04-0.10 (below the
11-12% dose at which a name can be written), take the smallest cap with the
largest net (fixed readout rescues minus losses, v2); move off 0.08 only if that
beats 0.08 by at least 2; if no cap has a positive net, keep 0.08. Results:
pending.

## Step 3: screen 2 and rescue split 2

The whole treatment (cap, datasheets, prompts, arms, k = 3, missing-data rule,
strata, comparisons, audit) is frozen from the verified sweep before screen 2
runs. Screen 2: seed 20261104, `s2_NNN` IDs, the first 12 boundary B0 failures
(rescue) and the first 24 boundary B0 successes (harm), in generation order.
Results: pending.

## Run hygiene

Runs refuse to start on battery, below 12 GiB free (6 GiB floor, the
user-directed deviation recorded since PR #39, plus 6 GiB load headroom), or from
a modified tree. They record argv, PID, HEAD, power, the `PYTORCH_MPS_*`
environment and the startup files every Python process runs, and log memory,
swap and disk per case. `examples/launch_detached.py` runs parts in a new
session under `caffeinate` and stops at the first failure.
