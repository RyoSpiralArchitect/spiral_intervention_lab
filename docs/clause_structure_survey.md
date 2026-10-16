# Clause Structure Survey and Structure Actuators

Date: 2026-10-16. Worker: local Llama-3.2-3B-Instruct (fp16, MPS), checkpoint
verified against the 10-14 retry manifest. Follows
[`operator_dose_response.md`](operator_dose_response.md), which found that the
L1 review failure is a clause-structure choice: at `Nora will take the sample to
Ivo`, ` before` beats `,` by 1.125 logit, and `,` lets the worker write the
missing clause itself.

## Questions

1. **Generality**: across all nine rewrite-ladder cases (L1/L2/L3 x delivery,
   review, workshop), which failures can a forced clause connective rescue?
2. **Structure actuators**: can a source-anchored operator, rather than readout
   steering, flip ` before` to `,` at the L1 review decision point, and how
   large is the edit?

Forcing a token is an operator-free ceiling. All edits are one-token replays in a
quarantined worker with released limits and state restored. No controller call,
rollout edit, production trial or apply permission is involved.

## 1. Survey (`results/clause_structure_survey_20261016/survey.json`)

`examples/survey_clause_structure.py` decodes each case through the worker
runtime, then walks every failed trajectory: at each prefix, each connective in
`, | while | ; | and | whereas | but` ranked within the top 10 is forced once and
the sentence finished greedily. A rescue needs score 1.0, no constraint
violations and correct roles (offline role-binding check).

The runtime path applies the word-budget guardrail, which ends a sentence at the
budget. L1 outputs match the recorded raw ladder screen exactly; three L2/L3
outputs differ only because the guardrail truncates them at 12 words (`...the
report to.`, `...sends the.`, `...contract to.`). The survey reports what B0/C1
runs actually decode.

| Case | Natural output (runtime) | Score / roles | Best rescue |
| --- | --- | --- | --- |
| L1 delivery | two clauses, `while Leena will bring...` | 1.0 / ok | (success) |
| L1 review | `...to Ivo before sending the report to Yara before dusk.` | 0.94 / Nora sends | `,` at `to Ivo` (rank 2, gap 1.13): `..., while Selim sends the report to Yara before dusk.` |
| L1 workshop | `..., while Rami sends the diagram to Vera.` | 1.0 / ok | (success) |
| L2 delivery | `Mira will send the budget to Omar before noon.` | 0.76 | none |
| L2 review | `...before sending the report to.` (truncated) | 0.82 | none |
| L2 workshop | `..., while Rami sends the.` (truncated) | 0.88 | none |
| L3 delivery | `...before noon, while Leena brings contract to.` (truncated) | 0.94 | `,` at `to Omar` (rank 2, gap 1.19): `..., Leena brings contract to Priya before noon.` |
| L3 review | `...before dusk, sends report to Yara.` | 0.94 / Nora sends | `;` at `to Ivo` (rank 8, gap 3.67): `...; Selim sends report to Yara before dusk.` |
| L3 workshop | `..., Rami sends diagram to Vera.` | 1.0 / ok | (success) |

- **Three of six failures share one decision.** Right after the first clause's
  recipient (`to Ivo`, `to Omar`), the natural token is ` before`, which attaches
  the deadline or a gerund to the first clause. A clause separator ranked 2-8
  (gap 1.1-3.7) rescues all three. L3 delivery looks like a word-budget failure,
  but its cause is the same choice: the early `before noon` leaves no room for
  `Priya`.
- **L3 review shows that a separator alone is not sufficient.** The natural
  output already has a comma (`before dusk, sends report`) but no new subject;
  the rescue is a separator before the deadline, where the next clause starts
  with its own agent.
- **The missing agent is most reachable in the wrong role.** Along the natural
  trajectories its first piece peaks at rank 1-3, but for review and L2
  delivery that peak is a recipient slot (`...the report to` -> ` Sel` rank 2;
  `...the budget to` -> `Leena` rank 2). This is why writing the name tends to
  produce role violations. Where a clause is already open (`..., while`), the
  agent is rank 1 and those failures are budget truncations.
- **L2 is a compression failure, not a structure failure.** With `the` and
  `will` allowed, no forced connective fits nine terms into 12 words; L2 needs a
  different intervention (or a terser register).

## 2. Structure actuators at L1 review, prefix 8 (`structure_patch.json`)

`examples/replay_structure_token_probe.py --patch-source-terms while,Selim` blends
the source activation of ` while` or `Selim` (from `...to Ivo, while Selim
should send...`) into the last position for every patchable surface x
{term token, centered +-1, span mean} x dose {0.08, 0.16, 0.32, 0.64, 1.0}, and
ranks `,` exactly over the full vocabulary.

- **Source-anchored patches can flip the structure decision.** 3 of 45 curves
  with the ` while` source and 4 of 45 with the `Selim` source make `,` the greedy
  token; every one of the seven yields `Nora will take the sample to Ivo, while
  Selim sends the report to Yara before dusk.` (score 1.0, roles correct). The
  same family never made ` Sel` itself win at prefixes 8-10.
- **Injecting the name's representation opens a clause rather than emitting the
  name.** With the `Selim` source, the worker does not write `Selim` next; it
  writes `,` and then `while Selim sends ...`.
- **The edits are large.** All wins are at L25-L27. The smallest is a 0.32 blend
  into L27 `resid_post`, moving that site by about 31% of its norm; `mlp_out`
  wins move 53-113%. Several wins lower the `,` logit and win by suppressing
  ` before`. By comparison, the readout push toward `,` wins at 6% of the L27
  `resid_pre` norm.
- Larger doses mostly hurt: the median `,` lift is -3.2 (` while`) and -1.4
  (`Selim`) at dose 0.64, and -9.5 / -6.5 at dose 1.0.

## Reading

- The structural lever generalizes beyond one case: half of the ladder failures
  turn on the same post-recipient choice, and a token already ranked 2-8 fixes
  them without writing any required term.
- It does not cover compression failures (L2), which need a different handle.
- Structure is reachable by both a readout push (6%) and source-anchored patches
  (31%+), while the name was reachable only by readout at 12-24%. No operator
  reaches either target within the old 0.08 cap.

## Narrowed next candidates

1. **Offer the structural decision to the controller.** When a missing term
   belongs to an unexpressed clause, surface the post-recipient prefix and the
   separator rank as a diagnostic card, and measure whether Luna chooses it.
2. **Find a smaller structural actuator.** Sweep mid-layer attention/KV routes
   from the source ` while`/`,` positions; the current patches only win at the
   last layers with large blends.
3. **Norm-relative caps.** A 6% structural push is two orders of magnitude
   below answer-injection doses; decide whether such a budget is acceptable as
   "bounded".
4. **L2 compression** as a separate line (register/terseness rather than
   clause structure).

Reproduce:

```bash
python3 -m SpiralInterventionLab.examples.survey_clause_structure \
  --worker-model-path "$LLAMA_MODEL_PATH" \
  --checkpoint-manifest results/entity_probe_refresh_llama_20261014_retry1/manifest.json \
  --reference-conditions results/llama_rewrite_ladder_20260914/conditions.jsonl \
  --output results/<new_dir>/survey.json
SRC=results/entity_probe_refresh_llama_20261014_retry1
python3 -m SpiralInterventionLab.examples.replay_structure_token_probe \
  --source-jsonl $SRC/live/c1.jsonl --source-manifest $SRC/manifest.json \
  --worker-model-path "$LLAMA_MODEL_PATH" --positions 8 --dose-positions 8 \
  --patch-source-terms while,Selim --skip-forced-and-readout \
  --output results/<new_dir>/structure_patch.json
```

One prompt template, one worker, greedy decoding and fp16 MPS numerics: these
are mechanism observations for this ladder, not a general efficacy estimate.
