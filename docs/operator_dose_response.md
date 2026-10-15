# Operator Dose Response and Structure Tokens

Date: 2026-10-16. Worker: local Llama-3.2-3B-Instruct (fp16, MPS), checkpoint
verified against the source manifest. Source prefixes:
`results/entity_probe_refresh_llama_20261014_retry1/` (L1 rewrite, missing `Selim`).
Follows [`orphan_blueprint_ceiling.md`](orphan_blueprint_ceiling.md), which found
the bounded operators 31-41x too weak to make ` Sel` the greedy token.

## Questions

1. **Dose**: what dose makes ` Sel` win, for readout steering and for
   source-anchored activation patches, and is the finished sentence correct?
2. **Role check**: the evaluator gives 1.0 to role-incorrect rewrites. Which
   outcomes are actually correct?
3. **Structure**: is the bottleneck the missing name, or the clause structure
   that drops it?

All edits are one-token replays in a quarantined worker whose surface caps,
alpha budget and replay policy are raised; operator shape (direction
construction, norm clip) is unchanged and shared defaults are untouched. No
controller call, rollout edit, production trial or apply permission is involved.

## Role check

`SpiralInterventionLab/tasks/role_binding.py` binds each source clause frame
(agent, action, theme, recipient) to the output: the action's nearest preceding
agent name (a possessive owns, it does not act), the first theme after the
action, and the first `to <name>` after the theme. It is a surface heuristic,
not a grammar check. Ladder tasks expose it as `role_binding_report()`; it is
**not** part of `score()`, `task_feedback()` or controller packets, so live
runs and earlier receipts stay comparable.

Applied to the ceiling receipts, only forced insertions at tokens 8 and 9 are
role-correct. The recorded natural output itself is a role violation: Nora,
not Selim, sends the report (`before sending the report to Yara`).

## Result 1: name dose

Readout steering (`results/operator_dose_response_20261016/dose.json`, two
recipes x 10 doses; the edited residual is `blocks.27.hook_resid_pre`, norm
42-45 at these prefixes; the added vector's norm equals the dose):

| Tokens | ` Sel` rank, gap | Minimal winning dose | Dose / residual norm | Finished sentence ends | Score | Roles |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 66, 9.55 | 10.24 | 24% | `Ivo Selim will send the report to Yara before dusk.` | 1.0 | correct |
| 9 | 5, 3.90 | 5.12 | 12% | `before Selim sends the report to Yara.` | 0.938889 | correct |
| 10 | 5, 3.37 | 5.12 | 12% | `before sending Selim's report to Yara before dusk.` | 1.0 | violated |
| 13 | 2, 4.44 | 5.12 | 12% | `before sending the report to Selim before dusk.` | 0.938889 | violated |

- The response is close to linear, about +1.3 to +1.5 logit per unit dose; the
  old 0.08 cap is about 0.2% of the residual norm.
- At every winning dose the finished sentence is identical across doses and
  identical to the forced-token ceiling. With a one-token edit, a winning readout
  push is functionally the same as writing ` Sel` into the output.

Source-anchored activation patches (`dose_patch.json`, 45 curves = 15 surfaces x
3 localizations, doses 0.08-1.0, step size 1000 so nothing is clipped):

- No win at tokens 8, 9 or 10. Best lifts are +1.2 to +2.2 logit at mid layers
  (e.g. L13 `resid_pre` span mean at token 9, rank 5 to 3).
- One win at token 13 (`mlp_out` L26 centered +-1, dose 0.64) lifts ` Sel` by only
  +1.05 and wins by depressing ` Y`; the sentence is role-incorrect.
- Larger doses mostly hurt: the median target-logit change at dose 1.0 is -5 to
  -12, and 44-91 of 225 rows per prefix change the top token. Blending in the
  source activation corrupts the state more than it carries `Selim`.
- The first patch attempt (`dose.json`, 900 patch rows) failed entirely with
  `simulate_decode_failed`: a 1e9 step size exceeded the worker's packet
  edit-cost budget. Those rows are superseded by `dose_patch.json`; the readout
  rows in `dose.json` are valid.

## Result 2: structure, not the name

`structure_probe.json` forces clause connectives and pushes a connective with the
same readout operator, retargeted to that token id, with exact full-vocabulary
ranks.

At token 8 (`Nora will take the sample to Ivo`) the natural next token is
` before` (p 0.56), and `,` is **rank 2 (p 0.18), only 1.125 logit behind**.
Forcing it lets the worker write the missing clause itself:

> Nora will take the sample to Ivo, while Selim sends the report to Yara before dusk.

Score 1.0, roles correct, no constraint violations, 16 words. ` while`, ` whereas`
and `;` also give role-correct 1.0 rewrites; ` and` keeps Nora as the sender.
At token 9 the connectives are far down (ranks 61-5889) and every forced
continuation drops `before dusk`.

The readout push toward `,` wins at dose 2.56 (6% of the residual norm), a
quarter of the 10.24 the name needed at the same prefix, with the same correct
sentence at 2.56 and 5.12. Here the controller nudges punctuation only; `Selim`
comes from the worker's own copy of the source.

## Reading

- The family that reaches the name is the one closest to answer injection: at a
  winning dose, readout steering selects the output token. Raising its cap by
  two orders of magnitude to write names would bypass the design guardrail that
  the controller chooses bounded interventions, not answers.
- The source-anchored family that would count as internal leverage does not
  reach at all, and larger doses damage the state.
- The L1 failure is better described as clause planning: the worker merges the
  second clause into a gerund (`before sending`) and drops its agent. A small
  structural nudge at the right prefix recovers the full, correct rewrite.

## Narrowed next candidates

1. **Structure-target actuators.** Offer clause-boundary connectives as
   diagnostic targets where a missing term belongs to an unexpressed clause
   (missing agent with a present action). Test whether a non-readout family
   (e.g. attention or mid-layer patches) can flip ` before` to `,` at token 8.
2. **A norm-relative dose contract.** Express caps as a fraction of the edited
   residual norm. 6% suffices here for a structure token; 0.2% (the old cap)
   does not; 12-24% for names is in answer-injection territory.
3. **Promote the role check into evaluation** for ladder runs as a reported
   field next to the legacy score, then decide whether controller feedback
   should see it.
4. **Generality.** Repeat the connective probe on the L1 delivery/workshop cases
   and L2/L3 to see whether clause merging is the common failure.

Reproduce (one tool per process; the patch family takes about 70 minutes on MPS):

```bash
SRC=results/entity_probe_refresh_llama_20261014_retry1
python3 -m SpiralInterventionLab.examples.replay_operator_dose_response \
  --source-jsonl $SRC/live/c1.jsonl --source-manifest $SRC/manifest.json \
  --worker-model-path "$LLAMA_MODEL_PATH" --positions 8,9,10,13 --families readout \
  --output results/<new_dir>/dose_readout.json
python3 -m SpiralInterventionLab.examples.replay_structure_token_probe \
  --source-jsonl $SRC/live/c1.jsonl --source-manifest $SRC/manifest.json \
  --worker-model-path "$LLAMA_MODEL_PATH" --output results/<new_dir>/structure_probe.json
```

Use `--families patch` for the activation-patch curves. One trajectory, one
prompt and fp16 MPS numerics: these are mechanism observations for this case,
not a general efficacy estimate.
