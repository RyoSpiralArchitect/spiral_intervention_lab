# Orphan Blueprint Ceiling Replay

Date: 2026-10-15. Worker: local Llama-3.2-3B-Instruct (fp16, MPS). Source:
`results/entity_probe_refresh_llama_20261014_retry1/` (Luna/Llama L1).

## Question

The 2026-10-14 probe-refresh retry emitted `report`, `send` and `Selim`
blueprints at token 6, then exhausted its 12-slot budget before replaying any
of them. Would more budget or better routing have mattered? This replay
separates two questions that a single live run cannot:

1. **Insertion value**: if the objective's first piece (` Sel`) were emitted at
   a given prefix, does the finished sentence improve? This is an operator-free
   ceiling: force the piece, decode greedily, score with the unchanged task
   evaluator.
2. **Actuator reach**: can any existing bounded operator make ` Sel` the greedy
   token at that prefix? 93 candidates per prefix: three readout-steering recipes
   and every activation-patch surface x {term token, centered +-1, span mean} x
   alpha {0.04, 0.08}, all under the existing 0.08 cap. Only a candidate that
   wins the first token is decoded to the end.

The three recorded blueprints are also replayed through the runtime's own
materialization paths with an unconstrained diagnostic budget. Everything is a
quarantined replay with state restored: no controller call, rollout edit,
production trial or apply permission. All 12 recorded prefixes were rebuilt
without drift, and the unforced branch reproduces the recorded output and score.

## Result

| Tokens | Prefix ends | ` Sel` rank | Gap to top-1 (logit) | Best operator lift | Forced ` Sel` sentence ends | Score |
| --- | --- | --- | --- | --- | --- | --- |
| 6 | `sample to` | 3 | 4.13 | +0.109 | `to Selim before dusk, who will send the report to Yara.` | 0.938889 |
| 8 | `to Ivo` | 66 | 9.55 | +0.117 | `Ivo Selim will send the report to Yara before dusk.` | 1.0 |
| 9 | `Ivo before` | 5 | 3.90 | +0.094 | `before Selim sends the report to Yara.` | 0.938889 |
| 10 | `before sending` | 5 | 3.37 | +0.094 | `sending Selim's report to Yara before dusk.` | 1.0 |
| 11 | `sending the` | 131 | 12.71 | +0.125 | `sending the Selim's report to Yara before dusk.` | 1.0 |
| 13 | `report to` | 2 | 4.44 | +0.141 | `report to Selim before dusk.` | 0.938889 |

Positions 7, 12 and 14-17 put ` Sel` beyond rank 160 (gaps 12.6-27.6) and are
in the [receipts](../results/orphan_blueprint_ceiling_20261015/). Forcing there
scores 0.938889, except 14 (`to Y Selim.`, 0.877778) and 17 (`before dusk
Selim.`, 1.0). Unforced, every prefix ends at 0.938889 with `Selim` missing.

- **No bounded operator wins anywhere.** 0 of 1,116 first-token replays made
  ` Sel` the greedy token. At the four prefixes where ` Sel` is within rank 5,
  the best lift at that prefix is 31-41x smaller than its gap (e.g. 0.094
  against 3.90 at the role-correct prefix 9). This is a shortfall of operator
  strength, not of routing or budget. Best post-edit ranks equal the unedited
  ranks there. Target-logit lift is reported, not gap closure; competitors can
  move too.
- **Activation patches were movement-limited.** All 1,080 activation-patch rows
  are `activation_patch_step_size_clip_saturated`. The step size defaults to
  alpha, so each arm was clipped at its own limit: doubling alpha from 0.04 to
  0.08 changed the target-logit lift in 304 of 540 pairs, but only slightly
  (e.g. token 9 `resid_pre` term token, 0.016 to 0.023). No patch exceeded
  |0.055| logit; the strongest rows are readout steering at alpha 0.08.
- **The orphan blueprints would not have changed the outcome.** Readout replays
  report `target_lift` (`send`, `Selim`) and `readout_gap_movement` (`report`)
  without rank change. Of 24 activation-patch rows, 23 are dead or neutral; one
  `report` row is labeled `rank_carrier`, also without a rank change.
- **Blueprint ranks were stale when issued.** The blueprints carry ` Sel`
  rank 119 from the token-5 probe; at the token-6 review prefix it is rank 3.
- **The evaluator rewards the wrong insertions.** Its four 1.0 sentences are
  missing a comma (8), reassign the report from Selim to Nora (10), or are
  ungrammatical (11, 17). The only role-correct insertion (9: `before Selim
  sends the report to Yara`) drops `before dusk` and scores 0.938889. A 1.0
  score from this task is not, by itself, evidence of a correct rewrite.
- **The budget premise depends on the code revision.** The retry ran an
  intermediate accounting revision that charged the token-4 review of an
  unchanged `sample` blueprint. The merged novelty rule would not, leaving one
  slot after the token-6 review. Per the results above, one replay of any of
  the three blueprints would not have changed the outcome.

## Interview comparison

The run's post-run interview attributed failure to insufficiently reproducible
efficacy, dead actuators and a weak corrective signal, and suggested rechecking
candidates at a comparable prefix. The ceiling supports only the last point
(stale ranks). Repeat/no-edit variation was zero; the limit is effect size,
not noise. The interview never mentioned the missing `Selim` or the refresh
blueprints because its sampled view omitted those lines.

## Narrowed improvement candidates

1. **Dose gap first.** A diagnostic-only dose response at tokens 9, 10 and 13
   would show whether any dose flips ` Sel` before the text degrades. Raise
   the step-size limit together with alpha for activation patches (every arm
   here was clipped) and raise alpha for readout steering. If none does, intervention within this operator family cannot
   solve L1, and the next step is a different family (e.g. multi-token or
   attention-route source copy), not more accounting.
2. **Position-aware value as an offer.** The forced-first-piece branch costs one
   greedy decode and tells the controller where insertion pays. Blueprint ranks
   should be restamped at the prefix where they are reviewed.
3. **Effect size in the interview.** Expose "required gap vs achieved lift" so
   the interview cannot read small effects as noise, and keep missing-term and
   refresh anchors in its view.
4. **Role-aware task check.** Before treating 1.0 as success, check who does
   what (Nora takes the sample to Ivo; Selim sends the report to Yara), e.g. via
   the existing observer critic.

Reproduce (one part per process to bound MPS memory):

```bash
python3 -m SpiralInterventionLab.examples.replay_orphan_blueprint_ceiling \
  --source-jsonl results/entity_probe_refresh_llama_20261014_retry1/live/c1.jsonl \
  --source-manifest results/entity_probe_refresh_llama_20261014_retry1/manifest.json \
  --worker-model-path "$LLAMA_MODEL_PATH" --parts ceiling \
  --output results/<new_dir>/ceiling.json
```

Run `--parts orphan` and `--parts sweep` the same way. The sweep took about
65 minutes on MPS. MPS numerics have not been independently qualified against
CPU; gaps of 3+ logits are far outside the fp16 differences measured earlier.
