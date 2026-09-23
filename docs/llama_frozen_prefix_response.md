# Frozen Candidate, Different Prefixes

Follow-up: [entry competition versus name completion](llama_competition_completion.md)
fixes the `before` prefix and separates the competitor gap, natural generation,
and a clearly labeled conditional name-completion branch.

A subsequent [live Luna pair](llama_live_review_reuse.md) observed four reused
reviews, but no explicit new-prefix measurement or task/API-token improvement.

## Question and Boundary

The L1/review/seed-0 Luna run finished with twelve charged diagnostics, no
rollout edits, and Selim still missing. Repeated activation-patch reviews
returned already-measured evidence. This follow-up separates two questions:

1. Can the runtime reuse a closed review without spending another diagnostic?
2. Does the same measured activation patch respond differently around a
   connective, without changing source, binding, or dose?

This is a local diagnostic experiment, not another Luna run or a production
trial. No controller API calls, answer hints, stronger doses, or new apply
permission were introduced. The lexical scorer is unchanged; it does not
certify semantic faithfulness or preservation of the two source actors.

## Implementation Contract

`runtime/diagnostic_reuse.py` caches only closed negative activation-patch
candidate/runtime-support reviews. The key includes objective, normalized
request intent, and the available evidence including materialization
descriptors. New evidence or changed intent reopens the review. Failed or
incomplete reviews, measurements, and promotion/trial decisions are not cached.
An episode reset clears the cache.

A reuse result is explicitly `historical_review_not_current_measurement`,
reports whether the prefix changed, charges zero diagnostic budget, and grants
no permission. Compact packets and the prompt distinguish reading old evidence
from explicitly requesting `matched_response_probe` at the new prefix. Reused
events remain in logs but do not enter the charged diagnostic history.

In a scripted replay of recorded command metadata at controller steps 4-8,
steps 5 and 8 reused reviews: **two of five review executions were avoided**.
Step 6 requested a different review type; its changed evidence reopened step 7.
This conservative cache is not a blanket ban on exploring a previous candidate.
The audit substitutes recorded results and omits packet-derived canonical
requests. It validates accounting on that fixed sequence, not future Luna
behavior, API-token savings, or the outcome of a changed controller trajectory.

`runtime/prefix_probe.py` freezes an actual source tensor from a recorded
candidate and keeps its hash, operator, site, alpha, cap, and target token
identical across prefixes. Its companion CLI checks the checkpoint, tokenizer,
prompt, every rebuilt prefix, anchor state identity, and source tensor hash.
It fails closed on mismatches. A temporary trace carries the frozen source;
this trace and all hooks are removed after each measurement.

The plan uses the recorded candidate anchor and the positions immediately
before/after the first subsequent connective or punctuation match. It does not
choose prefixes by response, rank, or desired answer. This is a lexical boundary
heuristic, not a validated syntactic clause parser. Each point has no-edit,
no-edit repeat, edited, and edited repeat conditions. The edit is active for
**one token only**; the suffix is then generated unedited, with a fixed maximum
16-token horizon and the original sentence/EOS stopping rule.

## Observation: 2026-09-21

Full-depth Llama, 28 layers, MPS/FP16, raw prompt, greedy decoding, L1/review/
seed 0 and the original 64-token ceiling were preserved. The source is
`resid_pre L27 / source_term_token`; `alpha=0.04`, step-size cap `0.16`, TTL 1.
The canonical frozen observable is token 24082, ` Sel`, not the whole name.
The historical recipe ID ends in `step_size=0.0800`, but the measured dose field
is `0.16`. The new frozen candidate ID includes the actual materialized budget
rather than trusting that legacy recipe name.

| Prefix ending | Worker step | No-edit rank | Edited rank | No-edit piece probability | Piece probability delta |
| --- | --- | --- | --- | --- | --- |
| `the sample` | 5 | 119 | 119 | 0.000003938 | +0.0000000343 |
| `to Ivo` | 8 | 66 | 65 | 0.000040020 | +0.0000006669 |
| `Ivo before` | 9 | 5 | 5 | 0.010285018 | +0.0001621729 |

All three points have identical target logit delta `+0.0078125`, unchanged
top-20 threshold, gap delta `-0.0078125`, and top-20 hit delta zero. At step 9
the piece was **already** in the top 20 without intervention. This is not a new
top-20 entry. Bound-piece probability is not whole-term probability or an
ownership-certified multi-term effect.

The twelve replay trajectories completed in 198 seconds, with 128 generated
token forward steps inside the probes. All six no-edit/edited terminal outputs
were identical:

> Nora will take the sample to Ivo before sending the report to Yara before dusk.

Score remains `0.938889`; Selim is absent. Every trajectory stopped at sentence
punctuation, not the lookahead ceiling. No-edit and edited-repeat logit controls
both had zero maximum difference at every point. All three state-restoration
checks passed, and each reported edit telemetry row recorded one hook call and
step-size saturation. The source tensor hash was identical across points and
matched the previous live measurement.

The first attempt reached the correct anchor/source but exceeded the fixed MPS
memory ceiling during a multi-token suffix simulation. It is retained as an
incomplete resource failure, not a dead operator. The retry used one-token
suffix simulations with unoccupied-buffer release, retaining horizon, model,
candidate, dose, and the 0.7 MPS memory fraction. Recorded boundary allocations
in the successful retry peaked at 10.87 GB; that is not continuous peak-memory
instrumentation. The anchor logit/probability deltas exactly reproduced the
earlier Luna-run measurement.

## Interpretation and Next Test

The same patch has a much larger absolute bound-piece probability effect after
`before`, where the unedited distribution already makes that piece plausible.
It does **not** have a larger measured logit effect there. Prefix-dependent
baseline accessibility and operator response must therefore remain separate.
The equal logit increments are only one local FP16 spacing; deterministic repeat
controls do not establish higher-precision fidelity or a general mechanism.

This narrows the question from "does the hook work?" to "can a context-local
token response change the competing continuation and realize the whole actor?"
It does not demonstrate task uplift, a certified actuator, or a universal best
insertion position. Only one source, dose, binding, task, and three recorded
prefixes were tested; later connective boundaries were not searched.

The next bounded comparison can keep the `before` prefix fixed and inspect the
actual next-token competitor (`sending`) and the continuation needed to finish
Selim, then confirm any promising effect under the existing normal-budget and
ownership gates. Do not convert the positive piece delta or cached review into
apply permission. Separately, a future Luna run can test whether the compact
reuse notice changes diagnostic requests; that behavioral claim remains open.

## Artifacts and Reproduction

- [Frozen plan and source/checkpoint hashes](../results/llama_l1_frozen_prefix_20260921_r2/manifest.json).
- [Candidate and tensor identity](../results/llama_l1_frozen_prefix_20260921_r2/frozen_candidate.json).
- [All measured rows](../results/llama_l1_frozen_prefix_20260921_r2/rows.jsonl),
  [summary](../results/llama_l1_frozen_prefix_20260921_r2/summary.json), and
  [completion/hash record](../results/llama_l1_frozen_prefix_20260921_r2/status.json).
- [Scripted review accounting](../results/llama_l1_frozen_prefix_20260921_r2/review_reuse_audit.json)
  and its [reproduction script](../results/llama_l1_frozen_prefix_20260921_r2/audit_review_reuse.py).
- [Incomplete first attempt](../results/llama_l1_frozen_prefix_20260921/status.json).
- [Verification record](../results/llama_l1_frozen_prefix_20260921_r2/verification.json):
  artifact hashes and the post-measurement diagnostic-hints type guard, which
  does not change forward/replay behavior.

```bash
python3 -m SpiralInterventionLab.examples.replay_prefix_response \
  --source-jsonl results/llama_l1_luna_20260921_r2/live/c1.jsonl \
  --source-manifest results/llama_l1_luna_20260921_r2/manifest.json \
  --worker-model-path "${WORKER_MODEL_PATH:?Set the matching local HF checkpoint}" \
  --controller-step 4 --recorded-dose 0.16 --horizon 16 \
  --output-dir results/llama_l1_frozen_prefix_repeat
```

The output directory must not exist. Model paths are never hardcoded into the
CLI. Results describe the current working tree, not an already merged release.

Verification: `python3 -m pytest SpiralInterventionLab/tests -q` passed
**346 tests and 2 subtests** with 16 warnings. `git diff --check` passed.
