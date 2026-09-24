# Early activation-patch seed discovery (2026-09-25)

## Question

The source GPT-2 run exposed source-body Mira and Omar entity blueprints after
diagnostic call 2, but first exposed activation-patch evidence only at call 10.
Can an activation-patch review turn that already-visible blueprint into a
diagnostic card sooner, without granting apply permission or changing the
controller's canonical frontier?

## Fixed-prefix checks

- Source episode: `results/candidate_handoff_early_gpt2_soft_20260925/live/c1.jsonl`.
  Same constrained-rewrite prompt, model path supplied through CLI, seed 7,
  float32 MPS, and `activation_patch_expanded` surface profile.
- At prefix ` In`, inject only the two recorded early diagnostic results and
  force one diagnostic-only `activation_patch_candidate_review`. This is a
  counterfactual harness run, **not** an observed controller choice.
- The review materialized six blueprint recipes. Six of eight Mira evidence
  rows reached an activation hook, but all eight classified `dead_actuator` at
  this early prefix. Its matched-response subcheck produced four frozen
  diagnostic cards. The review used one diagnostic call, but at least 12
  physical replays (six hooked activation rows plus six matched-response
  replays). Fewer diagnostic calls do not imply less compute time.
- Repeating the review at ` In the case of`, still with only those two prior
  diagnostics, reproduced the source run's Mira `resid_pre L11
  source_term_token` response: `target_mass_delta=0.000039` and
  `target_top20_threshold_gap_delta=-0.000967`. Thus the intervening diagnostic
  history was not necessary for that response in this fixed case; the answer
  position matters.
- The early frozen `matched_recipe` for the same source recipe, `step_size=0.04`,
  and bound piece ` Mir` was remeasured at ` In the case of` without another
  activation-patch review. Four physical replays restored state and gave
  `target_piece_prob_delta=0.0000386458` and gap delta `-0.000967`. The bound
  piece was already rank 3 and its top-20 hit delta was zero. This is not
  answer-level success or production certification.

The replay script is
`SpiralInterventionLab/examples/replay_candidate_seed_discovery.py`. Its
results are under `results/candidate_seed_discovery_early_gpt2_20260925/`;
`mira_prefix1_offer.json` records the new packet option and physical-cost
lower bound, while `mira_prefix1_to_prefix4_remeasured.json` records the
current-prefix card measurement. The source run and replay share an MPS backend
for relative comparison. TransformerLens warns that this PyTorch MPS version
may silently produce incorrect results, so no cross-device correctness claim
is made.

## Runtime boundary

In `candidate_handoff_mode=soft`, a source-body exact-span blueprint now
exposes `activation_patch_candidate_review` as a diagnostic-only *option*.
The existing `operator_diagnostic_replay` remains the canonical frontier.
The option requires a live last-token activation-patch surface, remaining
diagnostic budget, a missing objective term, and no prior review or measured
card for that objective. It is not a measured seed, a candidate card, or an
automatic request. The controller still chooses whether to spend the call.

The matched-response rows carry both their executable `matched_recipe` ID and
their parent `seed_operator_recipe_id` into the frozen card descriptor. This
provenance was previously lost when worker results were compacted. No
production apply permission or trial gate changed.

## Live controller check

The same GPT-2 checkpoint, constrained-rewrite fixture, seed 7, float32 MPS,
Luna controller, 12 diagnostic calls, and soft handoff settings were run with
the new option. The new option was visible for Mira and Omar immediately after
the entity blueprint review (two calls spent), but the controller first chose
the canonical `operator_diagnostic_replay` and other diagnostics. Visibility
alone did not make the controller select early activation-patch review.

The first live run exposed a separate request-alias bug: one command named
`activation_patch_candidate_review` both with an explicit mode and via
`next_action` without a mode. The loop treated these as distinct and spent two
calls on the same Mira review. The corrected run normalizes the omitted mode
before deduplication.

| Run | First frozen card | Remaining calls at first card | Final c1 score |
| --- | --- | ---: | ---: |
| Source soft run | After call 11 | 1 | 0.5875 |
| Early-option run, before alias fix | After call 12 | 0 | 0.5875 |
| Early-option run, alias fixed | After call 10 | 2 | 0.5875 |

The one-call difference from the source run is observational, not a causal
estimate of the new option: the controller's diagnostic sequence also varied.
In the corrected run the controller reviewed Mira at call 10, then spent the
last two calls on budget gap confirmation and Omar review rather than frozen
card remeasurement. The Mira `resid_pre L11 source_term_token` row again had
`target_mass_delta=0.000039`, gap delta `-0.000967`, and bound piece ` Mir`
already at rank 3; its top-20 hit delta was zero. This is a small diagnostic
response, not production certification. All 17 controller commands were
`noop`; c1 and no-edit b0 both emitted
` In the case of a rewrite, the budget draft.` at score 0.5875.

Raw local evidence: `results/candidate_seed_discovery_soft_live_20260925/live/c1.jsonl`
(SHA-256 `4bff2450b78a8e6d71f5d6acdc4b5420062e4524d12fb0d75510b085e1bccdc5`)
and `results/candidate_seed_discovery_dedup_live_20260925/live/c1.jsonl`
(SHA-256 `ae7e9ebab3a9fba5da49b47d1fa752199cad231b53ecf3bf6f6ea4f7148bd6f5`).
The source soft run is
`results/candidate_handoff_early_gpt2_soft_20260925/live/c1.jsonl`
(SHA-256 `493e493b500ae361a64299c2f71e7e8d648eff30d8bf42c3c3800043e2521ddc`).
The full JSONL files remain local; this note records their comparison rather
than treating it as a controlled multi-seed estimate.

To repeat the corrected live condition with a locally supplied checkpoint,
set `MODEL_PATH` to the GPT-2 model directory and choose a fresh `LOG_DIR`:

```sh
python3 -m SpiralInterventionLab.examples.digit_transform_e2e \
  --provider openai --controller-model gpt-5.6-luna \
  --worker-model gpt2 --worker-model-path "$MODEL_PATH" --worker-hf-offline \
  --worker-device mps --worker-dtype float32 --worker-mps-mode conservative \
  --task constrained_rewrite --seed 7 --no-b1 \
  --controller-prompt-profile compact --controller-packet-view compact \
  --readout-analyzer sae_scaffold --readout-analyzer-rerank-mode apply \
  --activation-surface-profile activation_patch_expanded \
  --max-diagnostic-calls-per-run 12 --diagnostic-result-window 12 \
  --candidate-handoff-rounds 2 --candidate-handoff-mode soft \
  --post-run-debrief controller --post-run-debrief-max-output-tokens 1200 \
  --log-dir "$LOG_DIR"
```

Next question: why does the controller defer an executable but expensive
activation-patch review until late despite early visibility? A fixed-prefix
off/soft choice comparison should isolate the offer's effect before changing
priorities or apply authority. The MPS warning above also still applies.
