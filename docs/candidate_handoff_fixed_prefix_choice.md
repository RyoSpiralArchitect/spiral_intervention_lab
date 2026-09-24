# Fixed-prefix candidate handoff choice (2026-09-25)

## Question

Does the controller request an exact candidate measurement when a recorded,
preflight-measurable seed is visible with diagnostic budget still available?
This is a controller-choice question, not a worker-outcome or apply-permission
test.

## Method

- Source: `results/candidate_handoff_early_gpt2_soft_20260925/live/c1.jsonl`.
- GPT-2, constrained-rewrite seed 7, greedy prefix ` In the case of` after four
  generated tokens. The worker checkpoint remains a CLI input.
- Reuse the single recorded Mira `resid_pre L11 source_term_token` row whose
  target-piece binding was measured at that exact prefix (` Mir`, binding ID
  `tpb:3bb1c7f23ba3b882ac98`). The row is retained as a diagnostic seed;
  it is not granted production permission.
- Counterfactually set seven of twelve diagnostic calls used, leaving five.
  The original run first exposed the seed after diagnostic ten, leaving two.
  This is **not** evidence that the existing runtime discovers the seed early.
- Build off and soft packets from one worker state. Full-packet differences are
  confined to `strategy_hints.candidate_handoff` and
  `strategy_hints.available_next_diagnostics`; compact packets additionally
  differ in their source-packet digest. The same compact controller prompt and
  GPT-5.6 Luna provider are used for both conditions.
- Call the controller once per condition, then repeat with the call order
  reversed. Do not dispatch diagnostics, compile edits, or generate another
  token after the fixed prefix.

The reusable shadow harness is
`SpiralInterventionLab/examples/replay_candidate_handoff_choice.py`. It rejects
prompt/prefix drift, a missing or ambiguous recorded seed, and packet drift
outside the handoff fields. Outputs are
`results/candidate_handoff_fixed_prefix_gpt2_20260925/luna_choice.json` and
`results/candidate_handoff_fixed_prefix_gpt2_20260925/luna_choice_reversed.json`.

## Observation

| Condition | First call order | Reversed call order |
| --- | --- | --- |
| off | `target_entity_insertion_probe` | `target_entity_insertion_probe` |
| soft | exact `matched_response_probe` | exact `matched_response_probe` |

Both soft requests matched the offered objective, recipe ID, comparison axis,
focus term, and numeric dose grid `[0.04]`. Both commands were `noop` with a
diagnostic request; no edit was applied. The soft handoff remained explicitly
`production_apply_allowed=false`. This small, counterbalanced sample supports
the narrow claim that *visibility can change the controller's diagnostic
choice at this prefix*. It does not establish measurement success, improved
generation, or general reliability across prefixes/models.

In the source live run, the first activation-patch rows appeared only in the
tenth charged diagnostic, `activation_patch_production_trial_gate_review`.
Earlier `operator_diagnostic_replay`, gap confirmation, and conversion results
contained readout carriers or operator previews but no measured
activation-patch seed row. The live bottleneck is thus seed discovery latency,
not a demonstrated refusal to choose a visible seed.

## Follow-up

The diagnostic-only early blueprint review, frozen-card replay, and live
controller check are recorded in `docs/candidate_seed_discovery_early_review.md`.
The new option was visible after the entity blueprint review, but it was not
chosen immediately. A fixed-prefix early off/soft choice test remains useful
before changing diagnostic priorities or claiming general scheduling gains.
