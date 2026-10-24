# Fixed-boundary source-component experiment (2026-10-23)

Recorded before model measurement. This follows the source-response trajectory
audit, not a new source/site search. No provider calls or production applies.

- Full Llama-3.2-3B checkpoint, verified against the retained manifest.
- Naturally reconstruct position 8, ` Nora will take the sample to Ivo`.
- Exact input-token hash, SOURCE comma (11, position 76), SOURCE ` while`
  (1418, position 86), currently offered resid_pre L14 must match the reference.
- Same alpha 0.04, step cap 0.08, norm cap 1.0 and TTL 1. No cap override.
- In order: comma source, while source, self-copy, radial shrink, common,
  positive difference, negative difference. Two no-edit baselines and two
  edited replays per condition. A condition is one diagnostic slot.
- Common displacement is `(S_comma - T + S_while - T)/2`; difference is
  `(S_comma - S_while)/2`. Difference blend sources explicitly add `T` back.
- No source normalization or forced norm matching. Inspect raw displacement,
  clipping scale, source rounding and actual written norms. The raw average and
  half-difference need not be orthogonal. Separately capped responses are not an
  additive contribution decomposition.
- Seven conditions plus alignment and observation: nine of 12 slots, 30
  explicit diagnostic replays at one site. Count all packet canaries separately.
- Preserve all last-position stage vectors and raw logits. Validate post-hook
  blending, repeats, no-edit drift, self-copy, state restoration and cap identity.
- Primary continuous readouts: comma and while probability deltas, and change
  in `logit(comma) - logit(while)`. Also retain every observed structural token,
  including before and period, and raw greedy token IDs. No target-owner label
  is attached to the common component.
- Compare common with the two original sources at the same dose; compare signed
  differences with each other and null. Inspect whether common preserves shared
  comma response and whether difference reverses token preference. Do not infer
  semantic ownership, mediation, task success or apply permission.
- Run MPS/float16 once, then CPU/float32 once. Repeatability is not accuracy;
  backend and dtype remain confounded. No adaptive threshold/cap/source changes.
- Resource supervisor: MPS preflight 20 GiB, stop below 12 GiB or 600 s;
  CPU preflight 32 GiB, stop below 16 GiB or 900 s. Stop only the owned child.

Reference: `results/source_clause_specificity_llama_20261021_aligned/specificity.json`.
Source profile: `results/entity_probe_refresh_llama_20261014_retry1/manifest.json`.
Model path is provided by CLI and never embedded into the runtime or controller.
