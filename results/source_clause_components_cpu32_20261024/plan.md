# CPU/float32 replication of the fixed source-component experiment

Declared before model execution on 2026-10-24. This resumes the blocked numerical
replication; all earlier blocked receipts and the MPS error/reanalysis are retained.
The original design is `../source_clause_components_20261023/plan.md`.

- Full Llama-3.2-3B checkpoint, local model path supplied by CLI.
- Natural prefix at position 8: ` Nora will take the sample to Ivo`.
- Same exact token prefix, SOURCE comma (11, position 76), SOURCE ` while`
  (1418, position 86), offered resid_pre L14, alpha 0.04, step cap 0.08,
  norm cap 1.0, TTL 1. Fail closed on identity/cap mismatch.
- Same ordered seven conditions: source:11, source:1418, self_copy,
  radial_shrink, common, difference_positive, difference_negative.
- Two no-edit and two edited one-token replays per condition. Nine of 12
  logical slots including alignment/observation; 30 explicit diagnostic
  replays. Count packet canaries and actual forwards separately.
- CPU/float32, four Torch threads. No provider calls, production applies,
  source/site search, threshold tuning, model downloads, or implicit retries.
- Preserve the original resource guard: start >=32 GiB free; stop below
  16 GiB or after 900 seconds, terminating only this experiment's child.
- Compare comma/while probability deltas and comma-minus-while raw-logit
  contrast, plus all structural-token readouts, greedy IDs, captured null /
  repeat floors, hook/dose validation and restoration checks.
- Prior MPS capture used zeros_like for radial shrink. The existing fix uses
  target * 0.0 to match the compiled source's signed zeros. Retain that exact
  fix; do not relax hash validation. The runner also records completed replay
  counts before analysis. Record code-hash differences explicitly rather than
  claiming that the pre-fix MPS and this run have identical source code.
- MPS/float16 versus CPU/float32 changes backend and dtype together. Contrast
  reversal is not absolute-lift ownership, additive mediation, greedy-token
  change, semantic success, or permission to apply.
