# Predeclared Common Radial Comparison

Question: does common's comma advantage depend on its component parallel to the
current activation, or on its orthogonal component? This is a geometric control,
not a semantic-subspace or causal-ownership claim.

- Keep the verified Llama-3.2-3B full 28-layer checkpoint, CPU/float32, four threads,
  seed 0, natural position 8 prefix, exact SOURCE IDs 11 and 1418 and source
  positions, residual-pre L14 last, alpha 0.04, step cap 0.08, norm clip 1, TTL 1.
- Preserve seven original conditions in order: source 11, source 1418, self-copy,
  radial shrink, common, positive difference, negative difference.
- Add exactly three conditions: negative common, common parallel to current T,
  common orthogonal to T. C=mean(S_left,S_right), d=C-T. Sources are T-d,
  T+project_parallel(d,T), T+project_orthogonal(d,T), using existing DSL.
- No rescaling of components, no adaptive sites, seeds, doses or additional
  permutations. Report actual written norms and rounding; separately clipped
  responses are not an additive attribution.
- Two no-edit and two edited replays per condition. Ten condition slots plus
  alignment and observation consume 12/12 existing diagnostic slots.
- Capture all 28 layers and raw logits. Compare full-vocabulary raw probabilities,
  comma/while contrast and margins to the fixed baseline greedy winner, repeats,
  self-copy and baseline drift floors. Record signed radial coefficient, not
  an assumed shrink label.
- Recheck unchanged checkpoint, natural prefix, actual tensor/source/site/cap
  identities before accepting the comparison. Retain partial/error receipts.
- One supervised attempt: start disk >=32 GiB; stop own child below 16 GiB or
  after 900 seconds. No downloads, implicit retries or other-process termination.
- No controller provider calls, production edits or task-success claim. Runtime
  diagnostic cards are separately tested for exact ID selection and no authority.
- Raw vectors stay local; hashes and numerical summaries are reproducible.

Recorded before starting the model measurement. No outcome-based modifications
to this plan are permitted within this attempt.
