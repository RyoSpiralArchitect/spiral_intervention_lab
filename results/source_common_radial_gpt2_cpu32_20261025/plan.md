# GPT-2 fixed-text-prefix radial comparison

Authorized on 2026-10-25. Run only after the Llama retry exits; no concurrent
model loads. The user explicitly chose the same visible-text prefix, supplied
as diagnostic prefill, NOT a claim of natural GPT-2 generation.

- Same task fixture/prompt and seed 0 as the Llama reference.
- Fixed output prefill: ` Nora will take the sample to Ivo`.
- Full GPT-2, CPU/float32, four threads, local checkpoint supplied by CLI.
- Exact source lexemes comma and ` while`, token IDs derived before replay by
  the current tokenizer. Require single-token encoding and an exact SOURCE
  occurrence; do not select by measured readout response.
- Same ten conditions, two repeats, twelve diagnostic slots, unchanged normal
  runtime surface selection and caps (alpha .04, step .08, norm_clip 1, TTL 1).
- Alignment and structure observation cost two slots. Count all actual forward
  attempts, including packet canaries, separately from condition replays.
- Same 900-second wall limit, >=32 GiB start / 16 GiB stop disk floors, one
  attempt, no implicit retry or download, no controller calls or production apply.
- Capture source/target identity, post-hook vectors, full-vocabulary logits,
  null/repeat floors, and actual writes. No component re-normalization.

Cross-model differences include tokenizer, prefix token count, baseline readout,
depth/site and activation norms. This is a same-visible-context diagnostic,
not a model-only causal comparison, task benchmark or ownership certification.
The Llama natural prefix is provenance, not a GPT-2 checkpoint identity.

The standalone harness does not change the package code being measured by the
ongoing Llama run. Private output segments are populated explicitly as prefill;
all generation counters in shared receipts must be read with that distinction.
