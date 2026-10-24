# Same-Prefix Radial Comparison

Both separately supervised measurements completed on 2026-10-25 (JST).
The Llama run is an explicitly authorized retry, not an implicit retry of its
predecessor. GPT-2 uses explicitly authorized fixed-text diagnostic prefill.

## Inputs

- `../source_common_radial_cpu32_retry1_20261025/`: natural Llama prefix,
  ten conditions, 12 slots, 81 actual forwards, 842.52 seconds.
- `../source_common_radial_gpt2_cpu32_20261025/`: same visible-text GPT-2 prefill,
  ten conditions, 12 slots, 42 actual forwards, 20.01 seconds.
- `../source_common_radial_cpu32_20261025/`: preserved incomplete first attempt;
  not treated as a completed experiment.

Both retain plans, supervision, raw-vector hashes and compact trajectory receipts.
`source_snapshot.tgz` and raw `vectors_*.pt` files stay local. Package source is
identical across the two new model measurements; the GPT-2 prefill harness has
its own hash. Published checksum inventories exclude those binary archives.

## Offline Verification

From the repository root, with retained local raw vectors and recorded package
code, `analyze.py` recomputes geometry and full-vocabulary readout. It fails closed
on incomplete runs, changed code, raw-hash mismatch, or compact/numerical drift.
It refuses to overwrite an existing summary. `--verify-existing` checks that the
saved summary exactly reproduces without rewriting it. The included `analysis.log`
and `verification.log` files record successful reanalysis.

```sh
python3 results/source_common_radial_comparison_20261025/analyze.py \
  --verify-existing \
  --directory results/source_common_radial_cpu32_retry1_20261025 \
  --previous results/source_direction_controls_cpu32_20261024/cpu32/measurement/trajectory.json \
  --interrupted results/source_common_radial_cpu32_20261025/cpu32/measurement/trajectory.json

python3 results/source_common_radial_comparison_20261025/analyze.py \
  --verify-existing \
  --directory results/source_common_radial_gpt2_cpu32_20261025
```

`build_comparison.py` assembles `comparison.json` from verified summaries, without
model calls. This is implementation-level reanalysis, not independent replication.

## Boundaries

The same visible context, absolute caps and 12-slot budget do not make this a
model-only causal comparison: checkpoint, tokenizer, site, activation scale,
baseline winner and natural-generation versus prefill history differ. No token
choice changed within either run, no task continuation was evaluated, and no
production apply permission was created. Diagnostic forwards remain distinct
from generation-time edits.

See `../../docs/source_common_radial.md` for observations, interpretation and
the explicit source-pair binding caveat before controller use.

Verification: full suite 728 passed, 9 subtests passed, 15 dependency warnings;
manual harnesses compiled and both completed actual measurements. `git diff
--check` passed. Test output is retained in `tests.txt`.
