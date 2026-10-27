# Published Intervention Evidence

This is a derived public copy, not a new experiment. Local originals are unchanged.
`publication.json` records original and published SHA-256 hashes and every applied
transformation. These hashes detect changes; they are not signatures.

## Structural Rescue

- [Outcome and cost report](pilot/report.md)
- [Complete synthetic-task receipt](pilot/results.json)
- [Verified analysis](pilot/analysis.json)
- [Frozen manifest](pilot/manifest.json)
- [Recorded execution plan](pilot/execution_plan.json)
- [Initial memory-limited attempt](initial_attempt/results.json)

The completed pilot has 36 policy conditions and nine extra calibration branches.
Every policy obtains 2/3 calibration and 6/6 confirmation joint successes, with
unchanged full outputs. Nine API requests produce eight edits and one noop.
The original failed attempt is not treated as an efficacy observation.

The public source archive contains only the four historical files hashed by the
original execution plan, byte-identical to the local archive. It is deliberately
not a complete frozen runtime. Later runner/evaluator fixes in this PR were not
used to generate these live results. No additional API/model run was made during
PR preparation. New runs automatically archive and hash all package Python sources
and `pyproject.toml`, rather than relying on the original manual snapshot step.

Rebuild this report without a model or API:

```sh
python3 -S SpiralInterventionLab/examples/summarize_structural_rescue.py \
  results/published_intervention_20261027/pilot
```

## Source-Choice Audits

- [Original radial-only dispatch audit](source_choice/summary.json)
- [Exact recorded single-source replay](source_choice/selected_source_summary.json)
- [Fresh controller selection and execution audit](source_choice/fresh_choice_summary.json)

These are explicitly post-run derived audits, not live events. The JSONL audit
files retain that distinction. Raw choice receipts, host supervision, tensors,
and full source archives stay local; their original hashes remain in the audits.
The published audits alone cannot independently reproduce tensor-level results
or exact recorded-choice replay. Single-source `before` execution is verified
locally; historical comma/while comparison remains `not_covered`. No role rescue,
controller efficacy or production permission is inferred.

## Publication Boundary

Only repository-local path prefixes are removed from receipt keys. Numeric
measurements, controller choices, synthetic task content, failures and original
hash values are retained. The offline analysis is regenerated with the current
analyzer and records its hash. Model weights, raw vectors, unrelated experiments,
machine supervision logs and oracle output are not included.

`PUBLISHED_SHA256SUMS` covers this public subset, relative to the repository root.
The exporter refuses an existing destination; it never rewrites local originals.
