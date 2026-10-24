# Published source-response evidence

The source-response work publishes code, research notes, measurement JSON,
resource receipts, source snapshots, analysis scripts and checksum inventories
for the following experiments:

- `results/source_clause_trajectory_cpu32_20261022/`
- `results/source_clause_trajectory_mps16_20261022/`
- `results/source_clause_components_20261023/`
- `results/source_clause_components_cpu32_20261024/`
- `results/source_clause_choice_boundary9_cpu32_20261024/`
- `results/source_direction_controls_cpu32_20261024/` (fixed-coordinate follow-up)
- `results/source_common_radial_cpu32_20261025/` (preserved incomplete attempt)
- `results/source_common_radial_cpu32_retry1_20261025/` (authorized completed Llama retry)
- `results/source_common_radial_gpt2_cpu32_20261025/` (completed fixed-text GPT-2 prefill)
- `results/source_common_radial_comparison_20261025/` (offline paired analysis)

Raw `vectors_*.pt` tensors remain retained locally, unchanged, and are **not
included in Git**. The full `SHA256SUMS` inventories preserve their identities;
checking those inventories on a fresh clone will report missing local artifacts.
`PUBLISHED_SHA256SUMS` covers only each experiment's published file subset and can be
checked from the repository root. Neither inventory is a signature or independent
attestation.

The completed radial retry and GPT-2 experiments' `source_snapshot.tgz` archives
also remain local and are excluded from their `PUBLISHED_SHA256SUMS`. The earlier
experiments, including the incomplete radial attempt, publish the code snapshots
listed in their own inventories. Do not infer availability from the full local
inventory. Ignored analysis/run logs named in the published inventory are included
explicitly, so the published subset verifies on a fresh clone.

JSON receipts contain per-vector hashes and the derived measurements. Analysis
scripts require the retained tensors to rederive those measurements; the JSON
alone is not sufficient for independent tensor-level reproduction. Local
reanalysis verified those hashes and reproduced the reported summaries. The
CLI can produce a new attempt with a caller-supplied local checkpoint path, but
a new run is not an exact substitute for the original retained tensors.

Original blocked/error receipts are preserved alongside later completed runs
and explicitly qualified reanalysis. They are not rewritten as successes.
The tests and diagnostic measurements do not establish task success, broad
generalization, causal mediation, or production apply authorization.
