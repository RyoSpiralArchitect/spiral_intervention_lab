# Published source-response evidence

The source-response PR publishes code, research notes, measurement JSON,
resource receipts, source snapshots, analysis scripts and checksum inventories
for the following experiments:

- `results/source_clause_trajectory_cpu32_20261022/`
- `results/source_clause_trajectory_mps16_20261022/`
- `results/source_clause_components_20261023/`
- `results/source_clause_components_cpu32_20261024/`
- `results/source_clause_choice_boundary9_cpu32_20261024/`

Raw `vectors_*.pt` tensors remain retained locally, unchanged, and are **not
included in Git**. The full `SHA256SUMS` inventories preserve their identities;
checking those inventories on a fresh clone will report missing local artifacts.
`PUBLISHED_SHA256SUMS` covers only the files published in this PR and can be
checked from the repository root. Neither inventory is a signature or independent
attestation.

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
