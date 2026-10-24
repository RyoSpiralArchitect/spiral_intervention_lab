# Source-response trajectory audit

## Question and scope

The aligned SOURCE-clause experiment found that patches from comma and ` while`
both increased comma probability. That does not establish that the sources,
interventions, or downstream trajectories were equivalent. This audit separates:

1. The two cached source activations, `S_comma` and `S_while`.
2. The intended displacements, `S_comma - T` and `S_while - T`, where `T` is the
   same current answer-position activation.
3. The actual bounded, dtype-rounded post-hook displacements.
4. Their last-position differences from no-edit baselines at each residual/MLP
   site, final normalization and the raw output logits.

The sources and position are declared before measurement. This is a local
geometric audit, not a causal mediation experiment, semantic-role certificate,
controller-choice evaluation, or task-success claim. Similar output probability
changes need not imply the same hidden path. Large baseline activation similarity
must not be substituted for similarity of the intervention-induced differences.
The pair was chosen because of the previous screen's shared comma response:
this is a hypothesis-guided follow-up, not a held-out source-selection test.

## Measurement contract

The CLI is `SpiralInterventionLab.examples.replay_source_clause_trajectory`.
It reuses the normal-cap SOURCE patch and specificity-control implementations;
it does not add a runtime operator, controller card, or policy override.

- Reconstruct a recorded prefix naturally, without forced output tokens. Refuse
  a mismatch in either text or exact input-token identity.
- Verify checkpoint hashes and the previous receipt's source IDs, positions,
  localization, surfaces and caps. Device/dtype changes are explicit.
- Charge the existing no-edit cache alignment and structural observation.
- Run the predeclared pair, self-copy and radial-shrink controls, twice each,
  with two separate no-edit baselines for each condition.
- Capture existing forward results after intervention hooks. Do not execute
  extra forwards for vector collection or change hook registration order.
- Check the captured intervention-site difference against the actual bounded
  blend arithmetic. A pre-hook cache is not acceptable evidence.
- Restore the capture wrapper even on failure; retain normal state, cap and
  hook-execution checks. The diagnostic ledger/budget may change, not generation.
- Record all source-structure token probability effects in the existing receipts,
  not just the source's own token. Raw output logits are explicitly pre-constraint;
  receipt probabilities use the existing post-constraint readout path.

For a single offered site this declares **18 diagnostic replays and six of 12
diagnostic slots**, including alignment and observation. These are not total
model forwards: packet construction can also run an unedited answer-readout
canary. The instrument now tags packet preparation separately, counts all actual
forward attempts/completions, and reports prefix reconstruction separately. No
provider calls, production applies, or full continuation evaluations are included.

## Reading the geometry

For each site, compare the two vectors `edited - baseline`. Report both norms,
their cosine, and `norm(delta_a - delta_b) / max(norm(delta_a), norm(delta_b))`.
A zero vector has undefined cosine, represented as null, not perfect agreement.
Raw-logit differences also get a centered comparison to expose vocabulary-wide
constant offsets that softmax would ignore.

Retain the self-copy response, repeat-difference norm and cross-condition baseline
drift at every stage. These are observed noise/null controls, not confidence
intervals. Residual and MLP sites are different coordinates in the computation;
their absolute norms should not be read as a single conserved causal quantity.
Last-position collection does not establish the absence of changes at other
positions or isolate an attention/MLP causal mediator.

Each condition retains CPU copies of the last-position vectors and raw logits in
a `.pt` artifact, plus hashes and ordinary JSON receipts. The analyzer can be
recomputed without a new model run. Missing stages, incomplete repeats,
cross-context comparisons and unverified post-hook captures fail closed.

## Reproduction

```sh
python3 -m SpiralInterventionLab.examples.replay_source_clause_trajectory \
  --source-jsonl <completed-episode.jsonl> \
  --source-manifest <checkpoint-and-profile-manifest.json> \
  --reference-receipt <aligned-source-specificity.json> \
  --worker-model-path <local-model-directory> \
  --position 8 --source-token-ids 11,1418 \
  --device cpu --dtype float32 --torch-threads 4 \
  --output-dir <new-directory>
```

The numeric IDs above belong to this Llama checkpoint/tokenizer, not a portable
term mapping. Model paths remain CLI-supplied. Each run refuses an existing output
directory. Resource supervision should stop only its own child on disk pressure
or timeout and retain the failed/interrupted attempt rather than overwrite it.

## Results: 2026-10-22

Both numerical configurations naturally reconstructed the same eight-token prefix,
` Nora will take the sample to Ivo`, with identical 108 input-token IDs. Sources
were comma at position 76 and ` while` at position 86. Both used the already
offered `resid_pre L14`, alpha 0.04, step-size cap 0.08 and TTL 1. No caps changed.

### A measurement-construction correction

The first CPU/float32 run completed all four conditions, but the analyzer refused
them: it expected four forwards per condition and captured seven. The additional
three are answer-readout canaries from packet preparation (one preflight and one
before each edited replay). The original error receipt and vectors are retained.
This was not a failed operator replay or missing hook.

The CPU reanalysis is explicitly post-hoc. It partitions the known call path only
after checking **exact equality of every captured vector** within all five
baseline/preparation frames and within both edited frames. All four conditions
pass, as does the independent bounded-blend check. This salvages the geometric
comparison without pretending the original capture had call-site labels. No
noisy or ambiguous partition is accepted, and no raw receipt is rewritten.

A fresh MPS/float16 run uses capture-time replay/packet labels and completes
normally. It observes **31 forwards in the diagnostic interval**, of which 18 are
the declared diagnostic replays and 13 are packet canaries. The whole run has
39 forwards, including eight for prefix reconstruction. Both configurations spend
six diagnostic slots; slot count, replay count and physical forward count are
distinct measures. Earlier notes' reported forward counts should be read as
explicit diagnostic replay counts, not complete physical-compute accounting.

### Inputs differ, and differences survive downstream

Cosine between the two source conditions, measured on **differences from the
no-edit baseline** except for the first two rows:

| Comparison | CPU/float32 | MPS/float16 |
| --- | ---: | ---: |
| Source activations themselves | 0.4452 | 0.4452 |
| Source minus common current activation | 0.5897 | 0.5897 |
| L14 resid_pre displacement, after patch | 0.5897 | 0.5905 |
| L14 residual output displacement | 0.7343 | 0.7217 |
| L20 residual output displacement | 0.6947 | 0.7000 |
| L27 residual output displacement | 0.7053 | 0.6741 |
| Final normalized activation displacement | 0.7025 | 0.6727 |
| Raw output-logit displacement | 0.7438 | 0.7850 |
| Mean-centered output-logit displacement | 0.7063 | 0.6732 |

Subtracting the shared target already raises alignment before downstream
processing. L14 raises it further, but the progression is not monotonic and
differences do not disappear. At L27, between-condition distance is still 0.743
(CPU) or 0.783 (MPS) times the larger response norm. Even the raw-logit cosine
overstates alignment relative to the mean-centered, softmax-relevant comparison.

Every observed baseline-drift, repeat-difference, preparation-versus-baseline and
self-copy norm floor is zero. Captured patch displacements match the dtype-rounded
blend exactly. Requested cap 0.08 is not identical to the written fp16 difference:
the latter has norms 0.080025 and 0.080141 here, due to output rounding.

### Shared comma response is numerically sensitive

| Condition | Comma probability delta, CPU/float32 | MPS/float16 |
| --- | ---: | ---: |
| Comma source | +0.00126560 | +0.00394066 |
| ` while` source | +0.00142928 | +0.00402286 |
| Self-copy | 0 | 0 |
| Radial shrink | -0.00049202 | -0.00046511 |

The MPS readout deltas reproduce the previous aligned receipt. The CPU path
reproduces the positive comma direction for both sources and the negative radial
control, but the source effects are only about one third as large. Some small
cross-token signs change: comma-source to ` while` is negative on CPU and positive
on MPS. Backend and dtype changed together, so this does **not** isolate a PyTorch
MPS bug from precision or other numerical differences. The MPS loader warning
still applies; repeatability does not establish numerical accuracy.

Every condition still greedily selects ` before`. These are one-token diagnostics,
not full rewrite runs. Production apply remains false and provider calls remain
zero.

## Interpretation and next test

The binary question was too coarse. The evidence is consistent with **distinct
source-dependent displacements acquiring some shared components while both
favoring comma at readout**, not identical inputs or complete trajectory collapse.
It does not identify a causal mediator or prove an attractor-manifold mechanism.

Before increasing strength, a next bounded test can separate the paired input
directions into common and difference components, retain the same cap and null
controls, and test whether the common component accounts for the comma response
while the difference component changes comma-versus-while ownership. That would
be a new intervention experiment, not a conclusion licensed by cosine alone.

The [2026-10-23 component follow-up](source_response_components.md) now tests this
under the same caps. Its MPS result separates relative token preference from
shared probability lift, without changing the greedy token. CPU replication is
resource-blocked; the component experiment is not yet numerically replicated.

Artifacts:

- [CPU original error receipt](../results/source_clause_trajectory_cpu32_20261022/measurement/trajectory.json)
- [CPU explicitly qualified reanalysis](../results/source_clause_trajectory_cpu32_20261022/trajectory_reanalysis.json)
- [CPU reanalysis procedure](../results/source_clause_trajectory_cpu32_20261022/reanalyze_retained.py)
- [Labelled MPS receipt](../results/source_clause_trajectory_mps16_20261022/measurement/trajectory.json)
- [Joined numerical comparison](../results/source_clause_trajectory_mps16_20261022/comparison.json)
- [MPS supervision](../results/source_clause_trajectory_mps16_20261022/supervision.json)
- [Full test output](../results/source_clause_trajectory_mps16_20261022/tests.txt)

Validation: **672 passed, 9 subtests passed, 15 dependency warnings**. New tests
exercise post-hook TransformerLens capture, exact wrapper restoration, ambiguous
or missing captures, unchanged authority/caps, and packet canaries counted without
being mistaken for diagnostic replay frames. `git diff --check` passes.
