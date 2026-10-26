# Source Pair and Structural Rescue Review

Review scope: exact source-pair recipes, controller-selected diagnostic dispatch,
the frozen structural-rescue runner/evaluator, offline reporting, and public artifacts.
This is an assistant self-review of the implementation, not independent approval.

## Findings Corrected Before Publication

- P1: The documented runner/report sequence could not complete from a fresh run
  because source capture was manual. The runner now captures and hashes package
  sources and environment versions before model loading. A runner-to-analyzer
  regression verifies that this archive is readable and cannot be overwritten.
- P2: No-boundary fallback forwards were copied into every policy, overstating
  physical work. Fallback is now charged once as shared work; overall outcomes
  retain unavailable cases, with eligible-pair counts explicitly separate.
- P2: Failed branch work and pre-model failures could disappear from receipts.
  Branch failures now retain attempted/completed forwards and provider metadata;
  checkpoint/load failures persist an error receipt. Failed conditions remain
  outside efficacy estimates but inside cost accounting.
- P2: The single-sentence check accepted an unfinished suffix after a completed
  sentence. A terminal-punctuation requirement now rejects that case.
- P2: A missing fixed orthogonal recipe could raise or silently advance to a
  different available pair. The fixed identity is selected before filtering and
  an unavailable fixed choice becomes an explicit no-edit decision.

The pilot restricts its provider to the explicitly no-retry OpenAI client.
Time/disk limits are checked before a policy choice as well as before forwards.
No runtime apply/certification gate or intervention dose was increased.

## Evidence Boundary

All 45 historical outputs were re-evaluated with the corrected evaluator; legacy
score, role status, constraints, sentence status and joint success are unchanged.
The original live receipts, source archive and manifests remain unchanged locally.
The public bundle records original and public hashes and its restricted archive
scope. These code fixes have regression coverage, not a new live-model experiment.

Remaining research limitations: one model/site, one fixed natural boundary,
one-shot recipe selection, six related confirmation cases already at baseline
ceiling, a surface role heuristic, and no B1 prompt-help arm. The unchanged
outputs are a null result for this pilot, not a general impossibility result.
No independent review or merge approval is implied by passing tests or CI.

## Local Validation

The final suite passed **811 tests and nine subtests**, with 15 dependency
deprecation warnings, in 76.91 seconds. Offline report regeneration and all 27
published-file checksums passed. The staged publication audit covered 51 files
including all four historical archive members; no secret-like values or local
paths were found in the evidence bundle, and staged Markdown links resolve.
`git diff --cached --check` passed. CI status is reported on the PR, not inferred
from this local run.
