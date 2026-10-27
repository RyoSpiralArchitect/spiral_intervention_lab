# Structural Rescue Pilot

Run status: `complete`.

Frozen manifest SHA-256: `81326474db5ae819ba997612019c50f77b52a7ad057fc4c4567d786bce572324`.

## Paired Task Outcomes

| Split | Policy | Pairs | Eligible | Joint Success | Wins / Losses vs B0 | Output Changes | Selection Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | b0 | 3 | not recorded | 2 | 0 / 0 | 0 | 0 |
| calibration | fixed | 3 | not recorded | 2 | 0 / 0 | 0 | 0 |
| calibration | random | 3 | not recorded | 2 | 0 / 0 | 0 | 0 |
| calibration | controller | 3 | not recorded | 2 | 0 / 0 | 0 | 0 |
| confirmation | b0 | 6 | not recorded | 6 | 0 / 0 | 0 | 0 |
| confirmation | fixed | 6 | not recorded | 6 | 0 / 0 | 0 | 0 |
| confirmation | random | 6 | not recorded | 6 | 0 / 0 | 0 | 0 |
| confirmation | controller | 6 | not recorded | 6 | 0 / 0 | 0 | 0 |

Joint success requires lexical constraints, a single sentence and a passing role report.
Provider errors are excluded from paired efficacy estimates; noop is retained as a decision.
Unavailable-boundary cases remain in overall policy outcomes, not eligible intervention comparisons.

## Calibration Actuation

| Case | Component | Raw Winner Changed | Full Output Changed | Joint Success | Best Gap Closure |
| --- | --- | --- | --- | --- | ---: |
| cal_delivery | common | False | False | True | 0.000928 |
| cal_delivery | common_parallel | False | False | True | 0.000928 |
| cal_delivery | common_orthogonal | False | False | True | 0.000000 |
| cal_review | common | False | False | False | 0.027778 |
| cal_review | common_parallel | False | False | False | 0.000000 |
| cal_review | common_orthogonal | False | False | False | 0.027778 |
| cal_workshop | common | False | False | True | 0.010753 |
| cal_workshop | common_parallel | False | False | True | 0.001086 |
| cal_workshop | common_orthogonal | False | False | True | 0.010753 |

Best gap closure is the maximum among observed SOURCE structural tokens, not an ownership claim.

## Cost

Provider calls: 9. Shared setup forwards: 85.
Shared no-boundary fallback forwards: 0.
Calibration-matrix forwards: 144.
Provider token totals: `{"input_tokens": 52849, "output_tokens": 1846, "total_tokens": 54695}`.

| Policy | Actual Forward Attempts |
| --- | ---: |
| b0 | 126 |
| fixed | 144 |
| random | 144 |
| controller | 142 |

Per-policy ceilings: 24 diagnostic forwards plus 64 continuation forwards. Shared natural-prefix reconstruction/alignment/observation and calibration matrix are separately charged.

## Scope

- one worker and one predeclared boundary per case.
- six related confirmation variants, not independent population samples.
- role check is conservative surface analysis, not semantic ground truth.
- one-shot recipe selection, not a multi-step adaptive controller.
- restored diagnostic branches, not certified production applies.
- equal ceilings with separately reported actual consumption.
