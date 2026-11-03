# Structural Rescue Pilot

Run status: `complete`.

Frozen manifest SHA-256: `95f8b3715a0042653113069274073fe471d841a19fc2bf4ded934ccb65d61e58`.

## Paired Task Outcomes

| Split | Policy | Pairs | Eligible | Joint Success | Wins / Losses vs B0 | Output Changes | Selection Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rescue | b0 | 7 | 7 | 0 | 0 / 0 | 0 | 0 |
| rescue | fixed | 7 | 7 | 0 | 0 / 0 | 0 | 0 |
| rescue | fixed_readout | 7 | 7 | 4 | 4 / 0 | 4 | 0 |
| rescue | random | 7 | 7 | 1 | 1 / 0 | 1 | 0 |
| rescue | controller | 7 | 7 | 3 | 3 / 0 | 3 | 0 |

Joint success requires lexical constraints, a single sentence and a passing role report.
Provider errors are excluded from paired efficacy estimates; noop is retained as a decision.
Unavailable-boundary cases remain in overall policy outcomes, not eligible intervention comparisons.

## B0 Failures by Type (exploratory)

| Split | B0 failure | Cases | Joint successes by policy |
| --- | --- | ---: | --- |
| rescue | constraint | 5 | b0 0, fixed 0, fixed_readout 3, random 0, controller 2 |
| rescue | role_unknown | 2 | b0 0, fixed 0, fixed_readout 1, random 1, controller 1 |

Grouping by failure type was not part of any pre-registered selection rule.

## Calibration Actuation

| Case | Component | Raw Winner Changed | Full Output Changed | Joint Success | Best Gap Closure |
| --- | --- | --- | --- | --- | ---: |

Best gap closure is the maximum among observed SOURCE structural tokens, not an ownership claim.

## Cost

Provider calls: 7. Shared setup forwards: 67.
Shared no-boundary fallback forwards: 0.
Calibration-matrix forwards: 0.
Provider token totals: `{"input_tokens": 24873, "output_tokens": 1089, "total_tokens": 25962}`.

| Policy | Actual Forward Attempts |
| --- | ---: |
| b0 | 60 |
| fixed | 74 |
| fixed_readout | 95 |
| random | 75 |
| controller | 95 |

Per-policy ceilings: 24 diagnostic forwards plus 64 continuation forwards. Shared natural-prefix reconstruction/alignment/observation and calibration matrix are separately charged.

## Scope

- one worker and one predeclared boundary per case.
- six related confirmation variants, not independent population samples.
- role check is conservative surface analysis, not semantic ground truth.
- one-shot recipe selection, not a multi-step adaptive controller.
- restored diagnostic branches, not certified production applies.
- equal ceilings with separately reported actual consumption.
