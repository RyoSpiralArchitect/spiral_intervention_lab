# Structural Rescue Pilot

Run status: `complete`.

Frozen manifest SHA-256: `92f1138a7015ed0cbbed701340193d78e6e39df9aa9ce0a561248e7d843f4e99`.

## Paired Task Outcomes

| Split | Policy | Pairs | Eligible | Joint Success | Wins / Losses vs B0 | Output Changes | Selection Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibration | b0 | 3 | 3 | 2 | 0 / 0 | 0 | 0 |
| calibration | fixed | 3 | 3 | 2 | 0 / 0 | 0 | 0 |
| calibration | fixed_readout | 3 | 3 | 3 | 1 / 0 | 3 | 0 |
| calibration | random | 3 | 3 | 2 | 0 / 0 | 0 | 0 |
| calibration | controller | 3 | 3 | 2 | 0 / 0 | 2 | 0 |
| confirmation | b0 | 6 | 6 | 6 | 0 / 0 | 0 | 0 |
| confirmation | fixed | 6 | 6 | 6 | 0 / 0 | 0 | 0 |
| confirmation | fixed_readout | 6 | 6 | 6 | 0 / 0 | 3 | 0 |
| confirmation | random | 6 | 6 | 6 | 0 / 0 | 0 | 0 |
| confirmation | controller | 6 | 6 | 6 | 0 / 0 | 3 | 0 |

Joint success requires lexical constraints, a single sentence and a passing role report.
Provider errors are excluded from paired efficacy estimates; noop is retained as a decision.
Unavailable-boundary cases remain in overall policy outcomes, not eligible intervention comparisons.

## Calibration Actuation

| Case | Component | Raw Winner Changed | Full Output Changed | Joint Success | Best Gap Closure |
| --- | --- | --- | --- | --- | ---: |
| cal_delivery | patch common ','+' while' | False | False | True | 0.032520 |
| cal_delivery | patch common_parallel ','+' while' | False | False | True | 0.006494 |
| cal_delivery | patch common_orthogonal ','+' while' | False | False | True | 0.073171 |
| cal_delivery | patch common ','+'.\n' | False | False | True | 0.016260 |
| cal_delivery | patch common_parallel ','+'.\n' | False | False | True | 0.006494 |
| cal_delivery | patch common_orthogonal ','+'.\n' | False | False | True | 0.056911 |
| cal_delivery | patch common ' while'+'.\n' | False | False | True | 0.016260 |
| cal_delivery | patch common_parallel ' while'+'.\n' | False | False | True | 0.006494 |
| cal_delivery | patch common_orthogonal ' while'+'.\n' | False | False | True | 0.065041 |
| cal_delivery | readout readout ',' | True | True | True | 1.008130 |
| cal_delivery | readout readout ' while' | False | False | True | 0.960993 |
| cal_delivery | readout readout '.\n' | False | False | True | 0.428571 |
| cal_review | patch common ','+' while' | False | False | False | 0.097222 |
| cal_review | patch common_parallel ','+' while' | False | False | False | 0.013171 |
| cal_review | patch common_orthogonal ','+' while' | False | False | False | 0.125000 |
| cal_review | patch common ','+'.\n' | False | False | False | 0.069444 |
| cal_review | patch common_parallel ','+'.\n' | False | False | False | 0.012158 |
| cal_review | patch common_orthogonal ','+'.\n' | False | False | False | 0.083333 |
| cal_review | patch common ' while'+'.\n' | False | False | False | 0.069444 |
| cal_review | patch common_parallel ' while'+'.\n' | False | False | False | 0.012158 |
| cal_review | patch common_orthogonal ' while'+'.\n' | False | False | False | 0.097222 |
| cal_review | readout readout ',' | True | True | True | 1.500000 |
| cal_review | readout readout ' while' | False | False | False | 0.977778 |
| cal_review | readout readout '.\n' | False | False | False | 0.527778 |
| cal_workshop | patch common ','+' while' | False | False | True | 0.086022 |
| cal_workshop | patch common_parallel ','+' while' | False | False | True | 0.004343 |
| cal_workshop | patch common_orthogonal ','+' while' | False | False | True | 0.118280 |
| cal_workshop | patch common ','+'.\n' | False | False | True | 0.053763 |
| cal_workshop | patch common_parallel ','+'.\n' | False | False | True | 0.004343 |
| cal_workshop | patch common_orthogonal ','+'.\n' | False | False | True | 0.075269 |
| cal_workshop | patch common ' while'+'.\n' | False | False | True | 0.075269 |
| cal_workshop | patch common_parallel ' while'+'.\n' | False | False | True | 0.004343 |
| cal_workshop | patch common_orthogonal ' while'+'.\n' | False | False | True | 0.107527 |
| cal_workshop | readout readout ',' | True | True | True | 1.172043 |
| cal_workshop | readout readout ' while' | True | True | True | 1.098291 |
| cal_workshop | readout readout '.\n' | False | False | True | 0.435396 |

Best gap closure is the maximum among observed SOURCE structural tokens, not an ownership claim.

## Cost

Provider calls: 9. Shared setup forwards: 85.
Shared no-boundary fallback forwards: 0.
Calibration-matrix forwards: 577.
Provider token totals: `{"input_tokens": 32020, "output_tokens": 1675, "total_tokens": 33695}`.

| Policy | Actual Forward Attempts |
| --- | ---: |
| b0 | 126 |
| fixed | 144 |
| fixed_readout | 146 |
| random | 144 |
| controller | 143 |

Per-policy ceilings: 24 diagnostic forwards plus 64 continuation forwards. Shared natural-prefix reconstruction/alignment/observation and calibration matrix are separately charged.

## Scope

- one worker and one predeclared boundary per case.
- six related confirmation variants, not independent population samples.
- role check is conservative surface analysis, not semantic ground truth.
- one-shot recipe selection, not a multi-step adaptive controller.
- restored diagnostic branches, not certified production applies.
- equal ceilings with separately reported actual consumption.
