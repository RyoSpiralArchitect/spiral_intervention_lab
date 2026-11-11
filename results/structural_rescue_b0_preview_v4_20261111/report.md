# Structural Rescue Pilot

Run status: `complete`.

Frozen manifest SHA-256: `f2a71c4b28f936abdd041b50c227d3421781a59c3550084f975d266c87a6c3b4`.

## Paired Task Outcomes

| Split | Policy | Pairs | Eligible | Joint Success | Wins / Losses vs B0 | Output Changes | Selection Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rescue_v4 | b0 | 3 | 3 | 0 | 0 / 0 | 0 | 0 |
| rescue_v4 | fixed_routed | 3 | 3 | 3 | 3 / 0 | 3 | 0 |
| rescue_v4 | controller | 3 | 3 | 3 | 3 / 0 | 3 | 0 |
| rescue_v4 | controller_b0_preview | 3 | 3 | 3 | 3 / 0 | 3 | 0 |

Joint success requires lexical constraints, a single sentence and a passing role report.
Provider errors are excluded from paired efficacy estimates; noop is retained as a decision.
Unavailable-boundary cases remain in overall policy outcomes, not eligible intervention comparisons.

## Selector Pair: controller vs fixed_routed

| B0 outcome | Eligible pairs | Different choices | Reference only | Challenger only |
| --- | ---: | ---: | ---: | ---: |
| b0_pass | 0 | 0 | 0 | 0 |
| b0_fail | 3 | 0 | 0 | 0 |

Expected cases: 3; unavailable boundaries: 0; unpaired: 0.
A challenger-only win on a B0 pass avoids a regression; only a B0 fail can be rescued.

## Selector Pair: controller_b0_preview vs controller

| B0 outcome | Eligible pairs | Different choices | Reference only | Challenger only |
| --- | ---: | ---: | ---: | ---: |
| b0_pass | 0 | 0 | 0 | 0 |
| b0_fail | 3 | 0 | 0 | 0 |

Expected cases: 3; unavailable boundaries: 0; unpaired: 0.
A challenger-only win on a B0 pass avoids a regression; only a B0 fail can be rescued.

## Recorded Constraint Inconsistencies

These rows report no constraint violations and a valid sentence but fail constraints.
Recorded labels are retained. Do not treat these as established rescue opportunities.

| Case | Policy | Reason |
| --- | --- | --- |
| gen_008 | b0 | constraints_failed_without_reported_violation |

## B0 Failures by Type (exploratory)

| Split | B0 failure | Cases | Joint successes by policy |
| --- | --- | ---: | --- |
| rescue_v4 | constraint | 3 | b0 0, fixed_routed 3, controller 3, controller_b0_preview 3 |

Grouping by failure type was not part of any pre-registered selection rule.

## Calibration Actuation

| Case | Component | Raw Winner Changed | Full Output Changed | Joint Success | Best Gap Closure |
| --- | --- | --- | --- | --- | ---: |

Best gap closure is the maximum among observed SOURCE structural tokens, not an ownership claim.

## Cost

Provider calls: 6. Shared setup forwards: 65.
Shared no-boundary fallback forwards: 0.
Calibration-matrix forwards: 0.
Provider token totals: `{"input_tokens": 23066, "output_tokens": 831, "total_tokens": 23897}`.

| Policy | Actual Forward Attempts |
| --- | ---: |
| b0 | 42 |
| fixed_routed | 57 |
| controller | 57 |
| controller_b0_preview | 57 |

Per-policy ceilings: 24 diagnostic forwards plus 64 continuation forwards. Shared natural-prefix reconstruction/alignment/observation and calibration matrix are separately charged.

## Scope

- one worker and one predeclared boundary per case.
- predeclared generated or related cases, not a population sample.
- role check is conservative surface analysis, not semantic ground truth.
- one-shot recipe selection, not a multi-step adaptive controller.
- restored diagnostic branches, not certified production applies.
- equal ceilings with separately reported actual consumption.
