# Structural Rescue Pilot

Run status: `complete`.

Frozen manifest SHA-256: `af946393433bfd059e310e7123c867feebe350616418201f9f484e02f39a4ca2`.

## Paired Task Outcomes

| Split | Policy | Pairs | Eligible | Joint Success | Wins / Losses vs B0 | Output Changes | Selection Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| heldout_v1 | b0 | 10 | 9 | 10 | 0 / 0 | 0 | 0 |
| heldout_v1 | fixed_routed | 10 | 9 | 9 | 0 / 1 | 8 | 0 |
| heldout_v1 | controller | 10 | 9 | 10 | 0 / 0 | 8 | 0 |

Joint success requires lexical constraints, a single sentence and a passing role report.
Provider errors are excluded from paired efficacy estimates; noop is retained as a decision.
Unavailable-boundary cases remain in overall policy outcomes, not eligible intervention comparisons.

## Selector Pair: controller vs fixed_routed

| B0 outcome | Eligible pairs | Different choices | Reference only | Challenger only |
| --- | ---: | ---: | ---: | ---: |
| b0_pass | 9 | 3 | 0 | 1 |
| b0_fail | 0 | 0 | 0 | 0 |

Expected cases: 10; unavailable boundaries: 1; unpaired: 0.
A challenger-only win on a B0 pass avoids a regression; only a B0 fail can be rescued.

## Calibration Actuation

| Case | Component | Raw Winner Changed | Full Output Changed | Joint Success | Best Gap Closure |
| --- | --- | --- | --- | --- | ---: |

Best gap closure is the maximum among observed SOURCE structural tokens, not an ownership claim.

## Cost

Provider calls: 9. Shared setup forwards: 209.
Shared no-boundary fallback forwards: 0.
Calibration-matrix forwards: 0.
Provider token totals: `{"input_tokens": 37608, "output_tokens": 1380, "total_tokens": 38988}`.

| Policy | Actual Forward Attempts |
| --- | ---: |
| b0 | 125 |
| fixed_routed | 143 |
| controller | 140 |

Per-policy ceilings: 24 diagnostic forwards plus 64 continuation forwards. Shared natural-prefix reconstruction/alignment/observation and calibration matrix are separately charged.

## Scope

- one worker and one predeclared boundary per case.
- predeclared generated or related cases, not a population sample.
- role check is conservative surface analysis, not semantic ground truth.
- one-shot recipe selection, not a multi-step adaptive controller.
- restored diagnostic branches, not certified production applies.
- equal ceilings with separately reported actual consumption.
