# Phase 5 — Central OPDI Hypothesis Test

## Scientific question

Does cross-plane information from SWIS TH1/TH2 add useful ICME-detection information beyond conventional solar-wind plasma and magnetic measurements?

Phase 5 implements the requested three-way ablation before Phase 6 machine learning:

1. **Conventional** — proton bulk speed, density, thermal speed, alpha/proton ratio, |B|, Bx, By, Bz.
2. **OPDI only** — JS OPDI, Hellinger OPDI, Wasserstein OPDI, and `d_opdi_dt`.
3. **Combined** — equal-weight combination of the conventional-group and OPDI-group anomaly scores.

## Evaluation design

The experiment uses leave-one-independent-interval-out evaluation. Individual minutes are never randomly divided between train and test when they originate from the same independent source interval.

For each fold, robust medians and MAD-based scales are estimated only from negative/control rows in the training intervals. The alert threshold is the configured 99th percentile of training-negative scores. A three-minute persistence requirement is then applied within each event window.

This is intentionally a transparent statistical detector, not the Phase 6 logistic-regression/random-forest/boosting stage.

## Current exploratory result

The current build evaluates four ready independent intervals; the November orientation control remains excluded because required Aditya-L1 data are absent.

| Mode | Precision | Recall | F1 | PR-AUC | Detection rate | False alarms/day | Median valid delay |
|---|---:|---:|---:|---:|---:|---:|---:|
| Conventional | 0.772 | 0.497 | 0.605 | 0.652 | 1.00 | 12.26 | 145 min |
| OPDI only | 0.742 | 0.169 | 0.276 | 0.602 | 1.00 | 8.02 | 86 min |
| Combined | 0.746 | 0.354 | 0.480 | 0.706 | 1.00 | 19.81 | 77 min |

The result is therefore classified as **exploratory mixed evidence**. Combined features improve PR-AUC by about 0.054 over Conventional and reduce the valid August detection delay, but the current thresholded F1 is lower and false-alarm rate is higher. This is not sufficient to claim that OPDI has already been validated as universally beneficial.

Detection delay is calculated only when a labeled positive boundary occurs inside the registered event window. The October constant complex-ICME window does not provide an exact onset, so it is excluded from delay statistics.

## Outputs

```text
outputs/phase5/
├── phase5_predictions.csv
├── phase5_fold_metrics.csv
├── phase5_summary_metrics.csv
├── phase5_detection_delays.csv
└── phase5_report.json
```

## Run

```bash
python3 scripts/run_phase5_experiment.py
```

## Interpretation guardrails

- Results are exploratory, not confirmatory.
- Event-wise/source-interval separation is mandatory.
- Threshold calibration uses training negatives only.
- The November interval is not imputed.
- Approximate/constant event windows are not treated as exact onset times for delay metrics.
- Detection timing is not called “early warning.”
- Phase 6 should test proper baseline ML models separately after this ablation.
