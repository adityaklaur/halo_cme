# Phase 6 — Baseline Machine-Learning Models

Phase 6 moves beyond the Phase 5 transparent anomaly-score ablation and trains three standard supervised baselines on the completed Phase 4 feature matrix:

1. Logistic Regression
2. Random Forest
3. Histogram Gradient Boosting

## Scientific split

Evaluation uses leave-one-independent-interval-out folds. Minutes from the held-out interval never appear in model training. The currently usable intervals are August 2024, September 2024, October 2024, and March 2025. The November orientation-control interval remains excluded because Phase 2/3 mark its required Aditya-L1 SWIS/MAG modalities as incomplete.

## Feature policy

The model matrix is built only from columns declared `uses_ground_truth_label: false` by the Phase 4 feature dictionary. This includes conventional plasma/MAG measurements, their past-only derivatives and rolling statistics, compression features, OPDI features, OPDI temporal features, and TH1/TH2 spectral-shape relationships. Ground-truth state, research labels, eligibility fields, and target columns are never model inputs.

Within every outer fold, feature availability filtering, median imputation, scaling for Logistic Regression, class balancing, and fitting are learned using the training intervals only. A fixed probability threshold of 0.50 and three-minute persistence rule are applied consistently to all three baselines.

## Outputs

`outputs/phase6/` contains held-out predictions, fold and summary metrics, detection-delay rows, fold-wise feature selection, baseline feature importance, the exact candidate feature list, a JSON report, and three full-data exploratory `.joblib` model artifacts.

The saved full-data model artifacts are convenience artifacts for continued development. Scientific performance must be quoted from the held-out event-wise predictions, not from the fitted full-data models.

## Current status

Phase 6 is implemented but remains exploratory because only four independent source intervals are currently research-usable. Model ranking must not be presented as a generalized or deployment-ready result until more independently labeled ICME and non-ICME intervals are available.
