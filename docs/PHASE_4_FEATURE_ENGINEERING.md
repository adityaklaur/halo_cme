# Phase 4 — Complete Feature Engineering

## Purpose

Phase 4 converts the Phase 3 minute-level ground-truth table into the complete scientific feature matrix required for later experiments. The feature builder preserves all Phase 2/3 provenance and labels, but no ground-truth label is used to calculate any Phase 4 feature.

## Implemented feature groups

### Conventional plasma and MAG features

The table retains the measured proton density, proton bulk speed, proton thermal speed, alpha/proton density ratio, magnetic-field magnitude, and GSE components Bx, By, and Bz.

It adds one-minute derivatives for bulk speed, density, and |B|, plus past-only rolling mean, median, and variance at 5, 15, and 60 minute windows for the principal conventional parameters.

### Compression indicators

Phase 4 adds density, magnetic-field, speed, and dynamic-pressure compression ratios relative to a past-only 15 minute median baseline. A joint compression index combines positive density, |B|, and speed compression.

### Cross-plane OPDI features

The existing Jensen-Shannon, Hellinger, and Wasserstein OPDI values are retained. Phase 4 adds derivatives for all three metrics, a canonical `d_opdi_dt` based on JS OPDI, 15 minute rolling OPDI mean/median/variance, a past-only robust OPDI anomaly score, and OPDI anomaly persistence.

### TH1/TH2 spectral-shape relationships

The registered normalized TH1 and TH2 spectra are used to calculate cosine similarity, spectral angle, spectral centroids and widths, centroid/width differences, peak energies, and the cross-plane log peak-energy ratio.

## Outputs

```text
outputs/phase4/
├── phase4_feature_dataset.csv
├── phase4_feature_dictionary.csv
├── phase4_event_summary.csv
└── phase4_report.json
```

The current Phase 4 build contains 17,201 records, 78 derived columns, and 16,602 rows ready for the exploratory Phase 5 ablation. The November orientation-control rows are preserved but excluded because their required Aditya-L1 source modalities are still incomplete.

## Build

```bash
python3 scripts/build_phase4_features.py
```

To rebuild Phases 2 through 5 in dependency order:

```bash
python3 scripts/build_phases2_to5.py
```

## Scientific guardrails

- Rolling baselines are past-only and are reset at each registered event window.
- The current minute is not included in its own rolling baseline.
- Phase 3 labels are carried through for later evaluation only.
- Missing November Aditya-L1 measurements are not replaced with OMNI or synthetic values.
- Spectral-shape features are calculated from the registered TH1/TH2 probability spectra, not from labels.
