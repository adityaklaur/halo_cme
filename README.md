# TopoCross-SWIS Phases 2–6 Research Prototype

Research prototype for detecting, organizing, and labeling solar-wind disturbances across five independent source intervals using real Aditya-L1 ASPEX/SWIS TH1, TH2, BLK, MAG data and traceable NASA OMNI context.

## Quick Start

Run these commands from Terminal:

```bash
cd "Halo Aditya L2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/build_phases2_to6.py
streamlit run app.py
```

If `streamlit: command not found` appears, the wrong Python environment is active. Run `source .venv/bin/activate` again, then retry `streamlit run app.py`.

## What This Prototype Does

- Reads Level-2 SWIS CDF files from TH1, TH2, and BLK.
- Synchronizes TH1 and TH2 spectra on a common log-energy grid.
- Calculates OPDI metrics: Jensen-Shannon, Hellinger, and Wasserstein divergence.
- Merges one-minute MAG context.
- Calibrates a robust quiet-period baseline from 2024-08-09 to 2024-08-10 08:30 UTC.
- Runs a rule-based state machine: NORMAL, WATCH, ALERT, ICME CANDIDATE.
- Produces a Streamlit dashboard with replay, whole-event plots, validation, CME candidate ranking, refresh/rebuild controls, and custom time-range selection.
- Keeps raw transition buildup in WATCH and enters ALERT only after a persistent change point is confirmed.
- Ranks CME candidates against the detected transition time when available, not only the legacy configured benchmark.
- Registers positive and control windows in a traceable Phase 2 event catalog.
- Checks per-event TH1, TH2, BLK and MAG completeness.
- Preserves independent-interval identity so event-wise validation cannot mistake windows from one interval for separate events.
- Exposes Phase 2 readiness, acquisition gaps and downloadable scientific tables in the dashboard.
- Includes independent August/October ICME, September quiet, November orientation-control, and March CIR/HSS source intervals.
- Keeps NASA OMNI one-minute values in separately named reference columns; they never replace missing Aditya-L1 measurements.
- Applies a registry-driven Phase 3 ground-truth policy to every Phase 2 event window.
- Produces minute-level research labels, binary targets, event/boundary registers, confidence fields, and modeling-eligibility flags.
- Prevents approximate October reference times from being presented as exact shock labels.
- Builds the complete Phase 4 feature matrix: conventional derivatives/rolling statistics/compression indicators, OPDI derivatives/rolling anomaly/persistence, and TH1/TH2 spectral-shape relationships.
- Runs the Phase 5 event-wise Conventional vs OPDI-only vs Combined ablation without random minute-level leakage.
- Reports detection rate, false alarms, precision, recall, F1, PR-AUC, and valid detection delay with explicit onset-quality guardrails.
- Trains and compares Phase 6 Logistic Regression, Random Forest, and Histogram Gradient Boosting baselines using leave-one-independent-interval-out evaluation.
- Saves held-out predictions, model metrics, baseline feature importance/selection diagnostics, and full-data exploratory `.joblib` artifacts.

There is no deep-learning model in this version. Phase 6 is the deliberately simple supervised baseline stage before the more novel cross-plane model in Phase 7.

## Phase 2 Scientific Dataset

Build or refresh the Phase 2 outputs with:

```bash
python3 scripts/build_phase2_dataset.py
```

The event registry is `config/phase2_events.yaml`. Generated tables and the machine-readable readiness manifest are written to `data/scientific/`.

The packaged build contains seven registered windows from five independent source intervals. Six windows pass the synchronized Aditya-L1 modality contract. The November 25 orientation control is deliberately non-ready because the uploaded SWIS ZIP is corrupt and MAG for that date is absent. See `docs/PHASE2_SCIENTIFIC_DATASET.md`.

To regenerate the multi-event processed sources after restoring raw files under a per-event `swis/` and `mag/` layout, run:

```bash
python3 scripts/build_multievent_sources.py --raw-root /path/to/per-event-raw
```

## Phase 3 Ground Truth

Build Phase 3 alone with:

```bash
python3 scripts/phase3_ground_truth.py
```

The labeling policies are stored in `config/phase3_labels.yaml`. Outputs are written to `outputs/phase3/`, including the complete 17,201-row ground-truth table, label counts, event register, boundary register, and validation report.

Phase 3 labels seven windows from five independent intervals. Six events are ready for exploratory modeling. The November orientation-control label is retained but its rows are excluded from modeling until valid Aditya-L1 SWIS and MAG measurements cover 25 November. See `docs/PHASE_3_GROUND_TRUTH.md`.

## Phase 4 Feature Engineering

Build the complete feature matrix with:

```bash
python3 scripts/build_phase4_features.py
```

Outputs are written to `outputs/phase4/`. See `docs/PHASE_4_FEATURE_ENGINEERING.md`.

## Phase 5 OPDI Ablation

Run the central scientific comparison with:

```bash
python3 scripts/run_phase5_experiment.py
```

The experiment compares Conventional, OPDI-only, and Combined modes using leave-one-independent-interval-out evaluation. Current evidence is exploratory and mixed: Combined improves PR-AUC and valid detection timing, but not thresholded F1/false alarms. Outputs are in `outputs/phase5/`. See `docs/PHASE_5_OPDI_ABLATION.md`.

## Phase 6 Baseline Machine Learning

Run the three baseline models with:

```bash
python3 scripts/run_phase6_ml.py
```

Phase 6 trains Logistic Regression, Random Forest, and Histogram Gradient Boosting on label-free Phase 4 features. Evaluation is leave-one-independent-interval-out; preprocessing and feature filtering are fitted only on each training fold. Current results remain exploratory because only four independent intervals are research-usable. Outputs are in `outputs/phase6/`, including saved `.joblib` artifacts. See `docs/PHASE_6_BASELINE_ML.md`.

## Setup

Use the project-local `.venv` environment from the extracted project folder:

```bash
cd "Halo Aditya L2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is the dependency file used by this package.

## Build Processed Data

The packaged processed August outputs are ready to use. Run a full rebuild only after supplying matching raw TH1, TH2, BLK and MAG files for every configured date.

```bash
python3 scripts/rebuild_processed.py
```

The script can read the current extracted folders directly:

- `tha1/`
- `tha2/`
- `swis_BLK/`
- `mag_2026Aug23T210145602/`

It also supports a future normalized layout under `data/raw/th1`, `data/raw/th2`, `data/raw/blk`, and `data/raw/mag`.

The processed date range is configured in `config/prototype.yaml` under `processing.start_date` and `processing.end_date`.
The final scientific profile uses 2024-08-09 to 2024-08-15, matching the safer FINAL regression baseline. Aug 8 can still be added manually as a SWIS-only exploratory day by changing the config and enabling `allow_missing_mag`.

## Run Dashboard

```bash
source .venv/bin/activate
streamlit run app.py
```

Dashboard features:

- Refresh loaded processed files from the sidebar.
- Rebuild processed files from the running app after adding local raw data.
- Select either a replay window or an exact custom UTC time range.
- Use the Data Manager tab to inspect local raw-file coverage, upload more TH1/TH2/BLK/MAG files, open official data links, and evaluate uploaded processed CSV files.

## Add More Data From The UI

In the Streamlit app, open **Data Manager** and upload:

- TH1 SWIS `.cdf` files
- TH2 SWIS `.cdf` files
- BLK plasma `.cdf` files
- MAG L2 `.nc` files

The app saves these files into `data/raw/`, which is ignored by GitHub. After uploading, update `processing.start_date` and `processing.end_date` in `config/prototype.yaml` if the new dates are outside the current range, then click **Rebuild now from Data Manager**.

## Static Backup Dashboard

Build an offline HTML dashboard with:

```bash
python3 scripts/build_static_dashboard.py
```

Then open:

```text
outputs/TopoCross_dashboard.html
```

## Useful Checks

Inspect a CDF file:

```bash
python3 scripts/inspect_cdf.py tha1/AL1_ASW91_L2_TH1_20240810_UNP_9999_999999_V03.cdf
```

Run OPDI unit tests:

```bash
python3 -m pytest tests
```

The regression tests expect the FINAL-style scientific profile: `10,080` rows, primary transition at `2024-08-10 12:19:00`, and `-31 min` offset from the approximate shock benchmark.

Run the final submission QA:

```bash
python3 scripts/final_qa.py
```

Run the focused Phase 4–6 / saved-model QA:

```bash
python3 scripts/phase6_qa.py
```

This writes:

```text
outputs/final_qa_report.json
```

See also:

- `docs/PHASE_4_FEATURE_ENGINEERING.md`
- `docs/PHASE_5_OPDI_ABLATION.md`
- `docs/PHASE_6_BASELINE_ML.md`

- `docs/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/DATA_SOURCES_AND_LIMITS.md`
- `docs/PHASE2_SCIENTIFIC_DATASET.md`
- `docs/PHASE_3_GROUND_TRUTH.md`

## Additional Data

Official Aditya-L1 science data should be downloaded from ISRO/ISSDC PRADAN:

- https://pradan.issdc.gov.in/al1
- https://pradan1.issdc.gov.in/al1

These portals may require registration/login and support bulk download.

A public Zenodo SWIS Level-2 sample is also supported for SWIS-only OPDI experiments:

```bash
python3 scripts/download_zenodo_swis.py
python3 scripts/build_zenodo_swis_only.py
```

This creates `data/external/zenodo_swis_20231106_12/processed/zenodo_swis_only_features_1min.csv`.
That dataset is useful for showing OPDI portability, but it does not include the BLK/MAG context needed for full CME validation.

## Scientific Guardrails

- The configured shock reference is approximate.
- Detector calculations do not use ground-truth labels.
- OPDI separation statistics are exploratory for one event interval.
- CME source compatibility is a heuristic ranking, not a calibrated probability.
- Multiple August windows count as one independent interval.
- October 12 is an explicit unpublished SWIS gap; partial September dates are not used in the registered quiet-control core.
- The November orientation control does not pass until valid Aditya-L1 SWIS and MAG files cover 25 November.
- Exact Phase 3 shock/sheath/ejecta labels are used only for the configured August event.
- Phase 4 features never use ground-truth labels as inputs.
- Phase 5 holds out complete independent source intervals and never randomly splits individual minutes.
- Phase 5 detection delay is only calculated for event windows containing an internal labeled onset.
- Current Phase 5 results are exploratory mixed evidence, not a generalized OPDI validation claim.
- Phase 6 uses complete held-out source intervals, train-only preprocessing, and label-free Phase 4 features; its ranking is exploratory until more independent events are available.
- October remains a `COMPLEX_ICME` window-level label because the literature does not provide an exact Aditya-L1 shock minute.
- Do not claim generalized detector performance while any required event fails the modality contract.
