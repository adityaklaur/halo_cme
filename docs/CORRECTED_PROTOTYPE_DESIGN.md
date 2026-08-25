# Corrected Prototype Design

This document supersedes the early roadmap in `design.txt` for the current project folder.

## Current Reality

- The project already has real Level-2 Aditya-L1 SWIS data, not only Level-1 data.
- The available SWIS variables are sufficient for a prototype: TH1/TH2 timestamps, energy centers, integrated flux, BLK proton/alpha moments, and MAG vectors.
- The archived dashboard and report show that a v1.0-style pipeline existed once, but the source modules and processed outputs were missing.
- A deep-learning model is not required for this urgent prototype. The scientifically safer prototype uses robust quiet-period calibration and a rule-based detector.
- The current implementation now uses the safer FINAL-style Aug 9-15 scientific interval by default, with dashboard rebuild/refresh controls, a custom UTC time-range selector, and a SWIS-only external Zenodo comparison dataset.

## One-Line Project Definition

TopoCross-SWIS is a reproducible scientific prototype that compares Aditya-L1 SWIS TH1 and TH2 particle spectra, converts their disagreement into OPDI metrics, and explains whether the solar-wind environment is normal, disturbed, or compatible with an ICME candidate.

This is a prototype for scientific demonstration, not an operational space-weather warning product.

## Prototype Scope

The working v1.0 prototype should prove these items:

1. Read TH1, TH2, BLK, and MAG data.
2. Synchronize TH1 and TH2 on a one-minute grid.
3. Regrid both spectra to a common 320-2100 eV log-energy axis.
4. Calculate OPDI using Jensen-Shannon, Hellinger, and Wasserstein distances.
5. Calibrate baseline behavior from the quiet pre-event window.
6. Detect persistent changes without using ground-truth labels as inputs.
7. Explain the state using OPDI, plasma, MAG, and transition evidence.
8. Rank candidate CMEs using the detected transition time and a transparent heuristic compatibility score.
9. Show everything in a Streamlit replay dashboard.
10. Let users refresh/rebuild processed data from the dashboard after adding local files.
11. Let users select an exact UTC time range for focused event inspection.
12. Let users evaluate additional processed SWIS/OPDI datasets without claiming full CME validation.

## Data Contract

SWIS TH1/TH2 CDF:

- `epoch_for_cdf_mod`
- `energy_center_mod`
- `integrated_flux_mod`

SWIS BLK CDF:

- `proton_density`
- `proton_bulk_speed`
- `proton_thermal`
- `alpha_density`
- `alpha_bulk_speed`
- `alpha_thermal`

MAG NetCDF:

- `time`
- `Bx_gse`, `By_gse`, `Bz_gse`
- `Bx_gsm`, `By_gsm`, `Bz_gsm`
- `Quality_flag_10s_data`

## Implemented Software Components

- `app.py`: Streamlit dashboard with replay, whole-event overview, validation, CME ranking, and Data Manager.
- `scripts/rebuild_processed.py`: rebuilds the Aug 2024 processed dataset from local raw CDF/NetCDF files.
- `scripts/inspect_cdf.py`: inspects CDF variable names, shapes, units, and fill values.
- `scripts/download_zenodo_swis.py`: downloads the public Zenodo SWIS L2 sample.
- `scripts/build_zenodo_swis_only.py`: builds a SWIS-only OPDI comparison dataset from the Zenodo files.
- `src/swis_august.py`: reads SWIS CDF files, regrids spectra, synchronizes TH1/TH2, and calculates OPDI.
- `src/mag_reader.py`: reads and cleans MAG L2 NetCDF files.
- `src/detector_august.py`: calibrates quiet baseline statistics and assigns detector states.
- `src/source_matcher.py`: ranks candidate CMEs with a transparent compatibility score.
- `src/dashboard_utils.py`: builds reusable Plotly figures for the dashboard.

## Edge Cases To Handle

- CDF fill values such as `-1e31`.
- Zero or negative flux values.
- Missing bins and incomplete spectra.
- TH1/TH2 cadence mismatch.
- Energy-grid variation per record.
- MAG 10-second cadence vs SWIS 5-second cadence.
- Missing MAG values during one-minute merges.
- Detector flicker from noisy thresholds.
- Raw transition threshold crossing should remain WATCH until a persistent change point confirms ALERT.
- Dashboard display when values are NaN.

## Scientific Limits

- The shock reference at 2024-08-10T12:50:00 is approximate.
- The dashboard timing offset is exploratory, not validated early-warning lead time.
- The project is a single-event prototype, not a generalized ICME detector.
- CME source ranking is a compatibility score, not probability or proof of causation.
- CME ranking uses the detector's transition time when available; catalog/literature candidates should still be shown separately from heuristic rank.
- Official extended mission data should come from ISRO/ISSDC PRADAN after login.
- The public Zenodo November 2023 sample is SWIS-only in this project, so it can test OPDI portability but cannot validate a full BLK/MAG/CME detector.

## Environment and Runbook

Use the project-local virtual environment, not another environment such as `shailVirt`.

Correct run sequence:

```bash
cd "Halo Aditya L2"
source .venv/bin/activate
pip install -r requirement.txt
streamlit run app.py
```

If someone activates a different Python environment, then `streamlit run app.py` may fail with:

```text
zsh: command not found: streamlit
```

That means Streamlit is not installed in that other environment. The fix is either:

```bash
source .venv/bin/activate
```

or:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

`requirement.txt` is the single dependency file used by this package.

## Dashboard Features For Demo

- Select an exact UTC range instead of only using a fixed replay window.
- Refresh loaded processed files from the sidebar.
- Rebuild the processed dataset from the running app after adding local raw data.
- Inspect local raw-file coverage in the Data Manager tab.
- Upload additional TH1/TH2/BLK/MAG raw files from the Data Manager tab into `data/raw/`.
- Open official PRADAN and public Zenodo data sources from the app.
- Upload a processed feature CSV for quick OPDI/state evaluation.
- Show the public Zenodo SWIS-only dataset as an external comparison dataset.

## Data Expansion Plan

For official extended mission coverage:

1. Download matching TH1, TH2, BLK, and MAG L2 files from ISRO/ISSDC PRADAN.
2. Upload them from the Data Manager tab, or place them manually in `data/raw/th1`, `data/raw/th2`, `data/raw/blk`, and `data/raw/mag`.
3. Update `processing.start_date` and `processing.end_date` in `config/prototype.yaml`.
4. Run `python3 scripts/rebuild_processed.py` or use the dashboard rebuild button.
5. Check Data Manager coverage and validation output before claiming scientific conclusions.

For public OPDI-only experimentation:

1. Run `python3 scripts/download_zenodo_swis.py`.
2. Run `python3 scripts/build_zenodo_swis_only.py`.
3. Use the Data Manager tab to confirm the external dataset is available.
4. Treat it as SWIS-only OPDI portability evidence, not CME detection validation.

## Build Order

1. Activate `.venv` and install dependencies from `requirement.txt`.
2. Run `python3 scripts/rebuild_processed.py`.
3. Confirm `data/processed/aug2024_features_1min.csv` has 10,080 rows for the current Aug 9-15 final scientific range.
4. Run `python3 -m pytest tests`.
5. Launch `streamlit run app.py`.
6. Use the dashboard replay around the configured shock reference for the demo.
7. Optionally run `python3 scripts/download_zenodo_swis.py` and `python3 scripts/build_zenodo_swis_only.py` to add a public SWIS-only OPDI comparison dataset.

## Presentation Narrative

The safest demo story is:

1. We use real Aditya-L1 Level-2 spacecraft data.
2. TH1 and TH2 observe different planes of the solar-wind ion distribution.
3. OPDI measures how much those two views disagree over time.
4. A quiet baseline is learned from the pre-event interval.
5. The detector flags persistent OPDI/plasma/MAG changes without using labels as inputs.
6. The dashboard explains why a state changed and links it to candidate CME sources.
7. Additional data can be added through PRADAN downloads and rebuilt from the app.
