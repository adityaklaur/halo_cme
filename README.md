# TopoCross-SWIS Prototype

College prototype for detecting and explaining an August 2024 solar-wind disturbance using real Aditya-L1 ASPEX/SWIS TH1, TH2, BLK, and MAG data.

## Quick Start

Run these commands from Terminal:

```bash
cd "Halo Aditya L2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
python3 scripts/rebuild_processed.py
python3 scripts/build_static_dashboard.py
python3 scripts/final_qa.py
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

There is no neural-network model in this version. "Training" means baseline calibration from the quiet pre-event interval.

## Setup

Use the project-local `.venv` environment from the extracted project folder:

```bash
cd "Halo Aditya L2"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirement.txt
```

`requirement.txt` is the dependency file used by this package.

## Build Processed Data

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

This writes:

```text
outputs/final_qa_report.json
```

See also:

- `docs/FINAL_SUBMISSION_CHECKLIST.md`
- `docs/DATA_SOURCES_AND_LIMITS.md`

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
- Do not claim validated early warning or generalized ICME detection from this single-event prototype.
