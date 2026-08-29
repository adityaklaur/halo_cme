# Phase 2 Scientific Dataset Implementation

## What is implemented

Phase 2 adds a reusable multi-event dataset layer on top of the August 2024 detector prototype. It provides:

- a YAML event registry;
- event-window validation;
- explicit positive, control and clean-negative flags;
- independent-interval identifiers to prevent pseudo-replication;
- TH1, TH2, BLK and MAG completeness checks;
- source-spectrum and CME-candidate checks;
- a row-level event feature table;
- a source-gap queue for missing or partial mission products;
- machine-readable research-readiness guardrails;
- a Scientific Dataset tab in the Streamlit dashboard.

## Build command

```bash
python3 scripts/build_phase2_dataset.py
```

Use strict mode in automated research workflows:

```bash
python3 scripts/build_phase2_dataset.py --strict-research-ready
```

Strict mode intentionally exits with status 2 while a required registered event fails its synchronized Aditya-L1 modality contract.

## Registry

Edit `config/phase2_events.yaml` to add an independently sourced interval. Each event must include:

- unique `event_id`;
- `independent_interval_id`;
- event class and sample role;
- UTC start and end;
- processed feature CSV;
- synchronized TH1/TH2 spectra NPZ;
- label status and source;
- CME candidate file when relevant.

The builder rejects empty windows, duplicate event IDs, invalid time ranges, unsupported classes, duplicate timestamps and missing scientific feature columns.

## Generated outputs

The build writes the following files under `data/scientific/`:

- `phase2_event_catalog.csv` - one row per registered event window;
- `phase2_feature_table.csv` - one-minute features with event provenance;
- `phase2_modality_coverage.csv` - per-event TH1/TH2/BLK/MAG completeness plus optional OMNI reference coverage;
- `phase2_acquisition_queue.csv` - missing, partial or blocked source products;
- `phase2_manifest.json` - readiness status, counts and guardrails.

## Current scientific status

The package contains seven scientific windows from five independent intervals: the August 2024 ICME source, October 2024 ICME, September 2024 quiet control, November 2024 spacecraft-orientation control, and March 2025 CIR/HSS. The August subwindows retain one shared independent-interval ID.

Six registered windows pass the 90% TH1/TH2/BLK/MAG contract. The November orientation window remains non-ready because its uploaded SWIS archive is corrupt and its MAG archive contains only 24 November. NASA OMNI data are included for all four new sources as optional near-Earth context and are never substituted for missing Aditya-L1 observations.

The October 12 SWIS product is unavailable on PRADAN and remains an explicit gap. The September quiet registry uses the complete 23 September 15:00-23:59 UTC subwindow because the supplied 24 September SWIS and MAG products are partial.

No synthetic event, boundary, instrument measurement or CME association was added.
