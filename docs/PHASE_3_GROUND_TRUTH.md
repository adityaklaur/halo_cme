# Phase 3 — Multi-Event Ground Truth

## Purpose

Phase 3 converts every registered Phase 2 scientific window into an explicit minute-level research-label stream. It consumes the Phase 2 event catalog and feature table; it does not create synthetic measurements or override Phase 2 modality checks.

## Scope

- 17,201 one-minute records
- 7 registered scientific windows
- 5 independent source intervals
- 8 research labels
- 6 Phase 3-ready events
- 1 blocked event: `NOV2024_ORIENTATION_CONTROL_01`

## Label policy

| Phase 2 event | Phase 3 policy | Research labels |
|---|---|---|
| August quiet control | Constant | `QUIET` |
| August complex ICME | Configured substructure | `QUIET`, `SHOCK`, `SHEATH`, `ICME/EJECTA` |
| August post-event control | Constant | `POST-ICME` |
| October complex ICME | Constant event-window label | `COMPLEX_ICME` |
| September quiet control | Constant | `QUIET` |
| November rotation control | Constant | `ORIENTATION-CONTROL` |
| March CIR/HSS | Constant | `CIR/HSS` |

The August shock minute remains an explicitly approximate internal benchmark. October literature references are recorded in the boundary register, but they are not converted into exact minute-level shock or sheath labels.

## Outputs

```text
outputs/phase3/
├── phase3_ground_truth_dataset.csv
├── phase3_label_counts.csv
├── phase3_event_register.csv
├── phase3_boundary_register.csv
└── phase3_report.json
```

The ground-truth dataset preserves all Phase 2 features and provenance, and adds:

- `research_label`
- label policy, confidence, status, and source
- Phase 2/3 readiness flags
- row-level modality completeness
- exploratory and confirmatory modeling eligibility
- ICME, shock, solar-transient, and event-positive binary targets

## Build and validation

Run both linked phases:

```bash
python3 scripts/build_phases2_to5.py
```

Run Phase 3 only:

```bash
python3 scripts/phase3_ground_truth.py
```

Strict research mode exits with status 2 while the November source remains blocked:

```bash
python3 scripts/phase3_ground_truth.py --strict-research-ready
```

The current build validates record counts against the Phase 2 manifest, unique event/timestamp keys, event coverage, complete label assignment, ordered August boundaries, and binary-target consistency.

## Scientific limitation

Phase 3 is structurally complete and valid, but the combined dataset is not yet research-ready because the November SWIS archive is corrupt and MAG does not cover the 25 November rotation. OMNI context is retained separately and cannot satisfy the missing Aditya-L1 modalities.
