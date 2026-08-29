# Final Submission Checklist

Use this checklist before submitting or presenting the prototype.

## Must Pass

- If `.venv` is missing, run `python3 -m venv .venv`.
- Run `source .venv/bin/activate`.
- Run `pip install -r requirements.txt`.
- Run `python3 scripts/rebuild_processed.py`.
- Run `python3 scripts/build_phases2_to5.py`.
- Run `python3 scripts/build_static_dashboard.py`.
- Run `python3 scripts/final_qa.py`.
- Run `streamlit run app.py`.
- In the app, click **Refresh loaded data** if it was already open.

## Expected Final Anchors

- Main processed interval: `2024-08-09 00:00:00` to `2024-08-15 23:59:00`.
- One-minute records: `10,080`.
- Primary transition nearest configured shock: `2024-08-10 12:19:00`.
- Offset from approximate `2024-08-10 12:50:00` shock reference: `-31 min`.
- Persistent transition episodes: `34`.
- Detector states should include `NORMAL`, `WATCH`, `ALERT`, and `ICME CANDIDATE`.
- Phase 2/3 registered records: `17,201`.
- Phase 3 event windows: `7` across `5` independent intervals.
- Phase 3 unknown labels: `0`.
- Phase 3-ready events: `6 of 7`; November remains blocked by missing Aditya modalities.
- Phase 4 feature records: `17,201` with `78` derived feature columns.
- Phase 4 exploratory-ablation rows: `16,602`.
- Phase 5 ready independent intervals: `4`.
- Phase 5 modes: `Conventional`, `OPDI only`, `Combined`.
- Current Phase 5 evidence status: `EXPLORATORY_MIXED_EVIDENCE`.

## Files To Show Evaluators

- `app.py` for the interactive Streamlit dashboard.
- `outputs/TopoCross_dashboard.html` for offline backup.
- `data/processed/pipeline_report.json` for reproducible result summary.
- `outputs/final_qa_report.json` for final QA evidence.
- `outputs/phase3/phase3_ground_truth_dataset.csv` for minute-level labels.
- `outputs/phase3/phase3_event_register.csv` and `phase3_boundary_register.csv` for traceability.
- `outputs/phase4/phase4_feature_dataset.csv` and `phase4_feature_dictionary.csv` for the complete feature set.
- `outputs/phase5/phase5_summary_metrics.csv`, `phase5_fold_metrics.csv`, and `phase5_report.json` for the central OPDI ablation.
- `docs/CORRECTED_PROTOTYPE_DESIGN.md` for design explanation.
- `README.md` for setup and run commands.

## What To Say

- This prototype uses real Aditya-L1 Level-2 SWIS and MAG data.
- TH1 and TH2 observe different viewing planes of the ion distribution.
- OPDI measures how differently those two planes see the particle environment.
- A quiet baseline is calibrated before the event.
- The detector does not use ground-truth labels as inputs.
- The result is a research prototype, not an operational warning system.
- Phase 3 is built from the Phase 2 multi-event registry rather than from an isolated single-event copy.
- Phase 4 creates conventional, cross-plane, rolling, compression, anomaly, persistence, and spectral-shape features without using the labels as inputs.
- Phase 5 compares Conventional vs OPDI-only vs Combined using held-out independent source intervals.
- The present Phase 5 result is mixed: Combined PR-AUC and valid detection timing improve, but F1 and false alarms do not.

## What Not To Claim

- Do not claim validated early warning.
- Do not claim a trained neural-network model.
- Do not claim generalized ICME detection across many events.
- Do not claim the CME ranking is a probability.
- Do not hide that the shock reference is approximate.
- Do not claim an exact October shock minute; the registered literature references are explicitly non-labeling boundaries.
- Do not claim full research readiness until the November SWIS and MAG inputs are repaired.
- Do not describe the current Phase 5 mixed result as proof that OPDI universally improves ICME detection.
- Do not quote an October detection delay because the October positive window has no exact labeled onset.
