# Final Submission Checklist

Use this checklist before submitting or presenting the prototype.

## Must Pass

- If `.venv` is missing, run `python3 -m venv .venv`.
- Run `source .venv/bin/activate`.
- Run `pip install -r requirement.txt`.
- Run `python3 scripts/rebuild_processed.py`.
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

## Files To Show Evaluators

- `app.py` for the interactive Streamlit dashboard.
- `outputs/TopoCross_dashboard.html` for offline backup.
- `data/processed/pipeline_report.json` for reproducible result summary.
- `outputs/final_qa_report.json` for final QA evidence.
- `docs/CORRECTED_PROTOTYPE_DESIGN.md` for design explanation.
- `README.md` for setup and run commands.

## What To Say

- This prototype uses real Aditya-L1 Level-2 SWIS and MAG data.
- TH1 and TH2 observe different viewing planes of the ion distribution.
- OPDI measures how differently those two planes see the particle environment.
- A quiet baseline is calibrated before the event.
- The detector does not use ground-truth labels as inputs.
- The result is a research prototype, not an operational warning system.

## What Not To Claim

- Do not claim validated early warning.
- Do not claim a trained neural-network model.
- Do not claim generalized ICME detection across many events.
- Do not claim the CME ranking is a probability.
- Do not hide that the shock reference is approximate.
