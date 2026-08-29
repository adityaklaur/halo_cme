from pathlib import Path
import json

import pandas as pd

from src.phase5_experiment import build_phase5_experiment


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "phase5_experiment.yaml"
INPUT = ROOT / "outputs" / "phase4" / "phase4_feature_dataset.csv"
OUT = ROOT / "outputs" / "phase5"


def test_phase5_builder_runs_three_requested_modes_eventwise(tmp_path):
    report = build_phase5_experiment(ROOT, CONFIG, INPUT, tmp_path)
    predictions = pd.read_csv(tmp_path / "phase5_predictions.csv", low_memory=False)
    summary = pd.read_csv(tmp_path / "phase5_summary_metrics.csv")

    assert report["phase"] == 5
    assert set(summary["mode"]) == {"Conventional", "OPDI only", "Combined"}
    assert report["folds"] == 4
    assert all(not item["interval_leakage"] for item in report["leakage_checks"])
    assert "NOV2024_ORIENTATION_INTERVAL" not in report["independent_ready_intervals"]
    assert predictions.groupby(["mode", "fold"])["independent_interval_id"].nunique().eq(1).all()
    assert summary["detection_rate"].between(0, 1).all()


def test_phase5_delay_excludes_constant_positive_window_onset():
    delays = pd.read_csv(OUT / "phase5_detection_delays.csv")
    october = delays.loc[delays["event_id"] == "OCT2024_COMPLEX_ICME_01"]
    august = delays.loc[delays["event_id"] == "AUG2024_COMPLEX_ICME_01"]
    assert set(october["onset_reference_status"]) == {"WINDOW_START_NOT_EXACT_ONSET"}
    assert october["detection_delay_minutes"].isna().all()
    assert set(august["onset_reference_status"]) == {"INTERNAL_LABELED_BOUNDARY"}
    assert august["detection_delay_minutes"].notna().all()


def test_phase5_report_is_exploratory_not_confirmatory():
    report = json.loads((OUT / "phase5_report.json").read_text())
    assert report["status"] == "COMPLETE_EXPLORATORY_EVENTWISE_ABLATION"
    assert "EXPLORATORY" in report["evidence_status"]
    assert report["modes"]["OPDI only"] == [
        "js_opdi",
        "hellinger_opdi",
        "wasserstein_opdi",
        "d_opdi_dt",
    ]
