from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def test_processed_dataset_contract_matches_final_scientific_profile():
    df = pd.read_csv(PROC / "aug2024_features_1min.csv", parse_dates=["timestamp"])
    assert len(df) == 10080
    assert df.timestamp.min() == pd.Timestamp("2024-08-09 00:00:00")
    assert df.timestamp.max() == pd.Timestamp("2024-08-15 23:59:00")
    required = {
        "js_opdi",
        "hellinger_opdi",
        "wasserstein_opdi",
        "proton_density",
        "proton_bulk_speed",
        "proton_thermal",
        "alpha_proton_ratio",
        "Bx_gse",
        "By_gse",
        "Bz_gse",
        "bmag_gse",
        "opdi_anomaly_score",
        "conventional_anomaly_score",
        "combined_anomaly_score",
        "transition_score",
        "transition_threshold_exceeded",
        "transition_component_js_opdi",
        "is_change_point",
        "state",
        "ground_truth_state",
    }
    assert required.issubset(df.columns)
    assert df["state"].notna().all()


def test_primary_transition_regression_anchor():
    df = pd.read_csv(PROC / "aug2024_features_1min.csv", parse_dates=["timestamp"])
    shock = pd.Timestamp("2024-08-10 12:50:00")
    change_points = df[df["is_change_point"].astype(bool)]
    row = change_points.iloc[np.argmin(np.abs((change_points.timestamp - shock).dt.total_seconds()))]
    assert row.timestamp == pd.Timestamp("2024-08-10 12:19:00")
    assert (row.timestamp - shock).total_seconds() / 60 == -31


def test_alert_starts_after_persistent_confirmation():
    df = pd.read_csv(PROC / "aug2024_features_1min.csv", parse_dates=["timestamp"])
    around = df.set_index("timestamp").loc["2024-08-10 12:16:00":"2024-08-10 12:21:00"]
    assert around.loc[pd.Timestamp("2024-08-10 12:17:00"), "transition_threshold_exceeded"]
    assert around.loc[pd.Timestamp("2024-08-10 12:17:00"), "state"] == "WATCH"
    assert around.loc[pd.Timestamp("2024-08-10 12:18:00"), "state"] == "WATCH"
    assert around.loc[pd.Timestamp("2024-08-10 12:19:00"), "is_change_point"]
    assert around.loc[pd.Timestamp("2024-08-10 12:19:00"), "state"] == "ALERT"


def test_archived_phase_js_medians_reproduced():
    df = pd.read_csv(PROC / "aug2024_features_1min.csv", parse_dates=["timestamp"])
    quiet = df[df.timestamp < pd.Timestamp("2024-08-10 12:50")]["js_opdi"].median()
    sheath = df[
        (df.timestamp >= pd.Timestamp("2024-08-10 12:50"))
        & (df.timestamp < pd.Timestamp("2024-08-11 12:00"))
    ]["js_opdi"].median()
    ejecta = df[
        (df.timestamp >= pd.Timestamp("2024-08-11 12:00"))
        & (df.timestamp < pd.Timestamp("2024-08-14 00:00"))
    ]["js_opdi"].median()
    # Aditya L2 uses cdflib + tolerant TH1/TH2 pairing, so the quiet median is
    # slightly different from FINAL's native exact-sync reader while preserving
    # the same phase-separation story.
    assert abs(quiet - 0.0226663897) < 1e-6
    assert abs(sheath - 0.0324937593) < 1e-6
    assert abs(ejecta - 0.0609232640) < 1e-6
    assert quiet < sheath < ejecta


def test_pipeline_report_and_static_dashboard_dependencies_present():
    report = json.loads((PROC / "pipeline_report.json").read_text())
    assert report["one_minute_records"] == 10080
    assert report["evaluation_against_configured_reference"]["nearest_change_to_reference_shock"] == "2024-08-10 12:19:00"
    assert report["cme_source_match_reference"] == "detected_transition"
    assert report["cme_source_match_reference_time"] == "2024-08-10 12:19:00"
    for path in [
        ROOT / "app.py",
        PROC / "aug2024_spectra_1min.npz",
        PROC / "state_statistical_tests.csv",
        PROC / "cme_candidate_ranking.csv",
        ROOT / "data/labels/event_boundaries.csv",
        ROOT / "scripts/build_static_dashboard.py",
    ]:
        assert path.exists(), path


def test_cme_candidate_table_includes_nasa_ccmc_aug7_candidate():
    candidates = pd.read_csv(ROOT / "data" / "cme_candidates" / "candidates.csv", parse_dates=["cme_time"])
    assert pd.Timestamp("2024-08-07 03:24:00") in set(candidates["cme_time"])
