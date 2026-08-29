from pathlib import Path
import json

import pandas as pd

from src.phase4_features import build_phase4_features


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "phase4_features.yaml"
OUT = ROOT / "outputs" / "phase4"


def test_phase4_builder_creates_complete_feature_groups(tmp_path):
    report = build_phase4_features(ROOT, CONFIG, tmp_path)
    data = pd.read_csv(tmp_path / "phase4_feature_dataset.csv", low_memory=False)
    dictionary = pd.read_csv(tmp_path / "phase4_feature_dictionary.csv")

    assert report["phase"] == 4
    assert len(data) == 17201
    assert data[["event_id", "timestamp"]].duplicated().sum() == 0
    assert report["blocked_events_from_phase3"] == ["NOV2024_ORIENTATION_CONTROL_01"]
    required = {
        "d_proton_bulk_speed_dt",
        "d_proton_density_dt",
        "d_bmag_gse_dt",
        "d_opdi_dt",
        "opdi_rolling_mean",
        "opdi_rolling_median",
        "opdi_rolling_variance",
        "opdi_anomaly",
        "opdi_persistence",
        "density_compression_ratio",
        "bmag_compression_ratio",
        "joint_compression_index",
        "th1_th2_spectral_angle_rad",
        "crossplane_centroid_delta_ev",
        "crossplane_width_delta_ev",
        "crossplane_log_peak_energy_ratio",
    }
    assert required.issubset(data.columns)
    assert not dictionary["uses_ground_truth_label"].astype(bool).any()


def test_phase4_keeps_blocked_november_but_excludes_it_from_ablation():
    data = pd.read_csv(OUT / "phase4_feature_dataset.csv", low_memory=False)
    november = data.loc[data["event_id"] == "NOV2024_ORIENTATION_CONTROL_01"]
    assert len(november) == 461
    assert not november["phase4_ready_for_exploratory_ablation"].astype(bool).any()
    assert november[["js_opdi", "proton_density", "bmag_gse"]].isna().all().all()


def test_phase4_report_matches_current_dataset():
    report = json.loads((OUT / "phase4_report.json").read_text())
    assert report["records"] == 17201
    assert report["independent_intervals"] == 5
    assert report["derived_feature_columns"] >= 60
    assert report["exploratory_ablation_rows"] > 16000
