from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from src.scientific_dataset import build_phase2_dataset
from src.multievent_processor import read_omni_csv


ROOT = Path(__file__).resolve().parents[1]
SCI = ROOT / "data" / "scientific"


def test_phase2_builder_creates_traceable_event_dataset(tmp_path):
    manifest = build_phase2_dataset(ROOT, ROOT / "config" / "phase2_events.yaml", tmp_path)
    catalog = pd.read_csv(tmp_path / "phase2_event_catalog.csv", parse_dates=["start_utc", "end_utc"])
    features = pd.read_csv(tmp_path / "phase2_feature_table.csv", parse_dates=["timestamp"], low_memory=False)
    coverage = pd.read_csv(tmp_path / "phase2_modality_coverage.csv")

    assert len(catalog) == 7
    assert len(features) == 17201
    assert features["event_id"].notna().all()
    assert features[["event_id", "timestamp"]].duplicated().sum() == 0
    assert set(coverage["modality"]) == {"TH1", "TH2", "BLK", "MAG", "OMNI_REFERENCE"}
    required = coverage.loc[coverage["required_for_contract"].astype(bool)]
    assert required.loc[required["event_id"] != "NOV2024_ORIENTATION_CONTROL_01", "passes_threshold"].all()
    assert manifest["independent_intervals"] == 5
    assert manifest["unregistered_source_records"] == 20239
    assert manifest["registered_windows_pass_contract"] is False


def test_phase2_scientific_guardrails_prevent_false_completion_claim():
    manifest = json.loads((SCI / "phase2_manifest.json").read_text())
    catalog = pd.read_csv(SCI / "phase2_event_catalog.csv")

    assert manifest["research_ready"] is False
    assert manifest["status"] == "MULTI_EVENT_ASSEMBLED_WITH_DATA_GAPS"
    assert int(catalog["independent_interval_id"].nunique()) == 5
    assert int(catalog["negative_control"].astype(bool).sum()) == 2
    assert {"CIR_SIR", "ORIENTATION_CONTROL"}.issubset(set(catalog["event_class"]))
    non_ready = catalog.loc[~catalog["phase2_ready"].astype(bool), "event_id"].tolist()
    assert non_ready == ["NOV2024_ORIENTATION_CONTROL_01"]


def test_online_omni_csvs_are_continuous_and_traceable():
    expected = {"oct2024": 8640, "mar2025": 7200, "sep2024": 7200, "nov2024": 4320}
    for source_id, records in expected.items():
        frame = read_omni_csv(ROOT / "data" / "external" / "omni" / f"{source_id}_omni_1min.csv")
        assert len(frame) == records
        assert not frame["timestamp"].duplicated().any()
        assert (frame["timestamp"].diff().dropna() == pd.Timedelta(minutes=1)).all()
        assert set(frame["omni_source"]) == {"NASA_CDAWEB_OMNI_HRO_1MIN"}


def test_omni_never_replaces_missing_aditya_measurements():
    frame = pd.read_csv(ROOT / "data" / "processed" / "events" / "nov2024_features_1min.csv")
    assert frame["omni_complete"].astype(bool).any()
    assert not frame["usable_opdi"].astype(bool).any()
    assert frame[["proton_density", "proton_bulk_speed", "proton_thermal"]].isna().all().all()
    assert frame[["Bx_gse", "By_gse", "Bz_gse"]].loc[1440:].isna().all().all()
