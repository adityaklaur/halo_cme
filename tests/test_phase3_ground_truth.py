from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.ground_truth import build_phase3_ground_truth


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "phase3_labels.yaml"
OUT = ROOT / "outputs" / "phase3"


def test_phase3_builder_consumes_complete_phase2_registry(tmp_path):
    report = build_phase3_ground_truth(ROOT, CONFIG, tmp_path)
    data = pd.read_csv(tmp_path / "phase3_ground_truth_dataset.csv", parse_dates=["timestamp"], low_memory=False)
    events = pd.read_csv(tmp_path / "phase3_event_register.csv")

    assert report["phase3_valid"] is True
    assert report["research_ready"] is False
    assert len(data) == 17201
    assert data["event_id"].nunique() == 7
    assert data["independent_interval_id"].nunique() == 5
    assert data[["event_id", "timestamp"]].duplicated().sum() == 0
    assert (data["research_label"] == "UNKNOWN").sum() == 0
    assert events.loc[~events["phase3_ready"].astype(bool), "event_id"].tolist() == [
        "NOV2024_ORIENTATION_CONTROL_01"
    ]


def test_phase3_expected_multi_event_label_counts():
    counts = pd.read_csv(OUT / "phase3_label_counts.csv").set_index("research_label")["n"].to_dict()
    assert counts == {
        "QUIET": 2030,
        "SHOCK": 1,
        "SHEATH": 1389,
        "ICME/EJECTA": 3600,
        "COMPLEX_ICME": 2520,
        "POST-ICME": 2880,
        "CIR/HSS": 4320,
        "ORIENTATION-CONTROL": 461,
    }


def test_august_substructure_and_october_uncertainty_are_preserved():
    data = pd.read_csv(OUT / "phase3_ground_truth_dataset.csv", parse_dates=["timestamp"], low_memory=False)
    august = data.loc[data["event_id"] == "AUG2024_COMPLEX_ICME_01"].set_index("timestamp")
    assert august.loc[pd.Timestamp("2024-08-10 12:49:00"), "research_label"] == "QUIET"
    assert august.loc[pd.Timestamp("2024-08-10 12:50:00"), "research_label"] == "SHOCK"
    assert august.loc[pd.Timestamp("2024-08-10 12:51:00"), "research_label"] == "SHEATH"
    assert august.loc[pd.Timestamp("2024-08-11 12:00:00"), "research_label"] == "ICME/EJECTA"

    october = data.loc[data["event_id"] == "OCT2024_COMPLEX_ICME_01"]
    assert set(october["research_label"]) == {"COMPLEX_ICME"}
    boundaries = pd.read_csv(OUT / "phase3_boundary_register.csv")
    references = boundaries.loc[
        (boundaries["event_id"] == "OCT2024_COMPLEX_ICME_01")
        & boundaries["boundary_type"].isin(["SHOCK_AFTER_LOWER_BOUND", "SHEATH_MC_BOUNDARY_REFERENCE"])
    ]
    assert not references["used_for_minute_labeling"].astype(bool).any()


def test_november_label_does_not_override_phase2_data_block():
    data = pd.read_csv(OUT / "phase3_ground_truth_dataset.csv", low_memory=False)
    november = data.loc[data["event_id"] == "NOV2024_ORIENTATION_CONTROL_01"]
    assert set(november["research_label"]) == {"ORIENTATION-CONTROL"}
    assert not november["phase3_ready"].astype(bool).any()
    assert not november["eligible_for_exploratory_modeling"].astype(bool).any()
    assert november[["proton_density", "proton_bulk_speed", "proton_thermal"]].isna().all().all()


def test_phase3_policy_registry_must_match_phase2_catalog(tmp_path):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["events"] = config["events"][:-1]
    invalid = tmp_path / "invalid_phase3.yaml"
    invalid.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="policies do not match Phase 2 catalog"):
        build_phase3_ground_truth(ROOT, invalid, tmp_path / "out")
