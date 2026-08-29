from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml

from src.phase6_ml import build_phase6_ml


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_phase6_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    rows = []
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2025-01-01")
    for interval_index, interval in enumerate(["I1", "I2", "I3"]):
        for minute in range(80):
            target = int(25 <= minute < 55)
            rows.append(
                {
                    "event_id": f"E{interval_index + 1}",
                    "independent_interval_id": interval,
                    "timestamp": start + pd.Timedelta(days=interval_index, minutes=minute),
                    "research_label": "ICME/EJECTA" if target else "QUIET",
                    "icme_binary": target,
                    "eligible_for_exploratory_modeling": True,
                    "phase4_ready_for_exploratory_ablation": True,
                    "f_conventional": rng.normal(target * 2.0 + interval_index * 0.1, 0.5),
                    "f_opdi": rng.normal(target * 1.5, 0.5),
                    "f_constant": 1.0,
                }
            )
    data_path = tmp_path / "features.csv"
    pd.DataFrame(rows).to_csv(data_path, index=False)
    dictionary_path = tmp_path / "dictionary.csv"
    pd.DataFrame(
        [
            {"feature_group": "conventional_raw", "column": "f_conventional", "uses_ground_truth_label": False},
            {"feature_group": "cross_plane_opdi", "column": "f_opdi", "uses_ground_truth_label": False},
            {"feature_group": "conventional_raw", "column": "f_constant", "uses_ground_truth_label": False},
        ]
    ).to_csv(dictionary_path, index=False)
    config = {
        "schema_version": 1,
        "target_column": "icme_binary",
        "modeling_flag": "eligible_for_exploratory_modeling",
        "phase4_ready_flag": "phase4_ready_for_exploratory_ablation",
        "minimum_training_feature_fraction": 0.5,
        "prediction_threshold": 0.5,
        "persistence_minutes": 2,
        "random_state": 42,
        "primary_metric": "pr_auc",
        "models": {
            "Logistic Regression": {"type": "logistic_regression", "max_iter": 500, "class_weight": "balanced"},
            "Random Forest": {"type": "random_forest", "n_estimators": 20, "min_samples_leaf": 2, "class_weight": "balanced_subsample"},
            "Gradient Boosting": {"type": "hist_gradient_boosting", "max_iter": 20, "max_leaf_nodes": 7, "min_samples_leaf": 5, "class_weight": "balanced"},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return data_path, dictionary_path, config_path


def test_phase6_builder_runs_three_models_without_interval_leakage(tmp_path):
    data_path, dictionary_path, config_path = _synthetic_phase6_inputs(tmp_path)
    out = tmp_path / "out"
    report = build_phase6_ml(ROOT, config_path, data_path, dictionary_path, out)
    summary = pd.read_csv(out / "phase6_summary_metrics.csv")
    predictions = pd.read_csv(out / "phase6_predictions.csv")

    assert report["phase"] == 6
    assert report["folds"] == 3
    assert set(summary["model"]) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    assert all(not item["interval_leakage"] for item in report["leakage_checks"])
    assert predictions.groupby(["model", "fold"])["independent_interval_id"].nunique().eq(1).all()
    assert predictions["probability"].between(0, 1).all()


def test_phase6_feature_filter_is_train_only_and_drops_constant(tmp_path):
    data_path, dictionary_path, config_path = _synthetic_phase6_inputs(tmp_path)
    out = tmp_path / "out"
    build_phase6_ml(ROOT, config_path, data_path, dictionary_path, out)
    selection = pd.read_csv(out / "phase6_feature_selection.csv")
    constant = selection.loc[selection["feature"] == "f_constant"]
    assert not constant["selected"].astype(bool).any()
    assert set(constant["reason"]) == {"CONSTANT_OR_EMPTY"}


def test_actual_phase6_report_is_exploratory_and_uses_label_free_phase4_features():
    report = json.loads((ROOT / "outputs" / "phase6" / "phase6_report.json").read_text())
    features = pd.read_csv(ROOT / "outputs" / "phase6" / "phase6_feature_list.csv")
    assert report["status"] == "COMPLETE_EXPLORATORY_BASELINE_ML"
    assert report["evidence_status"] == "EXPLORATORY_NOT_RESEARCH_READY"
    assert "NOV2024_ORIENTATION_INTERVAL" not in report["independent_ready_intervals"]
    assert not features["uses_ground_truth_label"].astype(bool).any()
    assert "icme_binary" not in set(features["feature"])
