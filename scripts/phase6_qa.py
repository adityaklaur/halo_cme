#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "phase6" / "phase6_qa_report.json"


def main() -> None:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    phase6 = ROOT / "outputs" / "phase6"
    required = [
        "phase6_predictions.csv",
        "phase6_fold_metrics.csv",
        "phase6_summary_metrics.csv",
        "phase6_detection_delays.csv",
        "phase6_feature_importance.csv",
        "phase6_feature_selection.csv",
        "phase6_feature_list.csv",
        "phase6_report.json",
    ]
    add("all_phase6_outputs_exist", all((phase6 / name).exists() for name in required), ", ".join(required))

    report = json.loads((phase6 / "phase6_report.json").read_text())
    summary = pd.read_csv(phase6 / "phase6_summary_metrics.csv")
    predictions = pd.read_csv(phase6 / "phase6_predictions.csv")
    features = pd.read_csv(phase6 / "phase6_feature_list.csv")
    add("three_required_models", set(summary["model"]) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}, str(summary["model"].tolist()))
    add("four_eventwise_folds", report.get("folds") == 4, str(report.get("folds")))
    add("no_interval_leakage", all(not x.get("interval_leakage") for x in report.get("leakage_checks", [])), str(report.get("leakage_checks")))
    add("november_excluded", "NOV2024_ORIENTATION_INTERVAL" not in report.get("independent_ready_intervals", []), str(report.get("independent_ready_intervals")))
    add("label_free_features", not features["uses_ground_truth_label"].astype(bool).any() and "icme_binary" not in set(features["feature"]), f"features={len(features)}")
    add("probabilities_valid", predictions["probability"].between(0, 1).all(), f"min={predictions['probability'].min()}, max={predictions['probability'].max()}")
    add("exploratory_guardrail", report.get("status") == "COMPLETE_EXPLORATORY_BASELINE_ML" and report.get("evidence_status") == "EXPLORATORY_NOT_RESEARCH_READY", f"{report.get('status')} / {report.get('evidence_status')}")

    model_dir = phase6 / "models"
    artifacts = sorted(model_dir.glob("*.joblib"))
    add("three_model_artifacts", len(artifacts) == 3, str([p.name for p in artifacts]))
    artifact_ok = True
    artifact_details = []
    sample = pd.read_csv(ROOT / "outputs" / "phase4" / "phase4_feature_dataset.csv", nrows=5, low_memory=False)
    for artifact in artifacts:
        bundle = joblib.load(artifact)
        pipeline = bundle["pipeline"]
        model_features = bundle["features"]
        try:
            proba = pipeline.predict_proba(sample[model_features])[:, 1]
            valid = len(proba) == len(sample) and ((proba >= 0) & (proba <= 1)).all()
        except Exception as exc:
            valid = False
            artifact_details.append(f"{artifact.name}: {exc}")
        else:
            artifact_details.append(f"{artifact.name}: ok")
        artifact_ok &= valid
    add("saved_models_reload_and_predict", artifact_ok, "; ".join(artifact_details))

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_phase4_features.py", "tests/test_phase5_experiment.py", "tests/test_phase6_ml.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    add("phase4_to_phase6_tests_pass", tests.returncode == 0, (tests.stdout + tests.stderr).strip())

    result = {
        "project": "TopoCross-SWIS Phase 6 baseline ML QA",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "full_suite_environment_note": "The complete legacy test suite additionally requires cdflib, netCDF4, and streamlit. Those packages are declared in requirements.txt but could not be installed in this offline QA runtime.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
