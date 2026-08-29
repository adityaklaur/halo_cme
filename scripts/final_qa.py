#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "final_qa_report.json"


def main() -> None:
    checks = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    features_path = PROC / "aug2024_features_1min.csv"
    report_path = PROC / "pipeline_report.json"
    spectra_path = PROC / "aug2024_spectra_1min.npz"
    static_dashboard = ROOT / "outputs" / "TopoCross_dashboard.html"
    phase2_catalog_path = ROOT / "data" / "scientific" / "phase2_event_catalog.csv"
    phase2_features_path = ROOT / "data" / "scientific" / "phase2_feature_table.csv"
    phase2_coverage_path = ROOT / "data" / "scientific" / "phase2_modality_coverage.csv"
    phase2_manifest_path = ROOT / "data" / "scientific" / "phase2_manifest.json"
    phase3_dir = ROOT / "outputs" / "phase3"
    phase3_dataset_path = phase3_dir / "phase3_ground_truth_dataset.csv"
    phase3_counts_path = phase3_dir / "phase3_label_counts.csv"
    phase3_events_path = phase3_dir / "phase3_event_register.csv"
    phase3_boundaries_path = phase3_dir / "phase3_boundary_register.csv"
    phase3_report_path = phase3_dir / "phase3_report.json"
    phase4_dir = ROOT / "outputs" / "phase4"
    phase4_dataset_path = phase4_dir / "phase4_feature_dataset.csv"
    phase4_dictionary_path = phase4_dir / "phase4_feature_dictionary.csv"
    phase4_summary_path = phase4_dir / "phase4_event_summary.csv"
    phase4_report_path = phase4_dir / "phase4_report.json"
    phase5_dir = ROOT / "outputs" / "phase5"
    phase5_predictions_path = phase5_dir / "phase5_predictions.csv"
    phase5_folds_path = phase5_dir / "phase5_fold_metrics.csv"
    phase5_summary_path = phase5_dir / "phase5_summary_metrics.csv"
    phase5_delays_path = phase5_dir / "phase5_detection_delays.csv"
    phase5_report_path = phase5_dir / "phase5_report.json"
    phase6_dir = ROOT / "outputs" / "phase6"
    phase6_predictions_path = phase6_dir / "phase6_predictions.csv"
    phase6_folds_path = phase6_dir / "phase6_fold_metrics.csv"
    phase6_summary_path = phase6_dir / "phase6_summary_metrics.csv"
    phase6_delays_path = phase6_dir / "phase6_detection_delays.csv"
    phase6_importance_path = phase6_dir / "phase6_feature_importance.csv"
    phase6_features_path = phase6_dir / "phase6_feature_list.csv"
    phase6_report_path = phase6_dir / "phase6_report.json"
    zenodo_report = ROOT / "data" / "external" / "zenodo_swis_20231106_12" / "processed" / "zenodo_swis_only_report.json"

    add("feature_csv_exists", features_path.exists(), str(features_path))
    add("pipeline_report_exists", report_path.exists(), str(report_path))
    add("spectra_npz_exists", spectra_path.exists(), str(spectra_path))
    add("static_dashboard_exists", static_dashboard.exists(), str(static_dashboard))
    add("phase2_catalog_exists", phase2_catalog_path.exists(), str(phase2_catalog_path))
    add("phase2_features_exists", phase2_features_path.exists(), str(phase2_features_path))
    add("phase2_coverage_exists", phase2_coverage_path.exists(), str(phase2_coverage_path))
    add("phase2_manifest_exists", phase2_manifest_path.exists(), str(phase2_manifest_path))
    add("phase3_dataset_exists", phase3_dataset_path.exists(), str(phase3_dataset_path))
    add("phase3_counts_exists", phase3_counts_path.exists(), str(phase3_counts_path))
    add("phase3_events_exists", phase3_events_path.exists(), str(phase3_events_path))
    add("phase3_boundaries_exists", phase3_boundaries_path.exists(), str(phase3_boundaries_path))
    add("phase3_report_exists", phase3_report_path.exists(), str(phase3_report_path))
    add("phase4_dataset_exists", phase4_dataset_path.exists(), str(phase4_dataset_path))
    add("phase4_dictionary_exists", phase4_dictionary_path.exists(), str(phase4_dictionary_path))
    add("phase4_summary_exists", phase4_summary_path.exists(), str(phase4_summary_path))
    add("phase4_report_exists", phase4_report_path.exists(), str(phase4_report_path))
    add("phase5_predictions_exists", phase5_predictions_path.exists(), str(phase5_predictions_path))
    add("phase5_folds_exists", phase5_folds_path.exists(), str(phase5_folds_path))
    add("phase5_summary_exists", phase5_summary_path.exists(), str(phase5_summary_path))
    add("phase5_delays_exists", phase5_delays_path.exists(), str(phase5_delays_path))
    add("phase5_report_exists", phase5_report_path.exists(), str(phase5_report_path))
    add("phase6_predictions_exists", phase6_predictions_path.exists(), str(phase6_predictions_path))
    add("phase6_folds_exists", phase6_folds_path.exists(), str(phase6_folds_path))
    add("phase6_summary_exists", phase6_summary_path.exists(), str(phase6_summary_path))
    add("phase6_delays_exists", phase6_delays_path.exists(), str(phase6_delays_path))
    add("phase6_importance_exists", phase6_importance_path.exists(), str(phase6_importance_path))
    add("phase6_feature_list_exists", phase6_features_path.exists(), str(phase6_features_path))
    add("phase6_report_exists", phase6_report_path.exists(), str(phase6_report_path))
    add(
        "zenodo_external_dataset_optional",
        True,
        str(zenodo_report) if zenodo_report.exists() else "optional external dataset not present in GitHub-clean folder",
    )

    if features_path.exists() and report_path.exists():
        df = pd.read_csv(features_path, parse_dates=["timestamp"])
        report = json.loads(report_path.read_text())
        add("record_count_10080", len(df) == 10080, f"records={len(df)}")
        add("range_start", df.timestamp.min() == pd.Timestamp("2024-08-09 00:00:00"), str(df.timestamp.min()))
        add("range_end", df.timestamp.max() == pd.Timestamp("2024-08-15 23:59:00"), str(df.timestamp.max()))
        add(
            "primary_transition_anchor",
            report["evaluation_against_configured_reference"]["nearest_change_to_reference_shock"] == "2024-08-10 12:19:00",
            report["evaluation_against_configured_reference"]["nearest_change_to_reference_shock"],
        )
        add(
            "offset_anchor",
            report["evaluation_against_configured_reference"]["nearest_change_offset_minutes"] == -31.0,
            str(report["evaluation_against_configured_reference"]["nearest_change_offset_minutes"]),
        )
        add("change_points_34", int(df.is_change_point.astype(bool).sum()) == 34, str(int(df.is_change_point.astype(bool).sum())))
        add("states_present", df["state"].notna().all(), str(df["state"].value_counts().to_dict()))
        around = df.set_index("timestamp").loc["2024-08-10 12:17:00":"2024-08-10 12:19:00"]
        add(
            "watch_before_confirmed_alert",
            around.loc[pd.Timestamp("2024-08-10 12:17:00"), "state"] == "WATCH"
            and around.loc[pd.Timestamp("2024-08-10 12:18:00"), "state"] == "WATCH"
            and around.loc[pd.Timestamp("2024-08-10 12:19:00"), "state"] == "ALERT",
            str(around["state"].to_dict()),
        )
        add(
            "cme_match_uses_detected_transition",
            report.get("cme_source_match_reference") == "detected_transition"
            and report.get("cme_source_match_reference_time") == "2024-08-10 12:19:00",
            f"{report.get('cme_source_match_reference')} @ {report.get('cme_source_match_reference_time')}",
        )
        add("js_missing_under_5_percent", df["js_opdi"].isna().mean() < 0.05, f"{df['js_opdi'].isna().mean() * 100:.3f}%")
        add("mag_missing_under_1_percent", df["bmag_gse"].isna().mean() < 0.01, f"{df['bmag_gse'].isna().mean() * 100:.3f}%")

    if phase2_catalog_path.exists() and phase2_features_path.exists() and phase2_coverage_path.exists() and phase2_manifest_path.exists():
        phase2_catalog = pd.read_csv(phase2_catalog_path)
        phase2_features = pd.read_csv(phase2_features_path, parse_dates=["timestamp"], low_memory=False)
        phase2_coverage = pd.read_csv(phase2_coverage_path)
        phase2_manifest = json.loads(phase2_manifest_path.read_text())
        add("phase2_registered_windows", len(phase2_catalog) == 7, f"windows={len(phase2_catalog)}")
        add("phase2_registered_rows", len(phase2_features) == 17201, f"rows={len(phase2_features)}")
        required_coverage = phase2_coverage.loc[phase2_coverage["required_for_contract"].astype(bool)]
        complete_event_coverage = required_coverage.loc[
            required_coverage["event_id"] != "NOV2024_ORIENTATION_CONTROL_01"
        ]
        add(
            "phase2_complete_events_pass_modalities",
            bool(complete_event_coverage["passes_threshold"].all()),
            str(complete_event_coverage.groupby("modality")["coverage_fraction"].min().to_dict()),
        )
        non_ready = phase2_catalog.loc[~phase2_catalog["phase2_ready"].astype(bool), "event_id"].tolist()
        add(
            "phase2_only_november_is_blocked",
            non_ready == ["NOV2024_ORIENTATION_CONTROL_01"],
            str(non_ready),
        )
        add(
            "phase2_independence_guardrail",
            phase2_manifest["independent_intervals"] == 5
            and phase2_manifest["status"] == "MULTI_EVENT_ASSEMBLED_WITH_DATA_GAPS"
            and phase2_manifest["research_ready"] is False,
            f"independent_intervals={phase2_manifest['independent_intervals']}, research_ready={phase2_manifest['research_ready']}",
        )

        expected_omni_rows = {"oct2024": 8640, "mar2025": 7200, "sep2024": 7200, "nov2024": 4320}
        omni_details = {}
        omni_ok = True
        for source_id, expected_rows in expected_omni_rows.items():
            omni = pd.read_csv(ROOT / "data" / "external" / "omni" / f"{source_id}_omni_1min.csv")
            omni_details[source_id] = len(omni)
            omni_ok &= len(omni) == expected_rows and not omni["timestamp"].duplicated().any()
        add("online_omni_exports_validated", omni_ok, str(omni_details))

    if all(
        path.exists()
        for path in [
            phase3_dataset_path,
            phase3_counts_path,
            phase3_events_path,
            phase3_boundaries_path,
            phase3_report_path,
        ]
    ):
        phase3 = pd.read_csv(phase3_dataset_path, parse_dates=["timestamp"], low_memory=False)
        phase3_counts = pd.read_csv(phase3_counts_path)
        phase3_events = pd.read_csv(phase3_events_path)
        phase3_boundaries = pd.read_csv(phase3_boundaries_path)
        phase3_report = json.loads(phase3_report_path.read_text())
        add("phase3_valid", phase3_report["phase3_valid"] is True, phase3_report["status"])
        add("phase3_records_match_phase2", len(phase3) == 17201, f"records={len(phase3)}")
        add(
            "phase3_event_and_interval_counts",
            phase3["event_id"].nunique() == 7 and phase3["independent_interval_id"].nunique() == 5,
            f"events={phase3['event_id'].nunique()}, intervals={phase3['independent_interval_id'].nunique()}",
        )
        add(
            "phase3_labels_complete",
            int((phase3["research_label"] == "UNKNOWN").sum()) == 0
            and int(phase3[["event_id", "timestamp"]].duplicated().sum()) == 0,
            f"unknown={(phase3['research_label'] == 'UNKNOWN').sum()}, duplicates={phase3[['event_id', 'timestamp']].duplicated().sum()}",
        )
        expected_labels = {
            "QUIET": 2030,
            "SHOCK": 1,
            "SHEATH": 1389,
            "ICME/EJECTA": 3600,
            "COMPLEX_ICME": 2520,
            "POST-ICME": 2880,
            "CIR/HSS": 4320,
            "ORIENTATION-CONTROL": 461,
        }
        actual_labels = phase3_counts.set_index("research_label")["n"].astype(int).to_dict()
        add("phase3_label_counts", actual_labels == expected_labels, str(actual_labels))
        blocked = phase3_events.loc[~phase3_events["phase3_ready"].astype(bool), "event_id"].tolist()
        add(
            "phase3_only_november_blocked",
            blocked == ["NOV2024_ORIENTATION_CONTROL_01"] and phase3_report["research_ready"] is False,
            str(blocked),
        )
        october_refs = phase3_boundaries.loc[
            phase3_boundaries["boundary_type"].isin(
                ["SHOCK_AFTER_LOWER_BOUND", "SHEATH_MC_BOUNDARY_REFERENCE"]
            )
        ]
        add(
            "phase3_october_uncertain_boundaries_not_used",
            len(october_refs) == 2 and not october_refs["used_for_minute_labeling"].astype(bool).any(),
            f"reference_rows={len(october_refs)}",
        )

    if all(path.exists() for path in [phase4_dataset_path, phase4_dictionary_path, phase4_summary_path, phase4_report_path]):
        phase4 = pd.read_csv(phase4_dataset_path, low_memory=False)
        phase4_dictionary = pd.read_csv(phase4_dictionary_path)
        phase4_report = json.loads(phase4_report_path.read_text())
        add("phase4_records_match_phase3", len(phase4) == 17201, f"records={len(phase4)}")
        add("phase4_complete_feature_groups", {"d_opdi_dt", "opdi_anomaly", "opdi_persistence", "joint_compression_index", "th1_th2_spectral_angle_rad"}.issubset(phase4.columns), f"columns={len(phase4.columns)}")
        add("phase4_labels_not_used_as_features", not phase4_dictionary["uses_ground_truth_label"].astype(bool).any(), "feature dictionary label-use flags")
        add("phase4_only_november_blocked", phase4_report.get("blocked_events_from_phase3") == ["NOV2024_ORIENTATION_CONTROL_01"], str(phase4_report.get("blocked_events_from_phase3")))

    if all(path.exists() for path in [phase5_predictions_path, phase5_folds_path, phase5_summary_path, phase5_delays_path, phase5_report_path]):
        phase5_summary = pd.read_csv(phase5_summary_path)
        phase5_delays = pd.read_csv(phase5_delays_path)
        phase5_report = json.loads(phase5_report_path.read_text())
        add("phase5_three_required_modes", set(phase5_summary["mode"]) == {"Conventional", "OPDI only", "Combined"}, str(phase5_summary["mode"].tolist()))
        add("phase5_eventwise_no_interval_leakage", all(not item.get("interval_leakage") for item in phase5_report.get("leakage_checks", [])), str(phase5_report.get("leakage_checks", [])))
        add("phase5_november_excluded", "NOV2024_ORIENTATION_INTERVAL" not in phase5_report.get("independent_ready_intervals", []), str(phase5_report.get("independent_ready_intervals", [])))
        october_delays = phase5_delays.loc[phase5_delays["event_id"] == "OCT2024_COMPLEX_ICME_01", "detection_delay_minutes"]
        add("phase5_october_delay_guardrail", october_delays.isna().all(), str(october_delays.tolist()))
        add("phase5_exploratory_status", "EXPLORATORY" in str(phase5_report.get("evidence_status", "")), str(phase5_report.get("evidence_status")))

    if all(path.exists() for path in [phase6_predictions_path, phase6_folds_path, phase6_summary_path, phase6_delays_path, phase6_features_path, phase6_report_path]):
        phase6_predictions = pd.read_csv(phase6_predictions_path)
        phase6_summary = pd.read_csv(phase6_summary_path)
        phase6_features = pd.read_csv(phase6_features_path)
        phase6_report = json.loads(phase6_report_path.read_text())
        add(
            "phase6_three_required_models",
            set(phase6_summary["model"]) == {"Logistic Regression", "Random Forest", "Gradient Boosting"},
            str(phase6_summary["model"].tolist()),
        )
        add(
            "phase6_eventwise_no_interval_leakage",
            all(not item.get("interval_leakage") for item in phase6_report.get("leakage_checks", [])),
            str(phase6_report.get("leakage_checks", [])),
        )
        add(
            "phase6_november_excluded",
            "NOV2024_ORIENTATION_INTERVAL" not in phase6_report.get("independent_ready_intervals", []),
            str(phase6_report.get("independent_ready_intervals", [])),
        )
        add(
            "phase6_label_free_features",
            not phase6_features["uses_ground_truth_label"].astype(bool).any() and "icme_binary" not in set(phase6_features["feature"]),
            f"features={len(phase6_features)}",
        )
        add(
            "phase6_probabilities_valid",
            phase6_predictions["probability"].between(0, 1).all(),
            f"range=({phase6_predictions['probability'].min()}, {phase6_predictions['probability'].max()})",
        )
        add(
            "phase6_exploratory_status",
            phase6_report.get("status") == "COMPLETE_EXPLORATORY_BASELINE_ML" and "EXPLORATORY" in str(phase6_report.get("evidence_status", "")),
            f"{phase6_report.get('status')} / {phase6_report.get('evidence_status')}",
        )
        model_dir = phase6_dir / "models"
        model_files = sorted(path.name for path in model_dir.glob("*.joblib")) if model_dir.exists() else []
        add("phase6_model_artifacts", len(model_files) == 3, str(model_files))

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    add("pytest_passes", tests.returncode == 0, (tests.stdout + tests.stderr)[-1000:])

    result = {
        "project": "TopoCross-SWIS Aditya L2 Phases 2-6 final submission QA",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "scientific_guardrail": "Phases 4-6 are implemented and QA-tested, but research readiness remains false until valid Aditya-L1 SWIS and MAG data cover the November orientation control; Phase 5/6 evidence remains exploratory.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
