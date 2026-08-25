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
    zenodo_report = ROOT / "data" / "external" / "zenodo_swis_20231106_12" / "processed" / "zenodo_swis_only_report.json"

    add("feature_csv_exists", features_path.exists(), str(features_path))
    add("pipeline_report_exists", report_path.exists(), str(report_path))
    add("spectra_npz_exists", spectra_path.exists(), str(spectra_path))
    add("static_dashboard_exists", static_dashboard.exists(), str(static_dashboard))
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

    tests = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    add("pytest_passes", tests.returncode == 0, (tests.stdout + tests.stderr)[-1000:])

    result = {
        "project": "TopoCross-SWIS Aditya L2 final submission QA",
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "scientific_guardrail": "Single-event exploratory prototype. Do not claim validated early warning or generalized ICME detection.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
