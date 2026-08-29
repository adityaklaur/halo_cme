#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from scipy.stats import mannwhitneyu
except Exception:  # pragma: no cover - only used if scipy is unavailable
    mannwhitneyu = None

from src.detector_august import add_detector, add_ground_truth, evaluate_detection
from src.ground_truth import build_phase3_ground_truth
from src.mag_reader import resample_mag_minute
from src.scientific_dataset import build_phase2_dataset
from src.source_matcher import rank_cme_candidates
from src.swis_august import process_swis_day


LEGACY_RAW_DIRS = {
    "th1": "tha1",
    "th2": "tha2",
    "blk": "swis_BLK",
    "mag": "mag_2026Aug23T210145602",
}


def pick_one(kind: str, date: str, version: str) -> Path:
    directories = [ROOT / "data" / "raw" / kind, ROOT / LEGACY_RAW_DIRS[kind]]
    if kind == "mag":
        patterns = [f"*{date}*V00.nc"]
    else:
        patterns = [f"*{date}*{version}.cdf"]
    searched = []
    for directory in directories:
        if not directory.exists():
            searched.append(str(directory))
            continue
        for pattern in patterns:
            searched.append(str(directory / pattern))
            matches = sorted(directory.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(f"No {kind} file found for {date}. Searched: {searched}")


def date_strings(start_date: str, end_date: str) -> list[str]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    return [d.strftime("%Y%m%d") for d in pd.date_range(start, end, freq="D")]


def cliffs_delta_and_p(event, quiet) -> tuple[float, float]:
    a = pd.to_numeric(event, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(quiet, errors="coerce").dropna().to_numpy(float)
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    if mannwhitneyu is None:
        return np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    delta = 2.0 * u / (len(a) * len(b)) - 1.0
    return float(delta), float(p)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "prototype.yaml").read_text())
    grid = np.geomspace(
        cfg["swis"]["common_energy_min_ev"],
        cfg["swis"]["common_energy_max_ev"],
        cfg["swis"]["common_grid_points"],
    )
    daily = ROOT / "data" / "processed" / "daily"
    daily.mkdir(parents=True, exist_ok=True)
    processing = cfg.get("processing", {})
    dates = date_strings(processing.get("start_date", "2024-08-09"), processing.get("end_date", "2024-08-15"))
    allow_missing_mag = bool(processing.get("allow_missing_mag", False))

    scalar = []
    specs = []
    reports = {}
    for date in dates:
        print(f"Processing SWIS {date}...")
        p1 = pick_one("th1", date, cfg["swis"]["version"])
        p2 = pick_one("th2", date, cfg["swis"]["version"])
        pb = pick_one("blk", date, cfg["swis"]["version"])
        day_features, day_spectra, day_report = process_swis_day(
            p1,
            p2,
            pb,
            grid,
            cfg["swis"]["min_valid_points"],
        )
        day_features.to_csv(daily / f"{date}_features.csv", index=False)
        np.savez_compressed(daily / f"{date}_spectra.npz", **day_spectra)
        (daily / f"{date}_report.json").write_text(json.dumps(day_report, indent=2))
        scalar.append(day_features)
        specs.append(day_spectra)
        reports[date] = day_report

    swis = pd.concat(scalar, ignore_index=True).sort_values("timestamp")
    mags = []
    for date in dates:
        print(f"Processing MAG {date}...")
        try:
            mags.append(resample_mag_minute(pick_one("mag", date, cfg["swis"]["version"])))
        except FileNotFoundError:
            if not allow_missing_mag:
                raise
            print(f"WARNING: No MAG L2 file found for {date}; continuing with MAG columns empty for that day.")
    if mags:
        mag = pd.concat(mags, ignore_index=True).sort_values("timestamp")
        merged = swis.merge(mag, on="timestamp", how="left")
    else:
        merged = swis.copy()

    event = cfg["event"]
    out, detector_report = add_detector(
        merged,
        event["baseline_start"],
        event["baseline_end"],
        cfg["detector"],
    )
    out = add_ground_truth(out, event["shock_time"], event["sheath_end_icme_start"], event["icme_end"])

    proc = ROOT / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    out.to_csv(proc / "aug2024_features_1min.csv", index=False)

    combined = {}
    for key in specs[0]:
        combined[key] = specs[0][key] if key == "energy" else np.concatenate([s[key] for s in specs], axis=0)
    np.savez_compressed(proc / "aug2024_spectra_1min.npz", **combined)

    metric_cols = [
        "js_opdi",
        "hellinger_opdi",
        "wasserstein_opdi",
        "proton_bulk_speed",
        "proton_density",
        "proton_thermal",
        "alpha_proton_ratio",
        "bmag_gse",
    ]
    summary = out.groupby("ground_truth_state")[metric_cols].agg(["count", "median", "mean", "std"])
    summary.columns = ["__".join(x) for x in summary.columns]
    summary.reset_index().to_csv(proc / "ground_truth_state_summary.csv", index=False)

    quiet = out.loc[out.ground_truth_state == "QUIET/PRE-EVENT"]
    tests = []
    for state in ["SHEATH", "ICME/EJECTA", "POST-EVENT"]:
        group = out.loc[out.ground_truth_state == state]
        for metric in ["js_opdi", "hellinger_opdi", "wasserstein_opdi"]:
            delta, p = cliffs_delta_and_p(group[metric], quiet[metric])
            tests.append(
                {
                    "comparison": f"{state} vs QUIET/PRE-EVENT",
                    "metric": metric,
                    "quiet_median": float(quiet[metric].median()),
                    "event_median": float(group[metric].median()),
                    "cliffs_delta_event_vs_quiet": delta,
                    "mannwhitney_p_two_sided": p,
                    "n_quiet": int(quiet[metric].notna().sum()),
                    "n_event": int(group[metric].notna().sum()),
                }
            )
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(proc / "state_statistical_tests.csv", index=False)

    eval_report = evaluate_detection(out, event["shock_time"])
    cps = out.loc[out.is_change_point.astype(bool)].copy()
    shock = pd.Timestamp(event["shock_time"])
    if len(cps):
        idx = (cps.timestamp - shock).abs().idxmin()
        primary = cps.loc[idx]
        primary_transition = {
            "detected_at": str(primary.timestamp),
            "offset_from_configured_reference_minutes": float((primary.timestamp - shock).total_seconds() / 60),
            "transition_score": float(primary.transition_score),
            "js_opdi": float(primary.js_opdi) if pd.notna(primary.js_opdi) else None,
            "proton_bulk_speed_km_s": float(primary.proton_bulk_speed) if pd.notna(primary.proton_bulk_speed) else None,
            "proton_density_cm3": float(primary.proton_density) if pd.notna(primary.proton_density) else None,
            "magnetic_field_magnitude_nT": float(primary.bmag_gse) if pd.notna(primary.bmag_gse) else None,
        }
    else:
        primary_transition = None

    source_match_time = (
        pd.Timestamp(primary_transition["detected_at"])
        if primary_transition
        else pd.Timestamp(event["shock_time"])
    )
    candidates = pd.read_csv(ROOT / "data" / "cme_candidates" / "candidates.csv")
    ranked = rank_cme_candidates(source_match_time, candidates)
    ranked.to_csv(proc / "cme_candidate_ranking.csv", index=False)

    usable = [float(r.get("usable_fraction", np.nan)) for r in reports.values()]
    report = {
        "prototype": "TopoCross-SWIS August 2024 v1.0",
        "data_interval_utc": {"start": str(out.timestamp.min()), "end": str(out.timestamp.max())},
        "one_minute_records": int(len(out)),
        "swis_version": cfg["swis"]["version"],
        "common_energy_grid_ev": [float(grid[0]), float(grid[-1])],
        "common_energy_grid_points": int(len(grid)),
        "processed_dates": dates,
        "missing_mag_dates_allowed": allow_missing_mag,
        "mean_daily_usable_fraction": float(np.nanmean(usable)) if usable else None,
        "detector": detector_report,
        "evaluation_against_configured_reference": eval_report,
        "primary_transition_nearest_configured_reference": primary_transition,
        "cme_source_match_reference_time": str(source_match_time),
        "cme_source_match_reference": "detected_transition" if primary_transition else "configured_shock_reference",
        "configured_ground_truth": {
            "shock_reference": event["shock_time"],
            "shock_reference_is_approximate": bool(event.get("shock_reference_is_approximate", False)),
            "icme_ejecta_start": event["sheath_end_icme_start"],
            "icme_ejecta_end": event["icme_end"],
            "reference_markers": event.get("reference_markers", {}),
            "note": event.get("ground_truth_note", ""),
        },
        "state_counts": {k: int(v) for k, v in out.state.value_counts().items()},
        "ground_truth_counts": {k: int(v) for k, v in out.ground_truth_state.value_counts().items()},
        "exploratory_opdi_tests": tests_df.to_dict(orient="records"),
        "daily_quality": reports,
        "scientific_status": [
            "Detector calculations do not use the ground-truth labels.",
            "The configured shock reference is approximate and must be reconciled before claiming a validated lead time.",
            "OPDI separation statistics are exploratory results for one event interval, not evidence of multi-event generalization.",
            "CME source compatibility is a heuristic ranking, not a calibrated causal probability.",
        ],
    }
    (proc / "pipeline_report.json").write_text(json.dumps(report, indent=2, default=str))
    phase2_manifest = build_phase2_dataset(
        ROOT,
        ROOT / "config" / "phase2_events.yaml",
        ROOT / "data" / "scientific",
    )
    phase3_report = build_phase3_ground_truth(
        ROOT,
        ROOT / "config" / "phase3_labels.yaml",
        ROOT / "outputs" / "phase3",
    )
    print(f"Built {len(out):,} one-minute records into {proc}")
    if primary_transition:
        print(
            "Primary transition nearest configured shock reference:",
            primary_transition["detected_at"],
            f"({primary_transition['offset_from_configured_reference_minutes']:+.1f} min)",
        )
    print("State counts:", report["state_counts"])
    print(
        "Phase 2 scientific dataset:",
        f"{phase2_manifest['event_windows']} registered windows from",
        f"{phase2_manifest['independent_intervals']} independent interval(s)",
    )
    print(
        "Phase 3 ground truth:",
        f"{phase3_report['validation']['records']} labeled records;",
        f"{phase3_report['validation']['phase3_ready_events']} of {phase3_report['validation']['event_windows']} events ready",
    )


if __name__ == "__main__":
    main()
