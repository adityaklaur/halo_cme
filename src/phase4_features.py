from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml


CORE_CONVENTIONAL = [
    "proton_density",
    "proton_bulk_speed",
    "proton_thermal",
    "alpha_proton_ratio",
    "bmag_gse",
    "Bx_gse",
    "By_gse",
    "Bz_gse",
]
ROLLING_CONVENTIONAL = [
    "proton_density",
    "proton_bulk_speed",
    "proton_thermal",
    "alpha_proton_ratio",
    "bmag_gse",
]
OPDI_COLUMNS = ["js_opdi", "hellinger_opdi", "wasserstein_opdi"]


@dataclass(frozen=True)
class Phase4Paths:
    dataset: Path
    dictionary: Path
    event_summary: Path
    report: Path


def output_paths(output_dir: Path) -> Phase4Paths:
    return Phase4Paths(
        dataset=output_dir / "phase4_feature_dataset.csv",
        dictionary=output_dir / "phase4_feature_dictionary.csv",
        event_summary=output_dir / "phase4_event_summary.csv",
        report=output_dir / "phase4_report.json",
    )


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def _resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _time_derivative(values: pd.Series, timestamps: pd.Series) -> pd.Series:
    dt = timestamps.diff().dt.total_seconds().div(60.0)
    return values.diff().div(dt.where(dt > 0))


def _rolling_past(series: pd.Series, window: int, statistic: str, min_periods: int | None = None) -> pd.Series:
    shifted = series.shift(1)
    min_p = min_periods if min_periods is not None else max(3, min(window, window // 3 or 1))
    roll = shifted.rolling(window=window, min_periods=min_p)
    if statistic == "median":
        return roll.median()
    if statistic == "mean":
        return roll.mean()
    if statistic == "variance":
        return roll.var(ddof=0)
    if statistic == "mad":
        return roll.apply(lambda x: float(np.nanmedian(np.abs(x - np.nanmedian(x)))), raw=True)
    raise ValueError(f"Unsupported rolling statistic: {statistic}")


def _consecutive_true(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).astype(bool).to_numpy()
    out = np.zeros(len(values), dtype=int)
    run = 0
    for i, flag in enumerate(values):
        run = run + 1 if flag else 0
        out[i] = run
    return pd.Series(out, index=mask.index)


def _spectral_features(prob1: np.ndarray, prob2: np.ndarray, energy: np.ndarray) -> dict[str, np.ndarray]:
    p = np.asarray(prob1, dtype=float)
    q = np.asarray(prob2, dtype=float)
    e = np.asarray(energy, dtype=float)
    if p.ndim != 2 or q.ndim != 2 or p.shape != q.shape or p.shape[1] != len(e):
        raise ValueError("Invalid TH1/TH2 probability spectra dimensions")

    valid = np.isfinite(p).all(axis=1) & np.isfinite(q).all(axis=1)
    p_sum = np.nansum(p, axis=1)
    q_sum = np.nansum(q, axis=1)
    valid &= (p_sum > 0) & (q_sum > 0)

    p_norm = np.full_like(p, np.nan)
    q_norm = np.full_like(q, np.nan)
    p_norm[valid] = p[valid] / p_sum[valid, None]
    q_norm[valid] = q[valid] / q_sum[valid, None]

    dot = np.nansum(p_norm * q_norm, axis=1)
    norm_p = np.sqrt(np.nansum(p_norm**2, axis=1))
    norm_q = np.sqrt(np.nansum(q_norm**2, axis=1))
    cosine = dot / np.where(norm_p * norm_q > 0, norm_p * norm_q, np.nan)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.arccos(cosine)

    c1 = np.nansum(p_norm * e[None, :], axis=1)
    c2 = np.nansum(q_norm * e[None, :], axis=1)
    w1 = np.sqrt(np.nansum(p_norm * (e[None, :] - c1[:, None]) ** 2, axis=1))
    w2 = np.sqrt(np.nansum(q_norm * (e[None, :] - c2[:, None]) ** 2, axis=1))

    p_safe = np.where(np.isfinite(p_norm), p_norm, -np.inf)
    q_safe = np.where(np.isfinite(q_norm), q_norm, -np.inf)
    peak1 = np.where(valid, e[np.argmax(p_safe, axis=1)], np.nan)
    peak2 = np.where(valid, e[np.argmax(q_safe, axis=1)], np.nan)
    log_peak_ratio = np.log(np.where((peak1 > 0) & (peak2 > 0), peak1 / peak2, np.nan))

    result = {
        "th1_th2_cosine_similarity": cosine,
        "th1_th2_spectral_angle_rad": angle,
        "th1_spectral_centroid_ev": c1,
        "th2_spectral_centroid_ev": c2,
        "crossplane_centroid_delta_ev": c1 - c2,
        "th1_spectral_width_ev": w1,
        "th2_spectral_width_ev": w2,
        "crossplane_width_delta_ev": w1 - w2,
        "th1_peak_energy_ev": peak1,
        "th2_peak_energy_ev": peak2,
        "crossplane_log_peak_energy_ratio": log_peak_ratio,
    }
    for key in result:
        result[key] = np.where(valid, result[key], np.nan)
    return result


def _attach_spectral_shape_features(root: Path, frame: pd.DataFrame) -> pd.DataFrame:
    spectral_columns = [
        "th1_th2_cosine_similarity",
        "th1_th2_spectral_angle_rad",
        "th1_spectral_centroid_ev",
        "th2_spectral_centroid_ev",
        "crossplane_centroid_delta_ev",
        "th1_spectral_width_ev",
        "th2_spectral_width_ev",
        "crossplane_width_delta_ev",
        "th1_peak_energy_ev",
        "th2_peak_energy_ev",
        "crossplane_log_peak_energy_ratio",
    ]
    for column in spectral_columns:
        frame[column] = np.nan

    cache: dict[Path, pd.DataFrame] = {}
    for spectra_name, index in frame.groupby("source_spectra_file", dropna=False).groups.items():
        if not isinstance(spectra_name, str) or not spectra_name.strip():
            continue
        path = _resolve(root, spectra_name)
        if not path.exists():
            continue
        if path not in cache:
            with np.load(path) as npz:
                required = {"time", "energy", "th1_probability", "th2_probability"}
                if not required.issubset(npz.files):
                    continue
                times = pd.to_datetime(npz["time"].astype("datetime64[ms]"))
                features = _spectral_features(
                    npz["th1_probability"], npz["th2_probability"], npz["energy"]
                )
                spectra_frame = pd.DataFrame({"timestamp": times, **features})
                spectra_frame = spectra_frame.drop_duplicates("timestamp").set_index("timestamp")
                cache[path] = spectra_frame
        spectra_frame = cache[path]
        timestamps = frame.loc[index, "timestamp"]
        aligned = spectra_frame.reindex(pd.DatetimeIndex(timestamps))
        for column in spectral_columns:
            frame.loc[index, column] = aligned[column].to_numpy()
    return frame


def _feature_dictionary(windows: list[int], primary_window: int) -> pd.DataFrame:
    rows: list[dict] = []

    def add(group: str, column: str, description: str, units: str = "", formula: str = "") -> None:
        rows.append(
            {
                "feature_group": group,
                "column": column,
                "description": description,
                "units": units,
                "formula_or_definition": formula,
                "uses_ground_truth_label": False,
            }
        )

    for column, units in [
        ("proton_density", "cm^-3"),
        ("proton_bulk_speed", "km/s"),
        ("proton_thermal", "km/s"),
        ("alpha_proton_ratio", "ratio"),
        ("bmag_gse", "nT"),
        ("Bx_gse", "nT"),
        ("By_gse", "nT"),
        ("Bz_gse", "nT"),
    ]:
        add("conventional_raw", column, "Aditya-L1 conventional plasma or MAG measurement", units)
    for column, units in [
        ("d_proton_bulk_speed_dt", "km/s/min"),
        ("d_proton_density_dt", "cm^-3/min"),
        ("d_bmag_gse_dt", "nT/min"),
    ]:
        add("conventional_derivative", column, "One-minute time derivative within each event window", units)
    for base in ROLLING_CONVENTIONAL:
        for window in windows:
            for stat in ["mean", "median", "variance"]:
                add(
                    "conventional_rolling",
                    f"{base}_rolling_{stat}_{window}m",
                    f"Past-only {window}-minute rolling {stat} of {base}",
                )
    for column, description in [
        ("density_compression_ratio", "Density relative to its past rolling median"),
        ("bmag_compression_ratio", "Magnetic-field magnitude relative to its past rolling median"),
        ("speed_compression_ratio", "Bulk speed relative to its past rolling median"),
        ("dynamic_pressure_proxy", "Density times squared bulk speed; proportional pressure proxy"),
        ("dynamic_pressure_compression_ratio", "Pressure proxy relative to its past rolling median"),
        ("joint_compression_index", "Joint positive compression of density, |B| and speed"),
    ]:
        add("compression", column, description, formula=f"past baseline={primary_window} min")

    for column in OPDI_COLUMNS:
        add("cross_plane_opdi", column, "Cross-plane distribution divergence from Phase 1/2")
    for column in ["d_js_opdi_dt", "d_hellinger_opdi_dt", "d_wasserstein_opdi_dt", "d_opdi_dt"]:
        add("cross_plane_derivative", column, "Time derivative of OPDI; d_opdi_dt uses JS as canonical OPDI")
    for column in ["opdi_rolling_mean", "opdi_rolling_median", "opdi_rolling_variance"]:
        add("cross_plane_rolling", column, f"Past-only {primary_window}-minute rolling statistic of JS OPDI")
    add("cross_plane_anomaly", "opdi_anomaly", "Absolute robust z-score of JS OPDI against a past-only quiet-like local baseline")
    add("cross_plane_anomaly", "opdi_persistence", "Consecutive minutes for which OPDI anomaly exceeds the configured threshold", "min")
    for column, description, units in [
        ("th1_th2_cosine_similarity", "Cosine similarity between normalized TH1 and TH2 energy spectra", "ratio"),
        ("th1_th2_spectral_angle_rad", "Spectral angle between normalized TH1 and TH2 energy spectra", "rad"),
        ("crossplane_centroid_delta_ev", "TH1 minus TH2 spectral centroid", "eV"),
        ("crossplane_width_delta_ev", "TH1 minus TH2 spectral width", "eV"),
        ("crossplane_log_peak_energy_ratio", "Log ratio of TH1 to TH2 peak energy", "log ratio"),
    ]:
        add("cross_plane_shape", column, description, units)
    return pd.DataFrame(rows)



def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def build_phase4_features(root: Path, config_path: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_path = _resolve(root, config.get("input_dataset", "outputs/phase3/phase3_ground_truth_dataset.csv"))
    if not input_path.exists():
        raise FileNotFoundError(f"Phase 3 ground-truth dataset not found: {input_path}")

    frame = pd.read_csv(input_path, parse_dates=["timestamp"], low_memory=False)
    required = {
        "event_id",
        "independent_interval_id",
        "timestamp",
        "source_spectra_file",
        "eligible_for_exploratory_modeling",
        *CORE_CONVENTIONAL,
        *OPDI_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Phase 4 input is missing required columns: " + ", ".join(missing))
    if frame[["event_id", "timestamp"]].duplicated().any():
        raise ValueError("Phase 4 input contains duplicate event/timestamp rows")

    windows = [int(value) for value in config.get("rolling_windows_minutes", [5, 15, 60])]
    primary_window = int(config.get("primary_rolling_window_minutes", 15))
    anomaly_window = int(config.get("opdi_anomaly_baseline_minutes", 60))
    anomaly_min = int(config.get("opdi_anomaly_min_periods", 15))
    anomaly_threshold = float(config.get("opdi_anomaly_threshold", 3.0))
    compression_window = int(config.get("compression_baseline_minutes", 15))
    primary_opdi = str(config.get("primary_opdi", "js_opdi"))
    if primary_opdi not in OPDI_COLUMNS:
        raise ValueError(f"Unsupported primary_opdi: {primary_opdi}")

    frame = frame.sort_values(["event_id", "timestamp"]).reset_index(drop=True)
    for _, index in frame.groupby("event_id", sort=False).groups.items():
        g = frame.loc[index].copy()
        t = g["timestamp"]
        frame.loc[index, "d_proton_bulk_speed_dt"] = _time_derivative(g["proton_bulk_speed"], t).to_numpy()
        frame.loc[index, "d_proton_density_dt"] = _time_derivative(g["proton_density"], t).to_numpy()
        frame.loc[index, "d_bmag_gse_dt"] = _time_derivative(g["bmag_gse"], t).to_numpy()

        for opdi in OPDI_COLUMNS:
            name = f"d_{opdi}_dt"
            frame.loc[index, name] = _time_derivative(g[opdi], t).to_numpy()
        frame.loc[index, "d_opdi_dt"] = _time_derivative(g[primary_opdi], t).to_numpy()

        for base in ROLLING_CONVENTIONAL:
            for window in windows:
                for stat in ["mean", "median", "variance"]:
                    name = f"{base}_rolling_{stat}_{window}m"
                    frame.loc[index, name] = _rolling_past(g[base], window, stat).to_numpy()

        frame.loc[index, "opdi_rolling_mean"] = _rolling_past(g[primary_opdi], primary_window, "mean").to_numpy()
        frame.loc[index, "opdi_rolling_median"] = _rolling_past(g[primary_opdi], primary_window, "median").to_numpy()
        frame.loc[index, "opdi_rolling_variance"] = _rolling_past(g[primary_opdi], primary_window, "variance").to_numpy()

        baseline_median = _rolling_past(g[primary_opdi], anomaly_window, "median", anomaly_min)
        baseline_mad = _rolling_past(g[primary_opdi], anomaly_window, "mad", anomaly_min)
        robust_scale = 1.4826 * baseline_mad
        finite_primary = g[primary_opdi].dropna().to_numpy(float)
        if len(finite_primary):
            center = float(np.median(finite_primary))
            fallback = float(np.median(np.abs(finite_primary - center))) * 1.4826
        else:
            fallback = 1e-6
        if not np.isfinite(fallback) or fallback <= 0:
            fallback = 1e-6
        robust_scale = robust_scale.where(robust_scale > 1e-12, fallback)
        anomaly = (g[primary_opdi] - baseline_median).abs() / robust_scale
        frame.loc[index, "opdi_anomaly"] = anomaly.to_numpy()
        frame.loc[index, "opdi_persistence"] = _consecutive_true(anomaly >= anomaly_threshold).to_numpy()

        density_base = _rolling_past(g["proton_density"], compression_window, "median")
        bmag_base = _rolling_past(g["bmag_gse"], compression_window, "median")
        speed_base = _rolling_past(g["proton_bulk_speed"], compression_window, "median")
        pressure = g["proton_density"] * g["proton_bulk_speed"] ** 2
        pressure_base = _rolling_past(pressure, compression_window, "median")
        density_ratio = g["proton_density"] / density_base.replace(0, np.nan)
        bmag_ratio = g["bmag_gse"] / bmag_base.replace(0, np.nan)
        speed_ratio = g["proton_bulk_speed"] / speed_base.replace(0, np.nan)
        pressure_ratio = pressure / pressure_base.replace(0, np.nan)
        frame.loc[index, "density_compression_ratio"] = density_ratio.to_numpy()
        frame.loc[index, "bmag_compression_ratio"] = bmag_ratio.to_numpy()
        frame.loc[index, "speed_compression_ratio"] = speed_ratio.to_numpy()
        frame.loc[index, "dynamic_pressure_proxy"] = pressure.to_numpy()
        frame.loc[index, "dynamic_pressure_compression_ratio"] = pressure_ratio.to_numpy()
        compression = np.sqrt(
            np.maximum(density_ratio.to_numpy() - 1.0, 0.0) ** 2
            + np.maximum(bmag_ratio.to_numpy() - 1.0, 0.0) ** 2
            + np.maximum(speed_ratio.to_numpy() - 1.0, 0.0) ** 2
        )
        frame.loc[index, "joint_compression_index"] = compression

    frame = _attach_spectral_shape_features(root, frame)
    frame["phase4_conventional_available_fraction"] = frame[CORE_CONVENTIONAL].notna().mean(axis=1)
    phase5_opdi_core = ["js_opdi", "hellinger_opdi", "wasserstein_opdi", "d_opdi_dt"]
    frame["phase4_opdi_available_fraction"] = frame[phase5_opdi_core].notna().mean(axis=1)
    frame["phase4_spectral_shape_complete"] = frame[
        ["th1_th2_cosine_similarity", "crossplane_centroid_delta_ev", "crossplane_width_delta_ev"]
    ].notna().all(axis=1)
    base_eligible = _as_bool(frame["eligible_for_exploratory_modeling"])
    frame["phase4_ready_for_exploratory_ablation"] = (
        base_eligible
        & (frame["phase4_conventional_available_fraction"] >= 0.70)
        & (frame["phase4_opdi_available_fraction"] >= 0.75)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    frame.to_csv(paths.dataset, index=False)
    dictionary = _feature_dictionary(windows, primary_window)
    dictionary.to_csv(paths.dictionary, index=False)

    summary = (
        frame.groupby(["event_id", "independent_interval_id", "research_label"], dropna=False)
        .agg(
            records=("timestamp", "size"),
            exploratory_rows=("phase4_ready_for_exploratory_ablation", "sum"),
            conventional_available_fraction=("phase4_conventional_available_fraction", "mean"),
            opdi_available_fraction=("phase4_opdi_available_fraction", "mean"),
            spectral_shape_fraction=("phase4_spectral_shape_complete", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(paths.event_summary, index=False)

    event_eligibility = frame.assign(_base_eligible=base_eligible).groupby("event_id")["_base_eligible"].any()
    blocked = sorted(event_eligibility.index[~event_eligibility].tolist())
    derived_columns = [column for column in frame.columns if column not in pd.read_csv(input_path, nrows=0).columns]
    report = {
        "schema_version": int(config.get("schema_version", 1)),
        "phase": 4,
        "dataset_name": config.get("dataset_name", "TopoCross-SWIS Phase 4 complete feature dataset"),
        "status": "FEATURE_ENGINEERING_COMPLETE_WITH_EXISTING_PHASE2_DATA_GAP" if blocked else "FEATURE_ENGINEERING_COMPLETE",
        "records": int(len(frame)),
        "events": int(frame["event_id"].nunique()),
        "independent_intervals": int(frame["independent_interval_id"].nunique()),
        "derived_feature_columns": int(len(derived_columns)),
        "exploratory_ablation_rows": int(frame["phase4_ready_for_exploratory_ablation"].sum()),
        "spectral_shape_complete_rows": int(frame["phase4_spectral_shape_complete"].sum()),
        "blocked_events_from_phase3": blocked,
        "primary_opdi": primary_opdi,
        "rolling_windows_minutes": windows,
        "opdi_anomaly": {
            "past_only_baseline_minutes": anomaly_window,
            "minimum_periods": anomaly_min,
            "threshold": anomaly_threshold,
        },
        "scientific_guardrails": [
            "All rolling baselines are past-only within an event window; the current row is not included in its own baseline.",
            "Ground-truth labels are carried through for evaluation but are not used to calculate Phase 4 features.",
            "The November orientation-control rows remain in the table but are excluded from modeling because required Aditya-L1 modalities are absent.",
            "Spectral-shape features are computed from normalized TH1/TH2 distributions in the registered source spectra files.",
        ],
        "outputs": {key: _display_path(root, value) for key, value in paths.__dict__.items()},
    }
    paths.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
