from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.mag_reader import resample_mag_minute
from src.swis_august import process_swis_day


OMNI_COLUMNS = [
    "timestamp",
    "omni_Bx_gse",
    "omni_By_gse",
    "omni_Bz_gse",
    "omni_flow_speed",
    "omni_proton_density",
    "omni_temperature",
]
SWIS_SCALAR_COLUMNS = [
    "js_opdi",
    "hellinger_opdi",
    "wasserstein_opdi",
    "proton_density",
    "proton_bulk_speed",
    "proton_thermal",
    "alpha_density",
    "alpha_bulk_speed",
    "alpha_thermal",
    "alpha_proton_ratio",
    "usable_opdi",
]
MAG_COLUMNS = [
    "Bx_gse",
    "By_gse",
    "Bz_gse",
    "Bx_gsm",
    "By_gsm",
    "Bz_gsm",
    "mag_quality_flag",
    "bmag_gse",
    "bmag_gsm",
]


def _minute_index(date: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(date)
    return pd.date_range(start, start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq="min")


def _placeholder_swis(date: str, energy: np.ndarray) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    index = _minute_index(date)
    frame = pd.DataFrame({"timestamp": index})
    for column in SWIS_SCALAR_COLUMNS:
        frame[column] = False if column == "usable_opdi" else np.nan
    shape = (len(index), len(energy))
    spectra = {
        "time": index.to_numpy(dtype="datetime64[ms]"),
        "energy": np.asarray(energy, dtype=float),
        "th1_flux": np.full(shape, np.nan),
        "th2_flux": np.full(shape, np.nan),
        "th1_probability": np.full(shape, np.nan),
        "th2_probability": np.full(shape, np.nan),
    }
    return frame, spectra, {
        "records": 0,
        "start": None,
        "end": None,
        "native_cadence_seconds": None,
        "usable_fraction": 0.0,
        "th1_finite_flux_fraction": 0.0,
        "th2_finite_flux_fraction": 0.0,
        "status": "MISSING_ADITYA_SWIS",
    }


def _pick(raw_event_dir: Path, kind: str, date: str) -> Path | None:
    folder = raw_event_dir / ("mag" if kind == "mag" else "swis")
    if kind == "mag":
        matches = sorted(folder.glob(f"*{date}*V00.nc")) if folder.exists() else []
    else:
        matches = sorted(folder.glob(f"*_{kind.upper()}_{date}_*V03.cdf")) if folder.exists() else []
    return matches[0] if matches else None


def read_omni_csv(path: Path) -> pd.DataFrame:
    """Read a CDAWeb HAPI/listing export and convert official fill values to NaN."""
    first = path.open("r", encoding="utf-8").readline().strip().lower()
    has_header = first.startswith("timestamp") or first.startswith("time")
    frame = pd.read_csv(path, header=0 if has_header else None)
    if has_header and set(OMNI_COLUMNS).issubset(frame.columns):
        frame = frame[OMNI_COLUMNS].copy()
    elif len(frame.columns) == len(OMNI_COLUMNS):
        frame.columns = OMNI_COLUMNS
    else:
        raise ValueError(f"Expected seven OMNI columns in {path}; found {len(frame.columns)}")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame["timestamp"] = timestamps.dt.tz_convert(None)
    for column in OMNI_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["omni_Bx_gse", "omni_By_gse", "omni_Bz_gse"]:
        frame.loc[frame[column].abs() >= 9999.0, column] = np.nan
    frame.loc[frame["omni_flow_speed"] >= 99999.0, "omni_flow_speed"] = np.nan
    frame.loc[frame["omni_proton_density"] >= 999.0, "omni_proton_density"] = np.nan
    frame.loc[frame["omni_temperature"] >= 9_999_999.0, "omni_temperature"] = np.nan

    if frame["timestamp"].duplicated().any():
        raise ValueError(f"Duplicate OMNI timestamps in {path}")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["omni_complete"] = frame[OMNI_COLUMNS[1:]].notna().all(axis=1)
    frame["omni_source"] = "NASA_CDAWEB_OMNI_HRO_1MIN"
    return frame


def omni_quality(frame: pd.DataFrame, start_date: str, end_date: str) -> dict:
    expected_start = pd.Timestamp(start_date)
    expected_end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    steps = frame["timestamp"].diff().dropna()
    return {
        "records": int(len(frame)),
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
        "expected_start": str(expected_start),
        "expected_end": str(expected_end),
        "continuous_minute_index": bool((steps == pd.Timedelta(minutes=1)).all()),
        "duplicate_timestamps": int(frame["timestamp"].duplicated().sum()),
        "fully_valid_records": int(frame["omni_complete"].sum()),
        "fully_valid_fraction": float(frame["omni_complete"].mean()),
    }


def file_inventory(paths: Iterable[Path]) -> list[dict]:
    rows = []
    for path in sorted(paths):
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({"filename": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
    return rows


def process_event_source(
    raw_root: Path,
    raw_subdir: str,
    start_date: str,
    end_date: str,
    omni_csv: Path,
    energy: np.ndarray,
    min_valid_points: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    raw_event_dir = raw_root / raw_subdir
    dates = [day.strftime("%Y%m%d") for day in pd.date_range(start_date, end_date, freq="D")]
    scalar_parts: list[pd.DataFrame] = []
    spectra_parts: list[dict[str, np.ndarray]] = []
    daily_quality: dict[str, dict] = {}
    source_files: list[Path] = []

    for date in dates:
        th1 = _pick(raw_event_dir, "th1", date)
        th2 = _pick(raw_event_dir, "th2", date)
        blk = _pick(raw_event_dir, "blk", date)
        if th1 and th2 and blk:
            frame, spectra, report = process_swis_day(th1, th2, blk, energy, min_valid_points)
            report["status"] = "AVAILABLE"
            source_files.extend([th1, th2, blk])
        else:
            frame, spectra, report = _placeholder_swis(date, energy)
            report["missing_files"] = [
                kind for kind, path in [("TH1", th1), ("TH2", th2), ("BLK", blk)] if path is None
            ]
        frame["aditya_swis_files_available"] = bool(th1 and th2 and blk)
        scalar_parts.append(frame)
        spectra_parts.append(spectra)
        daily_quality[date] = report

    features = pd.concat(scalar_parts, ignore_index=True).sort_values("timestamp")
    mag_parts = []
    missing_mag_dates = []
    daily_mag_coverage = {}
    for date in dates:
        mag_path = _pick(raw_event_dir, "mag", date)
        if mag_path:
            mag_day = resample_mag_minute(mag_path)
            mag_parts.append(mag_day)
            source_files.append(mag_path)
            daily_mag_coverage[date] = float(
                mag_day[["Bx_gse", "By_gse", "Bz_gse"]].notna().all(axis=1).sum() / 1440
            )
        else:
            missing_mag_dates.append(date)
            daily_mag_coverage[date] = 0.0
    if mag_parts:
        mag = pd.concat(mag_parts, ignore_index=True).sort_values("timestamp")
        features = features.merge(mag, on="timestamp", how="left", validate="one_to_one")
    for column in MAG_COLUMNS:
        if column not in features:
            features[column] = np.nan
    features["aditya_mag_available"] = features[["Bx_gse", "By_gse", "Bz_gse"]].notna().all(axis=1)

    omni = read_omni_csv(omni_csv)
    features = features.merge(omni, on="timestamp", how="left", validate="one_to_one")
    features["aditya_modalities_complete"] = (
        features["usable_opdi"].fillna(False).astype(bool)
        & features[["proton_density", "proton_bulk_speed", "proton_thermal"]].notna().all(axis=1)
        & features[["Bx_gse", "By_gse", "Bz_gse"]].notna().all(axis=1)
    )

    combined_spectra = {}
    for key in spectra_parts[0]:
        combined_spectra[key] = spectra_parts[0][key] if key == "energy" else np.concatenate(
            [part[key] for part in spectra_parts], axis=0
        )
    report = {
        "date_range": {"start": start_date, "end": end_date},
        "processed_dates": dates,
        "missing_swis_dates": [date for date, item in daily_quality.items() if item["status"] != "AVAILABLE"],
        "partial_swis_dates": [
            date
            for date, item in daily_quality.items()
            if item["status"] == "AVAILABLE" and float(item["usable_fraction"]) < 0.90
        ],
        "missing_mag_dates": missing_mag_dates,
        "partial_mag_dates": [date for date, fraction in daily_mag_coverage.items() if 0.0 < fraction < 0.90],
        "daily_mag_coverage": daily_mag_coverage,
        "aditya_complete_fraction": float(features["aditya_modalities_complete"].mean()),
        "omni": omni_quality(omni, start_date, end_date),
        "omni_role": "External near-Earth conventional-data reference; never substituted for Aditya-L1 SWIS OPDI.",
        "daily_swis_quality": daily_quality,
        "source_files": file_inventory(source_files),
    }
    return features, combined_spectra, report
