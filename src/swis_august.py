from __future__ import annotations

from pathlib import Path
import re
from typing import Optional
import warnings

import cdflib
import numpy as np
import pandas as pd


FILL_LIMIT = -1e30
AU_KM = 149_597_870.7


def _cdf_time(values: np.ndarray) -> pd.Series:
    return pd.to_datetime(cdflib.cdfepoch.to_datetime(values))


def _clean_array(values: np.ndarray, positive: bool = False) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    out[~np.isfinite(out)] = np.nan
    out[out <= FILL_LIMIT] = np.nan
    if positive:
        out[out <= 0] = np.nan
    return out


def _date_from_path(path: Path, fallback_time: pd.Series) -> pd.Timestamp:
    match = re.search(r"(20\d{6})", path.name)
    if match:
        return pd.Timestamp(match.group(1))
    return pd.Timestamp(fallback_time.min()).floor("D")


def _read_spectrum_cdf(path: Path, grid: np.ndarray, min_valid_points: int) -> dict[str, np.ndarray]:
    cdf = cdflib.CDF(str(path))
    time = _cdf_time(cdf.varget("epoch_for_cdf_mod"))
    energy = _clean_array(cdf.varget("energy_center_mod"), positive=True)
    flux = _clean_array(cdf.varget("integrated_flux_mod"), positive=True)
    regridded = _regrid_flux(energy, flux, grid, min_valid_points)
    prob = _probabilities(regridded, min_valid_points)
    return {
        "time": time,
        "flux": regridded,
        "probability": prob,
        "finite_flux_fraction": float(np.isfinite(flux).mean()),
        "native_records": int(len(time)),
    }


def _read_blk_cdf(path: Path) -> pd.DataFrame:
    cdf = cdflib.CDF(str(path))
    data = pd.DataFrame({"timestamp": _cdf_time(cdf.varget("epoch_for_cdf_mod"))})
    for column in [
        "proton_density",
        "proton_bulk_speed",
        "proton_thermal",
        "alpha_density",
        "alpha_bulk_speed",
        "alpha_thermal",
    ]:
        data[column] = _clean_array(cdf.varget(column))
    data["alpha_proton_ratio"] = data["alpha_density"] / data["proton_density"]
    data.loc[~np.isfinite(data["alpha_proton_ratio"]), "alpha_proton_ratio"] = np.nan
    data.loc[data["alpha_proton_ratio"] < 0, "alpha_proton_ratio"] = np.nan
    return data.sort_values("timestamp")


def _regrid_flux(
    energy: np.ndarray,
    flux: np.ndarray,
    grid: np.ndarray,
    min_valid_points: int,
) -> np.ndarray:
    out = np.full((flux.shape[0], len(grid)), np.nan, dtype=float)
    log_grid = np.log(grid)
    for i in range(flux.shape[0]):
        e = energy[i]
        f = flux[i]
        valid = np.isfinite(e) & np.isfinite(f) & (e > 0) & (f > 0)
        if int(valid.sum()) < min_valid_points:
            continue
        order = np.argsort(e[valid])
        x = np.log(e[valid][order])
        y = np.log(f[valid][order])
        unique = np.concatenate(([True], np.diff(x) > 0))
        x = x[unique]
        y = y[unique]
        if len(x) < min_valid_points:
            continue
        inside = (log_grid >= x[0]) & (log_grid <= x[-1])
        row = np.full(len(grid), np.nan, dtype=float)
        row[inside] = np.exp(np.interp(log_grid[inside], x, y))
        out[i] = row
    return out


def _probabilities(flux: np.ndarray, min_valid_points: int) -> np.ndarray:
    clean = np.where(np.isfinite(flux) & (flux > 0), flux, np.nan)
    valid_count = np.isfinite(clean).sum(axis=1)
    sums = np.nansum(clean, axis=1)
    probs = clean / sums[:, None]
    bad = (valid_count < min_valid_points) | ~np.isfinite(sums) | (sums <= 0)
    probs[bad] = np.nan
    return probs


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    mask = np.isfinite(p) & np.isfinite(q) & (p >= 0) & (q >= 0)
    if int(mask.sum()) == 0:
        return np.nan
    pp = p[mask] / np.nansum(p[mask])
    qq = q[mask] / np.nansum(q[mask])
    if not np.isfinite(pp).all() or not np.isfinite(qq).all():
        return np.nan
    m = 0.5 * (pp + qq)
    kl_pm = np.sum(np.where(pp > 0, pp * np.log(pp / m), 0.0))
    kl_qm = np.sum(np.where(qq > 0, qq * np.log(qq / m), 0.0))
    return float(0.5 * (kl_pm + kl_qm))


def _hellinger(p: np.ndarray, q: np.ndarray) -> float:
    mask = np.isfinite(p) & np.isfinite(q) & (p >= 0) & (q >= 0)
    if int(mask.sum()) == 0:
        return np.nan
    pp = p[mask] / np.nansum(p[mask])
    qq = q[mask] / np.nansum(q[mask])
    return float(np.sqrt(0.5 * np.sum((np.sqrt(pp) - np.sqrt(qq)) ** 2)))


def _wasserstein_1d(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    mask = np.isfinite(p) & np.isfinite(q) & (p >= 0) & (q >= 0)
    if int(mask.sum()) < 2:
        return np.nan
    pp = p[mask] / np.nansum(p[mask])
    qq = q[mask] / np.nansum(q[mask])
    x = np.log(grid[mask])
    x = (x - x.min()) / max(x.max() - x.min(), 1e-12)
    cdf_delta = np.abs(np.cumsum(pp) - np.cumsum(qq))
    dx = np.diff(x, prepend=x[0])
    return float(np.sum(cdf_delta * dx))


def _opdi_for_pair(p1: np.ndarray, p2: np.ndarray, grid: np.ndarray, min_valid_points: int) -> tuple[float, float, float]:
    valid = np.isfinite(p1) & np.isfinite(p2)
    if int(valid.sum()) < min_valid_points:
        return np.nan, np.nan, np.nan
    return _js_divergence(p1, p2), _hellinger(p1, p2), _wasserstein_1d(p1, p2, grid)


def _nanmedian_stack(values: np.ndarray, indices: np.ndarray, groups: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.full((n_groups, values.shape[1]), np.nan, dtype=float)
    for group in range(n_groups):
        idx = indices[groups == group]
        if len(idx):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                out[group] = np.nanmedian(values[idx], axis=0)
    return out


def process_swis_day(
    th1_path: Path,
    th2_path: Path,
    blk_path: Optional[Path],
    grid: np.ndarray,
    min_valid_points: int = 8,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    th1 = _read_spectrum_cdf(Path(th1_path), grid, min_valid_points)
    th2 = _read_spectrum_cdf(Path(th2_path), grid, min_valid_points)
    if blk_path is None:
        blk = pd.DataFrame({"timestamp": th1["time"]})
        for column in [
            "proton_density",
            "proton_bulk_speed",
            "proton_thermal",
            "alpha_density",
            "alpha_bulk_speed",
            "alpha_thermal",
            "alpha_proton_ratio",
        ]:
            blk[column] = np.nan
    else:
        blk = _read_blk_cdf(Path(blk_path))

    th1_index = pd.DataFrame({"timestamp": th1["time"], "i1": np.arange(len(th1["time"]))})
    th2_index = pd.DataFrame({"timestamp": th2["time"], "i2": np.arange(len(th2["time"]))})
    paired = pd.merge_asof(
        th1_index.sort_values("timestamp"),
        th2_index.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=3),
    ).dropna(subset=["i2"])
    paired["i1"] = paired["i1"].astype(int)
    paired["i2"] = paired["i2"].astype(int)

    blk_merged = pd.merge_asof(
        paired[["timestamp"]].sort_values("timestamp"),
        blk.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=30),
    )
    for column in [c for c in blk_merged.columns if c != "timestamp"]:
        paired[column] = blk_merged[column].to_numpy()

    opdi = np.array(
        [
            _opdi_for_pair(th1["probability"][i1], th2["probability"][i2], grid, min_valid_points)
            for i1, i2 in zip(paired["i1"], paired["i2"])
        ],
        dtype=float,
    )
    paired["js_opdi"] = opdi[:, 0]
    paired["hellinger_opdi"] = opdi[:, 1]
    paired["wasserstein_opdi"] = opdi[:, 2]
    paired["timestamp_minute"] = paired["timestamp"].dt.floor("min")

    date = _date_from_path(Path(th1_path), th1["time"])
    minute_index = pd.date_range(date, date + pd.Timedelta(days=1) - pd.Timedelta(minutes=1), freq="min")
    grouped = paired.groupby("timestamp_minute", sort=True)
    scalar_cols = [
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
    ]
    scalar = grouped[scalar_cols].median().reindex(minute_index)
    scalar.index.name = "timestamp"
    scalar = scalar.reset_index()
    scalar["usable_opdi"] = scalar["js_opdi"].notna()

    minute_codes = pd.Categorical(paired["timestamp_minute"], categories=minute_index).codes
    valid_rows = minute_codes >= 0
    minute_codes = minute_codes[valid_rows]
    i1 = paired.loc[valid_rows, "i1"].to_numpy(int)
    i2 = paired.loc[valid_rows, "i2"].to_numpy(int)
    spectra = {
        "time": minute_index.to_numpy(dtype="datetime64[ms]"),
        "energy": np.asarray(grid, dtype=float),
        "th1_flux": _nanmedian_stack(th1["flux"], i1, minute_codes, len(minute_index)),
        "th2_flux": _nanmedian_stack(th2["flux"], i2, minute_codes, len(minute_index)),
        "th1_probability": _nanmedian_stack(th1["probability"], i1, minute_codes, len(minute_index)),
        "th2_probability": _nanmedian_stack(th2["probability"], i2, minute_codes, len(minute_index)),
    }

    report = {
        "records": int(min(th1["native_records"], th2["native_records"])),
        "start": str(pd.Timestamp(paired["timestamp"].min())) if len(paired) else None,
        "end": str(pd.Timestamp(paired["timestamp"].max())) if len(paired) else None,
        "native_cadence_seconds": float(paired["timestamp"].diff().dt.total_seconds().median()) if len(paired) > 1 else None,
        "usable_fraction": float(scalar["usable_opdi"].mean()),
        "th1_finite_flux_fraction": th1["finite_flux_fraction"],
        "th2_finite_flux_fraction": th2["finite_flux_fraction"],
    }
    return scalar, spectra, report
