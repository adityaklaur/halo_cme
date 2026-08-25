from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


def _clean(values, sentinel: bool = True) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    out[~np.isfinite(out)] = np.nan
    if sentinel:
        out[np.abs(out) >= 9999] = np.nan
    out[np.abs(out) > 1e20] = np.nan
    return out


def resample_mag_minute(path: Path) -> pd.DataFrame:
    """Read Aditya-L1 MAG L2 NetCDF and resample 10-second vectors to one minute."""
    with Dataset(str(path)) as ds:
        time = pd.to_datetime(_clean(ds.variables["time"][:], sentinel=False), unit="s", utc=True).tz_convert(None)
        data = pd.DataFrame({"timestamp": time})
        for column in ["Bx_gse", "By_gse", "Bz_gse", "Bx_gsm", "By_gsm", "Bz_gsm"]:
            data[column] = _clean(ds.variables[column][:])
        if "Quality_flag_10s_data" in ds.variables:
            data["mag_quality_flag"] = _clean(ds.variables["Quality_flag_10s_data"][:])

    if "mag_quality_flag" in data:
        bad_quality = data["mag_quality_flag"].isna() | (data["mag_quality_flag"] < 0.5)
        vector_columns = ["Bx_gse", "By_gse", "Bz_gse", "Bx_gsm", "By_gsm", "Bz_gsm"]
        data.loc[bad_quality, vector_columns] = np.nan

    data["bmag_gse"] = np.sqrt(data["Bx_gse"] ** 2 + data["By_gse"] ** 2 + data["Bz_gse"] ** 2)
    data["bmag_gsm"] = np.sqrt(data["Bx_gsm"] ** 2 + data["By_gsm"] ** 2 + data["Bz_gsm"] ** 2)
    data["timestamp"] = data["timestamp"].dt.floor("min")
    numeric = [c for c in data.columns if c != "timestamp"]
    return data.groupby("timestamp", as_index=False)[numeric].median()
