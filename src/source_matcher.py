from __future__ import annotations

import re

import numpy as np
import pandas as pd


AU_KM = 149_597_870.7


def _source_direction_score(location: object) -> float:
    text = "" if pd.isna(location) else str(location).upper().strip()
    match = re.match(r"([NS])(\d{1,2})([EW])(\d{1,2})", text)
    if not match:
        return 0.5
    lat = int(match.group(2))
    lon = int(match.group(4))
    # Earth-directed events near disk center are favored; west events still remain plausible.
    lat_score = max(0.0, 1.0 - lat / 90.0)
    lon_score = max(0.0, 1.0 - lon / 90.0)
    return float(0.5 * lat_score + 0.5 * lon_score)


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "halo"}


def rank_cme_candidates(shock_time: pd.Timestamp, candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    out["cme_time"] = pd.to_datetime(out["cme_time"])
    shock_time = pd.Timestamp(shock_time)
    out["observed_transit_hours"] = (shock_time - out["cme_time"]).dt.total_seconds() / 3600.0
    speed = pd.to_numeric(out.get("space_speed_km_s"), errors="coerce")
    speed = speed.fillna(pd.to_numeric(out.get("linear_speed_km_s"), errors="coerce"))
    out["ballistic_transit_hours"] = AU_KM / speed / 3600.0

    transit_error = (out["observed_transit_hours"] - out["ballistic_transit_hours"]).abs()
    out["transit_score"] = np.exp(-transit_error / 30.0).clip(0, 1)
    out["direction_score"] = out["source_location"].apply(_source_direction_score)
    out["halo_score"] = out["halo"].apply(lambda x: 1.0 if _truthy(x) else 0.2)
    out["speed_score"] = ((speed - 250.0) / 1000.0).clip(0, 1)
    out["compatibility_score"] = (
        0.45 * out["transit_score"]
        + 0.25 * out["direction_score"]
        + 0.20 * out["halo_score"]
        + 0.10 * out["speed_score"]
    )
    return out.sort_values("compatibility_score", ascending=False).reset_index(drop=True)
