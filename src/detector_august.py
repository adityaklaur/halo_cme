from __future__ import annotations

import numpy as np
import pandas as pd


ROBUST_COLUMNS = [
    "js_opdi",
    "hellinger_opdi",
    "wasserstein_opdi",
    "proton_bulk_speed",
    "proton_density",
    "proton_thermal",
    "alpha_proton_ratio",
    "bmag_gse",
    "Bx_gse",
    "By_gse",
    "Bz_gse",
]


def robust_center_scale(values) -> tuple[float, float]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if len(series) == 0:
        return np.nan, 1.0
    median = float(series.median())
    scale = float((series - median).abs().median() * 1.4826)
    if not np.isfinite(scale) or scale <= 0:
        scale = float(series.std()) if len(series) > 1 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return median, scale


def _transition_representatives(flag: np.ndarray, min_run: int = 2) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    change_points = np.zeros(len(flag), dtype=bool)
    runs = []
    start = None
    for i, value in enumerate(flag):
        if value and start is None:
            start = i
        at_end = i == len(flag) - 1
        if start is not None and ((not value) or at_end):
            end = i if (value and at_end) else i - 1
            if end - start + 1 >= min_run:
                # Upper midpoint: a 4-minute run at 12:17-12:20 is represented at 12:19.
                mid = (start + end + 1) // 2
                change_points[mid] = True
                runs.append((start, end, mid))
            start = None
    return change_points, runs


def _apply_state_machine(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    out = frame.copy()
    n = len(out)
    watch_threshold = float(config.get("watch_threshold_combined", 2.8122751666908123))
    transition_threshold = float(config.get("transition_threshold", 8.0))
    conventional_threshold = float(config.get("sustained_conventional_threshold", 3.0))
    alert_after = int(config.get("alert_after_transition_minutes", 1))
    candidate_window = int(config.get("candidate_window_minutes", 35))

    states = np.full(n, "NORMAL", dtype=object)
    raw_transition = (out["transition_score"] >= transition_threshold).fillna(False).to_numpy()
    watch = ((out["combined_anomaly_score"] >= watch_threshold).fillna(False).to_numpy()) | raw_transition
    states[watch] = "WATCH"

    # ALERT is based on confirmed persistent transitions, not the first raw
    # threshold crossing. This keeps early transition buildup in WATCH.
    alert = np.zeros(n, dtype=bool)
    for idx in np.where(out["is_change_point"].fillna(False).to_numpy())[0]:
        alert[idx : min(n, idx + alert_after + 1)] = True

    recent_transition = np.zeros(n, dtype=bool)
    for idx in np.where(out["is_change_point"].fillna(False).to_numpy())[0]:
        recent_transition[idx : min(n, idx + candidate_window + 1)] = True
    candidate = (
        recent_transition
        & (out["conventional_anomaly_score"].fillna(0).to_numpy() >= conventional_threshold)
        & ~alert
    )

    states[candidate] = "ICME CANDIDATE"
    states[alert] = "ALERT"
    out["state"] = states
    return out


def add_detector(
    frame: pd.DataFrame,
    baseline_start: str,
    baseline_end: str,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, object]]:
    out = frame.copy().sort_values("timestamp").reset_index(drop=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    baseline_start_ts = pd.Timestamp(baseline_start)
    baseline_end_ts = pd.Timestamp(baseline_end)
    baseline_mask = (out.timestamp >= baseline_start_ts) & (out.timestamp <= baseline_end_ts)

    baseline: dict[str, dict[str, float]] = {}
    for column in ROBUST_COLUMNS:
        median, scale = robust_center_scale(out.loc[baseline_mask, column])
        baseline[column] = {"median": median, "scale": scale}
        out["z_" + column] = (pd.to_numeric(out[column], errors="coerce") - median) / scale

    out["opdi_anomaly_score"] = (
        0.5 * out["z_js_opdi"].clip(lower=0)
        + 0.3 * out["z_hellinger_opdi"].clip(lower=0)
        + 0.2 * out["z_wasserstein_opdi"].clip(lower=0)
    )
    plasma_cols = ["proton_bulk_speed", "proton_density", "proton_thermal", "alpha_proton_ratio"]
    mag_cols = ["bmag_gse", "Bx_gse", "By_gse", "Bz_gse"]
    out["plasma_anomaly_score"] = pd.concat([out["z_" + c].abs() for c in plasma_cols], axis=1).mean(axis=1)
    out["mag_anomaly_score"] = pd.concat([out["z_" + c].abs() for c in mag_cols], axis=1).mean(axis=1)
    plasma_weight = float(config.get("conventional_plasma_weight", 0.925))
    out["conventional_anomaly_score"] = (
        plasma_weight * out["plasma_anomaly_score"]
        + (1.0 - plasma_weight) * out["mag_anomaly_score"]
    )
    out["combined_anomaly_score"] = pd.concat(
        [out["opdi_anomaly_score"], out["conventional_anomaly_score"]],
        axis=1,
    ).max(axis=1)

    transition = np.zeros(len(out), dtype=float)
    step_config = config["transition_step_baseline"]
    lag = int(config.get("transition_lag_minutes", 4))
    for column, step in step_config.items():
        delta = (pd.to_numeric(out[column], errors="coerce") - pd.to_numeric(out[column], errors="coerce").shift(lag)).abs()
        z = ((delta - float(step["median_step"])) / float(step["scale_step"])).clip(lower=0).fillna(0)
        component = float(step["weight"]) * z
        out[f"transition_component_{column}"] = component
        transition += component
    out["transition_score"] = transition

    transition_threshold = float(config.get("transition_threshold", 8.0))
    raw_transition = (out.transition_score >= transition_threshold).fillna(False).to_numpy()
    out["transition_threshold_exceeded"] = raw_transition
    change_points, runs = _transition_representatives(raw_transition, int(config.get("transition_min_run_minutes", 2)))
    out["is_change_point"] = change_points
    out = _apply_state_machine(out, config)

    opdi_q95 = float(out.loc[baseline_mask, "opdi_anomaly_score"].quantile(0.95))
    opdi_q99 = float(out.loc[baseline_mask, "opdi_anomaly_score"].quantile(0.99))
    conv_q99 = float(out.loc[baseline_mask, "conventional_anomaly_score"].quantile(0.99))
    combined_q95 = float(out.loc[baseline_mask, "combined_anomaly_score"].quantile(0.95))

    report = {
        "baseline_start": str(baseline_start_ts),
        "baseline_end": str(baseline_end_ts),
        "robust_baseline": baseline,
        "opdi_watch_threshold_q95": opdi_q95,
        "opdi_only_threshold_q99": opdi_q99,
        "conventional_threshold_q99": conv_q99,
        "combined_baseline_q95": combined_q95,
        "watch_threshold_combined": float(config.get("watch_threshold_combined", max(opdi_q95, combined_q95))),
        "transition_threshold": transition_threshold,
        "transition_lag_minutes": lag,
        "transition_min_run_minutes": int(config.get("transition_min_run_minutes", 2)),
        "transition_step_baseline": step_config,
        "persistent_transition_runs": int(len(runs)),
    }
    return out, report


def add_ground_truth(frame: pd.DataFrame, shock_time: str, sheath_end_icme_start: str, icme_end: str) -> pd.DataFrame:
    out = frame.copy()
    shock = pd.Timestamp(shock_time)
    ejecta_start = pd.Timestamp(sheath_end_icme_start)
    ejecta_end = pd.Timestamp(icme_end)
    out["ground_truth_state"] = "POST-EVENT"
    out.loc[out["timestamp"] < shock, "ground_truth_state"] = "QUIET/PRE-EVENT"
    out.loc[(out["timestamp"] >= shock) & (out["timestamp"] < ejecta_start), "ground_truth_state"] = "SHEATH"
    out.loc[(out["timestamp"] >= ejecta_start) & (out["timestamp"] < ejecta_end), "ground_truth_state"] = "ICME/EJECTA"
    return out


def evaluate_detection(frame: pd.DataFrame, shock_time: str) -> dict[str, object]:
    shock = pd.Timestamp(shock_time)
    cps = frame.loc[frame["is_change_point"].astype(bool), "timestamp"].copy()
    after = cps[cps >= shock]
    before = cps[cps < shock]
    nearest = None
    offset = None
    if len(cps):
        idx = (cps - shock).abs().idxmin()
        nearest = pd.Timestamp(cps.loc[idx])
        offset = float((nearest - shock).total_seconds() / 60)
    return {
        "change_points_total": int(len(cps)),
        "first_change_after_reference_shock": str(after.iloc[0]) if len(after) else None,
        "delay_minutes_first_after_reference": float((after.iloc[0] - shock).total_seconds() / 60) if len(after) else None,
        "nearest_change_to_reference_shock": str(nearest) if nearest is not None else None,
        "nearest_change_offset_minutes": offset,
        "change_points_before_reference_shock": int(len(before)),
        "change_points_after_reference_shock": int(len(after)),
    }
