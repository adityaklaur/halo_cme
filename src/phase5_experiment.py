from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Phase5Paths:
    predictions: Path
    fold_metrics: Path
    summary_metrics: Path
    detection_delays: Path
    report: Path


def output_paths(output_dir: Path) -> Phase5Paths:
    return Phase5Paths(
        predictions=output_dir / "phase5_predictions.csv",
        fold_metrics=output_dir / "phase5_fold_metrics.csv",
        summary_metrics=output_dir / "phase5_summary_metrics.csv",
        detection_delays=output_dir / "phase5_detection_delays.csv",
        report=output_dir / "phase5_report.json",
    )


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def _robust_parameters(frame: pd.DataFrame, features: list[str]) -> tuple[pd.Series, pd.Series]:
    center = frame[features].median(axis=0, skipna=True)
    mad = (frame[features] - center).abs().median(axis=0, skipna=True)
    scale = 1.4826 * mad
    iqr = frame[features].quantile(0.75) - frame[features].quantile(0.25)
    fallback_iqr = iqr / 1.349
    std = frame[features].std(axis=0, ddof=0)
    scale = scale.where(scale > 1e-12, fallback_iqr)
    scale = scale.where(scale > 1e-12, std)
    scale = scale.where(scale > 1e-12, 1.0).fillna(1.0)
    center = center.fillna(0.0)
    return center, scale


def _feature_score(frame: pd.DataFrame, features: list[str], center: pd.Series, scale: pd.Series, minimum_fraction: float) -> pd.Series:
    z = (frame[features] - center[features]) / scale[features]
    available = z.notna().mean(axis=1)
    score = z.abs().median(axis=1, skipna=True)
    return score.where(available >= minimum_fraction)


def _persist_predictions(frame: pd.DataFrame, raw: pd.Series, minutes: int) -> pd.Series:
    if minutes <= 1:
        return raw.fillna(False).astype(bool)
    result = pd.Series(False, index=frame.index)
    for _, index in frame.groupby("event_id", sort=False).groups.items():
        flags = raw.loc[index].fillna(False).astype(bool).to_numpy()
        run = 0
        out = np.zeros(len(flags), dtype=bool)
        for i, flag in enumerate(flags):
            run = run + 1 if flag else 0
            if run >= minutes:
                out[i] = True
        result.loc[index] = out
    return result


def _precision_recall_f1(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y = y.astype(int)
    pred = pred.astype(int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    y = y[valid].astype(int)
    score = score[valid].astype(float)
    positives = int(y.sum())
    if positives == 0 or positives == len(y):
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def _false_alarm_episodes(frame: pd.DataFrame) -> int:
    total = 0
    negatives = frame.loc[frame["target_binary"] == 0].copy()
    for _, g in negatives.groupby("event_id", sort=False):
        pred = g.sort_values("timestamp")["predicted_binary"].astype(bool).to_numpy()
        if len(pred):
            total += int(pred[0]) + int(np.sum(pred[1:] & ~pred[:-1]))
    return total


def _event_detection_rows(frame: pd.DataFrame, mode: str, fold: str) -> list[dict]:
    rows = []
    for event_id, event in frame.groupby("event_id", sort=False):
        event = event.sort_values("timestamp")
        positive = event.loc[event["target_binary"] == 1]
        if positive.empty:
            continue
        reference = positive["timestamp"].min()
        detected = event.loc[event["predicted_binary"].astype(bool)]
        detected_at = detected["timestamp"].min() if not detected.empty else pd.NaT
        # A delay is meaningful only when the positive boundary occurs inside the
        # registered event window. Constant positive windows (for example the
        # approximate October complex-ICME window) do not provide an exact onset.
        first_event_time = event["timestamp"].min()
        onset_is_internal = bool(reference > first_event_time)
        if onset_is_internal and pd.notna(detected_at):
            detected_after_reference = event.loc[
                (event["timestamp"] >= reference) & event["predicted_binary"].astype(bool)
            ]
            detected_for_delay = detected_after_reference["timestamp"].min() if not detected_after_reference.empty else pd.NaT
        else:
            detected_for_delay = pd.NaT
        delay = (detected_for_delay - reference).total_seconds() / 60.0 if pd.notna(detected_for_delay) else np.nan
        rows.append(
            {
                "mode": mode,
                "fold": fold,
                "event_id": event_id,
                "independent_interval_id": event["independent_interval_id"].iloc[0],
                "reference_positive_onset": reference if onset_is_internal else pd.NaT,
                "onset_reference_status": "INTERNAL_LABELED_BOUNDARY" if onset_is_internal else "WINDOW_START_NOT_EXACT_ONSET",
                "detected_at": detected_at,
                "detected": bool(pd.notna(detected_at)),
                "detection_delay_minutes": delay,
            }
        )
    return rows


def _metrics(frame: pd.DataFrame, mode: str, fold: str) -> tuple[dict, list[dict]]:
    valid = frame["score"].notna()
    y = frame.loc[valid, "target_binary"].astype(int).to_numpy()
    pred = frame.loc[valid, "predicted_binary"].astype(bool).to_numpy()
    score = frame.loc[valid, "score"].to_numpy(float)
    precision, recall, f1 = _precision_recall_f1(y, pred)
    delays = _event_detection_rows(frame.loc[valid], mode, fold)
    positive_events = len(delays)
    detected_events = sum(int(row["detected"]) for row in delays)
    delay_values = [row["detection_delay_minutes"] for row in delays if np.isfinite(row["detection_delay_minutes"])]
    false_episodes = _false_alarm_episodes(frame.loc[valid])
    negative_minutes = int((frame.loc[valid, "target_binary"] == 0).sum())
    negative_days = negative_minutes / 1440.0
    row = {
        "mode": mode,
        "fold": fold,
        "records": int(valid.sum()),
        "positive_minutes": int(y.sum()),
        "negative_minutes": int(len(y) - y.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": _pr_auc(y, score),
        "false_alarm_episodes": int(false_episodes),
        "false_alarms_per_day": false_episodes / negative_days if negative_days > 0 else np.nan,
        "positive_events": positive_events,
        "detected_events": detected_events,
        "detection_rate": detected_events / positive_events if positive_events else np.nan,
        "median_detection_delay_minutes": float(np.median(delay_values)) if delay_values else np.nan,
        "missed_events": positive_events - detected_events,
    }
    return row, delays



def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)

def build_phase5_experiment(root: Path, config_path: Path, input_path: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(input_path, parse_dates=["timestamp"], low_memory=False)
    target = str(config.get("target_column", "icme_binary"))
    modeling_flag = str(config.get("modeling_flag", "eligible_for_exploratory_modeling"))
    quantile = float(config.get("negative_score_quantile", 0.99))
    persistence = int(config.get("persistence_minutes", 3))
    minimum_fraction = float(config.get("minimum_feature_fraction", 0.70))
    modes: dict[str, list[str]] = config.get("modes") or {}
    if set(modes) != {"Conventional", "OPDI only", "Combined"}:
        raise ValueError("Phase 5 must define exactly Conventional, OPDI only, and Combined modes")

    required = {"event_id", "independent_interval_id", "timestamp", target, modeling_flag}
    for features in modes.values():
        required.update(features)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Phase 5 input is missing required columns: " + ", ".join(missing))

    eligible = _as_bool(frame[modeling_flag])
    if "phase4_ready_for_exploratory_ablation" in frame.columns:
        eligible &= _as_bool(frame["phase4_ready_for_exploratory_ablation"])
    data = frame.loc[eligible].copy().sort_values(["independent_interval_id", "event_id", "timestamp"])
    data[target] = pd.to_numeric(data[target], errors="coerce").fillna(0).astype(int)
    intervals = sorted(data["independent_interval_id"].unique().tolist())
    if len(intervals) < 3:
        raise ValueError("Phase 5 requires at least three independent ready intervals for exploratory event-wise evaluation")

    predictions = []
    fold_metrics = []
    delay_rows = []
    leakage_checks = []

    conventional_features = modes["Conventional"]
    opdi_features = modes["OPDI only"]

    for held_out in intervals:
        train = data.loc[data["independent_interval_id"] != held_out].copy()
        test = data.loc[data["independent_interval_id"] == held_out].copy()
        train_intervals = sorted(train["independent_interval_id"].unique().tolist())
        leakage_checks.append(
            {
                "held_out_interval": held_out,
                "train_intervals": train_intervals,
                "test_intervals": [held_out],
                "interval_leakage": held_out in train_intervals,
            }
        )
        if train[target].nunique() < 2:
            raise ValueError(f"Training fold for {held_out} does not contain both target classes")
        negative_train = train.loc[train[target] == 0]
        if negative_train.empty:
            raise ValueError(f"Training fold for {held_out} has no negative calibration rows")

        centers: dict[str, pd.Series] = {}
        scales: dict[str, pd.Series] = {}
        for name, features in [("Conventional", conventional_features), ("OPDI only", opdi_features)]:
            centers[name], scales[name] = _robust_parameters(negative_train, features)

        conv_train = _feature_score(train, conventional_features, centers["Conventional"], scales["Conventional"], minimum_fraction)
        conv_test = _feature_score(test, conventional_features, centers["Conventional"], scales["Conventional"], minimum_fraction)
        opdi_train = _feature_score(train, opdi_features, centers["OPDI only"], scales["OPDI only"], minimum_fraction)
        opdi_test = _feature_score(test, opdi_features, centers["OPDI only"], scales["OPDI only"], minimum_fraction)
        mode_scores_train = {
            "Conventional": conv_train,
            "OPDI only": opdi_train,
            "Combined": pd.concat([conv_train.rename("c"), opdi_train.rename("o")], axis=1).mean(axis=1, skipna=True),
        }
        mode_scores_test = {
            "Conventional": conv_test,
            "OPDI only": opdi_test,
            "Combined": pd.concat([conv_test.rename("c"), opdi_test.rename("o")], axis=1).mean(axis=1, skipna=True),
        }

        for mode in ["Conventional", "OPDI only", "Combined"]:
            train_score = mode_scores_train[mode]
            negative_scores = train_score.loc[train[target] == 0].dropna()
            if negative_scores.empty:
                raise ValueError(f"No finite negative calibration scores for {mode} / {held_out}")
            threshold = float(negative_scores.quantile(quantile))
            test_mode = test[["event_id", "independent_interval_id", "timestamp", "research_label", target]].copy()
            test_mode = test_mode.rename(columns={target: "target_binary"})
            test_mode["mode"] = mode
            test_mode["fold"] = held_out
            test_mode["score"] = mode_scores_test[mode].to_numpy()
            test_mode["threshold"] = threshold
            raw_pred = test_mode["score"] >= threshold
            test_mode["raw_threshold_exceeded"] = raw_pred.fillna(False)
            test_mode["predicted_binary"] = _persist_predictions(test_mode, raw_pred, persistence).to_numpy()
            predictions.append(test_mode)
            metric, delays = _metrics(test_mode, mode, held_out)
            metric["threshold"] = threshold
            metric["train_negative_rows"] = int(len(negative_train))
            metric["train_positive_rows"] = int((train[target] == 1).sum())
            fold_metrics.append(metric)
            delay_rows.extend(delays)

    prediction_table = pd.concat(predictions, ignore_index=True)
    fold_metric_table = pd.DataFrame(fold_metrics)
    delay_table = pd.DataFrame(delay_rows)
    summary_rows = []
    for mode, mode_frame in prediction_table.groupby("mode", sort=False):
        metric, delays = _metrics(mode_frame, mode, "ALL_EVENTWISE_FOLDS")
        summary_rows.append(metric)
    summary = pd.DataFrame(summary_rows)

    conventional = summary.set_index("mode").loc["Conventional"]
    opdi = summary.set_index("mode").loc["OPDI only"]
    combined = summary.set_index("mode").loc["Combined"]
    improvements = {
        "opdi_minus_conventional_f1": float(opdi["f1"] - conventional["f1"]),
        "combined_minus_conventional_f1": float(combined["f1"] - conventional["f1"]),
        "opdi_minus_conventional_pr_auc": float(opdi["pr_auc"] - conventional["pr_auc"]),
        "combined_minus_conventional_pr_auc": float(combined["pr_auc"] - conventional["pr_auc"]),
        "combined_minus_conventional_detection_rate": float(combined["detection_rate"] - conventional["detection_rate"]),
        "combined_minus_conventional_false_alarms_per_day": float(combined["false_alarms_per_day"] - conventional["false_alarms_per_day"]),
    }
    if improvements["combined_minus_conventional_f1"] > 0 and improvements["combined_minus_conventional_pr_auc"] > 0:
        evidence_status = "EXPLORATORY_SUPPORT_FOR_ADDED_OPDI_INFORMATION"
    elif improvements["combined_minus_conventional_f1"] < 0 and improvements["combined_minus_conventional_pr_auc"] < 0:
        evidence_status = "EXPLORATORY_RESULTS_DO_NOT_SUPPORT_ADDED_OPDI_INFORMATION"
    else:
        evidence_status = "EXPLORATORY_MIXED_EVIDENCE"

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    prediction_table.to_csv(paths.predictions, index=False)
    fold_metric_table.to_csv(paths.fold_metrics, index=False)
    summary.to_csv(paths.summary_metrics, index=False)
    delay_table.to_csv(paths.detection_delays, index=False)

    report = {
        "schema_version": int(config.get("schema_version", 1)),
        "phase": 5,
        "experiment_name": config.get("experiment_name", "Phase 5 OPDI central hypothesis ablation"),
        "status": "COMPLETE_EXPLORATORY_EVENTWISE_ABLATION",
        "evidence_status": evidence_status,
        "target": target,
        "eligible_rows": int(len(data)),
        "independent_ready_intervals": intervals,
        "folds": len(intervals),
        "modes": modes,
        "calibration": {
            "method": "robust distance from training negative/control distribution",
            "negative_score_quantile": quantile,
            "persistence_minutes": persistence,
            "minimum_feature_fraction": minimum_fraction,
            "combined_score": "equal-weight mean of conventional-group and OPDI-group robust anomaly scores",
        },
        "summary_metrics": summary.replace({np.nan: None}).to_dict(orient="records"),
        "improvements_vs_conventional": improvements,
        "leakage_checks": leakage_checks,
        "scientific_guardrails": [
            "Independent source intervals, not individual minutes, are held out for evaluation.",
            "Thresholds and robust scaling are calibrated only on training-fold negative/control rows.",
            "This is a transparent statistical ablation, not the Phase 6 machine-learning model.",
            "November orientation-control rows are excluded because Phase 2/3 mark their Aditya-L1 modalities as incomplete.",
            "Results are exploratory because the current Phase 2/3 dataset is not fully research-ready and some event boundaries are provisional or approximate.",
            "Detection delay is calculated only when a positive boundary occurs inside the registered event window; constant positive windows without an exact onset are excluded from delay statistics.",
            "Detection timing is not called early warning.",
        ],
        "outputs": {key: _display_path(root, value) for key, value in paths.__dict__.items()},
    }
    paths.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
