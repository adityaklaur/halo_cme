from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Phase6Paths:
    predictions: Path
    fold_metrics: Path
    summary_metrics: Path
    detection_delays: Path
    feature_importance: Path
    feature_selection: Path
    feature_list: Path
    report: Path
    models_dir: Path


def output_paths(output_dir: Path) -> Phase6Paths:
    return Phase6Paths(
        predictions=output_dir / "phase6_predictions.csv",
        fold_metrics=output_dir / "phase6_fold_metrics.csv",
        summary_metrics=output_dir / "phase6_summary_metrics.csv",
        detection_delays=output_dir / "phase6_detection_delays.csv",
        feature_importance=output_dir / "phase6_feature_importance.csv",
        feature_selection=output_dir / "phase6_feature_selection.csv",
        feature_list=output_dir / "phase6_feature_list.csv",
        report=output_dir / "phase6_report.json",
        models_dir=output_dir / "models",
    )


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False}).fillna(False)


def _safe_pr_auc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    y = y[valid].astype(int)
    score = score[valid].astype(float)
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(y, score)
    order = np.argsort(recall)
    return float(auc(recall[order], precision[order]))


def _safe_roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    valid = np.isfinite(score)
    y = y[valid].astype(int)
    score = score[valid].astype(float)
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


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


def _false_alarm_episodes(frame: pd.DataFrame) -> int:
    total = 0
    negatives = frame.loc[frame["target_binary"] == 0].copy()
    for _, g in negatives.groupby("event_id", sort=False):
        pred = g.sort_values("timestamp")["predicted_binary"].astype(bool).to_numpy()
        if len(pred):
            total += int(pred[0]) + int(np.sum(pred[1:] & ~pred[:-1]))
    return total


def _event_detection_rows(frame: pd.DataFrame, model: str, fold: str) -> list[dict]:
    rows: list[dict] = []
    for event_id, event in frame.groupby("event_id", sort=False):
        event = event.sort_values("timestamp")
        positive = event.loc[event["target_binary"] == 1]
        if positive.empty:
            continue
        reference = positive["timestamp"].min()
        first_event_time = event["timestamp"].min()
        onset_is_internal = bool(reference > first_event_time)
        detected = event.loc[event["predicted_binary"].astype(bool)]
        detected_at = detected["timestamp"].min() if not detected.empty else pd.NaT
        if onset_is_internal:
            detected_after = event.loc[
                (event["timestamp"] >= reference) & event["predicted_binary"].astype(bool)
            ]
            detected_for_delay = detected_after["timestamp"].min() if not detected_after.empty else pd.NaT
        else:
            detected_for_delay = pd.NaT
        delay = (
            (detected_for_delay - reference).total_seconds() / 60.0
            if pd.notna(detected_for_delay)
            else np.nan
        )
        rows.append(
            {
                "model": model,
                "fold": fold,
                "event_id": event_id,
                "independent_interval_id": event["independent_interval_id"].iloc[0],
                "reference_positive_onset": reference if onset_is_internal else pd.NaT,
                "onset_reference_status": (
                    "INTERNAL_LABELED_BOUNDARY" if onset_is_internal else "WINDOW_START_NOT_EXACT_ONSET"
                ),
                "detected_at": detected_at,
                "detected": bool(pd.notna(detected_at)),
                "detection_delay_minutes": delay,
            }
        )
    return rows


def _metrics(frame: pd.DataFrame, model: str, fold: str) -> tuple[dict, list[dict]]:
    valid = frame["probability"].notna()
    y = frame.loc[valid, "target_binary"].astype(int).to_numpy()
    pred = frame.loc[valid, "predicted_binary"].astype(int).to_numpy()
    score = frame.loc[valid, "probability"].to_numpy(float)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    if len(y):
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    else:
        tn = fp = fn = tp = 0
    delays = _event_detection_rows(frame.loc[valid], model, fold)
    positive_events = len(delays)
    detected_events = sum(int(row["detected"]) for row in delays)
    delay_values = [row["detection_delay_minutes"] for row in delays if np.isfinite(row["detection_delay_minutes"])]
    false_episodes = _false_alarm_episodes(frame.loc[valid])
    negative_minutes = int((frame.loc[valid, "target_binary"] == 0).sum())
    negative_days = negative_minutes / 1440.0
    return (
        {
            "model": model,
            "fold": fold,
            "records": int(valid.sum()),
            "positive_minutes": int(y.sum()) if len(y) else 0,
            "negative_minutes": int(len(y) - y.sum()) if len(y) else 0,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": _safe_pr_auc(y, score),
            "roc_auc": _safe_roc_auc(y, score),
            "false_alarm_episodes": int(false_episodes),
            "false_alarms_per_day": false_episodes / negative_days if negative_days > 0 else np.nan,
            "positive_events": int(positive_events),
            "detected_events": int(detected_events),
            "detection_rate": detected_events / positive_events if positive_events else np.nan,
            "median_detection_delay_minutes": float(np.median(delay_values)) if delay_values else np.nan,
            "missed_events": int(positive_events - detected_events),
        },
        delays,
    )


def _make_pipeline(model_name: str, spec: dict, random_state: int) -> Pipeline:
    model_type = str(spec.get("type", "")).strip().lower()
    if model_type == "logistic_regression":
        model = LogisticRegression(
            C=float(spec.get("C", 1.0)),
            max_iter=int(spec.get("max_iter", 2500)),
            class_weight=spec.get("class_weight", "balanced"),
            solver="lbfgs",
            random_state=random_state,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=int(spec.get("n_estimators", 300)),
            min_samples_leaf=int(spec.get("min_samples_leaf", 5)),
            max_features=spec.get("max_features", "sqrt"),
            class_weight=spec.get("class_weight", "balanced_subsample"),
            random_state=random_state,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if model_type == "hist_gradient_boosting":
        model = HistGradientBoostingClassifier(
            max_iter=int(spec.get("max_iter", 120)),
            learning_rate=float(spec.get("learning_rate", 0.06)),
            max_leaf_nodes=int(spec.get("max_leaf_nodes", 15)),
            min_samples_leaf=int(spec.get("min_samples_leaf", 20)),
            l2_regularization=float(spec.get("l2_regularization", 1.0)),
            class_weight="balanced",
            random_state=random_state,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(f"Unsupported Phase 6 model type for {model_name}: {model_type}")


def _select_features(train: pd.DataFrame, features: list[str], minimum_fraction: float) -> tuple[list[str], list[dict]]:
    selected: list[str] = []
    rows: list[dict] = []
    for feature in features:
        series = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        available = float(series.notna().mean())
        unique = int(series.nunique(dropna=True))
        keep = available >= minimum_fraction and unique > 1
        reason = "SELECTED" if keep else ("LOW_AVAILABILITY" if available < minimum_fraction else "CONSTANT_OR_EMPTY")
        rows.append(
            {
                "feature": feature,
                "training_available_fraction": available,
                "training_unique_values": unique,
                "selected": bool(keep),
                "reason": reason,
            }
        )
        if keep:
            selected.append(feature)
    if not selected:
        raise ValueError("No Phase 4 features survived the train-only Phase 6 feature filter")
    return selected, rows


def _feature_importance_rows(pipeline: Pipeline, selected: list[str], model_name: str, fold: str) -> list[dict]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "coef_"):
        signed = np.asarray(model.coef_).reshape(-1)
        raw = np.abs(signed)
    elif hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
        signed = np.full_like(raw, np.nan, dtype=float)
    else:
        return []
    total = float(np.nansum(raw))
    normalized = raw / total if total > 0 else raw
    return [
        {
            "model": model_name,
            "fold": fold,
            "feature": feature,
            "importance": float(value),
            "normalized_importance": float(norm),
            "signed_coefficient": float(sign) if np.isfinite(sign) else np.nan,
        }
        for feature, value, norm, sign in zip(selected, raw, normalized, signed)
    ]


def build_phase6_ml(
    root: Path,
    config_path: Path,
    input_path: Path,
    feature_dictionary_path: Path,
    output_dir: Path,
) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(input_path, parse_dates=["timestamp"], low_memory=False)
    dictionary = pd.read_csv(feature_dictionary_path)

    target = str(config.get("target_column", "icme_binary"))
    modeling_flag = str(config.get("modeling_flag", "eligible_for_exploratory_modeling"))
    ready_flag = str(config.get("phase4_ready_flag", "phase4_ready_for_exploratory_ablation"))
    minimum_fraction = float(config.get("minimum_training_feature_fraction", 0.50))
    threshold = float(config.get("prediction_threshold", 0.50))
    persistence = int(config.get("persistence_minutes", 3))
    random_state = int(config.get("random_state", 42))
    models: dict[str, dict] = config.get("models") or {}
    expected_models = {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    if set(models) != expected_models:
        raise ValueError("Phase 6 must define Logistic Regression, Random Forest, and Gradient Boosting")

    if "uses_ground_truth_label" not in dictionary.columns:
        raise ValueError("Phase 4 feature dictionary must declare uses_ground_truth_label")
    label_free = ~_as_bool(dictionary["uses_ground_truth_label"])
    features = dictionary.loc[label_free, "column"].astype(str).drop_duplicates().tolist()
    forbidden = {target, "research_label", "ground_truth_state", "phase3_label_confidence", "phase3_policy"}
    leak_features = sorted(set(features) & forbidden)
    if leak_features:
        raise ValueError("Ground-truth or target columns found in Phase 6 feature list: " + ", ".join(leak_features))

    required = {"event_id", "independent_interval_id", "timestamp", "research_label", target, modeling_flag, ready_flag, *features}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("Phase 6 input is missing required columns: " + ", ".join(missing))

    eligible = _as_bool(frame[modeling_flag]) & _as_bool(frame[ready_flag])
    data = frame.loc[eligible].copy().sort_values(["independent_interval_id", "event_id", "timestamp"])
    data[target] = pd.to_numeric(data[target], errors="coerce").fillna(0).astype(int)
    data[features] = data[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    intervals = sorted(data["independent_interval_id"].dropna().unique().tolist())
    if len(intervals) < 3:
        raise ValueError("Phase 6 requires at least three independent ready intervals")

    predictions: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    delay_rows: list[dict] = []
    importance_rows: list[dict] = []
    selection_rows: list[dict] = []
    leakage_checks: list[dict] = []

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
            raise ValueError(f"Training fold for {held_out} does not contain both classes")
        selected, selection = _select_features(train, features, minimum_fraction)
        for item in selection:
            item["fold"] = held_out
        selection_rows.extend(selection)
        X_train = train[selected]
        y_train = train[target].astype(int)
        X_test = test[selected]

        for model_name, model_spec in models.items():
            pipeline = _make_pipeline(model_name, model_spec, random_state)
            pipeline.fit(X_train, y_train)
            probability = pipeline.predict_proba(X_test)[:, 1]
            pred_frame = test[["event_id", "independent_interval_id", "timestamp", "research_label", target]].copy()
            pred_frame = pred_frame.rename(columns={target: "target_binary"})
            pred_frame["model"] = model_name
            pred_frame["fold"] = held_out
            pred_frame["probability"] = probability
            pred_frame["threshold"] = threshold
            raw = pd.Series(probability >= threshold, index=pred_frame.index)
            pred_frame["raw_predicted_binary"] = raw.to_numpy(bool)
            pred_frame["predicted_binary"] = _persist_predictions(pred_frame, raw, persistence).to_numpy(bool)
            predictions.append(pred_frame)

            metric, delays = _metrics(pred_frame, model_name, held_out)
            metric["selected_features"] = len(selected)
            metric["train_rows"] = int(len(train))
            metric["train_positive_rows"] = int(y_train.sum())
            metric["train_negative_rows"] = int(len(y_train) - y_train.sum())
            fold_metrics.append(metric)
            delay_rows.extend(delays)
            importance_rows.extend(_feature_importance_rows(pipeline, selected, model_name, held_out))

    prediction_table = pd.concat(predictions, ignore_index=True)
    fold_metric_table = pd.DataFrame(fold_metrics)
    delay_table = pd.DataFrame(delay_rows)
    selection_table = pd.DataFrame(selection_rows)
    importance_table = pd.DataFrame(importance_rows)

    summary_rows: list[dict] = []
    for model_name, model_frame in prediction_table.groupby("model", sort=False):
        metric, _ = _metrics(model_frame, model_name, "ALL_EVENTWISE_FOLDS")
        summary_rows.append(metric)
    summary = pd.DataFrame(summary_rows)
    primary_metric = str(config.get("primary_metric", "pr_auc"))
    if primary_metric not in summary.columns:
        raise ValueError(f"Unknown Phase 6 primary metric: {primary_metric}")
    ranking = summary.sort_values([primary_metric, "f1"], ascending=False, na_position="last")["model"].tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(paths.predictions, index=False)
    fold_metric_table.to_csv(paths.fold_metrics, index=False)
    summary.to_csv(paths.summary_metrics, index=False)
    delay_table.to_csv(paths.detection_delays, index=False)
    importance_table.to_csv(paths.feature_importance, index=False)
    selection_table.to_csv(paths.feature_selection, index=False)
    pd.DataFrame(
        {
            "feature": features,
            "feature_group": dictionary.set_index("column").reindex(features)["feature_group"].to_numpy(),
            "uses_ground_truth_label": False,
        }
    ).to_csv(paths.feature_list, index=False)

    # Fit deployable exploratory artifacts on all currently eligible data. These are
    # convenience artifacts only; scientific performance is taken from held-out folds above.
    final_selected, _ = _select_features(data, features, minimum_fraction)
    X_all = data[final_selected]
    y_all = data[target].astype(int)
    model_artifacts: dict[str, str] = {}
    for model_name, model_spec in models.items():
        pipeline = _make_pipeline(model_name, model_spec, random_state)
        pipeline.fit(X_all, y_all)
        slug = model_name.lower().replace(" ", "_")
        artifact = paths.models_dir / f"{slug}.joblib"
        joblib.dump({"pipeline": pipeline, "features": final_selected, "threshold": threshold, "persistence_minutes": persistence}, artifact)
        model_artifacts[model_name] = _display_path(root, artifact)

    report = {
        "schema_version": int(config.get("schema_version", 1)),
        "phase": 6,
        "experiment_name": config.get("experiment_name", "Phase 6 baseline machine-learning comparison"),
        "status": "COMPLETE_EXPLORATORY_BASELINE_ML",
        "evidence_status": "EXPLORATORY_NOT_RESEARCH_READY",
        "target": target,
        "eligible_rows": int(len(data)),
        "independent_ready_intervals": intervals,
        "folds": len(intervals),
        "models": models,
        "feature_source": "All label-free features declared by the Phase 4 feature dictionary",
        "candidate_feature_count": len(features),
        "final_selected_feature_count": len(final_selected),
        "prediction_threshold": threshold,
        "persistence_minutes": persistence,
        "primary_metric": primary_metric,
        "model_ranking": ranking,
        "summary_metrics": summary.replace({np.nan: None}).to_dict(orient="records"),
        "leakage_checks": leakage_checks,
        "model_artifacts": model_artifacts,
        "scientific_guardrails": [
            "Independent source intervals, not individual minutes, are held out for evaluation.",
            "Feature availability filtering, imputation, scaling, class balancing, and model fitting are learned only from each training fold.",
            "Only Phase 4 features explicitly marked as not using ground-truth labels are admitted to the model matrix.",
            "The same probability threshold and temporal persistence rule are applied to all three models for a transparent baseline comparison.",
            "November orientation-control rows remain excluded because required Aditya-L1 modalities are incomplete.",
            "The current four ready independent intervals are too few for confirmatory model-selection claims; Phase 6 results are exploratory.",
            "Saved full-data models are convenience artifacts, not deployment-ready or independently validated models.",
            "Detection timing is reported as detection delay/offset, not early warning.",
        ],
        "outputs": {
            key: _display_path(root, value)
            for key, value in paths.__dict__.items()
            if key != "models_dir"
        },
    }
    paths.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
