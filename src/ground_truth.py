from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ALLOWED_LABELS = {
    "QUIET",
    "SHOCK",
    "SHEATH",
    "ICME/EJECTA",
    "COMPLEX_ICME",
    "POST-ICME",
    "CIR/HSS",
    "ORIENTATION-CONTROL",
}
LABEL_ORDER = [
    "QUIET",
    "SHOCK",
    "SHEATH",
    "ICME/EJECTA",
    "COMPLEX_ICME",
    "POST-ICME",
    "CIR/HSS",
    "ORIENTATION-CONTROL",
]
REQUIRED_MODALITY_COLUMNS = [
    "proton_density",
    "proton_bulk_speed",
    "proton_thermal",
    "Bx_gse",
    "By_gse",
    "Bz_gse",
]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def _row_modalities_complete(frame: pd.DataFrame) -> pd.Series:
    usable = _as_bool(frame["usable_opdi"])
    computed = usable & frame[REQUIRED_MODALITY_COLUMNS].notna().all(axis=1)
    if "aditya_modalities_complete" not in frame:
        return computed
    supplied = frame["aditya_modalities_complete"]
    has_supplied = supplied.notna()
    supplied_bool = _as_bool(supplied)
    return supplied_bool.where(has_supplied, computed)


def _load_inputs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scientific = root / "data" / "scientific"
    features = pd.read_csv(
        scientific / "phase2_feature_table.csv",
        parse_dates=["timestamp"],
        low_memory=False,
    ).sort_values(["event_id", "timestamp"]).reset_index(drop=True)
    catalog = pd.read_csv(
        scientific / "phase2_event_catalog.csv",
        parse_dates=["start_utc", "end_utc"],
    )
    manifest = json.loads((scientific / "phase2_manifest.json").read_text(encoding="utf-8"))
    return features, catalog, manifest


def _load_policies(config_path: Path, catalog_ids: set[str]) -> tuple[dict, dict[str, dict]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rows = config.get("events") or []
    ids = [str(row.get("event_id", "")).strip() for row in rows]
    if any(not event_id for event_id in ids):
        raise ValueError("Every Phase 3 policy requires an event_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Phase 3 event_id values must be unique")
    configured = set(ids)
    if configured != catalog_ids:
        missing = sorted(catalog_ids - configured)
        extra = sorted(configured - catalog_ids)
        raise ValueError(f"Phase 3 policies do not match Phase 2 catalog; missing={missing}, extra={extra}")

    policies = {}
    for row in rows:
        policy = str(row.get("policy", "")).upper()
        if policy not in {"CONSTANT", "ICME_SUBSTRUCTURE"}:
            raise ValueError(f"Unsupported Phase 3 policy for {row['event_id']}: {policy}")
        normalized = dict(row)
        normalized["policy"] = policy
        if policy == "CONSTANT":
            label = str(row.get("constant_label", "")).upper()
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Unsupported constant label for {row['event_id']}: {label}")
            normalized["constant_label"] = label
        else:
            for name in ["shock_reference", "ejecta_start", "ejecta_end"]:
                normalized[name] = pd.Timestamp(row[name])
            if not (normalized["shock_reference"] < normalized["ejecta_start"] < normalized["ejecta_end"]):
                raise ValueError(f"Invalid substructure boundary order for {row['event_id']}")
        policies[row["event_id"]] = normalized
    return config, policies


def _assign_policy(window: pd.DataFrame, policy: dict) -> pd.Series:
    if policy["policy"] == "CONSTANT":
        return pd.Series(policy["constant_label"], index=window.index, dtype="object")

    shock = policy["shock_reference"]
    ejecta_start = policy["ejecta_start"]
    ejecta_end = policy["ejecta_end"]
    if not (
        window["timestamp"].min() <= shock < ejecta_start < ejecta_end
        <= window["timestamp"].max() + pd.Timedelta(minutes=1)
    ):
        raise ValueError(f"Substructure boundaries fall outside {policy['event_id']}")
    labels = pd.Series("UNKNOWN", index=window.index, dtype="object")
    labels.loc[window["timestamp"] < shock] = str(policy.get("pre_shock_label", "QUIET")).upper()
    labels.loc[window["timestamp"] == shock] = "SHOCK"
    labels.loc[(window["timestamp"] > shock) & (window["timestamp"] < ejecta_start)] = "SHEATH"
    labels.loc[(window["timestamp"] >= ejecta_start) & (window["timestamp"] < ejecta_end)] = "ICME/EJECTA"
    labels.loc[window["timestamp"] >= ejecta_end] = "POST-ICME"
    return labels


def _boundary_rows(catalog_row: pd.Series, policy: dict) -> list[dict]:
    source = catalog_row["label_source"]
    status = policy.get("boundary_status", catalog_row["label_status"])
    rows = [
        {
            "event_id": catalog_row["event_id"],
            "boundary_type": "EVENT_WINDOW_START",
            "boundary_time_utc": catalog_row["start_utc"],
            "scientific_status": status,
            "used_for_minute_labeling": True,
            "label_source": source,
            "notes": "Start of the registered Phase 2 scientific window.",
        },
        {
            "event_id": catalog_row["event_id"],
            "boundary_type": "EVENT_WINDOW_END",
            "boundary_time_utc": catalog_row["end_utc"],
            "scientific_status": status,
            "used_for_minute_labeling": True,
            "label_source": source,
            "notes": "Inclusive end of the registered Phase 2 scientific window.",
        },
    ]
    if policy["policy"] == "ICME_SUBSTRUCTURE":
        for key, boundary_type, status_key, note in [
            ("shock_reference", "SHOCK_REFERENCE", "shock_reference_status", "Minute isolated as SHOCK."),
            ("ejecta_start", "ICME_EJECTA_START", "ejecta_start_status", "Start of the half-open ICME/ejecta interval."),
            ("ejecta_end", "ICME_EJECTA_END", "ejecta_end_status", "End of ICME/ejecta; post-ICME begins here."),
        ]:
            rows.append(
                {
                    "event_id": catalog_row["event_id"],
                    "boundary_type": boundary_type,
                    "boundary_time_utc": policy[key],
                    "scientific_status": policy.get(status_key, status),
                    "used_for_minute_labeling": True,
                    "label_source": source,
                    "notes": note,
                }
            )
    for reference in policy.get("reference_boundaries") or []:
        rows.append(
            {
                "event_id": catalog_row["event_id"],
                "boundary_type": reference["boundary_type"],
                "boundary_time_utc": pd.Timestamp(reference["boundary_time_utc"]),
                "scientific_status": reference["scientific_status"],
                "used_for_minute_labeling": False,
                "label_source": source,
                "notes": reference.get("notes", ""),
            }
        )
    return rows


def build_phase3_ground_truth(root: Path, config_path: Path, output_dir: Path) -> dict:
    features, catalog, phase2_manifest = _load_inputs(root)
    if features[["event_id", "timestamp"]].duplicated().any():
        raise ValueError("Phase 2 feature table contains duplicate event/timestamp keys")
    config, policies = _load_policies(config_path, set(catalog["event_id"]))
    catalog = catalog.copy()
    catalog["phase2_ready"] = _as_bool(catalog["phase2_ready"])

    feature_parts = []
    event_rows = []
    boundary_rows = []
    for _, catalog_row in catalog.iterrows():
        event_id = catalog_row["event_id"]
        policy = policies[event_id]
        window = features.loc[features["event_id"] == event_id].copy()
        if len(window) != int(catalog_row["records_observed"]):
            raise ValueError(f"Phase 2 row count mismatch for {event_id}")
        window["prototype_state"] = window.get("ground_truth_state", pd.Series(index=window.index, dtype="object"))
        window["research_label"] = _assign_policy(window, policy)
        if (window["research_label"] == "UNKNOWN").any():
            raise ValueError(f"Unlabeled minutes remain for {event_id}")
        window["phase3_policy"] = policy["policy"]
        window["phase3_label_confidence"] = str(policy.get("label_confidence", "UNSPECIFIED")).upper()
        window["phase3_boundary_status"] = policy.get("boundary_status", catalog_row["label_status"])
        window["phase3_label_source"] = catalog_row["label_source"]
        window["phase2_window_ready"] = bool(catalog_row["phase2_ready"])
        window["row_modalities_complete"] = _row_modalities_complete(window)
        window["phase3_ready"] = bool(catalog_row["phase2_ready"])
        window["eligible_for_exploratory_modeling"] = window["phase3_ready"] & window["row_modalities_complete"]
        window["eligible_for_confirmatory_modeling"] = (
            window["eligible_for_exploratory_modeling"]
            & window["phase3_label_confidence"].eq("HIGH")
        )
        window["icme_binary"] = window["research_label"].isin(
            ["SHOCK", "SHEATH", "ICME/EJECTA", "COMPLEX_ICME"]
        ).astype(int)
        window["shock_binary"] = window["research_label"].eq("SHOCK").astype(int)
        window["solar_transient_binary"] = window["research_label"].isin(
            ["SHOCK", "SHEATH", "ICME/EJECTA", "COMPLEX_ICME", "CIR/HSS"]
        ).astype(int)
        window["event_positive_binary"] = window["sample_role"].eq("POSITIVE").astype(int)
        feature_parts.append(window)

        present_labels = [label for label in LABEL_ORDER if label in set(window["research_label"])]
        eligible = int(window["eligible_for_exploratory_modeling"].sum())
        event_rows.append(
            {
                "event_id": event_id,
                "title": catalog_row["title"],
                "independent_interval_id": catalog_row["independent_interval_id"],
                "event_class": catalog_row["event_class"],
                "sample_role": catalog_row["sample_role"],
                "event_start": window["timestamp"].min(),
                "event_end": window["timestamp"].max(),
                "records": len(window),
                "research_labels": "|".join(present_labels),
                "phase3_policy": policy["policy"],
                "label_confidence": str(policy.get("label_confidence", "UNSPECIFIED")).upper(),
                "boundary_status": policy.get("boundary_status", catalog_row["label_status"]),
                "phase2_ready": bool(catalog_row["phase2_ready"]),
                "phase3_valid": True,
                "phase3_ready": bool(catalog_row["phase2_ready"]),
                "eligible_records": eligible,
                "eligible_fraction": eligible / len(window) if len(window) else 0.0,
                "label_source": catalog_row["label_source"],
                "notes": policy.get("notes", catalog_row.get("notes", "")),
            }
        )
        boundary_rows.extend(_boundary_rows(catalog_row, policy))

    ground_truth = pd.concat(feature_parts, ignore_index=True).sort_values(
        ["independent_interval_id", "timestamp", "event_id"]
    ).reset_index(drop=True)
    event_register = pd.DataFrame(event_rows)
    boundary_register = pd.DataFrame(boundary_rows).sort_values(["event_id", "boundary_time_utc"])
    counts = (
        ground_truth.groupby("research_label", sort=False)
        .agg(
            n=("timestamp", "size"),
            events=("event_id", "nunique"),
            independent_intervals=("independent_interval_id", "nunique"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            phase3_ready_minutes=("phase3_ready", "sum"),
            eligible_minutes=("eligible_for_exploratory_modeling", "sum"),
        )
        .reset_index()
    )
    counts["research_label"] = pd.Categorical(counts["research_label"], LABEL_ORDER, ordered=True)
    counts = counts.sort_values("research_label").reset_index(drop=True)
    counts["research_label"] = counts["research_label"].astype(str)
    counts["fraction"] = counts["n"] / len(ground_truth)

    expected_event_counts = catalog.set_index("event_id")["records_observed"].astype(int).to_dict()
    actual_event_counts = ground_truth.groupby("event_id").size().astype(int).to_dict()
    checks = {
        "records": int(len(ground_truth)),
        "expected_phase2_records": int(phase2_manifest["one_minute_records"]),
        "record_count_matches_phase2": len(ground_truth) == int(phase2_manifest["one_minute_records"]),
        "event_windows": int(event_register["event_id"].nunique()),
        "independent_intervals": int(event_register["independent_interval_id"].nunique()),
        "duplicate_event_timestamps": int(ground_truth[["event_id", "timestamp"]].duplicated().sum()),
        "timestamps_monotonic_within_events": bool(
            ground_truth.groupby("event_id", sort=False)["timestamp"].apply(lambda values: values.is_monotonic_increasing).all()
        ),
        "unknown_labels": int((ground_truth["research_label"] == "UNKNOWN").sum()),
        "event_counts_match": expected_event_counts == actual_event_counts,
        "icme_binary_consistent": bool(
            (
                ground_truth["icme_binary"]
                == ground_truth["research_label"].isin(["SHOCK", "SHEATH", "ICME/EJECTA", "COMPLEX_ICME"]).astype(int)
            ).all()
        ),
        "shock_binary_consistent": bool(
            (ground_truth["shock_binary"] == ground_truth["research_label"].eq("SHOCK").astype(int)).all()
        ),
        "phase3_ready_events": int(event_register["phase3_ready"].sum()),
        "blocked_events": event_register.loc[~event_register["phase3_ready"], "event_id"].tolist(),
        "actual_label_counts": {key: int(value) for key, value in ground_truth["research_label"].value_counts().items()},
    }
    checks["phase3_valid"] = all(
        [
            checks["record_count_matches_phase2"],
            checks["duplicate_event_timestamps"] == 0,
            checks["timestamps_monotonic_within_events"],
            checks["unknown_labels"] == 0,
            checks["event_counts_match"],
            checks["icme_binary_consistent"],
            checks["shock_binary_consistent"],
        ]
    )
    research_ready = bool(checks["phase3_valid"] and event_register["phase3_ready"].all())
    status = "RESEARCH_READY" if research_ready else "COMPLETED_WITH_BLOCKED_PHASE2_SOURCE"
    report = {
        "schema_version": int(config.get("schema_version", 2)),
        "dataset_name": config.get("dataset_name", "TopoCross-SWIS Phase 3 multi-event ground-truth dataset"),
        "phase": 3,
        "status": status,
        "scope": "Phase 2 multi-event scientific dataset",
        "phase3_valid": bool(checks["phase3_valid"]),
        "research_ready": research_ready,
        "labels": [label for label in LABEL_ORDER if label in set(ground_truth["research_label"])],
        "validation": checks,
        "scientific_guardrails": [
            "Phase 3 consumes the Phase 2 event registry and never creates synthetic spacecraft measurements.",
            "Exact shock/sheath/ejecta substructure is assigned only to the configured August event.",
            "October retains a window-level COMPLEX_ICME label because its publication does not provide an exact Aditya-L1 shock minute.",
            "The November orientation label is retained, but its minutes are excluded from modeling while Phase 2 modalities are incomplete.",
            "NASA OMNI remains optional context and does not satisfy Aditya-L1 modality readiness.",
        ],
        "outputs": {
            "ground_truth_dataset": "outputs/phase3/phase3_ground_truth_dataset.csv",
            "label_counts": "outputs/phase3/phase3_label_counts.csv",
            "event_register": "outputs/phase3/phase3_event_register.csv",
            "boundary_register": "outputs/phase3/phase3_boundary_register.csv",
            "report": "outputs/phase3/phase3_report.json",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(output_dir / "phase3_ground_truth_dataset.csv", index=False)
    counts.to_csv(output_dir / "phase3_label_counts.csv", index=False)
    event_register.to_csv(output_dir / "phase3_event_register.csv", index=False)
    boundary_register.to_csv(output_dir / "phase3_boundary_register.csv", index=False)
    (output_dir / "phase3_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report

