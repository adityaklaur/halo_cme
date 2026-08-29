from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
import yaml


ALLOWED_EVENT_CLASSES = {
    "QUIET",
    "SHOCK",
    "SHEATH",
    "ICME",
    "COMPLEX_ICME",
    "POST_ICME",
    "CIR_SIR",
    "ORIENTATION_CONTROL",
    "OTHER_DISTURBANCE",
    "UNKNOWN",
}
ALLOWED_SAMPLE_ROLES = {"POSITIVE", "CONTROL"}
REQUIRED_MODALITY_COLUMNS = {
    "TH1": ["usable_opdi"],
    "TH2": ["usable_opdi"],
    "BLK": ["proton_density", "proton_bulk_speed", "proton_thermal"],
    "MAG": ["Bx_gse", "By_gse", "Bz_gse", "bmag_gse"],
}
REQUIRED_FEATURE_COLUMNS = {
    "timestamp",
    "js_opdi",
    "hellinger_opdi",
    "wasserstein_opdi",
    "proton_density",
    "proton_bulk_speed",
    "proton_thermal",
    "alpha_proton_ratio",
    "Bx_gse",
    "By_gse",
    "Bz_gse",
    "bmag_gse",
}
EXTERNAL_REFERENCE_COLUMNS = [
    "omni_Bx_gse",
    "omni_By_gse",
    "omni_Bz_gse",
    "omni_flow_speed",
    "omni_proton_density",
    "omni_temperature",
]


@dataclass(frozen=True)
class Phase2Paths:
    catalog: Path
    features: Path
    coverage: Path
    manifest: Path
    acquisition_queue: Path


def output_paths(output_dir: Path) -> Phase2Paths:
    return Phase2Paths(
        catalog=output_dir / "phase2_event_catalog.csv",
        features=output_dir / "phase2_feature_table.csv",
        coverage=output_dir / "phase2_modality_coverage.csv",
        manifest=output_dir / "phase2_manifest.json",
        acquisition_queue=output_dir / "phase2_acquisition_queue.csv",
    )


def _resolve(root: Path, configured_path: str | None) -> Path | None:
    if not configured_path:
        return None
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _validate_catalog(config: dict) -> list[dict]:
    events = config.get("events") or []
    if not events:
        raise ValueError("Phase 2 event registry must contain at least one event")

    ids = [str(event.get("event_id", "")).strip() for event in events]
    if any(not event_id for event_id in ids):
        raise ValueError("Every Phase 2 event requires a non-empty event_id")
    if len(ids) != len(set(ids)):
        raise ValueError("Phase 2 event_id values must be unique")

    normalized = []
    for event in events:
        row = dict(event)
        event_class = str(row.get("event_class", "")).upper()
        sample_role = str(row.get("sample_role", "")).upper()
        if event_class not in ALLOWED_EVENT_CLASSES:
            raise ValueError(f"Unsupported event_class for {row['event_id']}: {event_class}")
        if sample_role not in ALLOWED_SAMPLE_ROLES:
            raise ValueError(f"Unsupported sample_role for {row['event_id']}: {sample_role}")
        start = pd.Timestamp(row["start_utc"])
        end = pd.Timestamp(row["end_utc"])
        if start > end:
            raise ValueError(f"start_utc occurs after end_utc for {row['event_id']}")
        row["event_class"] = event_class
        row["sample_role"] = sample_role
        row["start_utc"] = start
        row["end_utc"] = end
        row["negative_control"] = bool(row.get("negative_control", False))
        normalized.append(row)

    for interval_id in sorted({row["independent_interval_id"] for row in normalized}):
        interval_events = sorted(
            (row for row in normalized if row["independent_interval_id"] == interval_id),
            key=lambda row: row["start_utc"],
        )
        for previous, current in zip(interval_events, interval_events[1:]):
            if current["start_utc"] <= previous["end_utc"]:
                raise ValueError(
                    f"Overlapping Phase 2 windows in {interval_id}: "
                    f"{previous['event_id']} and {current['event_id']}"
                )
    return normalized


def _modality_fraction(frame: pd.DataFrame, columns: list[str]) -> float:
    if any(column not in frame.columns for column in columns) or frame.empty:
        return 0.0
    if columns == ["usable_opdi"]:
        usable = frame["usable_opdi"]
        if usable.dtype == object:
            usable = usable.astype(str).str.lower().map({"true": True, "false": False})
        return float(usable.fillna(False).astype(bool).mean())
    return float(frame[columns].notna().all(axis=1).mean())


def _spectra_contract(path: Path | None) -> tuple[bool, str]:
    if path is None or not path.exists():
        return False, "missing"
    try:
        with np.load(path) as spectra:
            required = {"time", "energy", "th1_probability", "th2_probability"}
            missing = sorted(required - set(spectra.files))
            if missing:
                return False, "missing keys: " + ", ".join(missing)
            n = len(spectra["time"])
            if len(spectra["th1_probability"]) != n or len(spectra["th2_probability"]) != n:
                return False, "time/probability length mismatch"
    except Exception as exc:
        return False, f"unreadable: {exc}"
    return True, "available"


def build_phase2_dataset(root: Path, config_path: Path, output_dir: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    events = _validate_catalog(config)
    cadence = int(config.get("expected_cadence_minutes", 1))
    coverage_threshold = float(config.get("minimum_modality_coverage", 0.90))
    minimum_intervals = int(config.get("minimum_independent_intervals_for_research", 5))

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir)
    feature_cache: dict[Path, pd.DataFrame] = {}
    feature_parts = []
    catalog_rows = []
    coverage_rows = []
    selected_source_keys: set[tuple[str, pd.Timestamp]] = set()

    for event in events:
        feature_path = _resolve(root, event["feature_file"])
        assert feature_path is not None
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature table not found for {event['event_id']}: {feature_path}")
        if feature_path not in feature_cache:
            frame = pd.read_csv(feature_path, parse_dates=["timestamp"])
            missing_columns = sorted(REQUIRED_FEATURE_COLUMNS - set(frame.columns))
            if missing_columns:
                raise ValueError(
                    f"Feature table for {event['event_id']} is missing columns: {', '.join(missing_columns)}"
                )
            if frame["timestamp"].duplicated().any():
                raise ValueError(f"Duplicate timestamps found in {feature_path}")
            feature_cache[feature_path] = frame.sort_values("timestamp").reset_index(drop=True)
        source = feature_cache[feature_path]
        window = source.loc[
            (source["timestamp"] >= event["start_utc"]) & (source["timestamp"] <= event["end_utc"])
        ].copy()
        if window.empty:
            raise ValueError(f"No rows found for registered event {event['event_id']}")

        expected_records = int((event["end_utc"] - event["start_utc"]).total_seconds() // (cadence * 60)) + 1
        time_coverage = min(1.0, len(window) / expected_records) if expected_records else 0.0
        spectra_path = _resolve(root, event.get("spectra_file"))
        spectra_available, spectra_detail = _spectra_contract(spectra_path)
        cme_path = _resolve(root, event.get("cme_candidates_file"))
        cme_candidates = 0
        if cme_path is not None and cme_path.exists():
            cme_candidates = int(len(pd.read_csv(cme_path)))

        modality_fractions = {
            modality: _modality_fraction(window, columns)
            for modality, columns in REQUIRED_MODALITY_COLUMNS.items()
        }
        complete_modalities = all(value >= coverage_threshold for value in modality_fractions.values())
        event_ready = (
            time_coverage >= coverage_threshold
            and complete_modalities
            and spectra_available
            and bool(str(event.get("label_source", "")).strip())
        )

        window.insert(0, "event_id", event["event_id"])
        window.insert(1, "independent_interval_id", event["independent_interval_id"])
        window.insert(2, "event_class", event["event_class"])
        window.insert(3, "sample_role", event["sample_role"])
        window.insert(4, "negative_control", event["negative_control"])
        window.insert(5, "label_status", event.get("label_status", "PROVISIONAL"))
        window.insert(6, "label_source", event.get("label_source", ""))
        window.insert(7, "source_feature_file", _display_path(root, feature_path))
        window.insert(8, "source_spectra_file", _display_path(root, spectra_path) if spectra_path else "")
        selected_source_keys.update((_display_path(root, feature_path), timestamp) for timestamp in window["timestamp"])
        feature_parts.append(window)

        catalog_rows.append(
            {
                "event_id": event["event_id"],
                "title": event.get("title", event["event_id"]),
                "independent_interval_id": event["independent_interval_id"],
                "event_class": event["event_class"],
                "sample_role": event["sample_role"],
                "negative_control": event["negative_control"],
                "start_utc": event["start_utc"],
                "end_utc": event["end_utc"],
                "records_observed": len(window),
                "records_expected": expected_records,
                "time_coverage_fraction": time_coverage,
                "spectra_available": spectra_available,
                "cme_candidates": cme_candidates,
                "label_status": event.get("label_status", "PROVISIONAL"),
                "data_status": event.get("data_status", "UNSPECIFIED"),
                "label_source": event.get("label_source", ""),
                "phase2_ready": event_ready,
                "notes": event.get("notes", ""),
            }
        )
        for modality, fraction in modality_fractions.items():
            coverage_rows.append(
                {
                    "event_id": event["event_id"],
                    "event_class": event["event_class"],
                    "modality": modality,
                    "coverage_fraction": fraction,
                    "passes_threshold": fraction >= coverage_threshold,
                    "required_for_contract": True,
                    "detail": spectra_detail if modality in {"TH1", "TH2"} else "feature-column completeness",
                }
            )
        omni_fraction = _modality_fraction(window, EXTERNAL_REFERENCE_COLUMNS)
        coverage_rows.append(
            {
                "event_id": event["event_id"],
                "event_class": event["event_class"],
                "modality": "OMNI_REFERENCE",
                "coverage_fraction": omni_fraction,
                "passes_threshold": omni_fraction >= coverage_threshold,
                "required_for_contract": False,
                "detail": "Optional NASA near-Earth reference; never substitutes for Aditya-L1 modalities",
            }
        )

    features = pd.concat(feature_parts, ignore_index=True)
    catalog = pd.DataFrame(catalog_rows)
    coverage = pd.DataFrame(coverage_rows)
    queue = pd.DataFrame(config.get("acquisition_queue") or [])

    independent_intervals = int(catalog["independent_interval_id"].nunique())
    source_records_available = int(sum(len(frame) for frame in feature_cache.values()))
    unregistered_source_records = max(0, source_records_available - len(selected_source_keys))
    classes = sorted(catalog["event_class"].unique().tolist())
    controls = int((catalog["sample_role"] == "CONTROL").sum())
    positive_events = int((catalog["sample_role"] == "POSITIVE").sum())
    clean_negative_controls = int(catalog["negative_control"].astype(bool).sum())
    catalog_ready = bool(catalog["phase2_ready"].all())
    research_ready = (
        catalog_ready
        and independent_intervals >= minimum_intervals
        and positive_events >= 2
        and controls >= 2
        and clean_negative_controls >= 1
        and "CIR_SIR" in classes
        and ("OTHER_DISTURBANCE" in classes or "ORIENTATION_CONTROL" in classes)
    )

    if research_ready:
        status = "RESEARCH_READY_MULTI_EVENT"
    elif independent_intervals >= minimum_intervals:
        status = "MULTI_EVENT_ASSEMBLED_WITH_DATA_GAPS"
    else:
        status = "BOOTSTRAP_COMPLETE_REQUIRES_MORE_INDEPENDENT_EVENTS"

    manifest = {
        "schema_version": int(config.get("schema_version", 1)),
        "dataset_name": config.get("dataset_name", "TopoCross-SWIS Phase 2 scientific dataset"),
        "phase": 2,
        "status": status,
        "event_windows": int(len(catalog)),
        "independent_intervals": independent_intervals,
        "one_minute_records": int(len(features)),
        "source_records_available": source_records_available,
        "unregistered_source_records": unregistered_source_records,
        "positive_events": positive_events,
        "control_windows": controls,
        "clean_negative_controls": clean_negative_controls,
        "event_classes": classes,
        "minimum_modality_coverage": coverage_threshold,
        "minimum_independent_intervals_for_research": minimum_intervals,
        "registered_windows_pass_contract": catalog_ready,
        "research_ready": research_ready,
        "output_files": {
            "event_catalog": _display_path(root, paths.catalog),
            "feature_table": _display_path(root, paths.features),
            "modality_coverage": _display_path(root, paths.coverage),
            "acquisition_queue": _display_path(root, paths.acquisition_queue),
        },
        "scientific_guardrails": [
            "Multiple windows from the August 2024 source count as one independent interval.",
            "Literature-backed approximate labels remain distinct from exact catalog boundaries.",
            "The post-ICME control is not treated as a clean quiet negative.",
            "NASA OMNI is an external near-Earth reference and never fills missing Aditya-L1 TH1, TH2, BLK or MAG values.",
            "The unpublished 12 October SWIS day is preserved as a data gap and excluded from the registered ICME core window.",
            "The November orientation control remains non-ready until valid Aditya-L1 SWIS and MAG files covering 25 November are supplied.",
            "No multi-event detector performance claim is made until every required event passes the modality contract.",
        ],
    }

    catalog.to_csv(paths.catalog, index=False)
    features.to_csv(paths.features, index=False)
    coverage.to_csv(paths.coverage, index=False)
    queue.to_csv(paths.acquisition_queue, index=False)
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
