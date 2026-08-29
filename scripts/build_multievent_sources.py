#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.multievent_processor import process_event_source, read_omni_csv
from src.ground_truth import build_phase3_ground_truth
from src.scientific_dataset import build_phase2_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build independent Phase 2 event sources from Aditya-L1 and OMNI files")
    parser.add_argument("--raw-root", type=Path, required=True, help="Folder containing per-event swis/ and mag/ folders")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multievent_sources.yaml")
    args = parser.parse_args()

    source_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prototype_config = yaml.safe_load((ROOT / "config" / "prototype.yaml").read_text(encoding="utf-8"))
    swis = prototype_config["swis"]
    energy = np.geomspace(swis["common_energy_min_ev"], swis["common_energy_max_ev"], swis["common_grid_points"])
    output_dir = ROOT / "data" / "processed" / "events"
    omni_dir = ROOT / "data" / "external" / "omni"
    output_dir.mkdir(parents=True, exist_ok=True)
    omni_dir.mkdir(parents=True, exist_ok=True)

    combined_manifest = {"schema_version": 1, "sources": {}}
    for source in source_config["sources"]:
        source_id = source["source_id"]
        print(f"Processing {source_id}...")
        raw_omni = ROOT / source["omni_raw_file"]
        normalized_omni = omni_dir / f"{source_id}_omni_1min.csv"
        read_omni_csv(raw_omni).to_csv(normalized_omni, index=False)
        features, spectra, report = process_event_source(
            raw_root=args.raw_root,
            raw_subdir=source["raw_subdir"],
            start_date=source["start_date"],
            end_date=source["end_date"],
            omni_csv=normalized_omni,
            energy=energy,
            min_valid_points=swis["min_valid_points"],
        )
        feature_path = output_dir / f"{source_id}_features_1min.csv"
        spectra_path = output_dir / f"{source_id}_spectra_1min.npz"
        report_path = output_dir / f"{source_id}_report.json"
        features.to_csv(feature_path, index=False)
        np.savez_compressed(spectra_path, **spectra)
        report.update(
            {
                "source_id": source_id,
                "label_source": source["label_source"],
                "omni_source_url": source["omni_source_url"],
                "normalized_omni_file": str(normalized_omni.relative_to(ROOT)),
            }
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        combined_manifest["sources"][source_id] = {
            "feature_file": str(feature_path.relative_to(ROOT)),
            "spectra_file": str(spectra_path.relative_to(ROOT)),
            "report_file": str(report_path.relative_to(ROOT)),
            "missing_swis_dates": report["missing_swis_dates"],
            "partial_swis_dates": report["partial_swis_dates"],
            "missing_mag_dates": report["missing_mag_dates"],
            "partial_mag_dates": report["partial_mag_dates"],
            "aditya_complete_fraction": report["aditya_complete_fraction"],
            "omni_fully_valid_fraction": report["omni"]["fully_valid_fraction"],
        }

    (output_dir / "multievent_source_manifest.json").write_text(
        json.dumps(combined_manifest, indent=2), encoding="utf-8"
    )
    phase2 = build_phase2_dataset(ROOT, ROOT / "config" / "phase2_events.yaml", ROOT / "data" / "scientific")
    phase3 = build_phase3_ground_truth(
        ROOT,
        ROOT / "config" / "phase3_labels.yaml",
        ROOT / "outputs" / "phase3",
    )
    print(
        json.dumps(
            {"source_manifest": combined_manifest, "phase2_manifest": phase2, "phase3_report": phase3},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
