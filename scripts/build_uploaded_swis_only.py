#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.swis_august import process_swis_day


RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "external" / "uploaded_swis_only" / "processed"


def key_for(path: Path) -> tuple[str, str] | None:
    match = re.search(r"_(20\d{6})_.*_(V\d+)\.cdf$", path.name)
    if not match:
        return None
    return match.group(1), match.group(2)


def index_files(folder: Path) -> dict[tuple[str, str], Path]:
    files = {}
    if not folder.exists():
        return files
    for path in sorted(folder.glob("*.cdf")):
        key = key_for(path)
        if key:
            files[key] = path
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OPDI outputs from uploaded TH1/TH2 SWIS CDF pairs.")
    parser.add_argument("--pairs", default="", help="Comma-separated date_version keys, for example 20240812_V03.")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config" / "prototype.yaml").read_text())
    grid = np.geomspace(
        cfg["swis"]["common_energy_min_ev"],
        cfg["swis"]["common_energy_max_ev"],
        cfg["swis"]["common_grid_points"],
    )
    th1 = index_files(RAW / "th1")
    th2 = index_files(RAW / "th2")
    pairs = sorted(set(th1) & set(th2))
    requested = {item.strip() for item in args.pairs.split(",") if item.strip()}
    if requested:
        pairs = [pair for pair in pairs if f"{pair[0]}_{pair[1]}" in requested]
    if not pairs:
        raise SystemExit("No matching TH1/TH2 CDF pairs found in data/raw/th1 and data/raw/th2.")

    OUT.mkdir(parents=True, exist_ok=True)
    scalar = []
    specs = []
    reports = {}
    for date, version in pairs:
        print(f"Processing uploaded SWIS-only {date} {version}...")
        day_features, day_spectra, day_report = process_swis_day(
            th1[(date, version)],
            th2[(date, version)],
            None,
            grid,
            cfg["swis"]["min_valid_points"],
        )
        day_features["source_date"] = date
        day_features["source_version"] = version
        day_features.to_csv(OUT / f"{date}_{version}_swis_only_features.csv", index=False)
        np.savez_compressed(OUT / f"{date}_{version}_swis_only_spectra.npz", **day_spectra)
        scalar.append(day_features)
        specs.append(day_spectra)
        reports[f"{date}_{version}"] = day_report

    features = pd.concat(scalar, ignore_index=True).sort_values("timestamp")
    features.to_csv(OUT / "uploaded_swis_only_features_1min.csv", index=False)
    combined = {}
    for key in specs[0]:
        combined[key] = specs[0][key] if key == "energy" else np.concatenate([s[key] for s in specs], axis=0)
    np.savez_compressed(OUT / "uploaded_swis_only_spectra_1min.npz", **combined)
    report = {
        "dataset": "Uploaded TH1/TH2 SWIS-only OPDI evaluation",
        "mode": "SWIS-only",
        "records": int(len(features)),
        "start": str(features.timestamp.min()),
        "end": str(features.timestamp.max()),
        "pairs_processed": [f"{date}_{version}" for date, version in pairs],
        "note": "BLK plasma, MAG context, detector states, and CME validation are not computed in SWIS-only mode.",
        "daily_quality": reports,
    }
    (OUT / "uploaded_swis_only_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Built {len(features):,} uploaded SWIS-only one-minute records into {OUT}")


if __name__ == "__main__":
    main()
