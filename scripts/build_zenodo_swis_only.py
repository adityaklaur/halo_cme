#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.swis_august import process_swis_day


DATASET = ROOT / "data" / "external" / "zenodo_swis_20231106_12"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "prototype.yaml").read_text())
    grid = np.geomspace(
        cfg["swis"]["common_energy_min_ev"],
        cfg["swis"]["common_energy_max_ev"],
        cfg["swis"]["common_grid_points"],
    )
    out_dir = DATASET / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    dates = [f"202311{x:02d}" for x in range(6, 13)]
    scalar = []
    specs = []
    reports = {}
    for date in dates:
        th1 = sorted((DATASET / "th1").glob(f"*{date}*.cdf"))
        th2 = sorted((DATASET / "th2").glob(f"*{date}*.cdf"))
        if not th1 or not th2:
            print(f"WARNING: Missing TH1/TH2 pair for {date}; skipping.")
            continue
        print(f"Processing Zenodo SWIS-only {date}...")
        day_features, day_spectra, day_report = process_swis_day(th1[0], th2[0], None, grid, cfg["swis"]["min_valid_points"])
        day_features.to_csv(out_dir / f"{date}_swis_only_features.csv", index=False)
        np.savez_compressed(out_dir / f"{date}_swis_only_spectra.npz", **day_spectra)
        scalar.append(day_features)
        specs.append(day_spectra)
        reports[date] = day_report

    if not scalar:
        raise SystemExit("No Zenodo TH1/TH2 pairs were processed. Run scripts/download_zenodo_swis.py first.")

    features = pd.concat(scalar, ignore_index=True).sort_values("timestamp")
    features.to_csv(out_dir / "zenodo_swis_only_features_1min.csv", index=False)
    combined = {}
    for key in specs[0]:
        combined[key] = specs[0][key] if key == "energy" else np.concatenate([s[key] for s in specs], axis=0)
    np.savez_compressed(out_dir / "zenodo_swis_only_spectra_1min.npz", **combined)
    report = {
        "dataset": "Zenodo AL1-ASPEX-SWIS 06-12 Nov 2023 L2 sample",
        "doi": "10.5281/zenodo.15861770",
        "mode": "SWIS-only OPDI evaluation",
        "records": int(len(features)),
        "start": str(features.timestamp.min()),
        "end": str(features.timestamp.max()),
        "note": "This public cruise-phase dataset has TH1/TH2 spectra only in this package; BLK, MAG, detector states, and CME validation are not computed.",
        "daily_quality": reports,
    }
    (out_dir / "zenodo_swis_only_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"Built {len(features):,} SWIS-only one-minute records into {out_dir}")


if __name__ == "__main__":
    main()
