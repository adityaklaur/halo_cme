#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase5_experiment import build_phase5_experiment


def main() -> None:
    report = build_phase5_experiment(
        ROOT,
        ROOT / "config" / "phase5_experiment.yaml",
        ROOT / "outputs" / "phase4" / "phase4_feature_dataset.csv",
        ROOT / "outputs" / "phase5",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
