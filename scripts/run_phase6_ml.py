#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase6_ml import build_phase6_ml


def main() -> None:
    report = build_phase6_ml(
        ROOT,
        ROOT / "config" / "phase6_ml.yaml",
        ROOT / "outputs" / "phase4" / "phase4_feature_dataset.csv",
        ROOT / "outputs" / "phase4" / "phase4_feature_dictionary.csv",
        ROOT / "outputs" / "phase6",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
