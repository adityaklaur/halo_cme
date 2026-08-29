#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase4_features import build_phase4_features


def main() -> None:
    report = build_phase4_features(
        ROOT,
        ROOT / "config" / "phase4_features.yaml",
        ROOT / "outputs" / "phase4",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
