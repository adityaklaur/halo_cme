#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ground_truth import build_phase3_ground_truth
from src.phase4_features import build_phase4_features
from src.phase5_experiment import build_phase5_experiment
from src.scientific_dataset import build_phase2_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build linked TopoCross-SWIS Phases 2 through 5")
    parser.add_argument("--strict-research-ready", action="store_true")
    args = parser.parse_args()

    phase2 = build_phase2_dataset(ROOT, ROOT / "config" / "phase2_events.yaml", ROOT / "data" / "scientific")
    phase3 = build_phase3_ground_truth(ROOT, ROOT / "config" / "phase3_labels.yaml", ROOT / "outputs" / "phase3")
    phase4 = build_phase4_features(ROOT, ROOT / "config" / "phase4_features.yaml", ROOT / "outputs" / "phase4")
    phase5 = build_phase5_experiment(
        ROOT,
        ROOT / "config" / "phase5_experiment.yaml",
        ROOT / "outputs" / "phase4" / "phase4_feature_dataset.csv",
        ROOT / "outputs" / "phase5",
    )
    result = {"phase2": phase2, "phase3": phase3, "phase4": phase4, "phase5": phase5}
    print(json.dumps(result, indent=2))
    if not phase3["phase3_valid"]:
        raise SystemExit(1)
    if args.strict_research_ready and not (phase2["research_ready"] and phase3["research_ready"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
