#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ground_truth import build_phase3_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 3 ground-truth labels from the Phase 2 multi-event registry")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "phase3_labels.yaml",
        help="Phase 3 labeling-policy YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "phase3",
        help="Phase 3 output directory",
    )
    parser.add_argument(
        "--strict-research-ready",
        action="store_true",
        help="Exit with status 2 while any registered event is blocked by Phase 2 data readiness",
    )
    args = parser.parse_args()
    report = build_phase3_ground_truth(ROOT, args.config, args.output_dir)
    print(json.dumps(report, indent=2))
    if not report["phase3_valid"]:
        raise SystemExit(1)
    if args.strict_research_ready and not report["research_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
