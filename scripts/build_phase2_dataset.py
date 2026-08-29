#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scientific_dataset import build_phase2_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the TopoCross-SWIS Phase 2 event dataset")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "phase2_events.yaml",
        help="Phase 2 event registry YAML",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "scientific",
        help="Output folder for the event catalog, feature table and manifest",
    )
    parser.add_argument(
        "--strict-research-ready",
        action="store_true",
        help="Exit non-zero when the registered dataset is not yet sufficient for research evaluation",
    )
    args = parser.parse_args()

    manifest = build_phase2_dataset(ROOT, args.config, args.output_dir)
    print(json.dumps(manifest, indent=2))
    if args.strict_research_ready and not manifest["research_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
