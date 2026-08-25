#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
RECORD_API = "https://zenodo.org/api/records/15861770"
OUT = ROOT / "data" / "external" / "zenodo_swis_20231106_12"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"SKIP existing {destination}")
        return
    print(f"Downloading {destination.name}...")
    with urllib.request.urlopen(url, timeout=120) as response:
        with destination.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD_API, timeout=60) as response:
        record = json.loads(response.read().decode("utf-8"))
    (OUT / "zenodo_record_metadata.json").write_text(json.dumps(record, indent=2))
    for item in record["files"]:
        name = item["key"]
        if "_TH1_" in name:
            destination = OUT / "th1" / name
        elif "_TH2_" in name:
            destination = OUT / "th2" / name
        else:
            destination = OUT / name
        download(item["links"]["self"], destination)
    print(f"Downloaded Zenodo SWIS sample to {OUT}")
    print("Note: this public dataset contains TH1/TH2 spectra only; it is useful for OPDI experiments but not full BLK/MAG CME validation.")


if __name__ == "__main__":
    main()
