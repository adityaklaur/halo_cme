#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

import cdflib


def inspect(path: Path) -> None:
    cdf = cdflib.CDF(str(path))
    print(f"FILE: {path}")
    print("=" * 80)
    info = cdf.cdf_info()
    variables = list(info.zVariables) + list(info.rVariables)
    for name in variables:
        attrs = cdf.varattsget(name)
        try:
            values = cdf.varget(name)
            shape = getattr(values, "shape", ())
            dtype = getattr(values, "dtype", type(values).__name__)
        except Exception as exc:
            shape = "unreadable"
            dtype = str(exc)
        print(f"VARIABLE: {name}")
        print(f"  Shape: {shape}")
        print(f"  Dtype: {dtype}")
        for key in ["UNITS", "Units", "units", "FILLVAL", "VALIDMIN", "VALIDMAX", "DEPEND_0", "DEPEND_1", "LABLAXIS"]:
            if key in attrs:
                print(f"  {key}: {attrs[key]}")
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect variables and metadata inside an Aditya-L1 SWIS CDF file.")
    parser.add_argument("cdf", type=Path)
    args = parser.parse_args()
    inspect(args.cdf)


if __name__ == "__main__":
    main()
