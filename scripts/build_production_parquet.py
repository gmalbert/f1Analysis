#!/usr/bin/env python3
"""Build and validate the typed production copy of the analysis dataset.

The CSV remains the training/source compatibility format for now.  This script
adds a Parquet artifact for the web application without changing the source
data, column names, or row ordering.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


def build_parquet(source: Path, destination: Path) -> tuple[int, int, int]:
    """Convert *source* to compressed Parquet and return rows/columns/bytes."""
    if not source.exists():
        raise FileNotFoundError(f"Source dataset not found: {source}")

    frame = pd.read_csv(source, sep="\t", low_memory=False)
    if frame.empty:
        raise ValueError(f"Source dataset is empty: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        frame.to_parquet(
            temporary,
            engine="pyarrow",
            compression="zstd",
            index=False,
        )
        round_trip = pd.read_parquet(temporary)
        if list(round_trip.columns) != list(frame.columns):
            raise ValueError("Parquet column order does not match the CSV source")
        if len(round_trip) != len(frame):
            raise ValueError("Parquet row count does not match the CSV source")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return len(frame), len(frame.columns), destination.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data_files/f1ForAnalysis.csv"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data_files/f1ForAnalysis.parquet"),
    )
    args = parser.parse_args()

    rows, columns, size = build_parquet(args.source, args.destination)
    print(
        f"Built {args.destination}: {rows:,} rows × {columns:,} columns "
        f"({size / 1024 / 1024:.2f} MiB)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
