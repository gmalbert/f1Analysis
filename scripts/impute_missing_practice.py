#!/usr/bin/env python3
"""Simple imputation helpers for missing practice session data.

Usage examples:
  python scripts/impute_missing_practice.py \
      --input data_files/all_practice_results.csv \
      --output data_files/all_practice_results.imputed.csv \
      --method season_mean

The script tries to be conservative: it operates only on numeric columns and
prefers per-driver, per-season statistics. It will not invent driver ids or
change schema.
"""
from __future__ import annotations

import argparse
import sys
import pandas as pd
import numpy as np


def detect_driver_field(df: pd.DataFrame) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in ("driverid", "driver_number", "drivernumber", "number"):
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def impute_season_mean(df: pd.DataFrame, driver_col: str | None, year_col: str | None) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return df
    if driver_col and year_col and year_col in df.columns:
        grp = df.groupby([year_col, driver_col])[num_cols]
        fills = grp.transform("mean")
        return df.fillna(fills)
    if driver_col:
        grp = df.groupby(driver_col)[num_cols]
        fills = grp.transform("mean")
        return df.fillna(fills)
    # fallback: global median
    return df.fillna(df[num_cols].median())


def impute_median_by_driver(df: pd.DataFrame, driver_col: str | None) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if driver_col and driver_col in df.columns:
        grp = df.groupby(driver_col)[num_cols]
        fills = grp.transform("median")
        return df.fillna(fills)
    return df.fillna(df[num_cols].median())


def impute_last_race(df: pd.DataFrame, driver_col: str | None, date_col: str | None) -> pd.DataFrame:
    # forward/backward fill per driver using date ordering
    if driver_col and date_col and driver_col in df.columns and date_col in df.columns:
        df_sorted = df.sort_values(by=[driver_col, date_col])
        num_cols = df_sorted.select_dtypes(include=[np.number]).columns.tolist()
        df_sorted[num_cols] = df_sorted.groupby(driver_col)[num_cols].ffill().bfill()
        return df_sorted.sort_index()
    return df


def impute_practice(input_path: str, output_path: str | None = None, method: str = "season_mean", year_col: str = "Year", date_col: str = "short_date") -> pd.DataFrame:
    """Read `input_path`, perform imputation, write to `output_path` and return DataFrame.
    """
    out = output_path or input_path.replace('.csv', '.imputed.csv')

    try:
        df = pd.read_csv(input_path, sep='\t' if '\t' in open(input_path, 'r', encoding='utf-8').read(1024) else ',', low_memory=False)
    except Exception:
        df = pd.read_csv(input_path, low_memory=False)

    # normalize column name detection (case-insensitive)
    driver_col = detect_driver_field(df)
    ycol = next((c for c in df.columns if c.lower() == year_col.lower()), None)
    dcol = next((c for c in df.columns if c.lower() == date_col.lower()), None)

    # Try to coerce common time-like columns to numeric (seconds) so they can be imputed.
    time_like = {"q1", "q2", "q3", "q1_sec", "q2_sec", "q3_sec", "time", "lap_time", "bestlap", "best_qual_time", "bestlap_time"}

    def time_to_seconds(val):
        """Parse common lap/time string formats into seconds (float).

        Supports examples like:
        - "1:23.456" -> 83.456
        - "83.456" -> 83.456
        - "1:02:03.456" -> 3723.456 (hours:minutes:seconds)
        - strings with trailing 's' or leading words are tolerated
        Returns np.nan for unparsable values.
        """
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if s == '' or s.lower() in ('nan', 'none'):
            return np.nan
        # remove common suffix/prefix characters
        s = s.rstrip('sS')
        s = s.replace('m', ':') if 'm' in s and ':' not in s else s
        # if colon present, parse h:m:s or m:s
        if ':' in s:
            parts = s.split(':')
            try:
                parts_f = [float(p) for p in parts]
            except Exception:
                return np.nan
            # units: rightmost is seconds, next left is minutes, etc.
            total = 0.0
            multiplier = 1.0
            for p in reversed(parts_f):
                total += p * multiplier
                multiplier *= 60.0
            return total
        # otherwise try plain float
        try:
            return float(s)
        except Exception:
            return np.nan

    for col in list(df.columns):
        # if column name looks explicitly time-like, try coercion
        if col.lower() in time_like:
            # if already numeric-ish, coerce directly
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # if many NaNs remain but strings look like colon-formatted times, parse them
            if df[col].isna().sum() > 0:
                sample = df[col].dropna().head(20)
                # check original raw strings for colon presence
                raw = df[col].astype(str)
                if raw.str.contains(':').any():
                    df[col] = df[col].fillna(raw.map(time_to_seconds))
        else:
            # also detect object/string columns that contain time-like patterns and parse
            if df[col].dtype == object:
                raw = df[col].dropna().astype(str)
                if not raw.empty and raw.str.contains(':').any():
                    # attempt to parse into a new numeric column with suffix '_sec' if not present
                    target = col + '_sec' if (col + '_sec') not in df.columns else col
                    parsed = df[col].astype(str).map(time_to_seconds)
                    # only keep parsed if we got any non-null values
                    if parsed.notna().any():
                        df[target] = parsed

    before_missing = df.select_dtypes(include=[np.number]).isna().sum().sum()

    if method == "season_mean":
        df_imputed = impute_season_mean(df, driver_col, ycol)
    elif method == "median_by_driver":
        df_imputed = impute_median_by_driver(df, driver_col)
    else:
        df_imputed = impute_last_race(df, driver_col, dcol)

    after_missing = df_imputed.select_dtypes(include=[np.number]).isna().sum().sum()

    df_imputed.to_csv(out, index=False)
    print(f"Wrote imputed file to {out}")
    print(f"Numeric missing before={before_missing}, after={after_missing}")
    return df_imputed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Impute missing practice session numeric fields")
    parser.add_argument("--input", default="data_files/all_practice_results.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--method", choices=("season_mean", "median_by_driver", "last_race"), default="season_mean")
    parser.add_argument("--year-col", default="Year")
    parser.add_argument("--date-col", default="short_date")
    args = parser.parse_args(argv)

    impute_practice(args.input, args.output, method=args.method, year_col=args.year_col, date_col=args.date_col)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
