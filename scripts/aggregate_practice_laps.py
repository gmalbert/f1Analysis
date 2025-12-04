#!/usr/bin/env python3
"""Aggregate practice laps to one best lap per driver/session.

Writes `data_files/practice_best_by_session.csv` (tab-separated) containing
one row per (raceId, Session, driverId) with the fastest lap found.

Usage:
  python scripts/aggregate_practice_laps.py --input data_files/all_practice_laps.csv
"""
from __future__ import annotations

import argparse
import os
import pandas as pd


def read_maybe_tab(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8') as fh:
        sample = fh.read(2048)
    sep = '\t' if '\t' in sample else ','
    return pd.read_csv(path, sep=sep, low_memory=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Aggregate practice laps into best-by-session')
    parser.add_argument('--input', default='data_files/all_practice_laps.csv')
    parser.add_argument('--output', default='data_files/practice_best_by_session.csv')
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"Input file {args.input} not found")
        return 1

    df = read_maybe_tab(args.input)

    # ensure we have a numeric lap-time column; try common names
    if 'LapTime_sec' not in df.columns:
        if 'LapTime' in df.columns:
            try:
                df['LapTime_sec'] = pd.to_timedelta(df['LapTime']).dt.total_seconds()
            except Exception:
                df['LapTime_sec'] = pd.to_numeric(df['LapTime'], errors='coerce')
        else:
            # fallback: try common alternatives
            for cand in ('lap_time', 'lapTime', 'LapTimeSeconds'):
                if cand in df.columns:
                    df['LapTime_sec'] = pd.to_numeric(df[cand], errors='coerce')
                    break

    if 'LapTime_sec' not in df.columns:
        print('Could not find or compute LapTime_sec; aborting')
        return 1

    # detect grouping columns
    race_col = next((c for c in ('raceId', 'grandPrixId', 'race_id', 'raceid') if c in df.columns), None)
    session_col = next((c for c in ('Session', 'FP_Name', 'session', 'fp_name') if c in df.columns), None)
    driver_col = next((c for c in ('driverId', 'DriverId', 'driverid', 'driver') if c in df.columns), None)

    if race_col is None or session_col is None or driver_col is None:
        print(f'Missing required grouping columns: race={race_col}, session={session_col}, driver={driver_col}')
        return 1

    # group and pick index of minimum LapTime_sec per (race, session, driver)
    grp_keys = [race_col, session_col, driver_col]
    # drop rows without a valid lap time
    df_clean = df[df['LapTime_sec'].notna()].copy()
    if df_clean.empty:
        print('No valid lap times found; nothing to aggregate')
        return 0

    idx = df_clean.groupby(grp_keys)['LapTime_sec'].idxmin()
    best = df_clean.loc[idx].reset_index(drop=True)

    # select a conservative set of columns to keep (preserve if present)
    keep = []
    want = [race_col, session_col, driver_col, 'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'best_theory_lap_diff_sec', 'FastestPracticeLap_sec', 'constructorName', 'round', 'grandPrixYear', 'Driver']
    for c in want:
        if c in best.columns and c not in keep:
            keep.append(c)

    if not keep:
        # if nothing matched, just write minimal columns
        keep = [race_col, session_col, driver_col, 'LapTime_sec']

    out_df = best[keep]

    out_df.to_csv(args.output, sep='\t', index=False)
    print(f'Wrote aggregated best-practice file to {args.output} ({len(out_df)} rows)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
