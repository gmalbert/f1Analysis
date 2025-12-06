#!/usr/bin/env python3
"""Headless prediction runner

Creates a lightweight predictions CSV for the next upcoming race when no
predictions file exists. This is a pragmatic fallback to populate
`data_files/predictions_{raceId}_{year}.csv` for the email sender and other
scripts that expect predictions to be present.

The produced predictions are a simple baseline (ranking by available practice/grid
metrics) and a conservative MAE estimate using driver historical std or a fallback.

Usage:
    python scripts\headless_predict.py [--force]

"""
from __future__ import annotations
import os
import sys
from datetime import datetime
import argparse
import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(ROOT, 'data_files')
CSV_PATH = os.path.join(DATA_DIR, 'f1ForAnalysis.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--force', action='store_true', help='Overwrite existing predictions file')
args = parser.parse_args()

if not os.path.exists(CSV_PATH):
    print('f1ForAnalysis.csv not found at', CSV_PATH)
    sys.exit(1)

print('Loading analysis CSV...')
df = pd.read_csv(CSV_PATH, sep='\t', low_memory=False)

# Ensure short_date exists and is datetime
if 'short_date' not in df.columns:
    print('No short_date column in CSV; cannot determine next race')
    sys.exit(1)

df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
now = pd.to_datetime(datetime.now())
upcoming = df[df['short_date'] >= now]
if upcoming.empty:
    print('No upcoming races in data. Selecting most recent race instead.')
    # fallback to latest date
    latest_date = df['short_date'].max()
    if pd.isna(latest_date):
        print('No dates present in CSV; aborting')
        sys.exit(1)
    send_df = df[df['short_date'] == latest_date].copy()
else:
    next_date = upcoming['short_date'].min()
    send_df = upcoming[upcoming['short_date'] == next_date].copy()

if send_df.empty:
    print('No rows selected for the target race; aborting')
    sys.exit(1)

# Determine race id/year for filename (best-effort)
race_id = None
race_year = None
for c in ('grandPrixRaceId','raceId_results','raceId'):
    if c in send_df.columns:
        race_id = send_df[c].dropna().unique()[0]
        break
for c in ('grandPrixYear','grandPrixYear_results','year'):
    if c in send_df.columns:
        race_year = send_df[c].dropna().unique()[0]
        break

# file path for predictions
if race_id is None or race_year is None:
    # fallback: use date in filename
    date_str = pd.to_datetime(send_df['short_date'].dropna().iloc[0]).strftime('%Y%m%d')
    out_fname = f'predictions_{date_str}.csv'
else:
    out_fname = f'predictions_{race_id}_{race_year}.csv'

out_path = os.path.join(DATA_DIR, out_fname)
if os.path.exists(out_path) and not args.force:
    print('Predictions file already exists at', out_path)
    print('Use --force to overwrite')
    sys.exit(0)

# Build a baseline score for each active driver row
# Prefer practice position, then starting grid, then historical avg final pos
priority_cols = ['averagePracticePosition', 'resultsStartingGridPositionNumber', 'avg_final_position_per_track', 'driver_constructor_avg_final_position', 'recent_form_3_races']

def baseline_score(row):
    vals = []
    for c in priority_cols:
        v = row.get(c, np.nan) if c in row.index else np.nan
        if pd.notna(v):
            vals.append(float(v))
    if not vals:
        return np.nan
    return np.nanmean(vals)

print('Computing baseline scores...')
rows = send_df.copy()
rows['_baseline_score'] = rows.apply(baseline_score, axis=1)

# If baseline is all NaN for many drivers, fallback to ranking by driverTotalRaceStarts (more experienced drivers first)
if rows['_baseline_score'].isna().all():
    print('No usable baseline features found; falling back to driver experience')
    if 'driverTotalRaceStarts' in rows.columns:
        rows['_baseline_score'] = -rows['driverTotalRaceStarts'].fillna(0).astype(float)
    else:
        rows['_baseline_score'] = 0

# Rank ascending: lower score => better predicted finish
rows = rows.sort_values(by=['_baseline_score'], ascending=True, na_position='last').reset_index(drop=True)
num_drivers = len(rows)
print(f'Assigning predicted positions for {num_drivers} drivers...')
rows['PredictedFinalPosition'] = np.arange(1, num_drivers+1).astype(float)

# Estimate MAE: prefer finishing_position_std_driver if present, else fallback to 1.5
if 'finishing_position_std_driver' in rows.columns:
    rows['PredictedPositionMAE'] = rows['finishing_position_std_driver'].fillna(1.5).astype(float)
else:
    rows['PredictedPositionMAE'] = 1.5

# Keep minimal columns for prediction output
out_cols = ['resultsDriverName', 'resultsDriverId', 'constructorName', 'PredictedFinalPosition', 'PredictedPositionMAE', 'short_date']
available = [c for c in out_cols if c in rows.columns]
predicted_results = rows[available].copy()
# Add race metadata if available
if 'grandPrixName' in send_df.columns:
    predicted_results['grandPrixName'] = send_df['grandPrixName'].iloc[0]

predicted_results.to_csv(out_path, index=False)
print('Wrote predictions to', out_path)
print('Done')
