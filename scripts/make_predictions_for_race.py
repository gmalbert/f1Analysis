#!/usr/bin/env python3
"""Create a baseline predictions CSV for a specified race name/date.

If rows for the specified `grandPrixName` or `short_date` exist in
`data_files/f1ForAnalysis.csv`, those rows are used as the base. Otherwise
this script falls back to `data_files/active_drivers.csv` to build a minimal
prediction set for all active drivers.

Usage:
  python scripts\make_predictions_for_race.py --grandPrixName "Abu Dhabi Grand Prix" --date 2025-12-07

The script writes `data_files/predictions_<slug>_<year>.csv`.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import re

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data_files'
ANALYSIS_CSV = DATA_DIR / 'f1ForAnalysis.csv'
ACTIVE_DRIVERS_CSV = DATA_DIR / 'active_drivers.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--grandPrixName', required=True, help='Grand Prix name (e.g. "Abu Dhabi Grand Prix")')
parser.add_argument('--date', required=True, help='Race date in YYYY-MM-DD')
parser.add_argument('--force', action='store_true', help='Overwrite existing predictions file')
args = parser.parse_args()

gp = args.grandPrixName
short_date = args.date
try:
    dt = pd.to_datetime(short_date).date()
except Exception:
    print('Invalid date format; use YYYY-MM-DD')
    sys.exit(1)

# Friendly slug for filename
def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s[:160]

out_fname = f"predictions_{slugify(gp)}_{dt.year}.csv"
out_path = DATA_DIR / out_fname
if out_path.exists() and not args.force:
    print('Predictions already exist at', out_path)
    print('Use --force to overwrite')
    # still continue to allow sending if user wants to call send script separately

# Try to load analysis CSV and find matching rows
base_rows = None
if ANALYSIS_CSV.exists():
    try:
        df = pd.read_csv(ANALYSIS_CSV, sep='\t', low_memory=False)
        # normalize and match by grandPrixName (case-insensitive) or short_date
        mask_name = df['grandPrixName'].astype(str).str.lower() == gp.lower() if 'grandPrixName' in df.columns else pd.Series(False, index=df.index)
        if 'short_date' in df.columns:
            df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
            mask_date = df['short_date'].dt.date == dt
        else:
            mask_date = pd.Series(False, index=df.index)
        mask = mask_name | mask_date
        if mask.any():
            base_rows = df[mask].copy()
            print(f'Found {len(base_rows)} rows in analysis CSV matching race')
    except Exception as _e:
        print('Error reading analysis CSV:', _e)

# If not found, fall back to active drivers list
if base_rows is None or base_rows.empty:
    print('No matching rows found in analysis CSV; falling back to active drivers')
    if not ACTIVE_DRIVERS_CSV.exists():
        print('active_drivers.csv not found; cannot build fallback roster')
        sys.exit(1)
    adf = pd.read_csv(ACTIVE_DRIVERS_CSV)
    # Attempt to find reasonable columns; common names: driverId, resultsDriverId, driverName, resultsDriverName
    if 'resultsDriverName' in adf.columns:
        names_col = 'resultsDriverName'
    elif 'driverName' in adf.columns:
        names_col = 'driverName'
    elif 'fullName' in adf.columns:
        names_col = 'fullName'
    else:
        # try to construct from first/last if available
        if 'firstName' in adf.columns and 'lastName' in adf.columns:
            adf['resultsDriverName'] = adf['firstName'].fillna('') + ' ' + adf['lastName'].fillna('')
            names_col = 'resultsDriverName'
        else:
            # fallback to index-based names
            adf['resultsDriverName'] = adf.index.astype(str)
            names_col = 'resultsDriverName'
    # constructors: try constructorName else empty
    if 'constructorName' not in adf.columns:
        adf['constructorName'] = ''
    base_rows = pd.DataFrame({
        'resultsDriverName': adf[names_col].astype(str),
        'resultsDriverId': adf.get('driverId', adf.get('resultsDriverId', adf.get('driver_id', pd.Series([None]*len(adf)))))
    })
    base_rows['constructorName'] = adf.get('constructorName', '')
    # set short_date and grandPrixName
    base_rows['short_date'] = pd.to_datetime(short_date)
    base_rows['grandPrixName'] = gp

# Now compute baseline scores similar to headless_predict logic
priority_cols = ['averagePracticePosition', 'resultsStartingGridPositionNumber', 'avg_final_position_per_track', 'driver_constructor_avg_final_position', 'recent_form_3_races']

# ensure columns exist in base_rows; if missing, create NaNs
for c in priority_cols:
    if c not in base_rows.columns:
        base_rows[c] = np.nan

def baseline_score(row):
    vals = []
    for c in priority_cols:
        v = row.get(c, np.nan)
        if pd.notna(v):
            try:
                vals.append(float(v))
            except Exception:
                pass
    if not vals:
        return np.nan
    return float(np.nanmean(vals))

base_rows['_baseline_score'] = base_rows.apply(baseline_score, axis=1)
if base_rows['_baseline_score'].isna().all():
    print('No baseline features available; using driver experience fallback')
    if 'driverTotalRaceStarts' in base_rows.columns:
        base_rows['_baseline_score'] = -base_rows['driverTotalRaceStarts'].fillna(0).astype(float)
    else:
        base_rows['_baseline_score'] = 0.0

base_rows = base_rows.sort_values(by=['_baseline_score'], ascending=True, na_position='last').reset_index(drop=True)
num = len(base_rows)
base_rows['PredictedFinalPosition'] = np.arange(1, num+1).astype(float)
if 'finishing_position_std_driver' in base_rows.columns:
    base_rows['PredictedPositionMAE'] = base_rows['finishing_position_std_driver'].fillna(1.5).astype(float)
else:
    base_rows['PredictedPositionMAE'] = 1.5

# Keep only the requested fields
out_cols = ['resultsDriverName', 'resultsDriverId', 'constructorName', 'PredictedFinalPosition', 'PredictedPositionMAE', 'short_date', 'grandPrixName']
avail = [c for c in out_cols if c in base_rows.columns]
pred_df = base_rows[avail].copy()
# if short_date present, ensure formatting
if 'short_date' in pred_df.columns:
    pred_df['short_date'] = pd.to_datetime(pred_df['short_date']).dt.strftime('%Y-%m-%d')

pred_df.to_csv(out_path, index=False)
print('Wrote predictions to', out_path)
print('Rows:', len(pred_df))
print(pred_df.head().to_string(index=False))

# Optionally send immediately by calling existing send script
try:
    # prefer the rich sender so it will pick up this new file
    print('Invoking send_rich_email_now.py to email the generated file...')
    os.system(f'"{sys.executable}" "{ROOT/"scripts/send_rich_email_now.py"}"')
except Exception:
    pass
