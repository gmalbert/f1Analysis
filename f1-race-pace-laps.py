#!/usr/bin/env python3
"""Pull race lap-level pace data from FastF1 for all races 2018+.

Captures fuel-corrected pace, lap consistency, and race completion rate.
These fill a major gap: the current model has qualifying lap data but no
equivalent race-lap pace features.

Output: data_files/race_pace_lap_data.csv (tab-separated)

Usage:
    python f1-race-pace-laps.py
    python f1-race-pace-laps.py --year 2025   # Single year only
    python f1-race-pace-laps.py --dry-run     # Show what would be pulled
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import time
import os
import argparse

os.environ['LOCAL_RUN'] = '1'  # Enable FastF1 cache

DATA_DIR = Path('data_files')
OUTPUT_FILE = DATA_DIR / 'race_pace_lap_data.csv'
CACHE_DIR = DATA_DIR / 'f1_cache'

fastf1.Cache.enable_cache(str(CACHE_DIR))


def process_race(session) -> list[dict]:
    """Extract per-driver race lap pace metrics from a session."""
    rows = []
    try:
        laps = session.laps
    except Exception:
        return rows  # Session data not available (future/missing race)
    if laps is None or laps.empty:
        return rows

    total_race_laps = int(laps['LapNumber'].max()) if not laps.empty else 0

    for driver in laps['Driver'].unique():
        dl = laps[laps['Driver'] == driver].copy()
        dl['LapTime_sec'] = dl['LapTime'].dt.total_seconds()
        dl = dl[dl['LapTime_sec'].notna() & (dl['LapTime_sec'] > 0)]
        if len(dl) < 10:
            continue

        # Fuel-corrected pace: pace change from early to late stint
        # Positive = getting slower (tyre deg > fuel benefit), Negative = improving
        early_pace = dl.head(10)['LapTime_sec'].mean()
        late_pace = dl.tail(10)['LapTime_sec'].mean()
        fuel_corrected = (
            float((late_pace - early_pace) / early_pace * 100)
            if early_pace > 0 else np.nan
        )

        # Lap consistency - exclude pit laps
        regular = dl
        if 'PitInTime' in dl.columns:
            regular = dl[dl['PitInTime'].isna()]
        pace_std = float(regular['LapTime_sec'].std()) if len(regular) > 5 else np.nan

        # Best 5-lap window (peak race pace indicator)
        lap_secs = dl['LapTime_sec'].values
        best_5lap = np.nan
        if len(lap_secs) >= 5:
            best_5lap = float(pd.Series(lap_secs).rolling(5).mean().min())

        laps_completed = len(dl)
        completion_pct = float(laps_completed / total_race_laps) if total_race_laps > 0 else 0.0

        rows.append({
            'year': session.event['EventDate'].year,
            'round': session.event['RoundNumber'],
            'driver': str(driver),
            'fuel_corrected_pace_pct': fuel_corrected,
            'race_pace_std': pace_std,
            'avg_lap_time_sec': float(dl['LapTime_sec'].mean()),
            'best_5lap_avg_sec': best_5lap,
            'laps_completed': laps_completed,
            'total_race_laps': total_race_laps,
            'completion_pct': completion_pct,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description='Pull F1 race lap pace data from FastF1')
    parser.add_argument('--year', type=int, default=None, help='Single year to process')
    parser.add_argument('--dry-run', action='store_true', help='Show races to process without API calls')
    args = parser.parse_args()

    # Load existing data to skip already-processed races
    existing = set()
    if OUTPUT_FILE.exists():
        try:
            old = pd.read_csv(OUTPUT_FILE, sep='\t')
            existing = set(zip(old['year'].astype(int), old['round'].astype(int)))
            print(f"Loaded {len(old)} existing rows; {len(existing)} races already processed")
        except Exception as e:
            print(f"WARNING: Could not load existing file: {e}")

    years = [args.year] if args.year else list(range(2018, 2027))
    all_rows = []

    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  Skip year {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            rnd = int(event['RoundNumber'])
            if rnd == 0:
                continue
            if (year, rnd) in existing:
                print(f"  Skip {year} R{rnd} (already processed)")
                continue

            if args.dry_run:
                print(f"  Would process: {year} R{rnd} {event.get('EventName', '')}")
                continue

            try:
                session = fastf1.get_session(year, rnd, 'R')
                session.load(laps=True, telemetry=False, weather=False, messages=False)
            except Exception as e:
                print(f"  Skip {year} R{rnd}: {e}")
                time.sleep(1)
                continue

            rows = process_race(session)
            if not rows:
                print(f"  Skip {year} R{rnd}: no lap data available (future race?)")
                continue
            all_rows.extend(rows)
            print(f"  Processed {year} R{rnd} ({event.get('EventName', '')}): {len(rows)} drivers")
            time.sleep(0.3)

    if args.dry_run:
        print(f"\nDry run complete.")
        return

    if not all_rows:
        print("No new data to write.")
        return

    result = pd.DataFrame(all_rows)

    if OUTPUT_FILE.exists() and len(existing) > 0:
        try:
            old = pd.read_csv(OUTPUT_FILE, sep='\t')
            result = pd.concat([old, result], ignore_index=True).drop_duplicates(
                subset=['year', 'round', 'driver'], keep='last')
        except Exception as e:
            print(f"WARNING: Could not merge with existing: {e}")

    DATA_DIR.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print(f"\nSaved {len(result)} total rows to {OUTPUT_FILE}")
    print(f"New rows added: {len(all_rows)}")


if __name__ == '__main__':
    main()
