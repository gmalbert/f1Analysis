#!/usr/bin/env python3
"""Pull tire compound and stint data from FastF1 for all races 2018+.

Run once for historical backfill (~2-3 hours), then incrementally each week.
Output: data_files/tire_strategy_data.csv (tab-separated)

Usage:
    python f1-tire-strategy.py
    python f1-tire-strategy.py --year 2025   # Single year only
    python f1-tire-strategy.py --dry-run     # Show what would be pulled, no API calls
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
OUTPUT_FILE = DATA_DIR / 'tire_strategy_data.csv'
CACHE_DIR = DATA_DIR / 'f1_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))


def process_race(session) -> list[dict]:
    """Extract per-driver tire strategy rows from a race session."""
    rows = []
    try:
        laps = session.laps
    except Exception:
        return rows  # Session data not available (future/missing race)
    if laps is None or laps.empty:
        return rows

    total_race_laps = int(laps['LapNumber'].max()) if not laps.empty else 0

    for driver in laps['Driver'].unique():
        dl = laps[laps['Driver'] == driver].sort_values('LapNumber')
        if dl.empty:
            continue

        stints = dl.groupby('Stint').agg(
            compound=('Compound', 'first'),
            laps_in_stint=('LapNumber', 'count'),
        ).reset_index()

        num_stints = len(stints)
        starting_compound = str(stints.iloc[0]['compound']) if num_stints > 0 else 'UNKNOWN'

        # Tire degradation: pace diff between first 3 and last 3 laps per stint
        stint_degs = []
        for _, stint_row in stints.iterrows():
            sl = dl[dl['Stint'] == stint_row['Stint']].copy()
            lap_secs = sl['LapTime'].dt.total_seconds().dropna()
            if len(lap_secs) >= 6:
                stint_degs.append(float(lap_secs.iloc[-3:].mean() - lap_secs.iloc[:3].mean()))
        avg_degradation = float(np.mean(stint_degs)) if stint_degs else np.nan

        total_laps = int(dl['LapNumber'].max()) if not dl.empty else 0
        soft_laps = int((dl['Compound'] == 'SOFT').sum())
        soft_ratio = float(soft_laps / total_laps) if total_laps > 0 else 0.0

        rows.append({
            'year': session.event['EventDate'].year,
            'round': session.event['RoundNumber'],
            'event_name': str(session.event.get('EventName', '')),
            'driver': str(driver),
            'num_stints': num_stints,
            'starting_compound': starting_compound,
            'avg_stint_length': float(stints['laps_in_stint'].mean()),
            'max_stint_length': int(stints['laps_in_stint'].max()) if stints['laps_in_stint'].notna().any() else 0,
            'avg_tire_degradation_sec': avg_degradation,
            'soft_ratio': soft_ratio,
            'used_hard': int('HARD' in stints['compound'].values),
            'used_medium': int('MEDIUM' in stints['compound'].values),
            'used_soft': int('SOFT' in stints['compound'].values),
            'total_laps': total_laps,
            'total_race_laps': total_race_laps,
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description='Pull F1 tire strategy data from FastF1')
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
            time.sleep(0.3)  # Polite API pacing

    if args.dry_run:
        print(f"\nDry run complete. {len(years)} years, {sum(1 for y in years for _ in range(1))} seasons queued.")
        return

    if not all_rows:
        print("No new data to write.")
        return

    result = pd.DataFrame(all_rows)

    # Merge with existing data
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
