#!/usr/bin/env python3
"""Pull first-lap positions from the Jolpica mirror of the Ergast F1 API.

Output: data_files/first_lap_positions.csv (tab-separated)
Columns: year, round, driverId, driverIdNorm, first_lap_position

Run from the repo root:
    python f1-first-lap-positions.py

The script is incremental — already-fetched (year, round) pairs are skipped.
Jolpica rate-limit: ~4 req/s.  We sleep 0.2 s between requests to stay polite.
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data_files')
OUTPUT_FILE = DATA_DIR / 'first_lap_positions.csv'

BASE_URL = "https://api.jolpi.ca/ergast/f1/{year}/{round}/laps/1.json"

# ------------------------------------------------------------------
# Load already-fetched rows so we can skip them
# ------------------------------------------------------------------
existing: set[tuple[int, int]] = set()
if OUTPUT_FILE.exists():
    old = pd.read_csv(OUTPUT_FILE, sep='\t')
    existing = set(zip(old['year'].astype(int), old['round'].astype(int)))
    print(f"Loaded {len(existing)} existing (year, round) pairs — will skip these.")

all_rows: list[dict] = []

# ------------------------------------------------------------------
# Pull lap-1 timing for every race 2018–current year
# ------------------------------------------------------------------
current_year = pd.Timestamp.now().year

for year in range(2018, current_year + 1):
    print(f"\n=== {year} ===")
    for rnd in range(1, 25):
        if (year, rnd) in existing:
            continue
        url = BASE_URL.format(year=year, round=rnd)
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 404:
                # No more rounds this year
                print(f"  R{rnd}: 404 — end of season")
                break
            response.raise_for_status()
            data = response.json()
            races = data.get('MRData', {}).get('RaceTable', {}).get('Races', [])
            if not races:
                print(f"  R{rnd}: no races in response — end of season")
                break
            race = races[0]
            laps = race.get('Laps', [])
            if not laps:
                # Race may be scheduled but not yet run
                print(f"  R{rnd}: no lap data yet — skipping")
                time.sleep(0.2)
                continue
            lap1 = laps[0]
            timings = lap1.get('Timings', [])
            for timing in timings:
                ergast_id = timing.get('driverId', '')
                _raw_pos = timing.get('position')
                try:
                    _pos = int(_raw_pos)
                except (TypeError, ValueError):
                    continue  # skip if position is None (race not yet run)
                all_rows.append({
                    'year': year,
                    'round': rnd,
                    'driverId': ergast_id,
                    # Normalise Ergast underscores to hyphens to match f1db IDs
                    'driverIdNorm': ergast_id.replace('_', '-'),
                    'first_lap_position': _pos,
                })
            print(f"  R{rnd}: {len(timings)} drivers")
            time.sleep(0.2)   # polite delay
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"  R{rnd}: rate-limited — sleeping 5 s")
                time.sleep(5)
            else:
                print(f"  Error {year} R{rnd}: {e}")
                time.sleep(1)
        except Exception as e:
            print(f"  Error {year} R{rnd}: {e}")
            time.sleep(1)

# ------------------------------------------------------------------
# Merge with existing data and save
# ------------------------------------------------------------------
result = pd.DataFrame(all_rows)
if OUTPUT_FILE.exists():
    old = pd.read_csv(OUTPUT_FILE, sep='\t')
    result = pd.concat([old, result], ignore_index=True).drop_duplicates(
        subset=['year', 'round', 'driverId'], keep='last')
result = result.sort_values(['year', 'round', 'first_lap_position']).reset_index(drop=True)
result.to_csv(OUTPUT_FILE, sep='\t', index=False)
print(f"\nSaved {len(result)} rows to {OUTPUT_FILE}")
