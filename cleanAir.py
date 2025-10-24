import fastf1
from fastf1 import plotting
from fastf1.core import Laps
import pandas as pd

from fastf1.ergast import Ergast
from os import path
import os
import datetime

DATA_DIR = 'data_files/'


# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

year, gp, session = 2024, 'Silverstone', 'FP2'
session = fastf1.get_session(year, gp, session)
session.load()

# Pick a driver
driver = 'VER'
laps = session.laps.pick_drivers(driver).pick_accurate()

# Get all cars for track position comparison
all_laps = session.laps.pick_accurate()
telemetry_all = {}
for drv in session.drivers:
    laps_drv = all_laps.pick_drivers(drv)
    # Pick a representative lap (e.g., the fastest)
    if not laps_drv.empty:
        fastest_lap = laps_drv.pick_fastest()
        telemetry_all[drv] = fastest_lap.get_telemetry()

# Function to calculate air gap in front of driver
def get_air_gap(lap, telemetry_all):
    drv_tel = lap.get_telemetry()
    if drv_tel.empty:
        return None

    # Look at car's position at midpoint of lap
    midpoint = drv_tel['Distance'].max() / 2
    own_point = drv_tel.iloc[(drv_tel['Distance'] - midpoint).abs().argmin()]

    min_gap = float('inf')
    for drv, tel in telemetry_all.items():
        if drv == lap['Driver']:
            continue
        tel_point = tel[tel['Date'] == own_point['Date']]
        if tel_point.empty:
            continue

        other_pos = tel_point.iloc[0][['X', 'Y']]
        own_pos = own_point[['X', 'Y']]
        dist = ((own_pos - other_pos) ** 2).sum() ** 0.5
        min_gap = min(min_gap, dist)

    return min_gap

# Classify laps
clean, dirty = [], []
for _, lap in laps.iterlaps():
    air_gap = get_air_gap(lap, telemetry_all)
    if air_gap is None or pd.isna(lap['LapTime']):
        continue
    if air_gap > 300:  # Approx >2s on most tracks
        clean.append(lap['LapTime'].total_seconds())
    else:
        dirty.append(lap['LapTime'].total_seconds())

# Calculate the delta
if clean and dirty:
    clean_avg = sum(clean) / len(clean)
    dirty_avg = sum(dirty) / len(dirty)
    delta = dirty_avg - clean_avg

    print(f"{driver} Clean Air Avg: {clean_avg:.3f}s")
    print(f"{driver} Dirty Air Avg: {dirty_avg:.3f}s")
    print(f"Pace Delta (Dirty - Clean): {delta:.3f}s")
else:
    print("Not enough data to compute clean/dirty air delta.")
