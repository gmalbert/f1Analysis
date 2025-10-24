import fastf1
import pandas as pd
import numpy as np
import datetime
from fastf1.ergast import Ergast
from os import path
from pit_constants import PIT_LANE_TIME_S, TYPICAL_STATIONARY_TIME_S

DATA_DIR = 'data_files/'
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

ergast = Ergast(result_type='pandas', auto_cast=True)
current_year = datetime.datetime.now().year

results = []

for year in range(2025, current_year + 1):
    print(f"Processing year {year}")
    schedule = ergast.get_race_schedule(season=year)
    for rnd, row in schedule.iterrows():
        round_number = int(row['round'])
        event_name = row['raceName']
        circuit = row['circuitName']
        # Robustly get the race date column
        race_date = pd.to_datetime(row.get('raceDate', row.get('date')))
        # --- Skip races in the future
        if race_date > datetime.datetime.now():
            print(f"    Skipping future race on {race_date.date()}")
            continue
        print(f"  Round {round_number}: {event_name} ({circuit})")
        try:
            session = fastf1.get_session(year, round_number, 'R')
            session.load()
        except Exception as e:
            print(f"    Could not load session: {e}")
            continue

        pit_lane_time_constant = PIT_LANE_TIME_S.get(circuit, np.nan)
        if np.isnan(pit_lane_time_constant):
            print(f"  WARNING: No pit lane time constant for circuit '{circuit}'")
        for drv in session.drivers:
            driver_info = session.get_driver(drv)
            abbreviation = driver_info.get('Abbreviation', drv)
            laps = session.laps.pick_drivers(drv)
            for _, lap in laps.iterlaps():
                # Only process laps with a pit stop
                if pd.notnull(lap['PitInTime']) and pd.notnull(lap['PitOutTime']):
                    pit_in = lap['PitInTime']
                    pit_out = lap['PitOutTime']
                    # Convert to seconds if needed
                    if hasattr(pit_in, 'total_seconds'):
                        pit_in = pit_in.total_seconds()
                    if hasattr(pit_out, 'total_seconds'):
                        pit_out = pit_out.total_seconds()
                    pit_time = pit_out - pit_in
                    stationary_time = pit_time - pit_lane_time_constant
                    results.append({
                        'Year': year,
                        'Round': round_number,
                        'Event': event_name,
                        'Circuit': circuit,
                        'Driver': lap['Driver'],
                        'Abbreviation': abbreviation,
                        'LapNumber': lap['LapNumber'],
                        'PitInTime': pit_in,
                        'PitOutTime': pit_out,
                        'PitTime': pit_time,
                        'PitLaneTimeConstant': pit_lane_time_constant,
                        'StationaryTime': stationary_time,
                        'StationaryTimeTypical': TYPICAL_STATIONARY_TIME_S
                    })
                    print(f"    {lap['Driver']} Lap {lap['LapNumber']}: Stationary {stationary_time:.2f}s (timing only, typical: {TYPICAL_STATIONARY_TIME_S}s)")

# Save to CSV
df = pd.DataFrame(results)
df.to_csv(path.join(DATA_DIR, 'pit_stop_stationary_times_2018_present.csv'), index=False)
print("Saved pit stop stationary times to pit_stop_stationary_times_2018_present.csv")