import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
from datetime import date, timedelta
import datetime

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))


def time_to_seconds(time_str):
    try:
           # Check if the time_str is valid
        if pd.isnull(time_str) or not isinstance(time_str, str):
            return None
            # Split the time string into minutes and seconds
            minutes, seconds = time_str.split(':')
            return int(minutes) * 60 + float(seconds)
    except ValueError:
            # Handle invalid or missing time formats
            return None

def km_to_miles(km):
    return km * 0.621371


all_laps = []
ergast = Ergast(result_type='pandas', auto_cast=True)

# Loop through all seasons and rounds
for year in range(2025, current_year + 1):
    season_schedule = ergast.get_race_schedule(season=year)

    # Filter the season schedule to include only past races
    season_schedule = season_schedule[pd.to_datetime(season_schedule['raceDate']) < pd.to_datetime(date.today())]
    total_rounds = len(season_schedule)


    for round_number in range(1, total_rounds + 1):
        for session_type in ['FP1', 'FP2', 'FP3']:
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load()

                drivers = session.drivers

                for driver in drivers:
                    laps = session.laps.pick_drivers(driver)
                    fastest_lap = laps.pick_fastest()
                    if fastest_lap is not None and not fastest_lap.empty:
                        fastest_lap = fastest_lap.copy()
                        fastest_lap['Year'] = session.date.year  
                        fastest_lap['FP_Name'] = session.event['EventName']
                        fastest_lap['Round'] = session.event['RoundNumber']
                        fastest_lap['Session'] = session_type

                        # Safely get best sector times
                        if 'Sector1Time' in laps.columns and not laps['Sector1Time'].isnull().all():
                            best_s1 = laps.loc[laps['Sector1Time'].idxmin()]
                            fastest_lap['best_s1'] = best_s1['Sector1Time']
                        if 'Sector2Time' in laps.columns and not laps['Sector2Time'].isnull().all():
                            best_s2 = laps.loc[laps['Sector2Time'].idxmin()]
                            fastest_lap['best_s2'] = best_s2['Sector2Time']
                        if 'Sector3Time' in laps.columns and not laps['Sector3Time'].isnull().all():
                            best_s3 = laps.loc[laps['Sector3Time'].idxmin()]
                            fastest_lap['best_s3'] = best_s3['Sector3Time']

                        all_laps.append(fastest_lap)

            except Exception as e:
                print(f"Skipping {session_type} for {year} round {round_number}: {e}")
                continue  # Skip to the next session
            

        all_practice_laps_df = pd.DataFrame(all_laps)

# Only convert columns if they exist
        for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
            if col in all_practice_laps_df.columns:
                all_practice_laps_df[f'{col}_mph'] = all_practice_laps_df[col].apply(km_to_miles)

        for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'best_s1', 'best_s2', 'best_s3']:
            if col in all_practice_laps_df.columns:
                all_practice_laps_df[f'{col}_sec'] = pd.to_timedelta(all_practice_laps_df[col]).dt.total_seconds()

print(all_practice_laps_df.head())

all_practice_laps_df.to_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t', index=False)


