import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
from datetime import date, timedelta
import datetime

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
results = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t')
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

def km_to_miles(km):
    return km * 0.621371

all_laps = []
ergast = Ergast(result_type='pandas', auto_cast=True)

# Loop through all seasons and rounds
for year in range(2025, current_year + 1):
    season_schedule = ergast.get_race_schedule(season=year)
    # Filter the season schedule to include only past races
    #season_schedule = season_schedule[pd.to_datetime(season_schedule['raceDate']) <= pd.to_datetime(date.today())]
    total_rounds = len(season_schedule)

    for round_number in range(1, total_rounds + 1):
        for session_type in ['FP1', 'FP2', 'FP3']:
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load()
                session_drivers = session.drivers

                for driver in session_drivers:
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
                        else:
                            fastest_lap['best_s1'] = pd.NaT
                        if 'Sector2Time' in laps.columns and not laps['Sector2Time'].isnull().all():
                            best_s2 = laps.loc[laps['Sector2Time'].idxmin()]
                            fastest_lap['best_s2'] = best_s2['Sector2Time']
                        else:
                            fastest_lap['best_s2'] = pd.NaT
                        if 'Sector3Time' in laps.columns and not laps['Sector3Time'].isnull().all():
                            best_s3 = laps.loc[laps['Sector3Time'].idxmin()]
                            fastest_lap['best_s3'] = best_s3['Sector3Time']
                        else:
                            fastest_lap['best_s3'] = pd.NaT

                        # Only calculate theoretical lap if all sectors are present
                        if pd.notnull(fastest_lap['best_s1']) and pd.notnull(fastest_lap['best_s2']) and pd.notnull(fastest_lap['best_s3']):
                            fastest_lap['best_theory_lap'] = fastest_lap['best_s1'] + fastest_lap['best_s2'] + fastest_lap['best_s3']
                            fastest_lap['best_theory_lap_diff'] = fastest_lap['LapTime'] - (fastest_lap['best_s1'] + fastest_lap['best_s2'] + fastest_lap['best_s3'])
                        else:
                            fastest_lap['best_theory_lap'] = pd.NaT
                            fastest_lap['best_theory_lap_diff'] = pd.NaT

                        all_laps.append(fastest_lap)

            except Exception as e:
                print(f"Skipping {session_type} for {year} round {round_number}: {e}")
                continue  # Skip to the next session

# Convert all_laps to DataFrame after all loops
all_practice_laps_df = pd.DataFrame(all_laps)

# Modify speed to MPH from KM/h
for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
    if col in all_practice_laps_df.columns:
        all_practice_laps_df[f'{col}_mph'] = all_practice_laps_df[col].apply(km_to_miles)

# Convert time columns to seconds from timedelta
for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'best_s1', 'best_s2', 'best_s3', 'best_theory_lap', 'best_theory_lap_diff']:
    if col in all_practice_laps_df.columns:
        all_practice_laps_df[f'{col}_sec'] = pd.to_timedelta(all_practice_laps_df[col]).dt.total_seconds()

# Merge with driver info
active_drivers = pd.merge(results, drivers, left_on='resultsDriverId', right_on='id', how='inner')
active_drivers = active_drivers[active_drivers['activeDriver'] == True]
active_drivers = active_drivers[['id', 'name', 'abbreviation']].drop_duplicates()

print(active_drivers.head(10))

# Merge practice laps with driver names
all_practice_laps_with_driver_names = pd.merge(
    active_drivers, 
    all_practice_laps_df, 
    left_on='abbreviation', 
    right_on='Driver', 
    how='inner'
)

#print(type(all_practice_laps_with_driver_names))
#print(all_practice_laps_with_driver_names.head(10))
#print(all_practice_laps_with_driver_names.shape)

# Merge with races info
races_with_mapping = pd.merge(
    races, 
    all_practice_laps_with_driver_names, 
    left_on=['year', 'round'], 
    right_on=['Year', 'Round'], 
    how='inner', 
    suffixes=('_races', '_mapping')
).drop_duplicates()

#print(races_with_mapping.head(10))
#print(races_with_mapping.shape)

##  need to remove columns to reduce number of rows and drop duplicates

# Save to CSV
races_with_mapping.to_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t', index=False)


