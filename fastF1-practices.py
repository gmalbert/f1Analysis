import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
import datetime

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
results = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t')
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 
active_drivers = pd.read_csv(path.join(DATA_DIR, 'active_drivers.csv'), sep='\t')

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

def km_to_miles(km):
    return km * 0.621371

all_laps = []
ergast = Ergast(result_type='pandas', auto_cast=True)

# Check if the processed file exists and read it to avoid reprocessing
csv_path = os.path.join(DATA_DIR, 'all_practice_laps.csv')
if os.path.exists(csv_path):
    processed_df = pd.read_csv(csv_path, sep='\t')
    processed_sessions = set(
        zip(
            processed_df['Year'],
            processed_df['Round'],
            processed_df['Session']
        )
    )
else:
    processed_df = pd.DataFrame()
    processed_sessions = set()

# --- MAIN DATA COLLECTION LOOP ---
for year in range(2018, current_year + 1):
    season_schedule = ergast.get_race_schedule(season=year)
    total_rounds = len(season_schedule)
    for round_number in range(1, total_rounds + 1):
        for session_type in ['FP1', 'FP2', 'FP3']:
            session_key = (year, round_number, session_type)
            session_date = pd.to_datetime(season_schedule.iloc[round_number - 1]['raceDate'])
            if session_key in processed_sessions:
                print(f"Skipping session: {session_key} (already processed)")
                continue
            elif (session_date - datetime.datetime.now()).days > 3:
                print(f"Skipping session: {session_key} (more than three days in the future)")
                continue
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load()
                session_drivers = session.drivers

                for driver in session_drivers:
                    laps = session.laps.pick_drivers(driver)
                    fastest_lap = laps.pick_fastest()
                    
                    # print all information available from per driver for the session.drivers
                    # Uncomment the following lines to see all available information
                    #print(f"Driver: {driver}, DriverId: {session_drivers[driver]['driverId']}, FirstName: {session_drivers[driver]['FirstName']}, LastName: {session_drivers[driver]['LastName']}")

                    # fastest_lap['DriverId'] = session_drivers['driverId']
                    # fastest_lap['FirstName'] = session_drivers['FirstName']
                    # fastest_lap['LastName'] = session_drivers['LastName']
                    if fastest_lap is not None and not fastest_lap.empty:
                        
                        fastest_lap = fastest_lap.copy()
                        # fastest_lap['DriverId'] = session_drivers['id']
                        fastest_lap['Year'] = session.date.year  
                        fastest_lap['FP_Name'] = session.event['EventName']
                        fastest_lap['Round'] = session.event['RoundNumber']
                        fastest_lap['Session'] = session_type
                        #fastest_lap['Position'] = fastest_lap['Position']
                        print(fastest_lap['Position'])


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

# --- PROCESS AND SAVE AFTER ALL YEARS ---

if all_laps:
    new_laps_df = pd.DataFrame(all_laps)
    # Concatenate with existing processed_df (if any)
    if not processed_df.empty:
        all_practice_laps_df = pd.concat([processed_df, new_laps_df], ignore_index=True)
    else:
        all_practice_laps_df = new_laps_df
else:
    all_practice_laps_df = processed_df.copy()  # No new laps, just use existing

# Modify speed to MPH from KM/h
for col in ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST']:
    if col in all_practice_laps_df.columns:
        all_practice_laps_df[f'{col}_mph'] = all_practice_laps_df[col].apply(km_to_miles)

# Convert time columns to seconds from timedelta
for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'best_s1', 'best_s2', 'best_s3', 'best_theory_lap', 'best_theory_lap_diff']:
    if col in all_practice_laps_df.columns:
        all_practice_laps_df[f'{col}_sec'] = pd.to_timedelta(all_practice_laps_df[col]).dt.total_seconds()

# # Merge with driver info
# active_drivers = pd.merge(results, drivers, left_on='resultsDriverId', right_on='id', how='inner')
# active_drivers = active_drivers[active_drivers['activeDriver'] == True]
# active_drivers = active_drivers[['id', 'name', 'abbreviation']].drop_duplicates()

# Merge practice laps with driver names
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_df,
    active_drivers,
    left_on='Driver', 
    right_on='abbreviation',
    how='left'
)

# Drop columns with _x or _y suffixes that could cause merge conflicts BEFORE the next merge
for col in list(all_practice_laps_with_driver_names.columns):
    if col.endswith('_x') or col.endswith('_y'):
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[col])

# Now merge with races
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_with_driver_names,
    races[['id', 'round', 'year']],
    left_on=['Round', 'Year'],
    right_on=['round', 'year'],
    how='left'
).rename(columns={'id': 'raceId'})

all_practice_laps_with_driver_names['Round'] = all_practice_laps_with_driver_names['Round'].astype(int)
all_practice_laps_with_driver_names['Year'] = all_practice_laps_with_driver_names['Year'].astype(int)
races['round'] = races['round'].astype(int)
races['year'] = races['year'].astype(int)

# Drop 'round' and 'year' columns if they exist to avoid merge conflicts
for col in ['round', 'year']:
    if col in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[col])

# Drop columns with _x or _y suffixes that could cause merge conflicts
for col in list(all_practice_laps_with_driver_names.columns):
    if col.endswith('_x') or col.endswith('_y'):
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[col])

# print(all_practice_laps_df.columns)

# print(all_practice_laps_df.head(50))

# all_practice_laps_with_driver_names = pd.merge(
#     all_practice_laps_df,
#     drivers,
#     left_on=['abbreviation', 'LastName'],
#     right_on=['abbreviation', 'lastName'],
#     how='left'
# ).drop_duplicates(subset=['Year', 'Round', 'DriverNumber'])

# print(qualifying_with_driverId.columns)

# print(qualifying_with_driverId.head(50))
# qualifying_with_driverId['id']


# print(qualifying_with_driverId[qualifying_with_driverId['id'].isnull()])

# manual_id_map = {
#     ('Sergio', 'Perez'): 'sergio-perez',      
#     ('Nico', 'Hulkenberg'): 'nico-hulkenberg',
#     ('Carlos', 'Sainz'): 'carlos-sainz-jr',
#     ('Max', 'Verstappen'): 'max-verstappen',

# }

# def fill_manual_id(row):
#     if pd.isnull(row['id']):
#         key = (row['FirstName'], row['LastName'])
#         return manual_id_map.get(key, None)
#     return row['id']

# all_practice_laps_with_driver_names['id'] = all_practice_laps_with_driver_names.apply(fill_manual_id, axis=1)


# print(all_practice_laps_with_driver_names.columns)
# print(all_practice_laps_with_driver_names.head())

# Drop duplicates before saving the main file
all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop_duplicates(
    subset=['Year', 'Round', 'Session', 'Driver']
)

# Save all practice laps (with driver names) to a separate CSV
all_practice_laps_with_driver_names.to_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t', index=False)
print("Saved all practice laps to all_practice_laps.csv")

# Only keep FP1 and FP2
fp_laps = all_practice_laps_with_driver_names[all_practice_laps_with_driver_names['Session'].isin(['FP1', 'FP2'])].copy()

# Sort so FP2 comes first, then FP1, then by best lap time (ascending)
fp_laps['Session_priority'] = fp_laps['Session'].map({'FP2': 1, 'FP1': 2})
fp_laps = fp_laps.sort_values(['Year', 'Round', 'Driver', 'Session_priority', 'LapTime_sec'])

# Drop duplicates so you keep only the best FP2 lap if available, otherwise best FP1
fp_laps_deduped = fp_laps.drop_duplicates(subset=['Year', 'Round', 'Driver'], keep='first')

# print(fp_laps_deduped.columns.tolist())

print(fp_laps_deduped.columns.tolist())

# Select only the columns you want to keep
columns_to_keep = [
    'Year', 'Round', 'raceId', 'Driver', 'driverId', 'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec',
    'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'best_theory_lap_sec', 'Session'
]

# Rename driverId.1 to driverId if it exists
if 'driverId.1' in fp_laps_deduped.columns:
    fp_laps_deduped = fp_laps_deduped.rename(columns={'driverId.1': 'driverId'})

# Now select columns
fp_laps_final = fp_laps_deduped[columns_to_keep]

# Save to a new slimmed-down CSV
fp_laps_final.to_csv(path.join(DATA_DIR, 'practice_best_fp1_fp2.csv'), sep='\t', index=False)
print("Saved slimmed practice file to practice_best_fp1_fp2.csv")