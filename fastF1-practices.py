import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
import datetime
import numpy as np

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


def add_teammate_delta(df, group_cols, value_col, new_col):
    """
    Adds a column with the difference between each driver's value_col and their teammate's for each group.
    Only works for teams with exactly 2 drivers per group.
    """
    def teammate_diff(x):
        if len(x) != 2:
            return [None] * len(x)
        return [x.iloc[0] - x.iloc[1], x.iloc[1] - x.iloc[0]]
    df[new_col] = (
        df.groupby(group_cols)[value_col]
        .transform(lambda x: teammate_diff(x) if len(x) == 2 else [None]*len(x))
    )
    return df


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
# for year in range(2025, current_year + 1):
for year in range(2018, current_year + 1):
    season_schedule = ergast.get_race_schedule(season=year)
    total_rounds = len(season_schedule)
    # for round_number in range(11, 12):
    for round_number in range(1, total_rounds + 1):    
        # for session_type in ['FP1']:
        for session_type in ['FP1', 'FP2', 'FP3']:
            session_key = (year, round_number, session_type)
            session_date = pd.to_datetime(season_schedule.iloc[round_number - 1]['raceDate']).date()
            today = datetime.date.today()
            if session_key in processed_sessions:
                print(f"Skipping session: {session_key} (already processed)")
                continue
            # Skip if the race is later than today
            if session_date > today:
                print(f"Skipping session: {session_key} (race is in the future)")
                continue
            elif (session_date - datetime.datetime.now().date()).days > 3:
                print(f"Skipping session: {session_key} (more than three days in the future)")
                continue
            try:
                session = fastf1.get_session(year, round_number, session_type)
                session.load()
                session_drivers = session.drivers

                # --- Clean/Dirty Air Delta Calculation for All Drivers ---
                session_laps = session.laps.pick_accurate()
                telemetry_all = {}
                for drv in session.drivers:
                    abbreviation = session.get_driver(drv).get('Abbreviation', drv)
                    laps_drv = session_laps.pick_drivers(drv)
                    if not laps_drv.empty:
                        # Use all laps' telemetry, not just the fastest lap
                        tel = laps_drv.get_telemetry()
                        if not tel.empty:
                            telemetry_all[abbreviation] = tel

                def get_air_gap(lap, telemetry_all):
                    drv_tel = lap.get_telemetry()
                    if drv_tel.empty:
                        return None
                    midpoint = drv_tel['Distance'].max() / 2
                    own_point = drv_tel.iloc[(drv_tel['Distance'] - midpoint).abs().argmin()]
                    min_gap = float('inf')
                    for drv, tel in telemetry_all.items():
                        if drv == lap['Driver']:
                            continue
                        # tel_point = tel[tel['Date'] == own_point['Date']]
                        # if tel_point.empty:
                        #     continue
                        # other_pos = tel_point.iloc[0][['X', 'Y']]
                        # Find the nearest timestamp in the other driver's telemetry
                        idx = (tel['Date'] - own_point['Date']).abs().idxmin()
                        tel_point = tel.loc[idx]
                        other_pos = tel_point[['X', 'Y']]
                        own_pos = own_point[['X', 'Y']]
                        dist = ((own_pos - other_pos) ** 2).sum() ** 0.5
                        min_gap = min(min_gap, dist)
                    return min_gap

                def get_sampled_min_air_gap(lap, telemetry_all, n_points=3):
                    drv_tel = lap.get_telemetry()
                    if drv_tel.empty:
                        return None
                    distances = drv_tel['Distance']
                    sample_distances = [distances.min() + i*(distances.max()-distances.min())/(n_points-1) for i in range(n_points)]
                    min_gap = float('inf')
                    own_abbr = lap['Driver']  # This should now match the keys in telemetry_all
                    for d in sample_distances:
                        own_point = drv_tel.iloc[(drv_tel['Distance'] - d).abs().argmin()]
                        for drv, tel in telemetry_all.items():
                            if drv == own_abbr:
                                continue
                            idx = (tel['Date'] - own_point['Date']).abs().idxmin()
                            tel_point = tel.loc[idx]
                            other_pos = tel_point[['X', 'Y']]
                            own_pos = own_point[['X', 'Y']]
                            dist = ((own_pos - other_pos) ** 2).sum() ** 0.5
                            min_gap = min(min_gap, dist)
                    return min_gap

                # Store deltas for all drivers in this session
                clean_dirty_deltas = []

                for driver in session.drivers:
                    driver_info = session.get_driver(driver)
                    abbreviation = driver_info.get('Abbreviation', driver)
                    # laps = session.laps.pick_drivers(driver).pick_accurate()
                    laps = session.laps.pick_drivers(driver).pick_quicklaps()
                    # Filter out slow laps (e.g., keep only laps within 105% of the best lap)
                    # if not laps.empty and 'LapTime' in laps.columns:
                    #     best_lap_time = laps['LapTime'].min()
                    #     threshold = best_lap_time * 1.05  # 105%
                    #     laps = laps[laps['LapTime'] <= threshold]
                    air_gaps = []
                    for _, lap in laps.iterlaps():
                        air_gap = get_sampled_min_air_gap(lap, telemetry_all, n_points=3)
                        if air_gap is not None and not pd.isna(lap['LapTime']):
                            air_gaps.append(air_gap)

                    if air_gaps:
                        print(f"{abbreviation} air gap stats: min={min(air_gaps):.2f}, max={max(air_gaps):.2f}, avg={sum(air_gaps)/len(air_gaps):.2f}")
                        print(np.histogram(air_gaps, bins=[0, 5, 10, 20, 30, 50, 100]))
                    clean, dirty = [], []
                    clean_count = 0
                    dirty_count = 0
                    neutral_count = 0

                    for _, lap in laps.iterlaps():
                        # air_gap = get_air_gap(lap, telemetry_all)
                        # air_gap = get_min_air_gap(lap, telemetry_all)
                        air_gap = get_sampled_min_air_gap(lap, telemetry_all, n_points=3)
                        print(f"Driver: {abbreviation}, Lap: {lap['LapNumber']}, Air gap: {air_gap}")

                        if air_gap is None or pd.isna(lap['LapTime']):
                            continue
                        if air_gap > 20:  # the lower the better, so we consider 25 as clean air
                            clean.append(lap['LapTime'].total_seconds())
                            clean_count += 1
                        elif air_gap < 15:
                            dirty.append(lap['LapTime'].total_seconds())
                            dirty_count += 1
                        else:
                            neutral_count += 1
                    print(f"Clean: {clean_count}, Dirty: {dirty_count}, Neutral: {neutral_count}")        
                    if clean and dirty:
                        clean_avg = sum(clean) / len(clean)
                        dirty_avg = sum(dirty) / len(dirty)
                        delta = dirty_avg - clean_avg
                        clean_dirty_deltas.append({
                            'Year': year,
                            'Round': round_number,
                            'Session': session_type,
                            'Driver': abbreviation,  # <--- Always use abbreviation here!
                            'CleanAirAvg': clean_avg,
                            'DirtyAirAvg': dirty_avg,
                            'Delta': delta,
                            'NumCleanLaps': len(clean),
                            'NumDirtyLaps': len(dirty)
                        })
                        print(f"{year} R{round_number} {session_type} {driver}: Clean={clean_avg:.3f}s Dirty={dirty_avg:.3f}s Delta={delta:.3f}s")
                        
                    else:
                        print(f"{year} R{round_number} {session_type} {driver}: Not enough data for delta.")

                # Optionally: Save or merge clean_dirty_deltas into your main DataFrame or output CSV
                # Example: Append to a master list for all sessions, then save at the end
                delta_csv = path.join(DATA_DIR, 'clean_dirty_air_deltas.csv')
                if clean_dirty_deltas:
                    df_clean_dirty = pd.DataFrame(clean_dirty_deltas)
                    df_clean_dirty.to_csv(delta_csv, mode='a', header=not os.path.exists(delta_csv), index=False, sep='\t')

                # --- Continue with your existing fastest lap processing code below ---
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
                        driver_info = session.get_driver(driver)
                        abbreviation = driver_info.get('Abbreviation', driver)
                        fastest_lap['Driver'] = abbreviation  # <-- Force abbreviation here!
                        fastest_lap['Year'] = session.date.year  
                        fastest_lap['FP_Name'] = session.event['EventName']
                        fastest_lap['Round'] = session.event['RoundNumber']
                        fastest_lap['Session'] = session_type
                        #fastest_lap['Position'] = fastest_lap['Position']
                        print(fastest_lap['Position'])


                        # Safely get best sector times
                        if 'Sector1Time' in laps.columns and not laps['Sector1Time'].isnull().all():
                            # laps['Sector1Time'] = pd.to_timedelta(laps['Sector1Time'], errors='coerce')
                            laps.loc[:, 'Sector1Time'] = pd.to_timedelta(laps['Sector1Time'], errors='coerce')
                            best_s1 = laps.loc[laps['Sector1Time'].idxmin()]
                            fastest_lap['best_s1'] = best_s1['Sector1Time']
                        else:
                            fastest_lap['best_s1'] = pd.NaT
                        if 'Sector2Time' in laps.columns and not laps['Sector2Time'].isnull().all():
                            laps.loc[:, 'Sector2Time'] = pd.to_timedelta(laps['Sector2Time'], errors='coerce')
                            best_s2 = laps.loc[laps['Sector2Time'].idxmin()]
                            fastest_lap['best_s2'] = best_s2['Sector2Time']
                        else:
                            fastest_lap['best_s2'] = pd.NaT
                        if 'Sector3Time' in laps.columns and not laps['Sector3Time'].isnull().all():
                            laps.loc[:, 'Sector3Time'] = pd.to_timedelta(laps['Sector3Time'], errors='coerce')
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

# After collecting all_laps, but before creating new_laps_df:
# for lap in all_laps:
#     driver_info = session.get_driver(lap['Driver'])
#     lap['Driver'] = driver_info.get('Abbreviation', lap['Driver'])

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

all_practice_laps_with_driver_names.rename(columns={'driverId_y': 'resultsDriverId'}, inplace=True)

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



# # Create a column for each driver's best qualifying time (lowest non-null Q1/Q2/Q3)
# qualifying_with_driverId['best_qual_time'] = qualifying_with_driverId[['Q1_sec', 'Q2_sec', 'Q3_sec']].min(axis=1)

# qualifying_with_driverId = add_teammate_delta(
#     qualifying_with_driverId,
#     ['Year', 'Round', 'constructorName'],
#     'best_qual_time',
#     'teammate_qual_delta'
# )

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
##     ('Max', 'Verstappen'): 'max-verstappen',

# }

# def fill_manual_id(row):
#     if pd.isnull(row['id']):
#         key = (row['FirstName'], row['LastName'])
#         return manual_id_map.get(key, None)
#     return row['id']

# all_practice_laps_with_driver_names['id'] = all_practice_laps_with_driver_names.apply(fill_manual_id, axis=1)


# print(all_practice_laps_with_driver_names.columns)
# print(all_practice_laps_with_driver_names.head())

# combined_practices_with_deltas = pd.merge(
#     all_practice_laps_with_driver_names,
#     df_clean_dirty[['CleanAirAvg', 'DirtyAirAvg', 'Delta', 'NumCleanLaps', 'NumDirtyLaps']],
#     on=['Year', 'Round', 'Session', 'Driver'],
#     how='left'
# )

# # Drop duplicates before saving the main file
# combined_practices_with_deltas = combined_practices_with_deltas.drop_duplicates(
#     subset=['Year', 'Round', 'Session', 'Driver']
# )



all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_with_driver_names,
    results[['constructorName', 'grandPrixYear', 'round', 'resultsDriverId']],
    left_on=['Year', 'Round', 'resultsDriverId'],
    right_on=['grandPrixYear', 'round', 'resultsDriverId'],
    how='left'
)



# Rename columns to match the expected format
if 'constructorName' not in all_practice_laps_with_driver_names.columns:
    if 'constructorName_x' in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names.rename(columns={'constructorName_x': 'constructorName'}, inplace=True)
    elif 'constructor' in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names['constructorName'] = all_practice_laps_with_driver_names['constructor']
    elif 'constructorId' in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names['constructorName'] = all_practice_laps_with_driver_names['constructorId']
    else:
        raise KeyError("No constructorName or fallback column found!")

# print(all_practice_laps_with_driver_names.columns.tolist())
# print(all_practice_laps_with_driver_names.head(50))

all_practice_laps_with_driver_names = add_teammate_delta(
    all_practice_laps_with_driver_names,
    ['Year', 'Round', 'Session', 'constructorName'],
    'LapTime_sec',
    'teammate_practice_delta'
).drop_duplicates(subset=['raceId', 'resultsDriverId', 'Session'])

# drop columns if they exist
for col in ['raceId.1', 'raceId.2', 'raceId.3', 'raceId_1', 'PitOutTime', 'PitInTime', 'best_s1', 'best_s2', 'best_s3', 'Time', 'Laptime', 'constructorName_y', 'grandPrixYear_y', 'round.1', 'resultsDriverId_y']:
    if col in all_practice_laps_with_driver_names.columns:
        all_practice_laps_with_driver_names = all_practice_laps_with_driver_names.drop(columns=[col])

# Find the fastest practice lap for each driver across FP1, FP2, FP3
practice_sessions = ['FP1', 'FP2', 'FP3']
fastest_practice_laps = (
    all_practice_laps_with_driver_names[
        all_practice_laps_with_driver_names['Session'].isin(practice_sessions)
    ]
    .sort_values(['Year', 'Round', 'Driver', 'LapTime_sec'])
    .drop_duplicates(subset=['Year', 'Round', 'Driver'], keep='first')
    .copy()
)
fastest_practice_laps.rename(columns={'LapTime_sec': 'FastestPracticeLap_sec'}, inplace=True)

print(all_practice_laps_with_driver_names.columns.tolist())

# Find the best practice lap for each constructor in each event
best_constructor_practice = (
    all_practice_laps_with_driver_names[
        all_practice_laps_with_driver_names['Session'].isin(practice_sessions)
    ]
    .sort_values(['Year', 'Round', 'constructorName', 'LapTime_sec'])
    .drop_duplicates(subset=['Year', 'Round', 'constructorName'], keep='first')
    .copy()
)
best_constructor_practice.rename(columns={'LapTime_sec': 'BestConstructorPracticeLap_sec'}, inplace=True)

# Merge fastest driver practice lap (across all practice sessions)
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_with_driver_names,
    fastest_practice_laps[['Year', 'Round', 'Driver', 'FastestPracticeLap_sec']],
    on=['Year', 'Round', 'Driver'],
    how='left'
)

# Merge best constructor practice lap (across all practice sessions)
all_practice_laps_with_driver_names = pd.merge(
    all_practice_laps_with_driver_names,
    best_constructor_practice[['Year', 'Round', 'constructorName', 'BestConstructorPracticeLap_sec']],
    on=['Year', 'Round', 'constructorName'],
    how='left'
).drop_duplicates(subset=['raceId', 'resultsDriverId', 'Session'])

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
fp_laps_deduped.rename(columns={'resultsDriverId': 'driverId'}, inplace=True)
# print(fp_laps_deduped.columns.tolist())

# Select only the columns you want to keep
columns_to_keep = [
    'Year', 'Round', 'raceId', 'Driver', 'driverId', 'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec',
    'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'best_theory_lap_sec', 'Session'
]

# print(fp_laps_deduped.columns.tolist())

# Rename driverId.1 to driverId if it exists
# if 'driverId.1' in fp_laps_deduped.columns:
#     fp_laps_deduped = fp_laps_deduped.rename(columns={'driverId.1': 'driverId'})

# Now select columns
fp_laps_final = fp_laps_deduped[columns_to_keep]

# # After all_practice_laps_with_driver_names is created and before saving:
# delta_csv = path.join(DATA_DIR, 'clean_dirty_air_deltas.csv')
# if os.path.exists(delta_csv):
#     df_clean_dirty = pd.read_csv(delta_csv, sep='\t')
#     # Merge on Year, Round, Session, Driver
#     all_practice_laps_with_driver_names = pd.merge(
#         all_practice_laps_with_driver_names,
#         df_clean_dirty,
#         on=['Year', 'Round', 'Session', 'Driver'],
#         how='left'
#     )

# Save to a new slimmed-down CSV
if not fp_laps_deduped.empty:
    # Optionally, ensure all columns exist (fill missing with NaN)
    for col in columns_to_keep:
        if col not in fp_laps_deduped.columns:
            fp_laps_deduped[col] = pd.NA
    fp_laps_final = fp_laps_deduped[columns_to_keep]
    fp_laps_final.to_csv(path.join(DATA_DIR, 'practice_best_fp1_fp2.csv'), sep='\t', index=False)
    print("Saved slimmed practice file to practice_best_fp1_fp2.csv")
else:
    print("No new FP1/FP2 laps found. Nothing to save.")
