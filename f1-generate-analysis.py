#import datetime as dt
from datetime import date, timedelta
import datetime
import pandas as pd
from os import path
import os
import openmeteo_requests
import requests_cache
import numpy as np
from retry_requests import retry
from openmeteo_sdk.Variable import Variable
from pit_constants import PIT_LANE_TIME_S, TYPICAL_STATIONARY_TIME_S
import fastf1
from fastf1.ergast import Ergast

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year
raceNoEarlierThan = current_year - 10

if os.environ.get('LOCAL_RUN') == '1':

    fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

def quantile_bin_feature(df, feature, q=10, suffix='_bin', dropna=True):
    """
    Automatically bin a high-cardinality numeric feature into quantiles.

    Parameters:
    - df: pandas DataFrame containing the feature.
    - feature: str, name of the column to bin.
    - q: int, number of quantiles (default 10 for deciles).
    - suffix: str, suffix for the new binned column (default '_bin').
    - dropna: bool, if True, drop rows with NaN in the feature before binning.

    Returns:
    - df: DataFrame with a new column '<feature><suffix>' containing bin labels (0 to q-1).
    """

    # Optionally drop NaN values for binning
    if dropna:
        valid = df[feature].dropna()
    else:
        valid = df[feature]

    # Use pd.qcut to bin into quantiles
    try:
        bins = pd.qcut(valid, q=q, labels=False, duplicates='drop')
        # Reindex to original DataFrame
        df[feature + suffix] = bins.reindex(df.index)
    except ValueError as e:
        print(f"Could not bin feature '{feature}': {e}")
        df[feature + suffix] = np.nan

    return df


## Results and Qualifying
drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json')) 
race_results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json')) 
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 
constructors = pd.read_json(path.join(DATA_DIR, 'f1db-constructors.json')) 
qualifying_json = pd.read_json(path.join(DATA_DIR, 'f1db-races-qualifying-results.json')) 
qualifying_csv = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t') 
grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json')) 
fp1 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-1-results.json')) 
fp2 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-2-results.json')) 
fp3 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-3-results.json')) 
# fp4 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-4-results.json')) 
current_practices = pd.read_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t') 
practice_best = pd.read_csv(path.join(DATA_DIR, 'practice_best_fp1_fp2.csv'), sep='\t')
pitstops = pd.read_json(path.join(DATA_DIR, 'f1db-races-pit-stops.json'))
all_laps = pd.read_csv(path.join(DATA_DIR, 'all_laps.csv'), sep='\t')
constructor_standings = pd.read_csv(path.join(DATA_DIR, 'constructor_standings.csv'), sep='\t')
driver_standings = pd.read_csv(path.join(DATA_DIR, 'driver_standings.csv'), sep='\t')


# Standardize Andrea Kimi Antonelli to Kimi Antonelli everywhere based on [f1db/f1db] Release v2025.15.0
# After loading drivers
drivers['id'] = drivers['id'].str.replace('^andrea-', '', regex=True)
drivers['name'] = drivers['name'].replace({'Andrea Kimi Antonelli': 'Kimi Antonelli'})

qualifying_csv['driverId'] = qualifying_csv['driverId'].str.replace('^andrea-', '', regex=True)
qualifying_csv['FullName'] = qualifying_csv['FullName'].replace({'Andrea Kimi Antonelli': 'Kimi Antonelli'})

# After loading race_results
# race_results['driverId'] = race_results['resultsDriverId'].str.replace('^andrea-', '', regex=True)

# After loading practice files (if they have driverId or name)
current_practices['resultsDriverId'] = current_practices['resultsDriverId'].str.replace('^andrea-', '', regex=True)
# current_practices['name'] = current_practices['name'].replace({'Andrea Kimi Antonelli': 'Kimi Antonelli'})
# current_practices['firstName'] = current_practices['firstName'].replace({'Andrea': 'Kimi'})

# practice_best['driverId'] = practice_best['driverId'].str.replace('^andrea-', '', regex=True)
# practice_best['resultsDriverName'] = practice_best['resultsDriverName'].replace({'Andrea Kimi Antonelli': 'Kimi Antonelli'})

driver_standings['driverId'] = driver_standings['driverId'].str.replace('^andrea-', '', regex=True)
driver_standings['driverName'] = driver_standings['driverName'].replace({'Andrea Kimi Antonelli': 'Kimi Antonelli'})

qualifying = pd.merge(
    qualifying_json,
    qualifying_csv[['q1_sec', 'q1_pos', 'q2_sec', 'q2_pos', 'q3_sec', 'q3_pos', 'best_qual_time', 'teammate_qual_delta', 'raceId', 'driverId', ]],
    left_on=['raceId', 'driverId'],
    right_on=['raceId', 'driverId'],
    how='right',
    suffixes=('_json', '_csv')
)

# qualifying.to_csv(path.join(DATA_DIR, 'f1QualifyingTest.csv'), index=False, sep='\t')

##### Pit Stops

pitstops = pitstops[pitstops['year'].between(raceNoEarlierThan, current_year)]

pitstops = pd.merge(pitstops, races[['id', 'grandPrixId']], left_on='raceId', right_on='id', how='left')
race_control = pd.read_csv(path.join(DATA_DIR, 'all_race_control_messages.csv'), sep='\t')

# Rename columns first
race_control.rename(columns={'Lap': 'red_flag_lap', 'id': 'raceId'}, inplace=True)

# Now filter for red flags
red_flags = race_control[
    race_control['Flag'].str.contains('RED', case=False, na=False)
]

# Get raceId and lap for each red flag
red_flags_df = red_flags[['raceId', 'red_flag_lap']]
print(red_flags_df.head())

# Merge to mark pit stops on red flag laps
pitstops = pd.merge(
    pitstops,
    red_flags_df,
    how='left',
    left_on=['raceId', 'lap'],
    right_on=['raceId', 'red_flag_lap'],
    indicator=True
)
# Pit stops during red flag laps will have _merge == 'both'
pitstops['in_red_flag'] = pitstops['_merge'] == 'both'
pitstops_clean = pitstops[~pitstops['in_red_flag']]

# print(pitstops['timeMillis'].describe())
# print(pitstops['timeMillis'].head(10))
# print(pitstops[pitstops['timeMillis'] > 10000]) 

# Group by race and calculate the mean pit stop time
race_avg = pitstops_clean.groupby('raceId')['timeMillis'].mean().reset_index(name='race_avg_timeMillis')

# Merge the average back to the pitstops DataFrame
pitstops_clean = pitstops_clean.merge(race_avg, on='raceId', how='left')

# Calculate the delta (difference) from the race average
pitstops_clean['delta_from_race_avg'] = (pitstops_clean['timeMillis'] - pitstops_clean['race_avg_timeMillis'])/ 1000  # Convert to seconds

pitstops_grouped = pitstops_clean.groupby(['raceId', 'driverId', 'constructorId', 'round', 'year', 'grandPrixId']).agg(numberOfStops = ('stop', 'count'), averageStopTimeMillis = ('timeMillis', 'mean'), 
                    totalStopTimeMillis = ('timeMillis', 'sum'), delta_from_race_avg=('delta_from_race_avg', 'mean')).reset_index()

pitstops_grouped['averageStopTime'] = (pitstops_grouped['averageStopTimeMillis'] / 1000).round(2)
pitstops_grouped['totalStopTime'] = (pitstops_grouped['totalStopTimeMillis'] / 1000).round(2)

# Add a column with the pit lane time constant for each grandPrixId
pitstops_grouped['pit_lane_time_constant'] = pitstops_grouped['grandPrixId'].map(PIT_LANE_TIME_S).round(2)

# If you want to fill missing values with a default (e.g., np.nan or a specific value)
pitstops_grouped['pit_lane_time_constant'] = pitstops_grouped['pit_lane_time_constant'].fillna(np.nan)

pitstops_grouped['pit_stop_delta'] = (pitstops_grouped['averageStopTime'] - pitstops_grouped['pit_lane_time_constant']).round(2)

pitstops_grouped.to_csv(path.join(DATA_DIR, 'f1PitStopsData_Grouped.csv'), columns=['raceId', 'driverId', 'constructorId', 'numberOfStops', 'averageStopTime', 'totalStopTime', 'round', 'year', 'pit_lane_time_constant', 'grandPrixId', 'pit_stop_delta', 'delta_from_race_avg'], sep='\t', index=False)



# # Calculate historical averages per driver
# historical_avg = pivot.groupby('Driver')[[f'delta_lap_{lap}' for lap in lap_targets]].mean().reset_index()

# print(historical_avg.sort_values(f'delta_lap_5', ascending=False))

# print(historical_avg.head(50))
# print(historical_avg.columns.tolist())

dnf_by_driver = race_results[race_results['reasonRetired'].notna()].groupby(['driverId']).size().reset_index(name='DNFCount')
# print(dnf_by_driver.head(50))

races_and_grandPrix = pd.merge(races, grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_races', '_grandPrix'])
races_and_grandPrix.rename(columns={'id_races': 'raceIdFromGrandPrix', 'id_grandPrix': 'grandPrixRaceId', 'fullName': 'grandPrixName', 'laps': 'grandPrixLaps', 'year': 'grandPrixYear'}, inplace=True)

results_and_drivers = pd.merge(race_results, drivers, left_on='driverId', right_on='id', how='inner', suffixes=['_results', '_drivers'])
results_and_drivers.rename(columns={'year': 'resultsYear', 'driverId': 'resultsDriverId', 'qualificationPositionNumber': 'resultsQualificationPositionNumber', 'positionNumber': 'resultsFinalPositionNumber', 'name': 'resultsDriverName', 'gridPositionNumber': 'resultsStartingGridPositionNumber', 'reasonRetired': 'resultsReasonRetired'}, inplace=True)

results_and_drivers = results_and_drivers[results_and_drivers['resultsYear'] >= raceNoEarlierThan]

# results_and_drivers[['totalPolePositions', 'totalFastestLaps', 'totalRaceWins', 'totalRaceEntries', 'totalRaceStarts', 'totalPodiums', 'bestRaceResult', 'bestStartingGridPosition', 'totalRaceLaps']]

results_and_drivers_and_constructors = pd.merge(results_and_drivers, constructors, left_on='constructorId', right_on='id', how='inner', suffixes=['_results', '_constructors'])
results_and_drivers_and_constructors.rename(columns={'name': 'constructorName', 'totalRaceStarts_constructors': 'constructorTotalRaceStarts', 'totalRaceEntries_constructors': 'constructorTotalRaceEntries', 'totalRaceWins_constructors': 'constructorTotalRaceWins', 
                                                     'total1And2Finishes': 'constructorTotal1And2Finishes', 'bestChampionshipPosition_results': 'driverBestChampionshipPosition', 'bestStartingGridPosition_results': 'driverBestStartingGridPosition', 
                                                     'bestRaceResult_results': 'driverBestRaceResult', 'totalChampionshipWins_results': 'driverTotalChampionshipWins',
                                                     'totalRaceEntries_results': 'driverTotalRaceEntries', 'totalRaceStarts_results': 'driverTotalRaceStarts', 'totalRaceWins_results': 'driverTotalRaceWins', 'totalRaceLaps_results': 'driverTotalRaceLaps', 
                                                     'totalPolePositions_results': 'driverTotalPolePositions', 'timeMillis_results': 'timeMillis',  
                                                     'totalPodiums_results': 'driverTotalPodiums', 'totalPodiumRaces': 'constructorTotalPodiumRaces', 'totalPolePositions_constructors': 'constructorTotalPolePositions', 'totalFastestLaps_constructors': 'constructorTotalFastestLaps'}, inplace=True)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying.columns.tolist())

results_and_drivers_and_constructors_and_grandprix = pd.merge(results_and_drivers_and_constructors, races_and_grandPrix, left_on='raceId', right_on='raceIdFromGrandPrix', how='inner', suffixes=['_results', '_grandprix'])

results_and_drivers_and_constructors_and_grandprix_and_qualifying = pd.merge(results_and_drivers_and_constructors_and_grandprix, qualifying, left_on=['raceIdFromGrandPrix', 'resultsDriverId'], right_on=['raceId', 'driverId'], how='inner', suffixes=['_results', '_qualifying']) 

results_and_drivers_and_constructors_and_grandprix_and_qualifying.rename(columns={'constructorName_qualifying': 'constructorName'}, inplace=True)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying.columns.tolist())

# results_and_drivers_and_constructors_and_grandprix_and_qualifying[['grandPrixName', 'resultsQualificationPositionNumber', 'abbreviation_results', 'raceId_results', 'resultsDriverId', # 'q1', 'q2', 'q3', 
#                                                                    'resultsYear', 'constructorName','resultsDriverId', 'resultsDriverName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
#                                       'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 
#                                       'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'grandPrixYear', 'raceIdFromGrandPrix', 'grandPrixRaceId', 'circuitType']]

fp1_fp2 = pd.merge(fp1, fp2, on=['raceId', 'driverId'], how='left', suffixes=['_fp1', '_fp2'])
fp1_fp2_fp3 = pd.merge(fp1_fp2, fp3, on=['raceId', 'driverId'], how='left', suffixes=['_fp1_2', '_fp3'])
# fp1_fp2_fp3_fp4 = pd.merge(fp1_fp2_fp3, fp4, on=['raceId', 'driverId'], how='left', suffixes=['_fp1_fp2_fp3', '_fp4'])
# print(fp1_fp2_fp3.columns.tolist())

fp1_fp2_fp3.rename(columns={'driverId': 'fpDriverId', 'raceId': 'fpRaceId', 'positionNumber_fp1': 'fp1PositionNumber', 'time_fp1': 'fp1Time', 'gap_fp1': 'fp1Gap', 'interval_fp1': 'fp1Interval', 
'positionNumber_fp2': 'fp2PositionNumber', 'time_fp2': 'fp2Time', 'gap_fp2': 'fp2Gap', 'interval_fp2': 'fp2Interval', 
'positionNumber': 'fp3PositionNumber', 'gap': 'fp3Gap', 'interval': 'fp3Interval', 'time': 'fp3Time'
# 'positionNumber_fp4': 'fp4PositionNumber', 'time_fp4': 'fp4Time', 'gap_fp4': 'fp4Gap', 'interval_fp4': 'fp4Interval' 
}, inplace=True)

# Drop 'time' from the right DataFrame if you don't need it
fp1_fp2_fp3 = fp1_fp2_fp3.drop(columns=['time', 'round'], errors='ignore')
#fp1_fp2_fp3_fp4 = fp1_fp2_fp3_fp4.drop(columns=['time'], errors='ignore')


results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying, fp1_fp2_fp3, left_on=['raceId_results', 'resultsDriverId'], right_on=['fpRaceId','fpDriverId' ], how='left', suffixes=['_results', '_practices']) 
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices, dnf_by_driver, left_on='resultsDriverId', right_on='driverId', how='left', suffixes=['_results', '_dnf'])
# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices, pitstops_grouped
    [['numberOfStops', 'averageStopTime', 'totalStopTime', 'pit_lane_time_constant', 'pit_stop_delta', 'raceId','driverId', 'delta_from_race_avg']], left_on=['raceId_results', 'resultsDriverId'], right_on=['raceId','driverId'], how='left', suffixes=['', '_pitstops'])
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=['driverId_pitstops', 'raceId_pitstops', 'raceId', 'driverId'], inplace=True, errors='ignore')

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverAge'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .apply(lambda row: (pd.to_datetime(row['date']) - pd.to_datetime(row['dateOfBirth'])).days // 365, axis=1)
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['short_date'] = pd.to_datetime(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['date']).dt.strftime('%Y-%m-%d')
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsPodium'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsTop5'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=5)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsTop10'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=10)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['DNF'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsReasonRetired'].notnull())
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverDNFCount'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['DNFCount'].fillna(0).astype(int)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverDNFAvg'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['DNFCount'].fillna(0).astype(int) / results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalRaceEntries'].fillna(1).astype(int))
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber']].mean(axis=1)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['fp3PositionNumber', 'fp2PositionNumber', 'fp1PositionNumber']].bfill(axis=1).iloc[:, 0]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['q3_sec', 'q2_sec', 'q1_sec']].bfill(axis=1).iloc[:, 0]

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying.columns.tolist())

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['bestQualifyingTime']]

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying.columns.tolist())
# Calculate pole position time per race
pole_times = qualifying[qualifying['positionNumber'] == 1][['raceId', 'q1', 'q2', 'q3']].copy()
# pole_times = qualifying[qualifying['q3_pos'] == 1][['raceId', 'q3_sec', 'q2_sec', 'q1_sec']].copy()


# Convert qualifying times to seconds (reuse your time_to_seconds function)
def time_to_seconds(time_str):
    try:
        if pd.isnull(time_str) or not isinstance(time_str, str):
            return None
        minutes, seconds = time_str.split(':')
        return int(minutes) * 60 + float(seconds)
    except ValueError:
        return None

# for col in ['q3', 'q2', 'q1']:
#     pole_times[col + '_sec'] = pole_times[col].apply(time_to_seconds)

# Get the best pole time for each race (prefer q3, then q2, then q1)
pole_times['pole_time_sec'] = pole_times[['q3', 'q2', 'q1']].bfill(axis=1).iloc[:, 0]

# Merge pole time into your main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    pole_times[['raceId', 'pole_time_sec']],
    left_on='raceId_results',
    right_on='raceId',
    how='left'
)

# # Always create the column, even if all values are NaN
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime'].apply(time_to_seconds)
# )






# Ensure the 'bestQualifyingTime' column is not null or invalid
# if results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'].notnull().any():
    # Convert 'bestQualifyingTime' to seconds
    # def time_to_seconds(time_str):
    #     try:
    #         # Check if the time_str is valid
    #         if pd.isnull(time_str) or not isinstance(time_str, str):
    #             return None
    #         # Split the time string into minutes and seconds
    #         minutes, seconds = time_str.split(':')
    #         return int(minutes) * 60 + float(seconds)
    #     except ValueError:
    #         # Handle invalid or missing time formats
    #         return None

    # Apply the conversion function to the column
    # results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'].apply(time_to_seconds)
    # print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())
df_clean_dirty = pd.read_csv(path.join(DATA_DIR, 'clean_dirty_air_deltas.csv'), sep='\t')
df_clean_dirty.columns = df_clean_dirty.columns.str.strip()
# print("df_clean_dirty columns:", df_clean_dirty.columns.tolist())
# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['abbreviation_results'].unique())
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['abbreviation_results'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['abbreviation_results'].str.strip()

df_clean_dirty = pd.read_csv(path.join(DATA_DIR, 'clean_dirty_air_deltas.csv'), sep='\t')
df_clean_dirty_pivot = df_clean_dirty.pivot_table(
    index=['Year', 'Round', 'Driver'],
    columns='Session',
    values=['CleanAirAvg', 'DirtyAirAvg', 'Delta', 'NumCleanLaps', 'NumDirtyLaps']
).reset_index()

# Flatten the MultiIndex columns
df_clean_dirty_pivot.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0]
    for col in df_clean_dirty_pivot.columns.values
]

# print(df_clean_dirty.head(50))
# Merge on Year, Round, Session, Driver
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    df_clean_dirty_pivot,
    # df_clean_dirty, #[['CleanAirAvg', 'DirtyAirAvg', 'Delta', 'Session']],
    left_on=['resultsYear', 'round_results', 'abbreviation_results'],
    right_on=['Year', 'Round', 'Driver'],
    how='left'
)

    # print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.head(50))
    # results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1_with_clean.csv'), sep='\t', index=False)

    # Display the converted column
    # print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['bestQualifyingTime', 'bestQualifyingTime_sec']].head())

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.to_list()
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1Test_300.csv'))
# else:
#     print("No valid times found in 'bestQualifyingTime' column.")

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1_results_test_line300.csv'), sep='\t', index=False)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

# # Calculate qualifying gap to pole
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'] = pd.to_numeric(
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'], errors='coerce'
# )

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'].head())

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec'] = pd.to_numeric(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec'], errors='coerce'
)
# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec'].head())
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_gap_to_pole'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec']
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = quantile_bin_feature(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    'qualifying_gap_to_pole', q=5
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q3Top10'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] <=10)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q2End'] = ((results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] >10) & (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] <=15))
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q1End'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] >15)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['activeDriver'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixYear'] == current_year)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['streetRace'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['circuitType'] == 'STREET')

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['trackRace'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['circuitType'] == 'RACE')

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['avgLapPace'] = ((results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['timeMillis_results']/1000) / results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixLaps']  )
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['finishingTime'] = ((results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['timeMillis_results']/1000))

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualPos_x_practicePos'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsPracticePositionNumber']
# )

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualPos_x_avg_practicePos'] = (
    np.where(
        (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] > 0) &
        (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] > 0),
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] *
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'],
        np.nan
    )
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualPos_x_last_practicePos'] = (
    np.where(
        (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] > 0) &
        (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber'] > 0),
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] *
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber'],
        np.nan
    )
)

# Count the number of unique active years for each driver
yearsActive = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverId')['grandPrixYear'].nunique().reset_index()

yearsActiveGroup = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverId')['grandPrixYear'].nunique().reset_index()
yearsActiveGroup.rename(columns={'grandPrixYear': 'yearsActive'}, inplace=True)

# Merge the active years count back into the main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.merge(
    yearsActiveGroup, on='resultsDriverId', how='left')

# Rename columns to avoid conflicts because there were duplicates column names post merge
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'round_results': 'round', 'time_results': 'time'}, inplace=True)

# Ensure 'time' is a string
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['time'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['time'].astype(str)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

if current_practices['resultsDriverId'].isnull().all():
    current_practices['resultsDriverId'] = current_practices['resultsDriverId_1']

##### It appears the issue with blank best_s1, etc, is that when the best_s1_sec field has data, there is no driverId or raceId, so we are not bringing the data over correct
##### from the practices page.

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices, current_practices, 
        left_on=['raceId_results', 'resultsDriverId'], right_on=['raceId', 'resultsDriverId'], how='left', suffixes=['', '_practices'])
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['resultsDriverId', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec']].head(50)
# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())



# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['grandPrixName', 'best_s1_sec_practices', 'best_s2_sec_practices', 'best_s2_sec_practices', 'best_theory_lap_sec_practices', 'SpeedI1_mph_practices', 'SpeedI2_mph_practices', 'SpeedFL_mph_practices', 'SpeedST_mph_practices']]

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1Test1.csv'))

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['resultsDriverId', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec']].tail(50))

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'round_results': 'round', 'turns_results': 'turns', 'circuitId_results': 'circuitId',
#                                                                                                  'best_s1_sec_results': 'best_s1_sec', 'best_s2_sec_results': 'best_s2_sec', 'best_s3_sec_results': 'best_s3_sec', #'best_theory_lap_sec_results': 'best_theory_lap_sec',
#                                                                                                  'SpeedI1_mph_results': 'SpeedI1_mph', 'SpeedI2_mph_results': 'SpeedI2_mph', 'SpeedFL_mph_results': 'SpeedFL_mph', 'SpeedST_mph_results': 'SpeedST_mph'
#                                                                                                 }, inplace=True)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1Test2.csv'))

# Read your main DataFrame (assuming it's already loaded as results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['best_s2_sec']].head())

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.sort_values(['constructorName', 'grandPrixYear', 'raceId_results'])
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_recent_form_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('constructorName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_recent_form_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('constructorName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_positionsGained_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['positionsGained']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_positionsGained_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['positionsGained']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_dnf_rate_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['DNF']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_starting_position_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['resultsStartingGridPositionNumber']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_starting_position_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['resultsStartingGridPositionNumber']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)


# Rolling median of final position for last 3 races
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_median_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=3, min_periods=1).median().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_median_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=5, min_periods=1).median().shift(1))
)

# Rolling best (min) final position for last 3 races
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_best_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=3, min_periods=1).min().shift(1))
)

# Rolling worst (max) final position for last 3 races
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_worst_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['resultsFinalPositionNumber']
    .transform(lambda x: x.rolling(window=3, min_periods=1).max().shift(1))
)

# Rolling average of DNFs for last 3 races
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_dnf_rate_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['DNF']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_dnf_rate_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['DNF']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

# Rolling average of DNFs for last 3 races
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_dnf_rate_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['constructorName', 'grandPrixYear', 'raceId_results'])
    .groupby('constructorName')['DNF']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

# Rolling average of DNFs for last 5 races
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_dnf_rate_5_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['constructorName', 'grandPrixYear', 'raceId_results'])
    .groupby('constructorName')['DNF']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
)

# Rolling average of positions gained for last 3 races
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_positions_gained_3_races'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['positionsGained']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_vs_season'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_3_races'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverId')['resultsFinalPositionNumber'].transform('mean')
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_std'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
        ['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber']
    ].std(axis=1)
)

team_avg_practice = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(
    ['raceId_results', 'constructorId_results']
)['averagePracticePosition'].transform('mean')
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_avg_practice_position'] = team_avg_practice



# Calculate practice session improvements for each driver
for session_pair in [('fp1PositionNumber', 'fp2PositionNumber'), ('fp2PositionNumber', 'fp3PositionNumber'), ('fp1PositionNumber', 'fp3PositionNumber')]:
    col_name = f'practice_position_improvement_{session_pair[0][2:4]}_{session_pair[1][2:4]}'
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[col_name] = (
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[session_pair[0]] -
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[session_pair[1]]
    )

for session_pair in [('fp1Time', 'fp2Time'), ('fp2Time', 'fp3Time'), ('fp1Time', 'fp3Time')]:
    time1_sec = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[session_pair[0]].apply(time_to_seconds)
    time2_sec = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[session_pair[1]].apply(time_to_seconds)
    col_name = f'practice_time_improvement_{session_pair[0][2:4]}_{session_pair[1][2:4]}'
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[col_name] = time1_sec - time2_sec

    time1_sec = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[session_pair[0]].apply(time_to_seconds)
    time2_sec = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[session_pair[1]].apply(time_to_seconds)
    col_name = f'practice_time_improvement_{session_pair[0][-4:].lower()}_{session_pair[1][-4:].lower()}'
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[col_name] = time1_sec - time2_sec

# Calculate average and last final position per driver per track
track_position_stats = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverId', 'circuitId', 'grandPrixYear', 'raceId_results'])
    .groupby(['resultsDriverId', 'circuitId'])
    .agg(
        avg_final_position_per_track=('resultsFinalPositionNumber', 'mean'),
        last_final_position_per_track=('resultsFinalPositionNumber', 'last')
    )
    .reset_index()
)

# Merge these stats into your main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    track_position_stats,
    on=['resultsDriverId', 'circuitId'],
    how='left'
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    constructor_standings[['id', 'Points', 'constructorRank']],
    left_on=['constructorId_results'],
    right_on=['id'],
    how='left'
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'Points': 'constructorPoints'}, inplace=True)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    driver_standings[['driverId', 'points', 'driverRank']],
    left_on=['resultsDriverId'],
    right_on=['driverId'],
    how='left'
)

if 'points' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'points': 'driverPoints'}, inplace=True)
elif 'points_y' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'points_y': 'driverPoints'}, inplace=True)

print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_x_constructor_wins'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorTotalRaceWins']
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1PositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber']
)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_x_qual'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_vs_track_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['avg_final_position_per_track']
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_penalty'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_penalty_x_constructor'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_penalty'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorTotalRaceWins']
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_x_qual'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_3_races'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_std_x_qual'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_std'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_x_constructor_rank'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorRank']
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_rank_x_constructor_rank'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverRank'] * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorRank']
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_x_qual'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']) * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_gap_to_teammate'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['resultsQualificationPositionNumber'].transform('mean')
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_gap_to_teammate'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['averagePracticePosition'].transform('mean')
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['total_experience'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalRaceStarts'] + results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorTotalRaceStarts']
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['podium_potential'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalPodiums'] + results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorTotalRaceWins']
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['street_experience'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['streetRace'] * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalRaceStarts']
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['track_experience'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['trackRace'] * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalRaceStarts']
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_ratio'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_3_races'] / (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_5_races'] + 1e-6)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_form_ratio'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_recent_form_3_races'] / (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_recent_form_5_races'] + 1e-6)

# Lap time delta vs. session fastest (e.g., FP1)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1_lap_delta_vs_best'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1Time'].apply(time_to_seconds) /
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('raceId_results')['fp1Time'].transform(lambda x: x.apply(time_to_seconds).min())
)

# Starting position × average pit stop time
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_x_avg_pit_time'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averageStopTime']
)

# Pit stop count × pit stop delta
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_count_x_pit_delta'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['numberOfStops'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_stop_delta']
)

# Pit stop frequency rate
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_stop_rate'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['numberOfStops'] /
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixLaps']
)

# Last race pace / track typical pace
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['last_race_vs_track_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_form_3_races'] /
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['avg_final_position_per_track'] + 1e-6)
)
 
# Race Pace vs. Field Median
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['race_pace_vs_median'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['avgLapPace'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('raceId_results')['avgLapPace'].transform('median')
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['historical_race_pace_vs_median'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['race_pace_vs_median']
    .shift(1)
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['historical_avgLapPace'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['avgLapPace']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

# Top Speed Rank (lower is better)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['top_speed_rank'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('raceId_results')['SpeedFL_mph'].rank(ascending=False)
)


# Power-to-Corner Ratio
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['power_to_corner_ratio'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SpeedFL_mph'] /
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['turns']
)

# Downforce Demand Score (sector2_avg_time / track_length)
if 'best_s2_sec' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['downforce_demand_score'] = (
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_s2_sec'] /
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['courseLength']
    )

# Calculate average and last final position per constructor per track
constructor_track_position_stats = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['constructorId_results', 'circuitId', 'grandPrixYear', 'raceId_results'])
    .groupby(['constructorId_results', 'circuitId'])
    .agg(
        avg_final_position_per_track_constructor=('resultsFinalPositionNumber', 'mean'),
        last_final_position_per_track_constructor=('resultsFinalPositionNumber', 'last')
    )
    .reset_index()
)

# Merge these stats into your main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    constructor_track_position_stats,
    on=['constructorId_results', 'circuitId'],
    how='left'
)


results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1_features_before_practice_update.csv'), index=False, sep='\t')
# List of columns to update from the reference table
columns_to_update = [
    'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec',
    'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',
    'best_theory_lap_sec', 'Session', 
]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1_features_before_after_update.csv'), index=False, sep='\t')


# Drop all 'raceId' and 'driverId' columns except those with suffixes
for col in ['raceId', 'driverId']:
    cols = [c for c in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns if c == col]
    if len(cols) > 1:
        # Drop all but the first occurrence
        first = True
        for c in cols:
            if first:
                first = False
            else:
                results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=[c])

# Merge in the updated columns using raceId and driverId as keys
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    practice_best[['raceId', 'driverId'] + columns_to_update],
    left_on=['raceId_results', 'resultsDriverId'],
    right_on=['raceId', 'driverId'],
    how='left'
).drop_duplicates(['raceId_results', 'resultsDriverId'])


# (Optional) Drop the merge keys from the reference if you don't want them in your final DataFrame
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=['raceId', 'driverId'])

# Now your main DataFrame has the updated practice columns with no suffixes or duplicates!

# Read grouped race control messages
race_control = pd.read_csv(path.join(DATA_DIR, 'race_control_messages_grouped_with_dnf.csv'), sep='\t')

# Create binary SafetyCarStatus column
race_control['SafetyCarStatus'] = (race_control['SafetyCarStatus'] > 0).astype(int)

# Merge into your main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    race_control[['raceId', 'SafetyCarStatus']],
    left_on='raceId_results',
    right_on='raceId',
    how='left'
)

# Fill NaN with 0 for races with no safety car info
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SafetyCarStatus'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SafetyCarStatus'].fillna(0).astype(int)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'engineManufacturerId_results': 'engineManufacturerId'})

# Calculate standard deviation of finishing position by driver
driver_position_std = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverId')['resultsFinalPositionNumber'].std().reset_index()
driver_position_std.rename(columns={'resultsFinalPositionNumber': 'finishing_position_std_driver'}, inplace=True)

# Calculate standard deviation of finishing position by constructor
constructor_position_std = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('constructorId_results')['resultsFinalPositionNumber'].std().reset_index()
constructor_position_std.rename(columns={'resultsFinalPositionNumber': 'finishing_position_std_constructor'}, inplace=True)

# Merge both into the main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.merge(
    driver_position_std, on='resultsDriverId', how='left'
).merge(
    constructor_position_std, on='constructorId_results', how='left'
)

# Specify the laps you want to compare to the start
lap_targets = [2, 5, 10, 15, 20]

# Get only the laps of interest (start and targets)
laps_of_interest = [1] + lap_targets
lap_df = all_laps[all_laps['LapNumber'].isin(laps_of_interest)]

# Pivot so each row is a driver/race, columns for each lap's position
pivot = lap_df.pivot_table(
    index=['year', 'round', 'Driver'],
    columns='LapNumber',
    values='Position'
).reset_index()

pivot.rename(columns={1: 'start_position'}, inplace=True)

# Calculate deltas for each target lap
for lap in lap_targets:
    pivot[f'delta_lap_{lap}'] = pivot['start_position'] - pivot.get(lap, pd.NA)

# Calculate historical averages per driver
historical_avg = pivot.groupby('Driver')[[f'delta_lap_{lap}' for lap in lap_targets]].mean().reset_index()

# Rename columns to add "_historical" except for 'Driver'
historical_avg = historical_avg.rename(
    columns={col: f"{col}_historical" for col in historical_avg.columns if col != "Driver"}
)

# print(historical_avg.head(50))

# Merge these deltas into your main results DataFrame
# First, make sure you have the correct keys to merge on (year, round, driver name)
# Only keep the first 'round' column if duplicates exist
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.loc[:,~results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.duplicated()]
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    pivot[['year', 'round', 'Driver'] + [f'delta_lap_{lap}' for lap in lap_targets]],
    left_on=['grandPrixYear', 'round', 'abbreviation_results'],
    right_on=['year', 'round', 'Driver'],
    how='left'
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    historical_avg,
    left_on=['abbreviation_results'],
    right_on=['Driver'],
    how='left'
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    constructors[['id', 'name', 'totalRaceEntries', 'totalRaceStarts', 'totalRaceWins', 'total1And2Finishes', 'totalPodiumRaces', 'totalPolePositions', 'totalFastestLaps']],
    left_on='constructorId_results',
    right_on='id',
    how='left'
)#.drop(columns=['constructorId'])

# Standardize Andrea Kimi Antonelli to Kimi Antonelli everywhere based on [f1db/f1db] Release v2025.15.0
# Secondary check in case initial changes don't take as part of merges

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsDriverId'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsDriverId']
    .str.replace('^andrea-', '', regex=True)
)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsDriverName'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsDriverName']
    .replace({'Andrea Kimi Antonelli': 'Kimi Antonelli'})
)


# Positions Gained in First Lap %
# Historical Positions Gained in First Lap %
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['positions_gained_first_lap_pct'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverId', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverId', group_keys=False)[['delta_lap_2', 'resultsStartingGridPositionNumber']]
    .apply(lambda df: (df['delta_lap_2'] / df['resultsStartingGridPositionNumber']).shift(1))
    .reset_index(level=0, drop=True)
)


# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={
    'name': 'constructorName',
    'totalRaceEntries': 'constructorTotalRaceEntries',
    'totalRaceStarts': 'constructorTotalRaceStarts',
    'totalRaceWins': 'constructorTotalRaceWins',
    'total1And2Finishes': 'constructorTotal1And2Finishes',
    'totalPodiumRaces': 'constructorTotalPodiumRaces',
    'totalPolePositions': 'constructorTotalPolePositions',
    'totalFastestLaps': 'constructorTotalFastestLaps',
    'FastestPracticeLap_sec': 'driverFastestPracticeLap_sec',
    'BestConstructorPracticeLap_sec_x': 'BestConstructorPracticeLap_sec',
}, inplace=True)

# Calculate qualifying consistency (std) for each driver across all races and sessions
qualifying_cols = ['resultsQualificationPositionNumber', 'q1_pos', 'q2_pos', 'q3_pos']
qualifying_long = results_and_drivers_and_constructors_and_grandprix_and_qualifying.melt(
    id_vars=['resultsDriverId', 'resultsDriverName'],
    value_vars=[col for col in qualifying_cols if col in results_and_drivers_and_constructors_and_grandprix_and_qualifying.columns],
    var_name='QualifyingSession',
    value_name='QualifyingPosition'
)
qualifying_long = qualifying_long.dropna(subset=['QualifyingPosition'])

driver_qual_consistency = qualifying_long.groupby('resultsDriverId')['QualifyingPosition'].std().reset_index()
driver_qual_consistency.rename(columns={'QualifyingPosition': 'qualifying_consistency_std'}, inplace=True)

# Merge into your main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.merge(
    driver_qual_consistency, on='resultsDriverId', how='left'
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={
'abbreviation_results': 'abbreviation'
}, inplace=True)

# ...place before feature engineering block...

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .loc[:, ~results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.duplicated()]
)

# 1. Practice Position × Safety Car Status
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_x_safetycar'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SafetyCarStatus']
)

# 2. Pit Stop Delta × Driver Age
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_delta_x_driver_age'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_stop_delta'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverAge']
)

# 3. Constructor Points × Grid Position
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_points_x_grid'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorPoints'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber']
)

# 4. Driver DNF Rate × Practice Std
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['dnf_rate_x_practice_std'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverDNFAvg'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_std']
)

# 5. Constructor Recent Form × Track Experience
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_recent_x_track_exp'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_recent_form_3_races'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['track_experience']
)

# 6. Driver Rank × Years Active
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_rank_x_years_active'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverRank'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['yearsActive']
)

if 'SpeedFL_mph' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'SpeedI1_mph_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'SpeedI1_mph_x': 'SpeedFL_mph'}, inplace=True)

if 'SpeedST_mph' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'SpeedST_mph_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'SpeedST_mph_x': 'SpeedST_mph'}, inplace=True)

if 'SpeedI1_mph' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'SpeedI1_mph_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'SpeedI1_mph_x': 'SpeedI1_mph'}, inplace=True)

if 'SpeedI2_mph' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'SpeedI2_mph_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'SpeedI2_mph_x': 'SpeedI2_mph'}, inplace=True)

if 'best_s1_sec' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'best_s1_sec_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'best_s1_sec_x': 'best_s1_sec'}, inplace=True)

if 'best_s2_sec' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'best_s2_sec_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'best_s2_sec_x': 'best_s2_sec'}, inplace=True)

if 'best_s3_sec' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'best_s3_sec_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'best_s3_sec_x': 'best_s3_sec'}, inplace=True)

# 8. Top Speed × Turns
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['top_speed_x_turns'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SpeedFL_mph'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['turns']
)

# 9. Average Practice Position × Driver Podiums
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_practice_x_driver_podiums'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalPodiums']
)

# 10. Grid Penalty × Constructor Rank
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_penalty_x_constructor_rank'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_penalty'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorRank']
)

if 'bestQualifyingTime_sec' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time']


# 1. Relative Practice Improvement vs. Field
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_vs_field'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber'] -
     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition']) /
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['averagePracticePosition'].transform('mean') + 1e-6)
)

# 2. Constructor Win Rate (last 3 years)
constructor_win_rate_3y = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixYear'] >=
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixYear'].max() - 2
    ]
    .groupby('constructorName')['resultsFinalPositionNumber']
    .apply(lambda x: (x == 1).mean())
    .reset_index(name='constructor_win_rate_3y')
)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    constructor_win_rate_3y,
    on='constructorName',
    how='left'
)

# 3. Driver Podium Rate (last 3 years)
driver_podium_rate_3y = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixYear'] >=
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixYear'].max() - 2
    ]
    .groupby('resultsDriverName')['resultsPodium']
    .mean()
    .reset_index(name='driver_podium_rate_3y')
)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    driver_podium_rate_3y,
    on='resultsDriverName',
    how='left'
)

# 4. Practice Consistency (std of FP positions)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_consistency_std'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber']
].std(axis=1)

# 5. Qualifying Position Percentile (within race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_percentile'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['resultsQualificationPositionNumber']
    .transform(lambda x: x.rank(pct=True))
)

# 8. Constructor Podium Ratio (podiums / entries)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_podium_ratio'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('constructorName')['resultsPodium'].transform('mean')
)

# 9. Practice-to-Qualifying Delta
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_to_qualifying_delta'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)

# 10. Track Familiarity (number of races driver has at this circuit)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['track_familiarity'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['resultsDriverId', 'circuitId'])['raceId_results'].transform('count')
)

# 11. Weather Volatility (std of temp/wind/humidity for race)
if all(col in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns for col in ['average_temp', 'average_humidity', 'average_wind_speed']):
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['weather_volatility'] = (
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('raceId_results')[['average_temp', 'average_humidity', 'average_wind_speed']]
        .transform('std').mean(axis=1)
    )

# 12. Qualifying-to-Final Position Delta
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_to_final_delta'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber']
)

# 13. Constructor Experience Weighted by Driver
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_experience_weighted'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorTotalRaceStarts'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['yearsActive']
)

# 14. Recent Podium Streak (last 3 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_podium_streak'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
    .groupby('resultsDriverName')['resultsPodium']
    .transform(lambda x: x.rolling(window=3, min_periods=1).sum().shift(1))
)

# 15. Relative Grid Position (percentile within race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_position_percentile'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['resultsStartingGridPositionNumber']
    .transform(lambda x: x.rank(pct=True))
)


# 17. Driver Age Squared (nonlinear effect)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_age_squared'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverAge'] ** 2
)

# 18. Constructor Recent Win Streak (last 3 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_recent_win_streak'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['constructorName', 'grandPrixYear', 'raceId_results'])
    .groupby('constructorName')['resultsFinalPositionNumber']
    .transform(lambda x: (x == 1).rolling(window=3, min_periods=1).sum().shift(1))
)

# 19. Practice Position Improvement Rate (FP1 to FP3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_rate'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1PositionNumber'] -
     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3PositionNumber']) /
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1PositionNumber'] + 1e-6)
)

# 20. Driver-Constructor Synergy (years active × constructor win rate)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_constructor_synergy'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['yearsActive'] *
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_win_rate_3y']
)


# Calculate raw overtake potential for each race
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['overtake_potential_raw'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['positionsGained'] /
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] + 1e-6)
)

# Sort for rolling calculation
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
)

# Calculate 3-year lookback average for each driver (excluding current race)
def overtake_3yr(row, df):
    mask = (
        (df['resultsDriverName'] == row['resultsDriverName']) &
        (df['grandPrixYear'] < row['grandPrixYear']) &
        (df['grandPrixYear'] >= row['grandPrixYear'] - 2)
    )
    return df.loc[mask, 'overtake_potential_raw'].mean()

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['overtake_potential_3yr'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.apply(
        lambda row: overtake_3yr(row, results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices),
        axis=1
    )
)

# Optionally, drop the raw column if you don't want it
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=['overtake_potential_raw'], inplace=True)

# Calculate raw overtake potential for each race
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['overtake_potential_raw'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['positionsGained'] /
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] + 1e-6)
)

# Sort for rolling calculation
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
)

# Calculate 5-year lookback average for each driver (excluding current race)
def overtake_5yr(row, df):
    mask = (
        (df['resultsDriverName'] == row['resultsDriverName']) &
        (df['grandPrixYear'] < row['grandPrixYear']) &
        (df['grandPrixYear'] >= row['grandPrixYear'] - 4)
    )
    return df.loc[mask, 'overtake_potential_raw'].mean()

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['overtake_potential_5yr'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.apply(
        lambda row: overtake_5yr(row, results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices),
        axis=1
    )
)

# Optionally, drop the raw column if you don't want it
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=['overtake_potential_raw'], inplace=True)

# Calculate raw qualifying-to-final delta for each race
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_to_final_delta_raw'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber']
)

# Sort for rolling calculation
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
)

# 3-year lookback average (excluding current race)
def qual_to_final_3yr(row, df):
    mask = (
        (df['resultsDriverName'] == row['resultsDriverName']) &
        (df['grandPrixYear'] < row['grandPrixYear']) &
        (df['grandPrixYear'] >= row['grandPrixYear'] - 2)
    )
    return df.loc[mask, 'qual_to_final_delta_raw'].mean()

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_to_final_delta_3yr'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.apply(
        lambda row: qual_to_final_3yr(row, results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices),
        axis=1
    )
)

# 5-year lookback average (excluding current race)
def qual_to_final_5yr(row, df):
    mask = (
        (df['resultsDriverName'] == row['resultsDriverName']) &
        (df['grandPrixYear'] < row['grandPrixYear']) &
        (df['grandPrixYear'] >= row['grandPrixYear'] - 4)
    )
    return df.loc[mask, 'qual_to_final_delta_raw'].mean()

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_to_final_delta_5yr'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.apply(
        lambda row: qual_to_final_5yr(row, results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices),
        axis=1
    )
)

# Optionally, drop the raw column if you don't want it
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=['qual_to_final_delta_raw'], inplace=True)


# 1. Driver's average qualifying position at this circuit (historical, excluding current race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_avg_qual_pos_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverName', 'circuitId'])['resultsQualificationPositionNumber']
    .transform(lambda x: x.shift(1).mean())
)

# 2. Constructor's average qualifying position at this circuit (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_avg_qual_pos_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['constructorName', 'circuitId'])['resultsQualificationPositionNumber']
    .transform(lambda x: x.shift(1).mean())
)

# 3. Driver's average grid position at this circuit (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_avg_grid_pos_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverName', 'circuitId'])['resultsStartingGridPositionNumber']
    .transform(lambda x: x.shift(1).mean())
)

# 4. Constructor's average grid position at this circuit (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_avg_grid_pos_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['constructorName', 'circuitId'])['resultsStartingGridPositionNumber']
    .transform(lambda x: x.shift(1).mean())
)

# 5. Driver's average practice position at this circuit (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_avg_practice_pos_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverName', 'circuitId'])['averagePracticePosition']
    .transform(lambda x: x.shift(1).mean())
)

# 6. Constructor's average practice position at this circuit (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_avg_practice_pos_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['constructorName', 'circuitId'])['averagePracticePosition']
    .transform(lambda x: x.shift(1).mean())
)

# 7. Driver's qualifying improvement rate (last 3 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_qual_improvement_3r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby('resultsDriverName')['resultsQualificationPositionNumber']
    .transform(lambda x: x.diff().rolling(window=3, min_periods=1).mean().shift(1))
)

# 8. Constructor's qualifying improvement rate (last 3 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_qual_improvement_3r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby('constructorName')['resultsQualificationPositionNumber']
    .transform(lambda x: x.diff().rolling(window=3, min_periods=1).mean().shift(1))
)

# 9. Driver's practice improvement rate (last 3 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_practice_improvement_3r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby('resultsDriverName')['averagePracticePosition']
    .transform(lambda x: x.diff().rolling(window=3, min_periods=1).mean().shift(1))
)

# 10. Constructor's practice improvement rate (last 3 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_practice_improvement_3r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby('constructorName')['averagePracticePosition']
    .transform(lambda x: x.diff().rolling(window=3, min_periods=1).mean().shift(1))
)

# 11. Driver's average qualifying gap to teammate (last 3 races)
def teammate_qual_gap(row, df):
    mask = (
        (df['resultsDriverName'] != row['resultsDriverName']) &
        (df['constructorName'] == row['constructorName']) &
        (df['grandPrixYear'] < row['grandPrixYear']) &
        (df['grandPrixYear'] >= row['grandPrixYear'] - 2)
    )
    teammate_qual = df.loc[mask, 'resultsQualificationPositionNumber']
    if len(teammate_qual) > 0:
        return row['resultsQualificationPositionNumber'] - teammate_qual.mean()
    else:
        return np.nan
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_teammate_qual_gap_3r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.apply(
    lambda row: teammate_qual_gap(row, results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices),
    axis=1
    )
    )

# 12. Driver's average practice gap to teammate (last 3 races)
def teammate_practice_gap(row, df):
    mask = (
        (df['resultsDriverName'] != row['resultsDriverName']) &
        (df['constructorName'] == row['constructorName']) &
        (df['grandPrixYear'] < row['grandPrixYear']) &
        (df['grandPrixYear'] >= row['grandPrixYear'] - 2)
    )
    teammate_practice = df.loc[mask, 'averagePracticePosition']
    if len(teammate_practice) > 0:
        return row['averagePracticePosition'] - teammate_practice.mean()
    else:
        return np.nan
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_teammate_practice_gap_3r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.apply(
    lambda row: teammate_practice_gap(row, results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices),
    axis=1
    )
)

# 13. Driver's average qualifying position in street races (historical)
street_mask = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['streetRace'] == True
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_street_qual_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[street_mask]
    .groupby('resultsDriverName')['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 14. Driver's average qualifying position in track races (historical)
track_mask = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['trackRace'] == True
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_track_qual_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[track_mask]
    .groupby('resultsDriverName')['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 15. Driver's average practice position in street races (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_street_practice_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[street_mask]
    .groupby('resultsDriverName')['averagePracticePosition'].transform(lambda x: x.shift(1).mean())
)

# 16. Driver's average practice position in track races (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_track_practice_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[track_mask]
    .groupby('resultsDriverName')['averagePracticePosition'].transform(lambda x: x.shift(1).mean())
)

# Load grouped weather data
weather_grouped = pd.read_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), sep='\t')

# Merge weather data into your main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    weather_grouped[['short_date', 'circuitId', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed']],
    left_on=['short_date', 'circuitId'],
    right_on=['short_date', 'circuitId'],
    how='left'
)


# 17. Driver's average qualifying position in high wind races (historical)
wind_median = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_wind_speed'].median()
high_wind_mask = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_wind_speed'] > wind_median
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_high_wind_qual_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[high_wind_mask]
    .groupby('resultsDriverName')['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 18. Driver's average practice position in high wind races (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_high_wind_practice_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[high_wind_mask]
    .groupby('resultsDriverName')['averagePracticePosition'].transform(lambda x: x.shift(1).mean())
)

# 19. Driver's average qualifying position in high humidity races (historical)
humidity_median = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_humidity'].median()
high_humidity_mask = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_humidity'] > humidity_median
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_high_humidity_qual_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[high_humidity_mask]
    .groupby('resultsDriverName')['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 20. Driver's average practice position in high humidity races (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_high_humidity_practice_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[high_humidity_mask]
    .groupby('resultsDriverName')['averagePracticePosition'].transform(lambda x: x.shift(1).mean())
)

# 21. Driver's average qualifying position in races with precipitation (historical)
wet_mask = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['total_precipitation'] > 0
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_wet_qual_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[wet_mask]
    .groupby('resultsDriverName')['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 22. Driver's average practice position in races with precipitation (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_wet_practice_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[wet_mask]
    .groupby('resultsDriverName')['averagePracticePosition'].transform(lambda x: x.shift(1).mean())
)

# 23. Driver's average qualifying position in races with safety car (historical)
sc_mask = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SafetyCarStatus'] == 1
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_safetycar_qual_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[sc_mask]
    .groupby('resultsDriverName')['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 24. Driver's average practice position in races with safety car (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_safetycar_practice_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[sc_mask]
    .groupby('resultsDriverName')['averagePracticePosition'].transform(lambda x: x.shift(1).mean())
)


# --- Driver-Constructor Features ---

# Create a unique identifier for driver-constructor pair
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_constructor_id'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsDriverId'].astype(str) + '_' +
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorId_results'].astype(str)
)

# Number of races driver has done with current constructor
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['races_with_constructor'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverId', 'constructorId_results'])
    .cumcount() + 1
)

# Is this the first season with current constructor?
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['is_first_season_with_constructor'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverId', 'constructorId_results'])['grandPrixYear']
    .transform(lambda x: x == x.min()).astype(int)
)

# Average final position with current constructor (excluding current race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_constructor_avg_final_position'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverId', 'constructorId_results'])['resultsFinalPositionNumber']
    .transform(lambda x: x.shift(1).mean())
)

# Average qualifying position with current constructor (excluding current race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_constructor_avg_qual_position'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverId', 'constructorId_results'])['resultsQualificationPositionNumber']
    .transform(lambda x: x.shift(1).mean())
)

# Podium rate with current constructor (excluding current race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_constructor_podium_rate'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverId', 'constructorId_results'])['resultsPodium']
    .transform(lambda x: x.shift(1).mean())
)

# --- Interaction Features ---
if 'SpeedI1_mph' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'SpeedI1_mph_y' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'SpeedI1_mph_y': 'SpeedI1_mph'}, inplace=True)

if 'SpeedI2_mph' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'SpeedI2_mph_y' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'SpeedI2_mph_y': 'SpeedI2_mph'}, inplace=True)

if 'LapTime_sec' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'LapTime_sec_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'LapTime_sec_x': 'LapTime_sec'}, inplace=True)

if 'Session' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'Session_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'Session_x': 'Session'}, inplace=True)

if 'best_theory_lap_sec' not in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
    if 'best_theory_lap_sec_x' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'best_theory_lap_sec_x': 'best_theory_lap_sec'}, inplace=True)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.to_list())

print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsDriverName'].str.contains('Antonelli', case=False, na=False)
][['resultsDriverId', 'resultsDriverName']])

current_season_race_count = races[races['year'] == current_year]['grandPrixId'].nunique()
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['currentRookie'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalRaceStarts'] < current_season_race_count)

# 1. Practice-to-Qualifying Improvement Rate
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_to_qual_improvement_rate'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']) /
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] + 1e-6)
)

# 2. Practice Consistency Relative to Teammate
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_consistency_vs_teammate'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_std'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['practice_position_std'].transform('mean')
)

# 3. Qualifying Position vs. Constructor Historical Average
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_vs_constructor_avg_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_avg_qual_pos_at_track']
)

# 4. Practice Lap Time Delta to Session Fastest (FP1)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1_lap_time_delta_to_best'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1Time'].apply(time_to_seconds) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('raceId_results')['fp1Time'].transform(lambda x: x.apply(time_to_seconds).min())
)

# 5. Qualifying Lap Time Delta to Pole (Q3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q3_lap_time_delta_to_pole'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q3_sec'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec']
)

# 6. Practice Position Percentile (FP3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3_position_percentile'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['fp3PositionNumber'].transform(lambda x: x.rank(pct=True))
)

# 7. Qualifying Position Percentile
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_percentile'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['resultsQualificationPositionNumber'].transform(lambda x: x.rank(pct=True))
)

# 8. Constructor Practice Improvement Rate (FP1 to FP3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_practice_improvement_rate'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['fp1PositionNumber'].transform('mean') -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['fp3PositionNumber'].transform('mean')
)

# 9. Practice-to-Qualifying Consistency (last 5 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_qual_consistency_5r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby('resultsDriverName', group_keys=False)[['averagePracticePosition', 'resultsQualificationPositionNumber']]
    .apply(lambda df: (df['averagePracticePosition'] - df['resultsQualificationPositionNumber']).rolling(window=5, min_periods=1).std().shift(1))
    .reset_index(level=0, drop=True)
)

# 10. Track-Specific Practice Improvement (FP1 to FP3, historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['track_fp1_fp3_improvement'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverName', 'circuitId'])['fp1PositionNumber']
    .transform(lambda x: x.shift(1).mean()) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['resultsDriverName', 'circuitId'])['fp3PositionNumber']
    .transform(lambda x: x.shift(1).mean())
)

# 11. Teammate Practice Delta at Track (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['teammate_practice_delta_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId', 'constructorName'])['averagePracticePosition'].transform('mean')
)

# 12. Constructor Qualifying Consistency (std last 5 races)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_qual_consistency_5r'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby('constructorName')['resultsQualificationPositionNumber']
    .transform(lambda x: x.rolling(window=5, min_periods=1).std().shift(1))
)

# 13. Practice Position vs. Historical Track Median
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_vs_track_median'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['averagePracticePosition'].transform('median')
)

# 14. Qualifying Position vs. Historical Track Median
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_vs_track_median'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['resultsQualificationPositionNumber'].transform('median')
)

# 15. Practice Lap Time Improvement Rate (FP1 to FP3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_lap_time_improvement_rate'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1Time'].apply(time_to_seconds) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3Time'].apply(time_to_seconds)
)

# 16. Practice Position Improvement vs. Field Average (FP1 to FP3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_vs_field_avg'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1PositionNumber'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3PositionNumber']) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['fp1PositionNumber'].transform('mean') +
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['fp3PositionNumber'].transform('mean')
)

# 17. Qualifying Position Improvement vs. Field Average (Q1 to Q3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_improvement_vs_field_avg'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q1_pos'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q3_pos']) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['q1_pos'].transform('mean') +
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['q3_pos'].transform('mean')
)

# 18. Practice-to-Qualifying Position Delta (current race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_to_qual_position_delta'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)

# 19. Constructor Podium Rate at Track (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructor_podium_rate_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['resultsPodium'].transform(lambda x: x.shift(1).mean())
)

# 20. Driver Podium Rate at Track (historical)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driver_podium_rate_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['resultsDriverName', 'circuitId'])['resultsPodium'].transform(lambda x: x.shift(1).mean())
)

# 21. Practice Position vs. Constructor Average (FP3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3_vs_constructor_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3PositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['fp3PositionNumber'].transform('mean')
)

# 22. Qualifying Position vs. Constructor Average
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_vs_constructor_avg'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['resultsQualificationPositionNumber'].transform('mean')
)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_lap_time_consistency'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['fp1Time', 'fp2Time', 'fp3Time']]
    .apply(lambda col: col.map(time_to_seconds)).std(axis=1)
)

# 24. Qualifying Lap Time Consistency (Q1-Q3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_lap_time_consistency'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['q1_sec', 'q2_sec', 'q3_sec']].std(axis=1)
)

# 25. Practice Position Improvement vs. Teammate (FP1-FP3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_vs_teammate'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1PositionNumber'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3PositionNumber']) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['fp1PositionNumber'].transform('mean') +
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['fp3PositionNumber'].transform('mean')
)

# 26. Qualifying Position Improvement vs. Teammate (Q1-Q3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_improvement_vs_teammate'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q1_pos'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q3_pos']) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['q1_pos'].transform('mean') +
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['q3_pos'].transform('mean')
)

# 27. Practice Position vs. Historical Best at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_vs_best_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['resultsDriverName', 'circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).min())
)

# 28. Qualifying Position vs. Historical Best at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_vs_best_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['resultsDriverName', 'circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).min())
)

# 29. Practice Position vs. Historical Worst at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_vs_worst_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['resultsDriverName', 'circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).max())
)

# 30. Qualifying Position vs. Historical Worst at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qual_vs_worst_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['resultsDriverName', 'circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).max())
)


# --- NEW LEAKAGE-FREE FEATURES (add after all merges, before .to_csv) ---

# 1. Practice Position Percentile vs Constructor
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_percentile_vs_constructor'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['averagePracticePosition'].transform(lambda x: x.rank(pct=True)) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['averagePracticePosition'].transform(lambda x: x.rank(pct=True))
)

# 2. Qualifying Position Percentile vs Constructor
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_percentile_vs_constructor'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['resultsQualificationPositionNumber'].transform(lambda x: x.rank(pct=True)) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['resultsQualificationPositionNumber'].transform(lambda x: x.rank(pct=True))
)

# 3. Practice Lap Time Delta to Constructor Best
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_lap_time_delta_to_constructor_best'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1Time'].apply(time_to_seconds) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['fp1Time'].transform(lambda x: x.apply(time_to_seconds).min())
)

# 4. Qualifying Lap Time Delta to Constructor Best
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_lap_time_delta_to_constructor_best'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['grandPrixName', 'constructorName'])['best_qual_time'].transform('min')
)

# 5. Practice Position vs Teammate Historical
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_teammate_historical'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).mean())
)

# 6. Qualifying Position vs Teammate Historical
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_teammate_historical'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 7. Practice Improvement vs Constructor Historical
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_vs_constructor_historical'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp1PositionNumber'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['fp3PositionNumber']) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['fp1PositionNumber'].transform(lambda x: x.shift(1).mean()) +
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['fp3PositionNumber'].transform(lambda x: x.shift(1).mean())
)

# 8. Qualifying Improvement vs Constructor Historical
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_improvement_vs_constructor_historical'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q1_pos'] - results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['q3_pos']) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['q1_pos'].transform(lambda x: x.shift(1).mean()) +
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['q3_pos'].transform(lambda x: x.shift(1).mean())
)

# 9. Practice Consistency vs Constructor Historical
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_consistency_vs_constructor_historical'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_std'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['practice_position_std'].transform(lambda x: x.shift(1).mean())
)

# 10. Qualifying Consistency vs Constructor Historical
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_consistency_vs_constructor_historical'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_consistency_std'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['qualifying_consistency_std'].transform(lambda x: x.shift(1).mean())
)

# 11. Practice Position vs Field Best at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_field_best_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).min())
)

# 12. Qualifying Position vs Field Best at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_field_best_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).min())
)

# 13. Practice Position vs Field Worst at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_field_worst_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).max())
)

# 14. Qualifying Position vs Field Worst at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_field_worst_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).max())
)

# 15. Practice Position vs Field Median at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_field_median_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).median())
)

# 16. Qualifying Position vs Field Median at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_field_median_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).median())
)

# Step 1: Calculate the delta as a new column
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_qual_delta'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)

# Step 2: Calculate the historical mean delta for each constructor/track (excluding current race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_to_qualifying_delta_vs_constructor_historical'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby(['constructorName', 'circuitId'])['practice_qual_delta']
    .transform(lambda x: x.shift(1).mean())
)

# 17. Practice-to-Qualifying Delta vs Constructor Historical
# Step 1: Calculate the delta as a new column
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_qual_delta'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber']
)

# Step 2: Calculate the historical mean delta for each constructor/track (excluding current race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_to_qualifying_delta_vs_constructor_historical'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
        .groupby(['constructorName', 'circuitId'])['practice_qual_delta']
        .transform(lambda x: x.shift(1).mean())
)

# 19. Practice Position vs Constructor Best at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_constructor_best_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).min())
)

# 20. Qualifying Position vs Constructor Best at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_constructor_best_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).min())
)

# 21. Practice Position vs Constructor Worst at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_constructor_worst_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).max())
)

# 22. Qualifying Position vs Constructor Worst at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_constructor_worst_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).max())
)

# 23. Practice Position vs Constructor Median at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_constructor_median_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['averagePracticePosition'].transform(lambda x: x.shift(1).median())
)

# 24. Qualifying Position vs Constructor Median at Track
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_constructor_median_at_track'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.shift(1).median())
)

# 25. Practice Lap Time Consistency vs Field
# Step 1: Calculate lap time consistency for each driver
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_lap_time_consistency'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['fp1Time', 'fp2Time', 'fp3Time']]
    .apply(lambda col: col.map(time_to_seconds)).std(axis=1)
)

# Step 2: Calculate the field average for each race
field_consistency = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
    .groupby('grandPrixName')['practice_lap_time_consistency']
    .mean()
    .rename('field_practice_lap_time_consistency')
)

# Step 3: Map the field average back to each row
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_lap_time_consistency_vs_field'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_lap_time_consistency'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixName'].map(field_consistency)
)

# 26. Qualifying Lap Time Consistency vs Field
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_lap_time_consistency_vs_field'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['q1_sec', 'q2_sec', 'q3_sec']].std(axis=1) -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')[['q1_sec', 'q2_sec', 'q3_sec']].transform('std').mean(axis=1)
)

# 27. Practice Position vs Constructor Recent Form
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_constructor_recent_form'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['averagePracticePosition'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

# 28. Qualifying Position vs Constructor Recent Form
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_constructor_recent_form'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['constructorName', 'circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

# 29. Practice Position vs Field Recent Form
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_position_vs_field_recent_form'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['averagePracticePosition'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

# 30. Qualifying Position vs Field Recent Form
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_vs_field_recent_form'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsQualificationPositionNumber'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby(['circuitId'])['resultsQualificationPositionNumber'].transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

# Quantile binning for high-cardinality numerical features
high_cardinality_features = [
    "LapTime_sec",
    "driverFastestPracticeLap_sec",
    "bestQualifyingTime_sec",
    "best_qual_time",
    "best_theory_lap_sec",
    "best_s3_sec",
    "delta_from_race_avg",
    "best_s2_sec",
    "best_s1_sec",
    "practice_qual_consistency_5r",
    "fp1_lap_delta_vs_best",
    "practice_lap_time_consistency_vs_field",
    "practice_lap_time_consistency",
    "grid_x_avg_pit_time",
    "pit_delta_x_driver_age",
    "practice_consistency_vs_teammate",
    "Delta_FP2",
    "CleanAirAvg_FP2",
    "DirtyAirAvg_FP2",
    "Delta_FP1",
    "DirtyAirAvg_FP1",
    "CleanAirAvg_FP1",
    "practice_consistency_vs_constructor_historical",
    "qualifying_lap_time_consistency_vs_field",
    "teammate_qual_delta",
    "fp1_lap_time_delta_to_best",
    "teammate_practice_delta",
    "practice_position_percentile_vs_constructor",
    "practice_time_improvement_1T_2T",
    "qualifying_lap_time_delta_to_constructor_best",
    "qual_lap_time_consistency",
    "practice_lap_time_delta_to_constructor_best",
    "practice_time_improvement_2T_3T",
    "practice_time_improvement_time_time",
    "practice_lap_time_improvement_rate",
    "practice_time_improvement_1T_3T",
    "last_race_vs_track_avg",
    "DirtyAirAvg_FP3",
    "CleanAirAvg_FP3",
    "Delta_FP3",
    "practice_gap_to_teammate",
    "qualifying_position_percentile_vs_constructor",
    "practice_position_vs_teammate_historical",
    "BestConstructorPracticeLap_sec",
    "teammate_practice_delta_at_track",
    "constructor_qual_consistency_5r",
    "historical_avgLapPace",
    "historical_race_pace_vs_median",
    "practice_position_vs_field_recent_form",
    "driver_teammate_practice_gap_3r",
    "qual_gap_to_teammate",
    "practice_improvement_vs_field",
    "practice_improvement_vs_constructor_historical",
    "dnf_rate_x_practice_std",
    "qual_vs_track_avg",
    "practice_improvement_vs_teammate",
    "qualifying_position_vs_teammate_historical",
    "qual_vs_constructor_avg_at_track",
    "practice_position_vs_constructor_recent_form",
    "driver_teammate_qual_gap_3r",
    "practice_to_qual_improvement_rate",
    "constructor_form_ratio",
    "practice_std_x_qual",
    "recent_form_ratio",
    "recent_vs_season",
    "qualifying_improvement_vs_constructor_historical",
    "qual_vs_constructor_avg",
    "qualifying_consistency_vs_constructor_historical",
    "qual_improvement_vs_teammate",
    "practice_improvement_vs_field_avg",
    "fp3_vs_constructor_avg",
    "constructor_recent_x_track_exp",
    "constructor_practice_improvement_3r",
    "average_practice_x_driver_podiums",
    "grid_position_percentile",
    "driver_practice_improvement_3r",
    "power_to_corner_ratio",
    "top_speed_x_turns",
    "qualifying_position_percentile",
    "qualPos_x_avg_practicePos",
    "fp3_position_percentile",
    "qual_improvement_vs_field_avg",
    "practice_position_vs_constructor_median_at_track",
    "practice_position_vs_field_median_at_track",
    "practice_improvement_rate",
    "practice_to_qualifying_delta_vs_constructor_historical",
    "recent_form_x_qual",
    "driver_avg_practice_pos_at_track",
    "practice_vs_track_median",
    "constructor_avg_practice_pos_at_track",
    "practice_vs_best_at_track",
    ]

for field in high_cardinality_features:
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = quantile_bin_feature(
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
        field, q=5
    )


# Check missing data before export
missing_summary = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.isnull().sum()
print("Missing values per column (only columns with missing data):")
print(missing_summary[missing_summary > 0].sort_values(ascending=False))

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['constructorName','resultsDriverId', 'resultsDriverName', 'grandPrixName', 'best_s1_sec', 'best_s2_sec', 'best_s2_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'q1', 'q2', 'q3', 'bestQualifyingTime', 'timeMillis_results', 'streetRace', 'trackRace', 'resultsDriverId', 'yearsActive', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 
#                                         'q1End', 'q2End', 'q3Top10', 'resultsDriverId', 'resultsReasonRetired','averagePracticePosition', 'raceId_results', 'resultsFinalPositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'fp1PositionNumber', 'fp1Time', 'fp1Gap', 
#                                         'fp1Interval', 'positionsGained', 'fp1PositionNumber', 'fp2PositionNumber','fp3PositionNumber', 'resultsYear',  'resultsStartingGridPositionNumber', 
#                                        'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 'round', 
#                                        'driverTotalRaceStarts', 'driverTotalPodiums', 'driverBestRaceResult',  'driverBestStartingGridPosition', 'driverTotalRaceLaps', 'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps',
#                                        'driverTotalPodiums', 'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'turns', 'grandPrixYear', 'raceIdFromGrandPrix', 'grandPrixRaceId', 'short_date', 'DNF', 'driverTotalPolePositions', 'activeDriver']]

bin_fields = [col for col in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns if col.endswith('_bin')]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.copy()

static_columns=['grandPrixYear', 'grandPrixName', 'raceId_results', 'circuitId', 'grandPrixRaceId', 'resultsDriverName', 'q1', 'q2', 'q3', 
                                        'fp1Time', 'fp1Gap', 'fp1Interval', 'fp1PositionNumber', 'fp2Time', 'fp2Gap', 'fp2Interval', 'fp2PositionNumber', 'fp3Time', 'fp3Gap', 'fp3Interval', 'fp3PositionNumber',#'fp4Time', 'fp4Gap', 'fp4Interval', 
                                         'resultsPodium', 'resultsTop5', 'resultsTop10', 'resultsYear', 'constructorName',  'resultsStartingGridPositionNumber',   'constructorId_results',
                                      'positionsGained', 'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 'round',
                                      'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums',
                                      'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'turns', 'short_date', 'DNF', 'fp1PositionNumber', 'fp2PositionNumber', 'streetRace', 'trackRace', 'avgLapPace', 'finishingTime', 'timeMillis_results', 'bestQualifyingTime',
                                     'fp3PositionNumber' ,'averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10','resultsDriverId', 'driverTotalPolePositions', 'activeDriver', 'yearsActive',
                                      'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'time', 'Session', 'driverDNFCount', 'driverDNFAvg', 'SafetyCarStatus', 
                                      'resultsFinalPositionNumber','recent_form_3_races', 'recent_form_5_races', 'constructor_recent_form_3_races', 'constructor_recent_form_5_races', 'courseLength',
                                      'CleanAirAvg_FP1', 'DirtyAirAvg_FP1', 'Delta_FP1', 'CleanAirAvg_FP2', 'DirtyAirAvg_FP2', 'Delta_FP2', 'CleanAirAvg_FP3', 'DirtyAirAvg_FP3','Delta_FP3',  
                                       'numberOfStops', 'averageStopTime', 'totalStopTime', 'pit_lane_time_constant', 'pit_stop_delta', 'engineManufacturerId', 'delta_from_race_avg', 'driverAge',
                                       'finishing_position_std_driver', 'finishing_position_std_constructor', 'delta_lap_2', 'delta_lap_5', 'delta_lap_10', 'delta_lap_15', 'delta_lap_20',
                                       'delta_lap_2_historical', 'delta_lap_5_historical', 'delta_lap_10_historical', 'delta_lap_15_historical', 'delta_lap_20_historical', 'abbreviation', 
                                       'driver_positionsGained_5_races',  'driver_dnf_rate_5_races', 'avg_final_position_per_track', 'last_final_position_per_track', 'q1_sec', 'q2_sec', 'q3_sec', 'q1_pos', 'q2_pos', 'q3_pos',
                                       'driver_positionsGained_3_races', 'driverFastestPracticeLap_sec', 'BestConstructorPracticeLap_sec', 'teammate_practice_delta', 'teammate_qual_delta', 'best_qual_time',
                                       'avg_final_position_per_track_constructor', 'last_final_position_per_track_constructor', 'bestQualifyingTime_sec', 'qualifying_gap_to_pole',
                                       'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P', 'practice_time_improvement_1T_2T', 'practice_time_improvement_time_time', 
                                       'practice_time_improvement_2T_3T', 'practice_time_improvement_1T_3T', 'qualifying_consistency_std', 'driver_starting_position_3_races', 'driver_starting_position_5_races',
                                       'qual_vs_track_avg', 'constructor_avg_practice_position', 'practice_position_std', 'recent_vs_season',
                                        'practice_improvement', 'qual_x_constructor_wins',  'qual_vs_track_avg', 'grid_penalty', 'grid_penalty_x_constructor', 'recent_form_x_qual', 'practice_std_x_qual',
                                       'qualPos_x_last_practicePos', 'driver_rank_x_constructor_rank','qualPos_x_avg_practicePos', 'recent_form_median_3_races','recent_form_median_5_races', 'recent_form_best_3_races', 'recent_form_worst_3_races', 'recent_dnf_rate_3_races', 'recent_positions_gained_3_races',
                                       'grid_x_constructor_rank', 'practice_improvement_x_qual','qual_gap_to_teammate', 'practice_gap_to_teammate',
                                        'recent_form_ratio', 'constructor_form_ratio','total_experience','podium_potential','street_experience','track_experience',
                                        'fp1_lap_delta_vs_best', 'grid_x_avg_pit_time', 'pit_count_x_pit_delta', 'pit_stop_rate', 'last_race_vs_track_avg',
                                        'race_pace_vs_median', 'top_speed_rank', 'positions_gained_first_lap_pct',  'power_to_corner_ratio', 'historical_avgLapPace',
                                         'practice_x_safetycar', 'pit_delta_x_driver_age', 'constructor_points_x_grid', 'dnf_rate_x_practice_std', 'constructor_recent_x_track_exp', 'driver_rank_x_years_active', 
                                         'top_speed_x_turns', 'grid_penalty_x_constructor_rank', 'average_practice_x_driver_podiums',
                                         'practice_improvement_vs_field', 'constructor_win_rate_3y', 'driver_podium_rate_3y', 'practice_consistency_std', 
                                         'constructor_podium_ratio','practice_to_qualifying_delta', 'track_familiarity', 
                                        'qualifying_position_percentile',   'historical_avgLapPace', 'top_speed_rank', 'qualifying_consistency_std', 
                                         'recent_podium_streak', 'grid_position_percentile', 'driver_age_squared', 'constructor_recent_win_streak', 'practice_improvement_rate', 'driver_constructor_synergy',
                                         'qual_to_final_delta_5yr', 'qual_to_final_delta_3yr', 'overtake_potential_3yr', 'overtake_potential_5yr',
                                         'driver_avg_qual_pos_at_track','constructor_avg_qual_pos_at_track','driver_avg_grid_pos_at_track','constructor_avg_grid_pos_at_track','driver_avg_practice_pos_at_track',
                                         'constructor_avg_practice_pos_at_track','driver_qual_improvement_3r','constructor_qual_improvement_3r','driver_practice_improvement_3r','constructor_practice_improvement_3r',
                                         'driver_teammate_qual_gap_3r','driver_teammate_practice_gap_3r','driver_street_qual_avg','driver_track_qual_avg','driver_street_practice_avg','driver_track_practice_avg',
                                         'driver_high_wind_qual_avg','driver_high_wind_practice_avg','driver_high_humidity_qual_avg','driver_high_humidity_practice_avg','driver_wet_qual_avg','driver_wet_practice_avg',
                                         'driver_safetycar_qual_avg','driver_safetycar_practice_avg',
                                         'driver_constructor_id','races_with_constructor','is_first_season_with_constructor','driver_constructor_avg_final_position','driver_constructor_avg_qual_position','driver_constructor_podium_rate',
                                         'constructor_dnf_rate_3_races', 'constructor_dnf_rate_5_races', 'recent_dnf_rate_5_races', 'historical_race_pace_vs_median',
                                         
                                         'practice_to_qual_improvement_rate','practice_consistency_vs_teammate','qual_vs_constructor_avg_at_track','fp1_lap_time_delta_to_best','q3_lap_time_delta_to_pole',
                                         'fp3_position_percentile','qualifying_position_percentile','constructor_practice_improvement_rate','practice_qual_consistency_5r','track_fp1_fp3_improvement',
                                         'teammate_practice_delta_at_track','constructor_qual_consistency_5r','practice_vs_track_median','qual_vs_track_median','practice_lap_time_improvement_rate',
                                         'practice_improvement_vs_field_avg','qual_improvement_vs_field_avg','practice_to_qual_position_delta','constructor_podium_rate_at_track','driver_podium_rate_at_track',
                                         'fp3_vs_constructor_avg','qual_vs_constructor_avg','practice_lap_time_consistency','qual_lap_time_consistency','practice_improvement_vs_teammate','qual_improvement_vs_teammate',
                                         'practice_vs_best_at_track','qual_vs_best_at_track','practice_vs_worst_at_track','qual_vs_worst_at_track',

                                         'practice_position_percentile_vs_constructor','qualifying_position_percentile_vs_constructor','practice_lap_time_delta_to_constructor_best','qualifying_lap_time_delta_to_constructor_best',
                                         'practice_position_vs_teammate_historical','qualifying_position_vs_teammate_historical','practice_improvement_vs_constructor_historical','qualifying_improvement_vs_constructor_historical',
                                        'practice_consistency_vs_constructor_historical','qualifying_consistency_vs_constructor_historical',
                                        'practice_position_vs_field_best_at_track','qualifying_position_vs_field_best_at_track','practice_position_vs_field_worst_at_track',
                                        'qualifying_position_vs_field_worst_at_track', 'practice_position_vs_field_median_at_track','qualifying_position_vs_field_median_at_track',
                                        'practice_to_qualifying_delta_vs_constructor_historical',
                                        'practice_position_vs_constructor_best_at_track','qualifying_position_vs_constructor_best_at_track',
                                        'practice_position_vs_constructor_worst_at_track','qualifying_position_vs_constructor_worst_at_track',
                                        'practice_position_vs_constructor_median_at_track','qualifying_position_vs_constructor_median_at_track',
                                        'practice_lap_time_consistency_vs_field','qualifying_lap_time_consistency_vs_field',
                                        'practice_position_vs_constructor_recent_form','qualifying_position_vs_constructor_recent_form','practice_position_vs_field_recent_form',
                                        'qualifying_position_vs_field_recent_form', 'currentRookie'
                                                                              ]

# Concatenate static columns and bin_fields
all_columns = static_columns + bin_fields

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(
    path.join(DATA_DIR, 'f1ForAnalysis.csv'),
    columns=all_columns,
    sep='\t',
    index=False
)

# Copy your main DataFrame for safety car feature engineering
safetycar_features = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.copy()

# --- SAFE SAFETY CAR FEATURES FOR POST-PRACTICE/QUALIFYING PREDICTION ---

# 1. Track and Weather Features
safetycar_features['turns_x_weather'] = safetycar_features['turns'] * safetycar_features['average_temp']
safetycar_features['turns_x_precip'] = safetycar_features['turns'] * safetycar_features['total_precipitation']
safetycar_features['turns_x_wind'] = safetycar_features['turns'] * safetycar_features['average_wind_speed']
safetycar_features['street_x_weather'] = safetycar_features['streetRace'].astype(int) * safetycar_features['average_temp']
safetycar_features['track_x_weather'] = safetycar_features['trackRace'].astype(int) * safetycar_features['average_temp']

# 2. Practice Features
safetycar_features['practice_position_std'] = safetycar_features[['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber']].std(axis=1)
safetycar_features['practice_improvement'] = safetycar_features['fp1PositionNumber'] - safetycar_features['lastFPPositionNumber']
safetycar_features['practice_improvement_x_qual'] = safetycar_features['practice_improvement'] * safetycar_features['resultsQualificationPositionNumber']
safetycar_features['practice_gap_to_teammate'] = safetycar_features['averagePracticePosition'] - safetycar_features.groupby(['grandPrixName', 'constructorName'])['averagePracticePosition'].transform('mean')
safetycar_features['practice_position_improvement_1P_2P'] = safetycar_features['fp1PositionNumber'] - safetycar_features['fp2PositionNumber']
safetycar_features['practice_position_improvement_2P_3P'] = safetycar_features['fp2PositionNumber'] - safetycar_features['fp3PositionNumber']
safetycar_features['practice_position_improvement_1P_3P'] = safetycar_features['fp1PositionNumber'] - safetycar_features['fp3PositionNumber']

# 3. Qualifying Features
safetycar_features['qualifying_gap_to_pole'] = safetycar_features['best_qual_time'] - safetycar_features['pole_time_sec']
safetycar_features['qualifying_position_percentile'] = safetycar_features.groupby('grandPrixName')['resultsQualificationPositionNumber'].transform(lambda x: x.rank(pct=True))
safetycar_features['qual_gap_to_teammate'] = safetycar_features['resultsQualificationPositionNumber'] - safetycar_features.groupby(['grandPrixName', 'constructorName'])['resultsQualificationPositionNumber'].transform('mean')
safetycar_features['qualPos_x_avg_practicePos'] = safetycar_features['resultsQualificationPositionNumber'] * safetycar_features['averagePracticePosition']
safetycar_features['qualPos_x_last_practicePos'] = safetycar_features['resultsQualificationPositionNumber'] * safetycar_features['lastFPPositionNumber']

# 4. Career/Constructor Features
safetycar_features['driver_experience'] = safetycar_features['driverTotalRaceStarts']
safetycar_features['constructor_experience'] = safetycar_features['constructorTotalRaceStarts']
safetycar_features['years_active'] = safetycar_features['yearsActive']
safetycar_features['driver_age_squared'] = safetycar_features['driverAge'] ** 2
safetycar_features['street_experience'] = safetycar_features['streetRace'].astype(int) * safetycar_features['driverTotalRaceStarts']
safetycar_features['track_experience'] = safetycar_features['trackRace'].astype(int) * safetycar_features['driverTotalRaceStarts']

# 5. Track Familiarity
safetycar_features['track_familiarity'] = safetycar_features.groupby(['resultsDriverId', 'circuitId'])['raceId_results'].transform('count')

# 6. Weather Volatility (std of temp/wind/humidity for race)
if all(col in safetycar_features.columns for col in ['average_temp', 'average_humidity', 'average_wind_speed']):
    safetycar_features['weather_volatility'] = (
        safetycar_features.groupby('raceId_results')[['average_temp', 'average_humidity', 'average_wind_speed']]
        .transform('std').mean(axis=1)
    )

# 7. Pit Stop Features (from practice/qualifying, not race)
safetycar_features['pit_stop_rate'] = safetycar_features['numberOfStops'] / (safetycar_features['grandPrixLaps'] + 1e-6)
safetycar_features['turns_x_pit_stop_rate'] = safetycar_features['turns'] * safetycar_features['pit_stop_rate']
safetycar_features['weather_x_pit_stop_rate'] = safetycar_features['average_temp'] * safetycar_features['pit_stop_rate']

# 8. Combined Features
safetycar_features['driver_experience_x_track_familiarity'] = safetycar_features['driverTotalRaceStarts'] * safetycar_features['track_familiarity']
safetycar_features['constructor_experience_x_track_familiarity'] = safetycar_features['constructorTotalRaceStarts'] * safetycar_features['track_familiarity']
safetycar_features['turns_x_weather_x_pit_stop_rate'] = safetycar_features['turns'] * safetycar_features['average_temp'] * safetycar_features['pit_stop_rate']

# --- END SAFE SAFETY CAR FEATURE BLOCK ---

# You can now use safetycar_features for modeling or save to CSV if desired
# Example:

# Define the columns you want to keep for safety car prediction
safetycar_feature_columns = [
    # Track & Weather
    'grandPrixYear', 'grandPrixName', 'raceId_results', 'resultsDriverId', 'circuitId', 'grandPrixRaceId',
    'grandPrixLaps', 'turns', 'streetRace', 'trackRace',
    'average_temp', 'average_humidity', 'average_wind_speed', 'total_precipitation',

    # Practice
    'fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber', #'fp4PositionNumber',
    'averagePracticePosition', 'lastFPPositionNumber', 'practice_position_std',
    'practice_improvement', 'practice_improvement_x_qual', 'practice_gap_to_teammate',
    'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P',

    # Qualifying
    'resultsQualificationPositionNumber', 'best_qual_time', 'pole_time_sec',
    'qualifying_gap_to_pole', 'qualifying_position_percentile', 'qual_gap_to_teammate',
    'qualPos_x_avg_practicePos', 'qualPos_x_last_practicePos',

    # Pit Stops
    'numberOfStops', 'averageStopTime', 'totalStopTime', 'pit_stop_rate', 'turns_x_pit_stop_rate', 'weather_x_pit_stop_rate',

    # Career/Constructor
    'driverTotalRaceStarts', 'constructorTotalRaceStarts', 'yearsActive', 'driverAge', 'driver_age_squared',
    'street_experience', 'track_experience', 'driver_experience', 'constructor_experience',

    # Track Familiarity
    'track_familiarity',

    # Weather Volatility
    'weather_volatility',

    # Combined/Engineered
    'turns_x_weather', 'turns_x_precip', 'turns_x_wind', 'street_x_weather', 'track_x_weather',
    'driver_experience_x_track_familiarity', 'constructor_experience_x_track_familiarity', 'turns_x_weather_x_pit_stop_rate',

    # Target
    'SafetyCarStatus'
]
# Remove duplicates based on key columns (e.g., raceId_results, resultsDriverId)
safetycar_features = safetycar_features.drop_duplicates(subset=['raceId_results', 'resultsDriverId'])
# Export only these columns
safetycar_features[safetycar_feature_columns].to_csv(path.join(DATA_DIR, 'f1SafetyCarFeatures.csv'), sep='\t', index=False)

# --- SAFETY CAR TRENDING & ENGINEERED FEATURES ---


positionCorrelation = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['driverTotalChampionshipWins',
        'resultsStartingGridPositionNumber', 'averagePracticePosition', 
        'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'constructorTotalRaceStarts', 
        'constructorTotalRaceWins', 'constructorTotalPolePositions', 'driverTotalRaceEntries', 'finishingTime',
         'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalPodiums', 'driverTotalPolePositions', 'SafetyCarStatus',
        'yearsActive', 'best_qual_time', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'LapTime_sec', 
        'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'trackRace', 'streetRace', 'turns', 'positionsGained', 'best_qual_time'
        ]].corr(method='pearson')

positionCorrelation.to_csv(path.join(DATA_DIR, 'f1PositionCorrelation.csv'), sep='\t')

### Weather

# Weather data for the last 10 years

races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 
circuits = pd.read_json(path.join(DATA_DIR, 'f1db-circuits.json')) 
weatherData = pd.read_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), sep='\t') 

races = races[races['year'].between(raceNoEarlierThan, current_year)]

circuits_and_races = pd.merge(races, circuits, left_on='circuitId', right_on='id', suffixes=['_races', '_circuits'])
circuits_and_races[['id_races', 'circuitId', 'year', 'date', 'grandPrixId', 'latitude', 'longitude']]

last_weather_date = datetime.datetime.strptime(weatherData['short_date'].iloc[-1],'%Y-%m-%d')

circuits_and_races_lat_long = circuits_and_races[['id_races', 'latitude', 'longitude', 'date', 'grandPrixId', 'circuitId']]

print(len(circuits_and_races_lat_long))
newRecords = True

weather_csv_path = os.path.join(DATA_DIR, 'f1WeatherData_AllData.csv')
if os.path.exists(weather_csv_path):
    processed_weather = pd.read_csv(weather_csv_path, sep='\t', usecols=['short_date'])
    # Convert to datetime and then to standard format for comparison
    processed_weather['short_date_compare'] = pd.to_datetime(processed_weather['short_date'], format='mixed').dt.strftime('%Y-%m-%d')
    processed_weather_set = set(processed_weather['short_date_compare'])
    
else:
    processed_weather_set = set()

circuits_and_races_lat_long = circuits_and_races_lat_long.copy()
circuits_and_races_lat_long['date'] = pd.to_datetime(circuits_and_races_lat_long['date'])
most_recent_date = circuits_and_races_lat_long['date'].max()

if most_recent_date >= pd.Timestamp.today().normalize():
    races_to_pull = circuits_and_races_lat_long[circuits_and_races_lat_long['date'] == most_recent_date]
else:
    races_to_pull = pd.DataFrame()  # No races to pull

# Only build params for races not already processed
full_params = []
# for race in circuits_and_races_lat_long.itertuples():
for race in races_to_pull.itertuples():
     # Standardize date for comparison
    short_date_compare = pd.to_datetime(race.date).strftime('%Y-%m-%d')
    # Keep original format for API and CSV
    short_date_original = pd.to_datetime(race.date).strftime('%m/%d/%Y')
    lat = race.latitude
    lon = race.longitude
    print("Checking:", (short_date_compare))
    if short_date_compare in processed_weather_set:
        print(f"Skipping weather for {short_date_compare} - already processed.")
        continue

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": short_date_original,
        "end_date": short_date_original,
        "hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m", "precipitation_probability"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch"
    }
    full_params.append((params, short_date_original, race.latitude, race.longitude))

all_hourly_data = []

if not full_params:
    print("All weather data is already up to date. No new API calls needed.")
else:
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Loop through the list of params
    for params, short_date, lat, lon in full_params:
        # use different URLs depending on whether we are seeking current or past weather
        # if datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') < datetime.datetime.now():
        if datetime.datetime.strptime(params['start_date'], '%m/%d/%Y') < datetime.datetime.now():
            url = "https://archive-api.open-meteo.com/v1/archive"
        elif datetime.datetime.strptime(params['start_date'], '%m/%d/%Y') >= datetime.datetime.now() and datetime.datetime.strptime(params['start_date'], '%m/%d/%Y') <= (datetime.datetime.now() + timedelta(days=16)):
            url = "https://api.open-meteo.com/v1/forecast"
        else:
            print("Break!")
            print(datetime.datetime.strptime(params['start_date'], '%m/%d/%Y'))
            break

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        # Process hourly data. The order of variables needs to be the same as requested.
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
        hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
        hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
        hourly_precipitation_probability = hourly.Variables(4).ValuesAsNumpy()

        hourly_data = {"date": pd.date_range(
            start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
            end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = hourly.Interval()),
            inclusive = "left"
        )}

        hourly_data["latitude"] = response.Latitude()
        hourly_data["longitude"] = response.Longitude()
        hourly_data["temperature_2m"] = hourly_temperature_2m
        hourly_data["hourly_precipitation"] = hourly_precipitation
        hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
        hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
        hourly_data["hourly_precipitation_probability"] = hourly_precipitation_probability
        hourly_data["short_date"] = pd.to_datetime(hourly_data["date"]).strftime('%Y-%m-%d')
        hourly_dataframe = pd.DataFrame(data = hourly_data)

        all_hourly_data = pd.DataFrame(data = all_hourly_data)
        all_hourly_data = pd.concat([all_hourly_data, hourly_dataframe], ignore_index=True)

circuits_and_races_lat_long = circuits_and_races_lat_long.copy()

# Merge new hourly data with circuits_and_races_lat_long
if len(all_hourly_data) > 0:
    circuits_and_races_lat_long['date'] = pd.to_datetime(circuits_and_races_lat_long['date']).dt.strftime('%Y-%m-%d')
    races_and_weather_for_concat = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])

    # Load existing weather data if present
    if os.path.exists(weather_csv_path):
        races_and_weather = pd.read_csv(weather_csv_path, sep='\t', usecols=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 
            'relative_humidity_2m', 'short_date', 'wind_speed_10m',  'id_races', 'grandPrixId', 'circuitId', 'hourly_precipitation_probability'])
        # Exclude rows in races_and_weather where short_date matches any value in races_and_weather_for_concat['short_date']
        races_and_weather = races_and_weather[~races_and_weather['short_date'].isin(races_and_weather_for_concat['short_date'])]
        print(f"Prior weather records were current: {len(races_and_weather_for_concat)} added.")
        # Merge the new data with the existing weatherData DataFrame
        races_and_weather = pd.concat([races_and_weather, races_and_weather_for_concat], ignore_index=True)
    else:
        races_and_weather = races_and_weather_for_concat
else:
    # If no new data, just load existing
    races_and_weather = pd.read_csv(weather_csv_path, sep='\t', usecols=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 
        'relative_humidity_2m', 'short_date', 'wind_speed_10m',  'id_races', 'grandPrixId', 'circuitId', 'hourly_precipitation_probability'])

races_and_weather.to_csv(path.join(DATA_DIR, 'f1WeatherData_AllData.csv'), columns=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 'relative_humidity_2m', 'short_date',
    'wind_speed_10m', 'id_races', 'hourly_precipitation_probability', 'grandPrixId', 'circuitId'], sep='\t', index=False)

races_and_weather_grouped = races_and_weather.groupby(['short_date', 'latitude_hourly', 'longitude_hourly', 'id_races', 'grandPrixId', 'circuitId']).agg(
    average_temp = ('temperature_2m', 'mean'), 
    total_precipitation = ('hourly_precipitation', 'sum'), 
    average_humidity = ('relative_humidity_2m', 'mean'), 
    average_wind_speed = ('wind_speed_10m', 'mean'),
    average_precipitation_probability = ('hourly_precipitation_probability', 'mean')
).reset_index()

races_and_weather_grouped.to_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), columns=['short_date', 'id_races', 'grandPrixId', 'circuitId', 'latitude_hourly', 'longitude_hourly', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed', 'average_precipitation_probability'], sep='\t', index=False)

# if the most recent date in the dataset is greater than today, that means that all of the other data in the weather dataset is current
# therefore, do no re-run the entire weather set, but instead re-run the weather for the upcoming race

# print(last_weather_date)
# print(last_weather_date >= datetime.datetime.now())
# print(last_weather_date <= (datetime.datetime.now() + timedelta(days=16)))

# weather_csv_path = os.path.join(DATA_DIR, 'f1WeatherData_AllData.csv')
# if os.path.exists(weather_csv_path):
#     processed_weather = pd.read_csv(weather_csv_path, sep='\t', usecols=['short_date', 'latitude_hourly', 'longitude_hourly'])
#     processed_weather_set = set(
#         zip(
#             processed_weather['short_date'],
#             processed_weather['latitude_hourly'],
#             processed_weather['longitude_hourly']
#         )
#     )
# else:
#     processed_weather_set = set()

# if last_weather_date >= datetime.datetime.now() and last_weather_date <= (datetime.datetime.now() + timedelta(days=16)):
#     print(f"Last weather date: {last_weather_date}")
#     newRecords = False

#     # Filter circuits_and_races_lat_long to only include records matching last_weather_date
#     circuits_and_races_lat_long = circuits_and_races_lat_long[
#     circuits_and_races_lat_long['date'] == last_weather_date.strftime('%Y-%m-%d')]
#     #print(len(circuits_and_races_lat_long))

# # Setup the Open-Meteo API client with cache and retry on error
# cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
# retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
# openmeteo = openmeteo_requests.Client(session = retry_session)

# # Make sure all required weather variables are listed here
# # The order of variables in hourly or daily is important to assign them correctly below

# all_hourly_data = []

# full_params = []

# for race in circuits_and_races_lat_long.itertuples():

#     short_date = pd.to_datetime(race.date).strftime('%Y-%m-%d')
#     lat = race.latitude
#     lon = race.longitude
#     if (short_date, lat, lon) in processed_weather_set:
#         print(f"Skipping weather for {short_date} ({lat}, {lon}) - already processed.")
#         continue  # Skip this race, already have weather data

#     params = {
#     "latitude": race.latitude,
# 	"longitude": race.longitude,
# 	"start_date": race.date.strftime('%Y-%m-%d'),
# 	"end_date": race.date.strftime('%Y-%m-%d'),
# 	"hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m", "precipitation_probability"],
#     "temperature_unit": "fahrenheit",
#     "wind_speed_unit": "mph",
#     "precipitation_unit": "inch"
# 	}

#     full_params.append((params, short_date, lat, lon))

# Loop through the list of params
# for params, short_date, lat, lon in full_params:

#     # use different URLs depending on whether we are seeking current or past weather
#     if datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') < datetime.datetime.now():
#         url = "https://archive-api.open-meteo.com/v1/archive"
#     elif datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') >= datetime.datetime.now() and datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') <= (datetime.datetime.now() + timedelta(days=16)):
#         url = "https://api.open-meteo.com/v1/forecast"   
#     else:
#         print("Break!")
#         print(datetime.datetime.strptime(params['start_date'], '%Y-%m-%d'))
#         break  
    
#     ### these next three lines can be removed if there are issues once new weather records are available
#     ### done to limit the number of calls to the API

#     #new_records = False

#     #if datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') > last_weather_date:
#     #    responses = openmeteo.weather_api(url, params=params)
#     responses = openmeteo.weather_api(url, params=params)
#     ## removed to allow rerun of weather for any missed data (4/16/2025)

# # Process first location. Add a for-loop for multiple locations or weather models
#     response = responses[0]

# # Process hourly data. The order of variables needs to be the same as requested.
#     hourly = response.Hourly()
#     hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
#     hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
#     hourly_relative_humidity_2m = hourly.Variables(2).ValuesAsNumpy()
#     hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
#     hourly_precipitation_probability = hourly.Variables(4).ValuesAsNumpy()

#     hourly_data = {"date": pd.date_range(
# 	    start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
# 	    end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
# 	    freq = pd.Timedelta(seconds = hourly.Interval()),
# 	    inclusive = "left"
# )}

#     hourly_data["latitude"] = response.Latitude()
#     hourly_data["longitude"] = response.Longitude()
#     hourly_data["temperature_2m"] = hourly_temperature_2m
#     hourly_data["hourly_precipitation"] = hourly_precipitation
#     hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
#     hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
#     hourly_data["hourly_precipitation_probability"] = hourly_precipitation_probability
#     hourly_data["short_date"] = pd.to_datetime(hourly_data["date"]).strftime('%Y-%m-%d')
#     hourly_dataframe = pd.DataFrame(data = hourly_data)
    
#     all_hourly_data = pd.DataFrame(data = all_hourly_data)

#     all_hourly_data = pd.concat([all_hourly_data, hourly_dataframe], ignore_index=True)

#     #    new_records = True

# circuits_and_races_lat_long.copy()
# circuits_and_races_lat_long['date'] = pd.to_datetime(circuits_and_races_lat_long['date']).dt.strftime('%Y-%m-%d')

# #races_and_weather = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])

# if newRecords:
#     ## Meaning that we need to re-run all data
#     races_and_weather = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])
#     print(f"New records added: {len(races_and_weather)}.")
# else:
#     ## meaning that we have all current data and don't need to rerun everything
#     ## in this case, we just want to add the new data to the end of the existing dataset
    
#     races_and_weather = pd.read_csv(path.join(DATA_DIR, 'f1WeatherData_AllData.csv'), sep='\t', usecols=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 
#     'relative_humidity_2m', 'short_date', 'wind_speed_10m',  'id_races', 'grandPrixId', 'circuitId'])
#     races_and_weather_for_concat = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])
#     #new_hourly_data = pd.concat(all_hourly_data, ignore_index=True)

#     # Exclude rows in races_and_weather where short_date matches any value in races_and_weather_for_concat['short_date']
#     races_and_weather = races_and_weather[~races_and_weather['short_date'].isin(races_and_weather_for_concat['short_date'])]
#     print(f"Prior weather records were current: {len(races_and_weather_for_concat)} added.")
#     # Merge the new data with the existing weatherData DataFrame

#     races_and_weather = pd.concat([races_and_weather, races_and_weather_for_concat], ignore_index=True)

# races_and_weather.to_csv(path.join(DATA_DIR, 'f1WeatherData_AllData.csv'), columns=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 'relative_humidity_2m', 'short_date',
# 'wind_speed_10m', 'id_races', 'hourly_precipitation_probability', 'grandPrixId', 'circuitId'], sep='\t')#)

# races_and_weather_grouped = races_and_weather.groupby(['short_date', 'latitude_hourly', 'longitude_hourly', 'id_races', 'grandPrixId', 'circuitId']).agg(average_temp = ('temperature_2m', 'mean'), 
#         total_precipitation = ('hourly_precipitation', 'sum'), average_humidity = ('relative_humidity_2m', 'mean'), average_wind_speed = ('wind_speed_10m', 'mean'),
#         average_precipitation_probability = ('hourly_precipitation_probability', 'mean')).reset_index()

# races_and_weather_grouped.to_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), columns=['short_date', 'id_races', 'grandPrixId', 'circuitId', 'latitude_hourly', 'longitude_hourly', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed', 'average_precipitation_probability'], sep='\t')#, mode='a', header=False)

raceNoEarlierThan = current_year - 10

race_results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json')) 
race_results = race_results[race_results['year'].between(raceNoEarlierThan, current_year-1)]
race_results_grouped = race_results.groupby(['raceId', 'year']).agg(totalFinishers = ('positionNumber', 'count'), totalParticipants = ('positionNumber', 'size')).reset_index()

# Add DNF column and calculate
race_results_grouped['DNF'] = race_results_grouped['totalParticipants'] - race_results_grouped['totalFinishers']
race_results_grouped.to_csv(path.join(DATA_DIR, 'f1RaceResultsData_Grouped.csv'), columns=['raceId', 'year', 'totalFinishers', 'totalParticipants', 'DNF'], sep='\t')  #, index=False)

### Active Drivers

all_practices = pd.concat([fp1, fp2, fp3], ignore_index=True)

all_practices =  all_practices[all_practices['year'] >= 2015]
active_drivers = pd.merge(
    drivers,
    all_practices,
    left_on='id',
    right_on='driverId',
    how='right',
    suffixes=['_drivers', '_all_practices']).drop_duplicates(subset=['driverId'])

active_drivers.to_csv(path.join(DATA_DIR, 'active_drivers.csv'), columns = ['driverId', 'abbreviation', 'name', 'firstName', 'lastName', 'driverNumber'], sep='\t', index=False)

# Show all rows where abbreviation and name are duplicated
duplicates = active_drivers[active_drivers.duplicated(subset=['abbreviation', 'name'], keep=False)]
print(f"Number of duplicate drivers: {len(duplicates)}")
print(duplicates)

print("All active drivers saved to active_drivers.csv")

print("Successfully generated F1 analysis data files.")
