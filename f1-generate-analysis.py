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

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year
raceNoEarlierThan = current_year - 10

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
fp4 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-4-results.json')) 
current_practices = pd.read_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t') 
practice_best = pd.read_csv(path.join(DATA_DIR, 'practice_best_fp1_fp2.csv'), sep='\t')
pitstops = pd.read_json(path.join(DATA_DIR, 'f1db-races-pit-stops.json'))
all_laps = pd.read_csv(path.join(DATA_DIR, 'all_laps.csv'), sep='\t')
constructor_standings = pd.read_csv(path.join(DATA_DIR, 'constructor_standings.csv'), sep='\t')
driver_standings = pd.read_csv(path.join(DATA_DIR, 'driver_standings.csv'), sep='\t')

# print(current_practices.columns.tolist())

qualifying = pd.merge(
    qualifying_json,
    qualifying_csv[['q1_sec', 'q1_pos', 'q2_sec', 'q2_pos', 'q3_sec', 'q3_pos', 'best_qual_time', 'teammate_qual_delta', 'raceId', 'driverId', ]],
    left_on=['raceId', 'driverId'],
    right_on=['raceId', 'driverId'],
    how='right',
    suffixes=('_json', '_csv')
)



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

results_and_drivers_and_constructors_and_grandprix_and_qualifying[['grandPrixName', 'resultsQualificationPositionNumber', 'abbreviation_results', 'raceId_results', 'resultsDriverId', # 'q1', 'q2', 'q3', 
                                                                   'resultsYear', 'constructorName','resultsDriverId', 'resultsDriverName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
                                      'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 
                                      'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'grandPrixYear', 'raceIdFromGrandPrix', 'grandPrixRaceId', 'circuitType']]

fp1_fp2 = pd.merge(fp1, fp2, on=['raceId', 'driverId'], how='left', suffixes=['_fp1', '_fp2'])
fp1_fp2_fp3 = pd.merge(fp1_fp2, fp3, on=['raceId', 'driverId'], how='left', suffixes=['_fp1_2', '_fp3'])
fp1_fp2_fp3_fp4 = pd.merge(fp1_fp2_fp3, fp4, on=['raceId', 'driverId'], how='left', suffixes=['_fp1_fp2_fp3', '_fp4'])


fp1_fp2_fp3_fp4.rename(columns={'driverId': 'fpDriverId', 'raceId': 'fpRaceId', 'positionNumber_fp1': 'fp1PositionNumber', 'time_fp1': 'fp1Time', 'gap_fp1': 'fp1Gap', 'interval_fp1': 'fp1Interval', 
'positionNumber_fp2': 'fp2PositionNumber', 'time_fp2': 'fp2Time', 'gap_fp2': 'fp2Gap', 'interval_fp2': 'fp2Interval', 
'positionNumber_fp1_fp2_fp3': 'fp3PositionNumber', 'time_fp1_fp2_fp3': 'fp3Time', 'gap_fp1_fp2_fp3': 'fp3Gap', 'interval_fp1_fp2_fp3': 'fp3Interval', 
'positionNumber_fp4': 'fp4PositionNumber', 'time_fp4': 'fp4Time', 'gap_fp4': 'fp4Gap', 'interval_fp4': 'fp4Interval' }, inplace=True)

# Drop 'time' from the right DataFrame if you don't need it
fp1_fp2_fp3_fp4 = fp1_fp2_fp3_fp4.drop(columns=['time', 'round'], errors='ignore')
#fp1_fp2_fp3_fp4 = fp1_fp2_fp3_fp4.drop(columns=['time'], errors='ignore')


results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying, fp1_fp2_fp3_fp4, left_on=['raceId_results', 'resultsDriverId'], right_on=['fpRaceId','fpDriverId' ], how='left', suffixes=['_results', '_practices']) 
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
    ['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber', 'fp4PositionNumber']].mean(axis=1)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['fp4PositionNumber', 'fp3PositionNumber', 'fp2PositionNumber', 'fp1PositionNumber']].bfill(axis=1).iloc[:, 0]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['q3_sec', 'q2_sec', 'q1_sec']].bfill(axis=1).iloc[:, 0]

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
if results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'].notnull().any():
    # Convert 'bestQualifyingTime' to seconds
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

    # Apply the conversion function to the column
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'].apply(time_to_seconds)
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
else:
    print("No valid times found in 'bestQualifyingTime' column.")

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

# # Calculate qualifying gap to pole
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'] = pd.to_numeric(
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'], errors='coerce'
# )

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'].head())

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec'] = pd.to_numeric(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec'], errors='coerce'
)
print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec'].head())
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_gap_to_pole'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_qual_time'] -
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pole_time_sec']
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

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices, current_practices, left_on=['raceId_results', 'resultsDriverId'], right_on=['raceId', 'resultsDriverId'], how='left', suffixes=['', '_practices'])

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['resultsDriverId', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec']]

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['grandPrixName', 'best_s1_sec_practices', 'best_s2_sec_practices', 'best_s2_sec_practices', 'best_theory_lap_sec_practices', 'SpeedI1_mph_practices', 'SpeedI2_mph_practices', 'SpeedFL_mph_practices', 'SpeedST_mph_practices']]

#results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1Test1.csv'))

print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'round_results': 'round', 'turns_results': 'turns', 'circuitId_results': 'circuitId',
                                                                                                 'best_s1_sec_results': 'best_s1_sec', 'best_s2_sec_results': 'best_s2_sec', 'best_s3_sec_results': 'best_s3_sec', 'best_theory_lap_sec_results': 'best_theory_lap_sec',
                                                                                                 'SpeedI1_mph_results': 'SpeedI1_mph', 'SpeedI2_mph_results': 'SpeedI2_mph', 'SpeedFL_mph_results': 'SpeedFL_mph', 'SpeedST_mph_results': 'SpeedST_mph'
                                                                                                }, inplace=True)

#results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1Test2.csv'))

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
        ['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber', 'fp4PositionNumber']
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

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())



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

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

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
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['wet_weather_experience'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_humidity'] * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['driverTotalRaceStarts']
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['wind_penalty'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_wind_speed'] * results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber']
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

# # Top speed × SafetyCarStatus (as a proxy for DRS laps)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['top_speed_x_safetycar'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SpeedFL_mph'] *
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SafetyCarStatus']
# )

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

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['historical_avgLapPace'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverName')['avgLapPace']
    .transform(lambda x: x.rolling(window=3, min_periods=1).mean().shift(1))
)

# Top Speed Rank (lower is better)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['top_speed_rank'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('raceId_results')['SpeedFL_mph'].rank(ascending=False)
)

# Grid Position Percentile
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_position_percentile'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] /
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['totalParticipants']
# )

# Grid Pos × Track Overtake Score (if you have 'track_overtake_score')
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grid_x_overtake_score'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] *
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['track_overtake_score']
# )

print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())


# Pit Count (already present as 'numberOfStops')

# Pit Loss Time (total time in pits)
# Already present as 'totalStopTime'

# Pit Timing Index (lap of first pit / total laps)
# if 'pit_lap' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_timing_index'] = (
#         results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_lap'] /
#         results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['grandPrixLaps']
#     )

# Power-to-Corner Ratio
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['power_to_corner_ratio'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['SpeedFL_mph'] /
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['turns']
)

# Downforce Demand Score (sector2_avg_time / track_length)
# if 'best_s2_sec' in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['downforce_demand_score'] = (
#         results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_s2_sec'] /
#         results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['track_length']
#     )

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

# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['average_safetycar_status_5_races'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['SafetyCarStatus']
#     .transform(lambda x: x.rolling(window=5, min_periods=1).mean().shift(1))
# )

# # See recent form for a single driver
# driver = "Max Verstappen"
# df = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
# print(df[df['resultsDriverName'] == driver][['grandPrixYear', 'raceId_results', 'resultsFinalPositionNumber', 'recent_form']])

# List of columns to update from the reference table
columns_to_update = [
    'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec',
    'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',
    'best_theory_lap_sec', 'Session', 
]

# Drop these columns from the main DataFrame if they exist
for col in columns_to_update:
    if col in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=[col])

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['best_s2_sec'])

# # Drop 'raceId_results' and 'driverId_results' from practice_best if they exist
# for col in ['raceId_results', 'driverId_results']:
#     if col in practice_best.columns:
#         practice_best = practice_best.drop(columns=[col])

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

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist()) 

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

# field_list = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist()

# for field in field_list:
#     print(field)

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

# print("Sample main DataFrame keys:")
# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['grandPrixYear', 'round', 'abbreviation_results']].drop_duplicates().head())

# print("Sample pivot DataFrame keys:")
# print(pivot[['year', 'round', 'Driver']].drop_duplicates().head())

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

## Update constructor name and associated columns
# constructor_names = constructors[['constructorId', 'name']].copy()
# constructor_names.rename(columns={'name': 'constructorName'}, inplace=True)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
#     constructor_names,
#     left_on='constructorId_results',
#     right_on='constructorId',
#     how='left'
# )
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=['constructorId'])
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'constructorName': 'constructorName_results'})

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices,
    constructors[['id', 'name', 'totalRaceEntries', 'totalRaceStarts', 'totalRaceWins', 'total1And2Finishes', 'totalPodiumRaces', 'totalPolePositions', 'totalFastestLaps']],
    left_on='constructorId_results',
    right_on='id',
    how='left'
)#.drop(columns=['constructorId'])

# Positions Gained in First Lap %
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['positions_gained_first_lap_pct'] = (
    (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['delta_lap_2']) /
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber']
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

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

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

print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['practice_improvement']].head())

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

# # 7. Practice Improvement × Constructor Podium Races
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement_x_constructor_podiums'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['practice_improvement'] *
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['constructorTotalPodiumRaces']
# )

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

# ...existing code to save to f1ForAnalysis.csv...

# ...place after all merges and before .to_csv()...

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
    ['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber', 'fp4PositionNumber']
].std(axis=1)

# 5. Qualifying Position Percentile (within race)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['qualifying_position_percentile'] = (
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('grandPrixName')['resultsQualificationPositionNumber']
    .transform(lambda x: x.rank(pct=True))
)

# # 6. Overtake Potential (positions gained / grid position)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['overtake_potential'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['positionsGained'] /
#     (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsStartingGridPositionNumber'] + 1e-6)
# )



# # 7. Recent DNF Streak (count DNFs in last 3 races)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['recent_dnf_streak'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices
#     .sort_values(['resultsDriverName', 'grandPrixYear', 'raceId_results'])
#     .groupby('resultsDriverName')['DNF']
#     .transform(lambda x: x.rolling(window=3, min_periods=1).sum().shift(1))
# )

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

# # 16. Pit Stop Efficiency (pit stop delta / number of stops)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_stop_efficiency'] = (
#     results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['pit_stop_delta'] /
#     (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['numberOfStops'] + 1e-6)
# )

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

# ...continue


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




# print("Constructor columns after merge:")
# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.to_list())

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.head(100))

# # Update driver names and associated columns
# driver_names = drivers[['id', 'name', 'totalRaceStarts', 'totalPodiums', 'bestRaceResult', 'bestStartingGridPosition', 'totalRaceLaps', 'totalChampionshipWins', 'totalRaceEntries', 'totalRaceWins', 'totalPolePositions']].copy()
# driver_names.rename(columns={'name': 'resultsDriverName', 
#                             'totalRaceStarts': 'driverTotalRaceStarts', 
#                             'totalPodiums': 'driverTotalPodiums', 
#                             'bestRaceResult': 'driverBestRaceResult', 
#                             'bestStartingGridPosition': 'driverBestStartingGridPosition', 
#                             'totalRaceLaps': 'driverTotalRaceLaps', 
#                             'totalChampionshipWins': 'driverTotalChampionshipWins',
#                             'totalRaceEntries': 'driverTotalRaceEntries',
#                             'totalRaceWins': 'driverTotalRaceWins',
#                             'totalPolePositions': 'driverTotalPolePositions'}, inplace=True)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['constructorName','resultsDriverId', 'resultsDriverName', 'grandPrixName', 'best_s1_sec', 'best_s2_sec', 'best_s2_sec', 'best_theory_lap_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'q1', 'q2', 'q3', 'bestQualifyingTime', 'timeMillis_results', 'streetRace', 'trackRace', 'resultsDriverId', 'yearsActive', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 
                                        'q1End', 'q2End', 'q3Top10', 'resultsDriverId', 'resultsReasonRetired','averagePracticePosition', 'raceId_results', 'resultsFinalPositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'fp1PositionNumber', 'fp1Time', 'fp1Gap', 
                                        'fp1Interval', 'positionsGained', 'fp1PositionNumber', 'fp2PositionNumber','fp3PositionNumber','fp4PositionNumber', 'resultsYear',  'resultsStartingGridPositionNumber', 
                                       'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 'round', 
                                       'driverTotalRaceStarts', 'driverTotalPodiums', 'driverBestRaceResult',  'driverBestStartingGridPosition', 'driverTotalRaceLaps', 'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps',
                                       'driverTotalPodiums', 'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'turns', 'grandPrixYear', 'raceIdFromGrandPrix', 'grandPrixRaceId', 'short_date', 'DNF', 'driverTotalPolePositions', 'activeDriver']]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), columns=['grandPrixYear', 'grandPrixName', 'raceId_results', 'circuitId', 'grandPrixRaceId', 'resultsDriverName', 'q1', 'q2', 'q3', 
                                        'fp1Time', 'fp1Gap', 'fp1Interval', 'fp1PositionNumber', 'fp2Time', 'fp2Gap', 'fp2Interval', 'fp2PositionNumber', 'fp3Time', 'fp3Gap', 'fp3Interval', 'fp3PositionNumber','fp4Time', 'fp4Gap', 'fp4Interval', 
                                        'fp4PositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'resultsYear', 'constructorName',  'resultsStartingGridPositionNumber',   'constructorId_results',
                                      'positionsGained', 'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 'round',
                                      'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums',
                                      'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'turns', 'short_date', 'DNF', 'fp1PositionNumber', 'fp2PositionNumber', 'streetRace', 'trackRace', 'avgLapPace', 'finishingTime', 'timeMillis_results', 'bestQualifyingTime',
                                     'fp3PositionNumber','fp4PositionNumber','averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10','resultsDriverId', 'driverTotalPolePositions', 'activeDriver', 'yearsActive',
                                      'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'time', 'Session', 'driverDNFCount', 'driverDNFAvg', 'SafetyCarStatus', 
                                      'resultsFinalPositionNumber','recent_form_3_races', 'recent_form_5_races', 'constructor_recent_form_3_races', 'constructor_recent_form_5_races', 
                                      'CleanAirAvg_FP1', 'DirtyAirAvg_FP1', 'Delta_FP1', 'CleanAirAvg_FP2', 'DirtyAirAvg_FP2', 'Delta_FP2', 'CleanAirAvg_FP3', 'DirtyAirAvg_FP3','Delta_FP3',  
                                       'numberOfStops', 'averageStopTime', 'totalStopTime', 'pit_lane_time_constant', 'pit_stop_delta', 'engineManufacturerId', 'delta_from_race_avg', 'driverAge',
                                       'finishing_position_std_driver', 'finishing_position_std_constructor', 'delta_lap_2', 'delta_lap_5', 'delta_lap_10', 'delta_lap_15', 'delta_lap_20',
                                       'delta_lap_2_historical', 'delta_lap_5_historical', 'delta_lap_10_historical', 'delta_lap_15_historical', 'delta_lap_20_historical', 'abbreviation', 
                                       'driver_positionsGained_5_races', 'driver_dnf_rate_5_races', 'avg_final_position_per_track', 'last_final_position_per_track', 
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
                                         'qual_to_final_delta_5yr', 'qual_to_final_delta_3yr', 'overtake_potential_3yr', 'overtake_potential_5yr'
                                          ], sep='\t', index=False)

# positionCorrelation = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[[
#     'lastFPPositionNumber', 'resultsFinalPositionNumber', 'resultsStartingGridPositionNumber','grandPrixLaps', 
#     'best_s1_sec', 'best_s2_sec', 'best_s2_sec', 'best_theory_lap_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',
#     'averagePracticePosition', 'DNF', 'resultsTop10', 'resultsTop5', 'resultsPodium', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 'positionsGained', 'q1End', 'q2End', 'q3Top10', 'streetRace', 'trackRace',
#      'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'driverTotalPolePositions', 'yearsActive']].corr(method='pearson')



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

# if the most recent date in the dataset is greater than today, that means that all of the other data in the weather dataset is current
# therefore, do no re-run the entire weather set, but instead re-run the weather for the upcoming race

# print(last_weather_date)
# print(last_weather_date >= datetime.datetime.now())
# print(last_weather_date <= (datetime.datetime.now() + timedelta(days=16)))

weather_csv_path = os.path.join(DATA_DIR, 'f1WeatherData_AllData.csv')
if os.path.exists(weather_csv_path):
    processed_weather = pd.read_csv(weather_csv_path, sep='\t', usecols=['short_date', 'latitude_hourly', 'longitude_hourly'])
    processed_weather_set = set(
        zip(
            processed_weather['short_date'],
            processed_weather['latitude_hourly'],
            processed_weather['longitude_hourly']
        )
    )
else:
    processed_weather_set = set()

if last_weather_date >= datetime.datetime.now() and last_weather_date <= (datetime.datetime.now() + timedelta(days=16)):
    print(f"Last weather date: {last_weather_date}")
    newRecords = False

    # Filter circuits_and_races_lat_long to only include records matching last_weather_date
    circuits_and_races_lat_long = circuits_and_races_lat_long[
    circuits_and_races_lat_long['date'] == last_weather_date.strftime('%Y-%m-%d')]
    #print(len(circuits_and_races_lat_long))

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below

all_hourly_data = []

full_params = []

for race in circuits_and_races_lat_long.itertuples():

    short_date = pd.to_datetime(race.date).strftime('%Y-%m-%d')
    lat = race.latitude
    lon = race.longitude
    if (short_date, lat, lon) in processed_weather_set:
        print(f"Skipping weather for {short_date} ({lat}, {lon}) - already processed.")
        continue  # Skip this race, already have weather data

    params = {
    "latitude": race.latitude,
	"longitude": race.longitude,
	"start_date": race.date.strftime('%Y-%m-%d'),
	"end_date": race.date.strftime('%Y-%m-%d'),
	"hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m", "precipitation_probability"],
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch"
	}

    full_params.append((params, short_date, lat, lon))

# Loop through the list of params
for params, short_date, lat, lon in full_params:



    # use different URLs depending on whether we are seeking current or past weather
    if datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') < datetime.datetime.now():
        url = "https://archive-api.open-meteo.com/v1/archive"
    elif datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') >= datetime.datetime.now() and datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') <= (datetime.datetime.now() + timedelta(days=16)):
        url = "https://api.open-meteo.com/v1/forecast"   
    else:
        print("Break!")
        print(datetime.datetime.strptime(params['start_date'], '%Y-%m-%d'))
        break  
    
    ### these next three lines can be removed if there are issues once new weather records are available
    ### done to limit the number of calls to the API

    #new_records = False

    #if datetime.datetime.strptime(params['start_date'], '%Y-%m-%d') > last_weather_date:
    #    responses = openmeteo.weather_api(url, params=params)
    responses = openmeteo.weather_api(url, params=params)
    ## removed to allow rerun of weather for any missed data (4/16/2025)

# Process first location. Add a for-loop for multiple locations or weather models
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

    #    new_records = True

circuits_and_races_lat_long['date'] = pd.to_datetime(circuits_and_races_lat_long['date']).dt.strftime('%Y-%m-%d')

#races_and_weather = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])

if newRecords:
    ## Meaning that we need to re-run all data
    races_and_weather = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])
    print(f"New records added: {len(races_and_weather)}.")
else:
    ## meaning that we have all current data and don't need to rerun everything
    ## in this case, we just want to add the new data to the end of the existing dataset
    
    races_and_weather = pd.read_csv(path.join(DATA_DIR, 'f1WeatherData_AllData.csv'), sep='\t', usecols=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 
    'relative_humidity_2m', 'short_date', 'wind_speed_10m',  'id_races', 'grandPrixId', 'circuitId'])
    races_and_weather_for_concat = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])
    #new_hourly_data = pd.concat(all_hourly_data, ignore_index=True)

    # Exclude rows in races_and_weather where short_date matches any value in races_and_weather_for_concat['short_date']
    races_and_weather = races_and_weather[~races_and_weather['short_date'].isin(races_and_weather_for_concat['short_date'])]
    print(f"Prior weather records were current: {len(races_and_weather_for_concat)} added.")
    # Merge the new data with the existing weatherData DataFrame

    races_and_weather = pd.concat([races_and_weather, races_and_weather_for_concat], ignore_index=True)

races_and_weather.to_csv(path.join(DATA_DIR, 'f1WeatherData_AllData.csv'), columns=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 'relative_humidity_2m', 'short_date',
'wind_speed_10m', 'id_races', 'hourly_precipitation_probability', 'grandPrixId', 'circuitId'], sep='\t')#)

races_and_weather_grouped = races_and_weather.groupby(['short_date', 'latitude_hourly', 'longitude_hourly', 'id_races', 'grandPrixId', 'circuitId']).agg(average_temp = ('temperature_2m', 'mean'), 
        total_precipitation = ('hourly_precipitation', 'sum'), average_humidity = ('relative_humidity_2m', 'mean'), average_wind_speed = ('wind_speed_10m', 'mean'),
        average_precipitation_probability = ('hourly_precipitation_probability', 'mean')).reset_index()

races_and_weather_grouped.to_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), columns=['short_date', 'id_races', 'grandPrixId', 'circuitId', 'latitude_hourly', 'longitude_hourly', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed', 'average_precipitation_probability'], sep='\t')#, mode='a', header=False)

raceNoEarlierThan = current_year - 10

race_results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json')) 
race_results = race_results[race_results['year'].between(raceNoEarlierThan, current_year-1)]
race_results_grouped = race_results.groupby(['raceId', 'year']).agg(totalFinishers = ('positionNumber', 'count'), totalParticipants = ('positionNumber', 'size')).reset_index()

# Add DNF column and calculate
race_results_grouped['DNF'] = race_results_grouped['totalParticipants'] - race_results_grouped['totalFinishers']
race_results_grouped.to_csv(path.join(DATA_DIR, 'f1RaceResultsData_Grouped.csv'), columns=['raceId', 'year', 'totalFinishers', 'totalParticipants', 'DNF'], sep='\t')  #, index=False)

### Active Drivers

all_practices = pd.concat([fp1, fp2, fp3, fp4], ignore_index=True)

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
