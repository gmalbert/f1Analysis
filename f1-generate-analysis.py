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

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year
raceNoEarlierThan = current_year - 10

##### Pit Stops

pitstops = pd.read_json(path.join(DATA_DIR, 'f1db-races-pit-stops.json')) 

pitstops = pitstops[pitstops['year'].between(raceNoEarlierThan, current_year)]

pitstops_grouped = pitstops.groupby(['raceId', 'driverId', 'constructorId']).agg(numberOfStops = ('stop', 'count'), averageStopTimeMillis = ('timeMillis', 'mean'), totalStopTimeMillis = ('timeMillis', 'sum')).reset_index()

pitstops_grouped['averageStopTime'] = (pitstops_grouped['averageStopTimeMillis'] / 1000).round(2)
pitstops_grouped['totalStopTime'] = (pitstops_grouped['totalStopTimeMillis'] / 1000).round(2)

pitstops_grouped.to_csv(path.join(DATA_DIR, 'f1PitStopsData_Grouped.csv'), columns=['raceId', 'driverId', 'constructorId', 'numberOfStops', 'averageStopTime', 'totalStopTime'], sep='\t')


## Results and Qualifying
drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json')) 
race_results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json')) 
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 
constructors = pd.read_json(path.join(DATA_DIR, 'f1db-constructors.json')) 
qualifying = pd.read_json(path.join(DATA_DIR, 'f1db-races-qualifying-results.json')) 
grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json')) 
fp1 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-1-results.json')) 
fp2 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-2-results.json')) 
fp3 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-3-results.json')) 
fp4 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-4-results.json')) 
current_practices = pd.read_csv(path.join(DATA_DIR, 'all_practice_laps.csv'), sep='\t') 
practice_best = pd.read_csv(path.join(DATA_DIR, 'practice_best_fp1_fp2.csv'), sep='\t')

races_and_grandPrix = pd.merge(races, grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_races', '_grandPrix'])
races_and_grandPrix.rename(columns={'id_races': 'raceIdFromGrandPrix', 'id_grandPrix': 'grandPrixRaceId', 'fullName': 'grandPrixName', 'laps': 'grandPrixLaps', 'year': 'grandPrixYear'}, inplace=True)

results_and_drivers = pd.merge(race_results, drivers, left_on='driverId', right_on='id', how='inner', suffixes=['_results', '_drivers'])
results_and_drivers.rename(columns={'year': 'resultsYear', 'driverId': 'resultsDriverId', 'qualificationPositionNumber': 'resultsQualificationPositionNumber', 'positionNumber': 'resultsFinalPositionNumber', 'name': 'resultsDriverName', 'gridPositionNumber': 'resultsStartingGridPositionNumber', 'reasonRetired': 'resultsReasonRetired'}, inplace=True)

results_and_drivers = results_and_drivers[results_and_drivers['resultsYear'] >= raceNoEarlierThan]

results_and_drivers[['totalPolePositions', 'totalFastestLaps', 'totalRaceWins', 'totalRaceEntries', 'totalRaceStarts', 'totalPodiums', 'bestRaceResult', 'bestStartingGridPosition', 'totalRaceLaps']]

results_and_drivers_and_constructors = pd.merge(results_and_drivers, constructors, left_on='constructorId', right_on='id', how='inner', suffixes=['_results', '_constructors'])
results_and_drivers_and_constructors.rename(columns={'name': 'constructorName', 'totalRaceStarts_constructors': 'constructorTotalRaceStarts', 'totalRaceEntries_constructors': 'constructorTotalRaceEntries', 'totalRaceWins_constructors': 'constructorTotalRaceWins', 
                                                     'total1And2Finishes': 'constructorTotal1And2Finishes', 'bestChampionshipPosition_results': 'driverBestChampionshipPosition', 'bestStartingGridPosition_results': 'driverBestStartingGridPosition', 
                                                     'bestRaceResult_results': 'driverBestRaceResult', 'totalChampionshipWins_results': 'driverTotalChampionshipWins',
                                                     'totalRaceEntries_results': 'driverTotalRaceEntries', 'totalRaceStarts_results': 'driverTotalRaceStarts', 'totalRaceWins_results': 'driverTotalRaceWins', 'totalRaceLaps_results': 'driverTotalRaceLaps', 
                                                     'totalPolePositions_results': 'driverTotalPolePositions', 'timeMillis_results': 'timeMillis',  
                                                     'totalPodiums_results': 'driverTotalPodiums', 'totalPodiumRaces': 'constructorTotalPodiumRaces', 'totalPolePositions_constructors': 'constructorTotalPolePositions', 'totalFastestLaps_constructors': 'constructorTotalFastestLaps'}, inplace=True)

results_and_drivers_and_constructors_and_grandprix = pd.merge(results_and_drivers_and_constructors, races_and_grandPrix, left_on='raceId', right_on='raceIdFromGrandPrix', how='inner', suffixes=['_results', '_grandprix'])

results_and_drivers_and_constructors_and_grandprix_and_qualifying = pd.merge(results_and_drivers_and_constructors_and_grandprix, qualifying, left_on=['raceIdFromGrandPrix', 'resultsDriverId'], right_on=['raceId', 'driverId'], how='inner', suffixes=['_results', '_qualifying']) 

results_and_drivers_and_constructors_and_grandprix_and_qualifying[['grandPrixName', 'resultsQualificationPositionNumber', 'raceId_results', 'resultsDriverId',  'q1', 'q2', 'q3', 'resultsYear', 'constructorName','resultsDriverId', 'resultsDriverName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
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

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['short_date'] = pd.to_datetime(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['date']).dt.strftime('%Y-%m-%d')

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsPodium'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsTop5'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=5)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsTop10'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=10)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['DNF'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsReasonRetired'].notnull())
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['averagePracticePosition'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['fp1PositionNumber', 'fp2PositionNumber', 'fp3PositionNumber', 'fp4PositionNumber']].mean(axis=1)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['lastFPPositionNumber'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['fp4PositionNumber', 'fp3PositionNumber', 'fp2PositionNumber', 'fp1PositionNumber']].bfill(axis=1).iloc[:, 0]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[
    ['q3', 'q2', 'q1']].bfill(axis=1).iloc[:, 0]

# Ensure the 'bestQualifyingTime' column is not null or invalid
if results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime'].notnull().any():
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
    results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime_sec'] = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['bestQualifyingTime'].apply(time_to_seconds)

    # Display the converted column
    print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['bestQualifyingTime', 'bestQualifyingTime_sec']].head())
else:
    print("No valid times found in 'bestQualifyingTime' column.")

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

# Count the number of unique active years for each driver
yearsActive = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverId')['grandPrixYear'].nunique().reset_index()

yearsActiveGroup = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.groupby('resultsDriverId')['grandPrixYear'].nunique().reset_index()
yearsActiveGroup.rename(columns={'grandPrixYear': 'yearsActive'}, inplace=True)

# Merge the active years count back into the main DataFrame
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.merge(
    yearsActiveGroup, on='resultsDriverId', how='left')

# Rename columns to avoid conflicts because there were duplicates column names post merge
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'round_results': 'round', 'time_results': 'time'}, inplace=True)

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices, current_practices, left_on=['raceId_results', 'resultsDriverId'], right_on=['raceId', 'driverId'], how='left', suffixes=['_results', '_practices'])

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns.tolist())

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns)
# results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['grandPrixName', 'best_s1_sec_practices', 'best_s2_sec_practices', 'best_s2_sec_practices', 'best_theory_lap_sec_practices', 'SpeedI1_mph_practices', 'SpeedI2_mph_practices', 'SpeedFL_mph_practices', 'SpeedST_mph_practices']]

#results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1Test1.csv'))

# print(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.rename(columns={'round_results': 'round', 'turns_results': 'turns', 'circuitId_results': 'circuitId',
                                                                                                 'best_s1_sec_results': 'best_s1_sec', 'best_s2_sec_results': 'best_s2_sec', 'best_s3_sec_results': 'best_s3_sec', 'best_theory_lap_sec_results': 'best_theory_lap_sec',
                                                                                                 'SpeedI1_mph_results': 'SpeedI1_mph', 'SpeedI2_mph_results': 'SpeedI2_mph', 'SpeedFL_mph_results': 'SpeedFL_mph', 'SpeedST_mph_results': 'SpeedST_mph'
                                                                                                }, inplace=True)

#results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1Test2.csv'))

# Read your main DataFrame (assuming it's already loaded as results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices)


# List of columns to update from the reference table
columns_to_update = [
    'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec',
    'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',
    'best_theory_lap_sec', 'Session'
]

# Drop these columns from the main DataFrame if they exist
for col in columns_to_update:
    if col in results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.columns:
        results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.drop(columns=[col])

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

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['grandPrixName', 'best_s1_sec', 'best_s2_sec', 'best_s2_sec', 'best_theory_lap_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'avgLapPace', 'q1', 'q2', 'q3', 'bestQualifyingTime', 'timeMillis_results', 'streetRace', 'trackRace', 'resultsDriverId', 'yearsActive', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 
                                        'q1End', 'q2End', 'q3Top10', 'resultsDriverId', 'resultsReasonRetired','averagePracticePosition', 'raceId_results', 'resultsFinalPositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'fp1PositionNumber', 'fp1Time', 'fp1Gap', 
                                        'fp1Interval', 'positionsGained', 'fp1PositionNumber', 'fp2PositionNumber','fp3PositionNumber','fp4PositionNumber', 'resultsYear', 'constructorName','resultsDriverId', 'resultsDriverName',  'resultsStartingGridPositionNumber', 
                                       'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 'round', 'bestQualifyingTime_sec',
                                       'driverTotalRaceStarts', 'driverTotalPodiums', 'driverBestRaceResult',  'driverBestStartingGridPosition', 'driverTotalRaceLaps', 'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps',
                                       'driverTotalPodiums', 'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'turns', 'grandPrixYear', 'raceIdFromGrandPrix', 'grandPrixRaceId', 'short_date', 'DNF', 'driverTotalPolePositions', 'activeDriver']]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), columns=['grandPrixYear', 'grandPrixName', 'raceId_results', 'circuitId', 'grandPrixRaceId', 'resultsDriverName', 'q1', 'q2', 'q3', 
                                        'fp1Time', 'fp1Gap', 'fp1Interval', 'fp1PositionNumber', 'fp2Time', 'fp2Gap', 'fp2Interval', 'fp2PositionNumber', 'fp3Time', 'fp3Gap', 'fp3Interval', 'fp3PositionNumber','fp4Time', 'fp4Gap', 'fp4Interval', 
                                        'fp4PositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'resultsYear', 'constructorName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 'bestQualifyingTime_sec', 'constructorId_results',
                                      'positionsGained', 'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 'round',
                                      'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums',
                                      'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'turns', 'short_date', 'DNF', 'fp1PositionNumber', 'fp2PositionNumber', 'streetRace', 'trackRace', 'avgLapPace', 'timeMillis_results', 'bestQualifyingTime',
                                     'fp3PositionNumber','fp4PositionNumber','averagePracticePosition', 'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'q1End', 'q2End', 'q3Top10','resultsDriverId', 'driverTotalPolePositions', 'activeDriver', 'yearsActive',
                                      'LapTime_sec', 'best_s1_sec', 'best_s2_sec', 'best_s3_sec', 'best_theory_lap_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph', 'Session' ], sep='\t')

positionCorrelation = results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[[
    'lastFPPositionNumber', 'resultsFinalPositionNumber', 'resultsStartingGridPositionNumber','grandPrixLaps', 
    'best_s1_sec', 'best_s2_sec', 'best_s2_sec', 'best_theory_lap_sec', 'SpeedI1_mph', 'SpeedI2_mph', 'SpeedFL_mph', 'SpeedST_mph',
    'averagePracticePosition', 'DNF', 'resultsTop10', 'resultsTop5', 'resultsPodium', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotalPolePositions', 'turns', 'positionsGained', 'q1End', 'q2End', 'q3Top10', 'streetRace', 'trackRace',
     'driverBestStartingGridPosition', 'driverBestRaceResult', 'driverTotalChampionshipWins', 'driverTotalRaceEntries', 'driverTotalRaceStarts', 'driverTotalRaceWins', 'driverTotalRaceLaps', 'driverTotalPodiums', 'driverTotalPolePositions', 'yearsActive']].corr(method='pearson')

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

print(last_weather_date)
print(last_weather_date >= datetime.datetime.now())
print(last_weather_date <= (datetime.datetime.now() + timedelta(days=16)))


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
         
    params = {
    "latitude": race.latitude,
	"longitude": race.longitude,
	"start_date": race.date.strftime('%Y-%m-%d'),
	"end_date": race.date.strftime('%Y-%m-%d'),
	"hourly": ["temperature_2m", "precipitation", "relative_humidity_2m", "wind_speed_10m"],
    "temperature_unit": "fahrenheit",
    "wind_speed_unit": "mph",
    "precipitation_unit": "inch"
	}

    full_params.append(params)

# Loop through the list of params
for params in full_params:
    
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
    'relative_humidity_2m', 'short_date', 'wind_speed_10m', 'id_races', 'grandPrixId', 'circuitId'])
    races_and_weather_for_concat = pd.merge(all_hourly_data, circuits_and_races_lat_long, left_on='short_date', right_on='date', how='inner', suffixes=['_hourly', '_lat_long'])
    #new_hourly_data = pd.concat(all_hourly_data, ignore_index=True)

    # Exclude rows in races_and_weather where short_date matches any value in races_and_weather_for_concat['short_date']
    races_and_weather = races_and_weather[~races_and_weather['short_date'].isin(races_and_weather_for_concat['short_date'])]
    print(f"Prior weather records were current: {len(races_and_weather_for_concat)} added.")
    # Merge the new data with the existing weatherData DataFrame

    races_and_weather = pd.concat([races_and_weather, races_and_weather_for_concat], ignore_index=True)

races_and_weather.to_csv(path.join(DATA_DIR, 'f1WeatherData_AllData.csv'), columns=['date_hourly', 'latitude_hourly', 'longitude_hourly', 'temperature_2m', 'hourly_precipitation', 'relative_humidity_2m', 'short_date',
'wind_speed_10m', 'id_races', 'grandPrixId', 'circuitId'], sep='\t')#)

races_and_weather_grouped = races_and_weather.groupby(['short_date', 'latitude_hourly', 'longitude_hourly', 'id_races', 'grandPrixId', 'circuitId']).agg(average_temp = ('temperature_2m', 'mean'), total_precipitation = ('hourly_precipitation', 'sum'), average_humidity = ('relative_humidity_2m', 'mean'), average_wind_speed = ('wind_speed_10m', 'mean')).reset_index()

races_and_weather_grouped.to_csv(path.join(DATA_DIR, 'f1WeatherData_Grouped.csv'), columns=['short_date', 'id_races', 'grandPrixId', 'circuitId', 'latitude_hourly', 'longitude_hourly', 'average_temp', 'total_precipitation', 'average_humidity', 'average_wind_speed'], sep='\t')#, mode='a', header=False)

# drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
# results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json'))
# races = pd.read_json(path.join(DATA_DIR, 'f1db-races-race_results.json')) 

# active_drivers = active_drivers[active_drivers['activeDriver'] == True]

# races = races[races['year'] >= 2015]
# results =  results[results['year'] >= 2015]
# active_drivers = pd.merge(
#     drivers,
#     results,
#     left_on='id',
#     right_on='driverId',
#     how='inner',
#     suffixes=['drivers', 'results']).drop_duplicates(subset=['driverId'])

# active_drivers.to_csv(path.join(DATA_DIR, 'active_drivers.csv'), columns = ['driverId', 'abbreviation', 'name', 'firstName', 'lastName', 'driverNumber'], sep='\t', index=False)

# # Show all rows where abbreviation and name are duplicated
# duplicates = active_drivers[active_drivers.duplicated(subset=['abbreviation', 'name'], keep=False)]
# print(duplicates)

raceNoEarlierThan = current_year - 10

race_results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json')) 
race_results = race_results[race_results['year'].between(raceNoEarlierThan, current_year-1)]
race_results_grouped = race_results.groupby(['raceId', 'year']).agg(totalFinishers = ('positionNumber', 'count'), totalParticipants = ('positionNumber', 'size')).reset_index()

# Add DNF column and calculate
race_results_grouped['DNF'] = race_results_grouped['totalParticipants'] - race_results_grouped['totalFinishers']
race_results_grouped.to_csv('f1RaceResultsData_Grouped.csv', columns=['raceId', 'year', 'totalFinishers', 'totalParticipants', 'DNF'], sep='\t')  #, index=False)

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



