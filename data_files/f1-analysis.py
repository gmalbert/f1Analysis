import datetime as dt
import pandas as pd
from os import path
import os

DATA_DIR = '/Users/gmalb/Downloads/f1'

current_year = dt.datetime.now().year
raceNoEarlierThan = current_year - 10

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


races_and_grandPrix = pd.merge(races, grandPrix, left_on='grandPrixId', right_on='id', how='inner', suffixes=['_races', '_grandPrix'])
races_and_grandPrix.rename(columns={'id_races': 'raceIdFromGrandPrix', 'id_grandPrix': 'grandPrixRaceId', 'fullName': 'grandPrixName', 'laps': 'grandPrixLaps', 'year': 'grandPrixYear'}, inplace=True)

results_and_drivers = pd.merge(race_results, drivers, left_on='driverId', right_on='id', how='inner', suffixes=['_results', '_drivers'])
results_and_drivers.rename(columns={'year': 'resultsYear', 'driverId': 'resultsDriverId', 'qualificationPositionNumber': 'resultsQualificationPositionNumber', 'positionNumber': 'resultsFinalPositionNumber', 'name': 'resultsDriverName', 'gridPositionNumber': 'resultsStartingGridPositionNumber', 'reasonRetired': 'resultsReasonRetired'}, inplace=True)
results_and_drivers = results_and_drivers[results_and_drivers['resultsYear'] >= raceNoEarlierThan]

results_and_drivers_and_constructors = pd.merge(results_and_drivers, constructors, left_on='constructorId', right_on='id', how='inner', suffixes=['_results', '_constructors'])
results_and_drivers_and_constructors.rename(columns={'name': 'constructorName', 'totalRaceStarts_constructors': 'constructorTotalRaceStarts', 'totalRaceEntries_constructors': 'constructorTotalRaceEntries', 'totalRaceWins_constructors': 'constructorTotalRaceWins', 
                                                     'total1And2Finishes': 'constructorTotal1And2Finishes', 
                                                     'totalPodiumRaces': 'constructorTotalPodiumRaces', 'totalPolePositions_constructors': 'constructorTotalPolePositions', 'totalFastestLaps_constructors': 'constructorTotalFastestLaps'}, inplace=True)

results_and_drivers_and_constructors_and_grandprix = pd.merge(results_and_drivers_and_constructors, races_and_grandPrix, left_on='raceId', right_on='raceIdFromGrandPrix', how='inner', suffixes=['_results', '_grandprix'])

results_and_drivers_and_constructors_and_grandprix_and_qualifying = pd.merge(results_and_drivers_and_constructors_and_grandprix, qualifying, left_on=['raceIdFromGrandPrix', 'resultsDriverId'], right_on=['raceId', 'driverId'], how='inner', suffixes=['_results', '_qualifying']) 

results_and_drivers_and_constructors_and_grandprix_and_qualifying[['grandPrixName', 'raceId_results', 'resultsDriverId',  'q1', 'q2', 'q3', 'resultsYear', 'constructorName','resultsDriverId', 'resultsDriverName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
                                      'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 
                                      'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'grandPrixYear', 'raceIdFromGrandPrix', 'grandPrixRaceId']]


fp1_fp2 = pd.merge(fp1, fp2, on=['raceId', 'driverId'], how='left', suffixes=['_fp1', '_fp2'])
fp1_fp2_fp3 = pd.merge(fp1_fp2, fp3, on=['raceId', 'driverId'], how='left', suffixes=['_fp1_2', '_fp3'])
fp1_fp2_fp3_fp4 = pd.merge(fp1_fp2_fp3, fp4, on=['raceId', 'driverId'], how='left', suffixes=['_fp1_fp2_fp3', '_fp4'])


fp1_fp2_fp3_fp4.rename(columns={'driverId': 'fpDriverId', 'raceId': 'fpRaceId', 'positionNumber_fp1': 'fp1PositionNumber', 'time_fp1': 'fp1Time', 'gap_fp1': 'fp1Gap', 'interval_fp1': 'fp1Interval', 
'positionNumber_fp2': 'fp2PositionNumber', 'time_fp2': 'fp2Time', 'gap_fp2': 'fp2Gap', 'interval_fp2': 'fp2Interval', 
'positionNumber_fp1_fp2_fp3': 'fp3PositionNumber', 'time_fp1_fp2_fp3': 'fp3Time', 'gap_fp1_fp2_fp3': 'fp3Gap', 'interval_fp1_fp2_fp3': 'fp3Interval', 
'positionNumber_fp4': 'fp4PositionNumber', 'time_fp4': 'fp4Time', 'gap_fp4': 'fp4Gap', 'interval_fp4': 'fp4Interval'}, inplace=True)


results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices = pd.merge(results_and_drivers_and_constructors_and_grandprix_and_qualifying, fp1_fp2_fp3_fp4, left_on=['raceId_results', 'resultsDriverId'], right_on=['fpRaceId','fpDriverId' ], how='left', suffixes=['_results', '_practices']) 

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['short_date'] = pd.to_datetime(results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['date']).dt.strftime('%Y-%m-%d')

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsPodium'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=3)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsTop5'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=5)
results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsTop10'] = (results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices['resultsFinalPositionNumber'] <=10)

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices[['grandPrixName', 'resultsDriverId', 'raceId_results', 'resultsFinalPositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'fp1PositionNumber', 'fp1Time', 'fp1Gap', 
                                        'fp1Interval', 'positionsGained', 'q1', 'q2', 'q3', 'resultsYear', 'constructorName','resultsDriverId', 'resultsDriverName',  'resultsStartingGridPositionNumber', 
                                      'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 
                                      'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'grandPrixYear', 'raceIdFromGrandPrix', 'grandPrixRaceId', 'short_date']]

results_and_drivers_and_constructors_and_grandprix_and_qualifying_and_practices.to_csv('f1ForAnalysis.csv', columns=['grandPrixYear', 'grandPrixName', 'raceId_results', 'circuitId', 'grandPrixRaceId', 'resultsDriverName', 'q1', 'q2', 'q3', 
                                        'fp1Time', 'fp1Gap', 'fp1Interval', 'fp1PositionNumber', 'fp2Time', 'fp2Gap', 'fp2Interval', 'fp2PositionNumber', 'fp3Time', 'fp3Gap', 'fp3Interval', 'fp3PositionNumber','fp4Time', 'fp4Gap', 'fp4Interval', 
                                        'fp4PositionNumber', 'resultsPodium', 'resultsTop5', 'resultsTop10', 'resultsYear', 'constructorName',  'resultsStartingGridPositionNumber', 'resultsFinalPositionNumber', 
                                      'positionsGained', 'resultsReasonRetired', 'constructorTotalRaceEntries', 'constructorTotalRaceStarts', 'constructorTotalRaceWins', 'constructorTotal1And2Finishes', 'constructorTotalPodiumRaces', 
                                      'constructorTotalPolePositions', 'constructorTotalFastestLaps', 'grandPrixLaps', 'short_date' ], sep='\t')

