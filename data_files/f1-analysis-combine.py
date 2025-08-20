import datetime as dt
import pandas as pd
from os import path
import os

DATA_DIR = '/Users/gmalb/Downloads/f1'

current_year = dt.datetime.now().year
raceNoEarlierThan = current_year - 10

altResults = pd.read_csv(path.join(DATA_DIR, 'f1SessionList.csv'), sep='\t') 
race_results = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t') 

altResults.columns
race_results.columns

#race_results['date'] = pd.to_datetime(circuits_and_races_lat_long['date']).dt.strftime('%Y-%m-%d')

raceWithAltResults = pd.merge(altResults, race_results, on='short_date', how='inner', suffixes=['_alt', '_orig'])
raceWithAltResults.columns
raceWithAltResults = raceWithAltResults.drop(columns=['Unnamed: 0_alt', 'Unnamed: 0'])
raceWithAltResults[['session_key', 'circuit_key', 'circuit_short_name', 'location', 'raceId_results', 'short_date', 'year', 'circuitId', 'grandPrixRaceId']]

raceWithAltResults.to_csv('f1RaceWithAltResults.csv', sep='\t', columns=['session_key', 'circuit_key', 'circuit_short_name', 'location', 'raceId_results', 'short_date', 'year', 'circuitId', 'grandPrixRaceId']).index=False

raceWithAltResults.head(100)

