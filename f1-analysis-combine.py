import datetime as dt
import pandas as pd
from os import path
import os

DATA_DIR = 'data_files/'

current_year = dt.datetime.now().year
raceNoEarlierThan = current_year - 10

altResults = pd.read_csv(path.join(DATA_DIR, 'f1SessionList.csv'), sep='\t') 
race_results = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t') 
race_control_messages = pd.read_csv(path.join(DATA_DIR, 'all_race_control_messages.csv'), sep='\t') 

altResults.columns
race_results.columns

#race_results['date'] = pd.to_datetime(circuits_and_races_lat_long['date']).dt.strftime('%Y-%m-%d')

raceWithAltResults = pd.merge(altResults, race_results, on='short_date', how='inner', suffixes=['_alt', '_orig'])
raceWithAltResults.columns
raceWithAltResults = raceWithAltResults.drop(columns=['Unnamed: 0_alt', 'Unnamed: 0'])
raceWithAltResults[['session_key', 'circuit_key', 'circuit_short_name', 'location', 'raceId_results', 'short_date', 'year', 'circuitId', 'grandPrixRaceId']]

raceWithAltResults.to_csv(path.join(DATA_DIR, 'f1RaceWithAltResults.csv'), sep='\t', columns=['session_key', 'circuit_key', 'circuit_short_name', 'location', 'raceId_results', 'short_date', 'year', 'circuitId', 'grandPrixRaceId'])

raceWithAltResults.head(100)

## combine race ID with race control messages

raceWithControlMessages = pd.merge(race_results, race_control_messages, left_on=['grandPrixYear', 'round'], right_on=['Year', 'Round'], how='inner', suffixes=['_results', '_messages'])

# Create a DataFrame with unique values for the specified columns
unique_race_control_messages = raceWithControlMessages[
    ['Time', 'Category', 'Message', 'Status', 'Flag', 'Scope', 'Sector', 'RacingNumber', 'Lap', 'Round', 'Year', 'raceId_results']
].drop_duplicates()

# Display the unique DataFrame
print(unique_race_control_messages.head(50))

# Save the unique DataFrame to a CSV file (optional)
unique_race_control_messages.to_csv(path.join(DATA_DIR, 'unique_race_control_messages.csv'), sep='\t', index=False)

