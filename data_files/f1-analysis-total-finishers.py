import datetime
import pandas as pd
from os import path
import os

DATA_DIR = '/Users/gmalb/Downloads/f1'

current_year = datetime.datetime.now().year
raceNoEarlierThan = current_year - 10

race_results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json')) 

race_results = race_results[race_results['year'].between(raceNoEarlierThan, current_year-1)]

race_results_grouped = race_results.groupby(['raceId', 'year']).agg(totalFinishers = ('positionNumber', 'count'), totalParticipants = ('positionNumber', 'size')).reset_index()

# Add DNF column and calculate
race_results_grouped['DNF'] = race_results_grouped['totalParticipants'] - race_results_grouped['totalFinishers']

print(race_results_grouped.tail(50))

race_results_grouped.to_csv('f1RaceResultsData_Grouped.csv', columns=['raceId', 'year', 'totalFinishers', 'totalParticipants', 'DNF'], sep='\t')  #, index=False)