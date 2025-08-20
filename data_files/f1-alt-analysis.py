import datetime as dt
import pandas as pd
from os import path
import os
from urllib.request import urlopen
import json

DATA_DIR = '/Users/gmalb/Downloads/f1'

current_year = dt.datetime.now().year
raceNoEarlierThan = current_year - 10
print(raceNoEarlierThan)

url = 'https://api.openf1.org/v1/sessions?date_start>=' + str(raceNoEarlierThan) + '-01-01&date_end<=' + str(current_year-1) + '-12-31&session_name=Race'

print(url)
response = urlopen(url)
data = json.loads(response.read().decode('utf-8'))

sessionList = pd.DataFrame(data)
sessionList['short_date'] = pd.to_datetime(sessionList['date_start']).dt.strftime('%Y-%m-%d')
print(sessionList.head(100))

sessionList_grouped = sessionList.groupby(['session_key', 'year', 'circuit_key', 'circuit_short_name', 'location', 'short_date']).agg({'country_code': 'size'}).reset_index()
#print(sessionList_grouped)
#print(type(sessionList_grouped))

sessionList_grouped.to_csv(path.join(DATA_DIR, 'f1SessionList.csv'), columns=['session_key', 'year', 'circuit_key', 'circuit_short_name', 'location', 'short_date'], sep='\t')

circuitList_grouped = sessionList.groupby(['circuit_key', 'circuit_short_name', 'location']).agg({ 'year': 'size' }).reset_index()
circuitList_grouped.to_csv(path.join(DATA_DIR, 'f1CircuitList.csv'), columns=['circuit_key', 'circuit_short_name', 'location'], sep='\t')