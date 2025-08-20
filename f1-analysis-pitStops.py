
import datetime as dt
import numpy as np
import pandas as pd
from os import path
import os

DATA_DIR = '/Users/gmalb/Downloads/f1'

current_year = dt.datetime.now().year
raceNoEarlierThan = current_year - 10

pitstops = pd.read_json(path.join(DATA_DIR, 'f1db-races-pit-stops.json')) 

pitstops = pitstops[pitstops['year'].between(raceNoEarlierThan, current_year)]

pitstops_grouped = pitstops.groupby(['raceId', 'driverId', 'constructorId']).agg(numberOfStops = ('stop', 'count'), averageStopTimeMillis = ('timeMillis', 'mean'), totalStopTimeMillis = ('timeMillis', 'sum')).reset_index()

pitstops_grouped['averageStopTime'] = pitstops_grouped['averageStopTimeMillis'] / 1000
pitstops_grouped['totalStopTime'] = pitstops_grouped['totalStopTimeMillis'] / 1000

pitstops_grouped.to_csv('f1PitStopsData_Grouped.csv', columns=['raceId', 'driverId', 'constructorId', 'numberOfStops', 'averageStopTime', 'totalStopTime'], sep='\t')
