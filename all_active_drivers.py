import pandas as pd
from os import path
import os
import datetime

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
# results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json'))
fp1 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-1-results.json')) 
fp2 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-2-results.json')) 
fp3 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-3-results.json')) 
fp4 = pd.read_json(path.join(DATA_DIR, 'f1db-races-free-practice-4-results.json')) 

all_practices = pd.concat([fp1, fp2, fp3, fp4], ignore_index=True)
# races = pd.read_json(path.join(DATA_DIR, 'f1db-races-race_results.json')) 

# active_drivers = active_drivers[active_drivers['activeDriver'] == True]

# races = races[races['year'] >= 2015]
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

####### modify this file to look at practices because there are lots of missing values
####### may need to concatenate with practices 1, 2, 3, and 4 and then de-dup.