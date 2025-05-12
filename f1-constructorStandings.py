from matplotlib import pyplot as plt
import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
from datetime import date, timedelta
import datetime

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

constructors = pd.read_json(path.join(DATA_DIR, 'f1db-constructors.json')) 
drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
results = pd.read_csv(path.join(DATA_DIR, 'f1ForAnalysis.csv'), sep='\t')
grandPrix = pd.read_json(path.join(DATA_DIR, 'f1db-grands-prix.json')) 
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

# Initialize Ergast API
ergast = Ergast(result_type='pandas', auto_cast=True)

season_schedule = ergast.get_race_schedule(season=current_year)

# Filter the season schedule to include only past races
season_schedule = season_schedule[pd.to_datetime(season_schedule['raceDate']) < pd.to_datetime(date.today())]

total_rounds = len(season_schedule)
all_sessions = []

all_constructor_standings = []
all_driver_standings = []

for round_number in range(1, total_rounds + 1):
        # Load the race session
    session = fastf1.get_session(current_year, round_number, 'R')
    session.load()

    # Get race control messages as a DataFrame
    constructor_standings = session.results.groupby('TeamName')['Points'].sum()
    constructor_standings = pd.Series(constructor_standings)
    all_constructor_standings.append(constructor_standings)
    
    driver_standings = session.results.groupby('Abbreviation')['Points'].sum()
    driver_standings = pd.Series(driver_standings)
    all_driver_standings.append(driver_standings)
   # print(f"Round {round_number}")
   ## print(f"Length (Constr.): {len(constructor_standings)}")
#    print(f"Length: (Driver): {len(driver_standings)}")

 #   print(constructor_standings)
  #  print(driver_standings)

#######  Add round and year to dataframe for connecting in raceAnalysis.py
#######


# Combine all constructor standings into a single DataFrame
##all_constructor_standings_df = pd.DataFrame(all_constructor_standings)
all_constructor_standings_df = pd.concat(all_constructor_standings, axis=0)#.fillna(0)
all_constructor_standings_df = pd.DataFrame(all_constructor_standings_df)

all_driver_standings_df = pd.concat(all_driver_standings, axis=0)#.fillna(0)
all_driver_standings_df = pd.DataFrame(all_driver_standings_df)

# Group by 'TeamName'/'Abbreviation' and sum the points
all_constructor_standings_df = all_constructor_standings_df.groupby('TeamName').agg(Points=('Points', 'sum')).reset_index()
all_driver_standings_df = all_driver_standings_df.groupby('Abbreviation').agg(Points=('Points', 'sum')).reset_index()

# Sort by highest points
all_constructor_standings_df_sorted = all_constructor_standings_df.sort_values(by='Points', ascending=False).reset_index(drop=True)
all_driver_standings_df_sorted = all_driver_standings_df.sort_values(by='Points', ascending=False).reset_index(drop=True)

# Optionally add rank
all_constructor_standings_df_sorted['constructorRank'] = all_constructor_standings_df_sorted['Points'].rank(method='min', ascending=False).astype(int)
all_driver_standings_df_sorted['driverRank'] = all_driver_standings_df_sorted['Points'].rank(method='min', ascending=False).astype(int)

print(all_constructor_standings_df_sorted)
print(all_driver_standings_df_sorted)

def clean_constructor_names(name):
    # Remove any unwanted characters or spaces
    #name = name.replace(' ', '')
    name = name.replace('Red Bull Racing', 'Red Bull')
    name = name.replace('Haas F1 Team', 'Haas')
    return name

clean_constructor_names(all_constructor_standings_df_sorted['TeamName'])
all_constructor_standings_df_sorted['TeamName'] = all_constructor_standings_df_sorted['TeamName'].apply(clean_constructor_names)

## Limit the drivers to only those who are active in the current season
constructor_standings_with_mapping = pd.merge(constructors, all_constructor_standings_df_sorted, left_on='name', right_on='TeamName', how='right')
active_drivers = pd.merge(results, drivers, left_on='resultsDriverId', right_on='id', how='inner')
active_drivers = active_drivers[active_drivers['activeDriver'] == True]

driver_standings_with_mapping = pd.merge(active_drivers, all_driver_standings_df_sorted, left_on='abbreviation', right_on='Abbreviation', how='inner')
driver_standings_with_mapping = driver_standings_with_mapping[['id', 'name', 'Points', 'driverRank']].drop_duplicates()
driver_standings_with_mapping = driver_standings_with_mapping.rename(columns={'id': 'driverId', 'name': 'driverName', 'Points': 'points', 'driverRank': 'driverRank'})

constructor_standings_with_mapping = constructor_standings_with_mapping.to_csv(path.join(DATA_DIR, 'constructor_standings.csv'), sep='\t', index=False)
driver_standings_with_mapping = driver_standings_with_mapping.to_csv(path.join(DATA_DIR, 'driver_standings.csv'), sep='\t', index=False)
