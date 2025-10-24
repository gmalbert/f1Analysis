import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
from datetime import date, timedelta
import datetime

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

# Initialize Ergast API
ergast = Ergast(result_type='pandas', auto_cast=True)

#session = fastf1.get_session(2025, 7, 'Q')
#session.load()

# Initialize lists to store sprint results
sprint_results_list = []

# Loop through all seasons and rounds
# sprints were introduced in 2021
for i in range(2021, current_year + 1):
    # Get the number of rounds in each season
    season_schedule = ergast.get_race_schedule(season=i)

    # Filter the season schedule to include only past races
    #season_schedule = season_schedule[pd.to_datetime(season_schedule['raceDate']) <= pd.to_datetime(date.today())]
    #print(season_schedule['raceDate'])
    #print(date.today())
    total_rounds = len(season_schedule)

    for round_number in range(1, total_rounds + 1):
        try:
            # Load the sprint session
            sprint = fastf1.get_session(i, round_number, 'S')
            sprint.load()

            # Get sprint results as a DataFrame
            sprint_results = sprint.results
            if isinstance(sprint_results, pd.DataFrame):
                # Add metadata to the DataFrame
                sprint_results['Round'] = round_number
                sprint_results['Year'] = i
                sprint_results['Event'] = sprint.event['EventName']
                sprint_results_list.append(sprint_results)

                

        except Exception as e:
            print(f"Skipping round {round_number} for {i}: {e}")
            continue

# Combine all sprint results into a single DataFrame
if sprint_results_list:
    sprint_results_list_df = pd.concat(sprint_results_list, ignore_index=True)
    sprint_results_list_df['time_sec'] = pd.to_timedelta(sprint_results_list_df['Time']).dt.total_seconds()
    
    #for col in ['Q1', 'Q2', 'Q3']:
    #    if col in qualifying_results.columns:
     #       qualifying_results_list_df[f'{col}_sec'] = pd.to_timedelta(qualifying_results_list_df[col]).dt.total_seconds()
    
    #print(qualifying_results_list_df)

    # Save the combined DataFrame to a CSV file (optional)
    sprint_results_list_df.to_csv(path.join(DATA_DIR, 'all_sprint_races.csv'), sep='\t', index=False)
else:
    print("No sprint results were found.")