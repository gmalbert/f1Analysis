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

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json')) 

# Initialize Ergast API
ergast = Ergast(result_type='pandas', auto_cast=True)

# Initialize lists to store race control messages
all_race_control_messages = []

# Loop through all seasons and rounds
for i in range(2018, current_year):
    # Get the number of rounds in each season
    season_schedule = ergast.get_race_schedule(season=i)
    total_rounds = len(season_schedule)

    for round_number in range(1, total_rounds + 1):
        # Load the race session
        session = fastf1.get_session(i, round_number, 'R')
        session.load(messages=True)

        # Get race control messages as a DataFrame
        race_control_messages = session.race_control_messages
        if isinstance(race_control_messages, pd.DataFrame):
            # Add metadata to the DataFrame
            race_control_messages['Round'] = round_number
            race_control_messages['Year'] = i
            race_control_messages['Event'] = session.event['EventName']
            all_race_control_messages.append(race_control_messages)

# Combine all race control messages into a single DataFrame
if all_race_control_messages:
    all_race_control_messages_df = pd.concat(all_race_control_messages, ignore_index=True)
    #print(all_race_control_messages_df)

    # Save the combined DataFrame to a CSV file (optional)
    all_race_control_messages_df.to_csv(path.join(DATA_DIR, 'all_race_control_messages.csv'), sep='\t', index=False)

    race_control_messages_with_grandprix = pd.merge(all_race_control_messages_df, races, left_on=['Round', 'Year'], right_on=['round', 'year'], how='inner')
    
    # Filter rows where 'Category' is either 'Flag' or 'SafetyCar'
    race_control_messages_with_grandprix = race_control_messages_with_grandprix[
        race_control_messages_with_grandprix['Category'].isin(['Flag', 'SafetyCar'])
    ]

    # Group and aggregate the filtered DataFrame
    
    # df.rename(columns={'old_name': 'new_name'}, inplace=True)
    race_control_messages_with_grandprix.rename(columns={'id': 'raceId'}, inplace=True)
    race_control_messages_with_grandprix_grouped = race_control_messages_with_grandprix.groupby(
        ['Round', 'Year', 'raceId', 'grandPrixId']
    ).agg(
        SafetyCarStatus=('Status', lambda x: (x == 'DEPLOYED').sum()),
        redFlag=('Flag', lambda x: (x == 'RED').sum()),
        yellowFlag=('Flag', lambda x: ((x == 'DOUBLE YELLOW') | (x == 'YELLOW')).sum())
    ).reset_index()

    race_control_messages_with_grandprix_grouped.to_csv(path.join(DATA_DIR, 'grouped_race_control_messages.csv'), sep='\t', index=False)

else:
    print("No race control messages were found.")



#response_frame = ergast.get_circuits(season=2022, )
#print(response_frame)
#print(response_frame.columns)

#laps = session.laps
#print(laps.head())  # Ensure it's a DataFrame

#drivers = session.drivers

##stints = laps[["Driver", "DriverNumber", "Stint", "Compound", "LapNumber"]]
#stints = stints.groupby(["Driver", "DriverNumber", "Stint", "Compound"])
#stints = stints.count().reset_index()
#stints = stints.rename(columns={"LapNumber": "StintLength"})
#print(stints)

#hungary = fastf1.get_event(2022, "Hungary")
#print(hungary)

##schedule_2022 = fastf1.get_event_schedule(2022)
#print(schedule_2022)

#hungary_schedule = hungary.get_race()
#print(hungary_schedule)