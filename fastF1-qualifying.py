import fastf1
import pandas as pd
from fastf1.ergast import Ergast
from os import path
import os
from datetime import date, timedelta
import datetime
import numpy as np

DATA_DIR = 'data_files/'

current_year = datetime.datetime.now().year

# Enable FastF1 caching
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

# Initialize Ergast API
ergast = Ergast(result_type='pandas', auto_cast=True)

drivers = pd.read_json(path.join(DATA_DIR, 'f1db-drivers.json'))
active_drivers = pd.read_csv(path.join(DATA_DIR, 'active_drivers.csv'), sep='\t')
constructors = pd.read_json(path.join(DATA_DIR, 'f1db-constructors.json')) 

# Path to the qualifying results CSV
csv_path = path.join(DATA_DIR, 'all_qualifying_races.csv')

def add_teammate_delta(df, group_cols, value_col, new_col):
    """
    Adds a column with the difference between each driver's value_col and their teammate's for each group.
    Only works for teams with exactly 2 drivers per group.
    """
    def teammate_diff(x):
        if len(x) != 2:
            return [None] * len(x)
        return [x.iloc[0] - x.iloc[1], x.iloc[1] - x.iloc[0]]
    df[new_col] = (
        df.groupby(group_cols)[value_col]
        .transform(lambda x: teammate_diff(x) if len(x) == 2 else [None]*len(x))
    )
    return df


# Load existing qualifying results if they exist
if os.path.exists(csv_path):
    processed_df = pd.read_csv(csv_path, sep='\t')
    # Build a set of already processed session keys (Year, Round)
    processed_sessions = set(
        zip(
            processed_df['Year'],
            processed_df['Round']
        )
    )
else:
    processed_df = pd.DataFrame()
    processed_sessions = set()

# Initialize list to store new qualifying results
qualifying_results_list = []

today = datetime.datetime.now().date()

# Load races reference table (if not already loaded)
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json'))

# Make sure column names match for merging
races = races.rename(columns={'year': 'Year', 'round': 'Round', 'id': 'raceId'})

# Loop through all seasons and rounds
for i in range(current_year, current_year + 1):
    season_schedule = ergast.get_race_schedule(season=i)
    total_rounds = len(season_schedule)

    for round_number in range(1, total_rounds + 1):
        session_key = (i, round_number)
        if session_key in processed_sessions:
            print(f"Skipping session: {session_key} (already processed)")
            continue

        # Get the qualifying date for this round
        try:
            race_row = season_schedule[season_schedule['round'].astype(int) == round_number]
            if race_row.empty:
                continue
            qual_date_col = None
            # Try to find a qualifying date column
            for col in race_row.columns:
                if 'qualifying' in col.lower() and 'date' in col.lower():
                    qual_date_col = col
                    break
            if qual_date_col is None:
                # Fallback to race date if qualifying date not present
                qual_date_col = [col for col in race_row.columns if 'date' in col.lower()][0]
            qual_date = pd.to_datetime(race_row.iloc[0][qual_date_col]).date()
        except Exception as e:
            print(f"Could not determine qualifying date for {i} round {round_number}: {e}")
            continue

        # Only process sessions less than 3 days in the future
        if (qual_date - today).days > 0:
            print(f"Skipping session: {session_key} (qualifying in the future)")
            continue

        try:
            # Load the qualifying session
            qualifying = fastf1.get_session(i, round_number, 'Q')
            qualifying.load()

            # Get qualifying results as a DataFrame
            qualifying_results = qualifying.results
            # Add driverId as a hyphenated string for each row
            print(qualifying_results.head())
            if isinstance(qualifying_results, pd.DataFrame):
                #qualifying_results['driverId'] = qualifying_results['Driver'].apply(lambda d: d.driverId if hasattr(d, 'driverId') else None)
                qualifying_results['Round'] = round_number
                qualifying_results['Year'] = i
                qualifying_results['Event'] = qualifying.event['EventName']
                qualifying_results_list.append(qualifying_results)
        except Exception as e:
            print(f"Skipping round {round_number} for {i}: {e}")
            continue

# print(races[(races['Year'] == 2025) & (races['officialName'].str.contains('British|Belgian', case=False, na=False))])

# Combine all new qualifying results into a single DataFrame
if qualifying_results_list:
    new_results_df = pd.concat(qualifying_results_list, ignore_index=True)
    for col in ['Q1', 'Q2', 'Q3']:
        if col in new_results_df.columns:
            new_results_df[f'{col}_sec'] = pd.to_timedelta(new_results_df[col]).dt.total_seconds()
    # Combine with existing data and drop duplicates
    if not processed_df.empty:
        combined_df = pd.concat([processed_df, new_results_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Year', 'Round', 'DriverNumber'], keep='last')
    else:
        combined_df = new_results_df

    combined_df['Year'] = combined_df['Year'].astype(int)
    combined_df['Round'] = combined_df['Round'].astype(int)
    races['Year'] = races['Year'].astype(int)
    races['Round'] = races['Round'].astype(int)

    # # Remove records for round 12 or 13 in 2025
    # combined_df = combined_df[~((combined_df['Year'] == 2025) & (combined_df['Round'].isin([12, 13])))]

    # # Save the combined DataFrame to a CSV file
    # combined_df.to_csv(csv_path, sep='\t', index=False)

    # Merge raceId into your qualifying results DataFrame
    # (Assuming new_results_df is your qualifying results DataFrame)
    results_with_raceId = pd.merge(
        combined_df,
        races[['Year', 'Round', 'raceId']],
        left_on=['Year', 'Round'],
        right_on=['Year', 'Round'],
        how='left'
    )
    #print(results_with_raceId)


    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(csv_path, sep='\t', index=False)
else:
    print("No new qualifying results were found.")

qualifying = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')



# qualifying_with_driverId = qualifying_with_driverId.rename(columns={'id_races': 'raceId'})

# qualifying_with_driverId = pd.merge(
#     qualifying,
#     drivers[['abbreviation', 'id']],
#     left_on='Abbreviation',
#     right_on='abbreviation',
#     how='left'
# )
# dupes = qualifying_with_driverId.duplicated(subset=['Abbreviation'], keep=False)
# print(qualifying_with_driverId[dupes])

# qualifying_with_driverId[dupes].to_csv(path.join(DATA_DIR, 'dupes.csv'), sep='\t', index=False)


# drivers['mergeKey'] = (
#     drivers['firstName'].str[0].str.upper() + ' ' + drivers['lastName'].str.upper().str.strip()
# )

# print(drivers['mergeKey'].head(10))



qualifying_with_driverId = pd.merge(
    qualifying,
    active_drivers,
    left_on='Abbreviation', 
    right_on='abbreviation',
    how='left',
    suffixes=('', '_drivers')
).drop_duplicates(subset=['Year', 'Round', 'DriverNumber'])

print(qualifying_with_driverId.columns.to_list())

# qualifying_with_driverId['TeamId'] = qualifying_with_driverId['TeamId'].str.replace('_', '-', regex=False)

qualifying_with_driverId = pd.merge(
    qualifying_with_driverId,
    constructors[['id', 'name']],
    left_on='constructorId',
    right_on='id',
    how='left',
    suffixes=('', '_constructor')
).drop_duplicates(subset=['Year', 'Round', 'DriverNumber'])

print(qualifying_with_driverId.columns.to_list())
# print(qualifying_with_driverId[['TeamId', 'name_constructor']].head(50))

# Remove any existing 'constructorName' column to avoid DataFrame assignment issues
# Remove ALL columns named 'constructorName' (even if there are duplicates)
# while 'constructorName' in qualifying_with_driverId.columns:
#     qualifying_with_driverId = qualifying_with_driverId.drop(columns=['constructorName'])

# Also, if there are duplicate columns in general, keep only the first occurrence of each
_, idx = np.unique(qualifying_with_driverId.columns, return_index=True)
qualifying_with_driverId = qualifying_with_driverId.iloc[:, idx]

# Now assign 'constructorName' as a 1D Series from 'name_constructor'
# qualifying_with_driverId['constructorName'] = qualifying_with_driverId['name_constructor'].fillna(qualifying_with_driverId['TeamId'])

# print("Post merge:")
# print(qualifying_with_driverId[['TeamId', 'constructorName', 'name_constructor']].head(50))

# After merge, the constructor name column is likely 'name'
# if 'name' in qualifying_with_driverId.columns:
#     qualifying_with_driverId = qualifying_with_driverId.rename(columns={'name': 'constructorName'})

# # Remove any other columns that start with 'constructorName' except the main one
# cols = [col for col in qualifying_with_driverId.columns if col.startswith('constructorName')]
# if len(cols) > 1:
#     qualifying_with_driverId = qualifying_with_driverId.drop(columns=[c for c in cols if c != 'constructorName'])

# print(qualifying_with_driverId.columns)
# print(type(qualifying_with_driverId['constructorName']))
# print(qualifying_with_driverId['constructorName'].head())

# # Now check
# assert qualifying_with_driverId['constructorName'].ndim == 1

# Example: Use Q3_sec if available, else Q2_sec or Q1_sec
if 'Q3_sec' in qualifying_with_driverId.columns and not qualifying_with_driverId['Q3_sec'].isnull().all():
    qual_col = 'Q3_sec'
elif 'Q2_sec' in qualifying_with_driverId.columns and not qualifying_with_driverId['Q2_sec'].isnull().all():
    qual_col = 'Q2_sec'
else:
    qual_col = 'Q1_sec'

# qualifying_with_driverId = add_teammate_delta(
#     qualifying_with_driverId,
#     ['Year', 'Round', 'constructorName'],
#     qual_col,
#     'teammate_qual_delta'
# )
qualifying_with_driverId.rename(columns={'Q1_sec': 'q1_sec', 'Q2_sec': 'q2_sec', 'Q3_sec': 'q3_sec'}, inplace=True)
# Create a column for each driver's best qualifying time (lowest non-null Q1/Q2/Q3)
qualifying_with_driverId['best_qual_time'] = qualifying_with_driverId[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)

qualifying_with_driverId = add_teammate_delta(
    qualifying_with_driverId,
    ['Year', 'Round', 'constructorName'],
    'best_qual_time',
    'teammate_qual_delta'
)

########### need to redo this for all of the qualifying results

# qualifying_with_driverId = pd.merge(
#     qualifying,
#     drivers,
#     left_on=['Abbreviation', 'LastName'],
#     right_on=['abbreviation', 'lastName'],
#     how='left'
# ).drop_duplicates(subset=['Year', 'Round', 'DriverNumber'])

# # print(qualifying_with_driverId.columns)

# # print(qualifying_with_driverId.head(50))
# # qualifying_with_driverId['id']


# # print(qualifying_with_driverId[qualifying_with_driverId['id'].isnull()])

# manual_id_map = {
#     ('Sergio', 'Perez'): 'sergio-perez',      
#     ('Nico', 'Hulkenberg'): 'nico-hulkenberg',
#     ('Carlos', 'Sainz'): 'carlos-sainz-jr',
#     ('Max', 'Verstappen'): 'max-verstappen',

# }

# def fill_manual_id(row):
#     if pd.isnull(row['id']):
#         key = (row['FirstName'], row['LastName'])
#         return manual_id_map.get(key, None)
#     return row['id']

# qualifying_with_driverId['id'] = qualifying_with_driverId.apply(fill_manual_id, axis=1)
#qualifying_with_driverId.drop(columns='DriverId', inplace=True)


qualifying_with_driverId.rename(columns={'driverId_drivers': 'driverId'}, inplace=True)

qualifying_with_driverId = pd.merge(
    qualifying,
    races[['Year', 'Round', 'raceId']],
    left_on=['Year', 'Round'],
    right_on=['Year', 'Round'],
    how='left',
    suffixes=('', '_races')
)

# Calculate position (rank) for Q1, Q2, Q3 times (lower time = better rank)
for q in ['q1_sec', 'q2_sec', 'q3_sec']:
    if q in qualifying_with_driverId.columns:
        pos_col = q.lower().replace('_sec', '_pos')
        qualifying_with_driverId[pos_col] = (
            qualifying_with_driverId
            .groupby(['Year', 'Round'])[q]
            .rank(method='min', ascending=True)
        )

# Optionally, rename for consistency with your output columns
qualifying_with_driverId.rename(columns={
    # 'Q1_sec': 'q1_sec', 'Q2_sec': 'q2_sec', 'Q3_sec': 'q3_sec',
    'Q1_pos': 'q1_pos', 'Q2_pos': 'q2_pos', 'Q3_pos': 'q3_pos'
}, inplace=True)

# qualifying_with_driverId.rename(columns={'Q1_sec': 'q1_sec', 'Q2_sec': 'q2_sec', 'Q3_sec': 'q3_sec'}, inplace=True)

qualifying_with_driverId.to_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t', index=False, columns=[
    'Year', 'Round', 'Event', 'raceId', 'DriverNumber', 'Abbreviation', 'FullName', 
    'LastName', 'driverId', 'constructorName', 'q1_sec', 'q1_pos', 'q2_sec', 'q2_pos', 'q3_sec', 'q3_pos', #'q1', 'q2', 'q3',
    'best_qual_time', 'teammate_qual_delta', #'Position', 'Points',
    'totalChampionshipPoints', 'totalChampionshipWins', 'totalFastestLaps', 'totalGrandSlams', 'totalPodiums', 
    'totalPoints',  'totalPolePositions',  'totalRaceEntries',  'totalRaceLaps', 
      'totalRaceStarts',  'totalRaceWins', 'bestChampionshipPosition',  'bestRaceResult', 'bestStartingGridPosition',  'constructorId', 
])

print("Qualifying results processed and saved to all_qualifying_races.csv")

# print(
#     qualifying_with_driverId.groupby(['Year', 'Round', 'constructorName'])
#     .size()
#     .value_counts()
# )