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
# Exclude 'jos-verstappen' as he is not Max Verstappen
drivers = drivers[drivers['id'] != 'jos-verstappen']

active_drivers = pd.read_csv(path.join(DATA_DIR, 'active_drivers.csv'), sep='\t')
constructors = pd.read_json(path.join(DATA_DIR, 'f1db-constructors.json')) 
race_results = pd.read_json(path.join(DATA_DIR, 'f1db-races-race-results.json'))

# Path to the qualifying results CSV
csv_path = path.join(DATA_DIR, 'all_qualifying_races.csv')

def add_teammate_delta(df, group_cols, value_col, new_col):
    """
    Adds a column with the difference between each driver's value_col and their teammate's for each group.
    Computes, for each row, the difference between the driver's value and the mean of their teammates'
    values within the same group (group_cols). Leaves NaN when there is no teammate data.
    """
    # Create a canonical numeric series for the value to use
    vals = pd.to_numeric(df[value_col], errors='coerce') if value_col in df.columns else pd.Series([np.nan]*len(df), index=df.index)

    # per-group count of non-null values and sum
    team_count = df.groupby(group_cols)[vals.name if hasattr(vals, 'name') else value_col].transform(lambda s: pd.to_numeric(s, errors='coerce').notna().sum())
    team_sum = df.groupby(group_cols)[vals.name if hasattr(vals, 'name') else value_col].transform(lambda s: pd.to_numeric(s, errors='coerce').fillna(0).sum())

    # compute teammates' mean excluding self: (team_sum - my_val) / (team_count - 1)
    my_val = vals
    other_mean = pd.Series(index=df.index, dtype='float')
    valid_mask = (team_count > 1) & my_val.notna()
    other_mean.loc[valid_mask] = (team_sum.loc[valid_mask] - my_val.loc[valid_mask]) / (team_count.loc[valid_mask] - 1)

    # teammate delta = my_val - other_mean
    teammate_delta = pd.Series(index=df.index, dtype='float')
    teammate_delta.loc[valid_mask] = my_val.loc[valid_mask] - other_mean.loc[valid_mask]

    df[new_col] = teammate_delta
    return df


# Load existing qualifying results if they exist
if os.path.exists(csv_path):
    processed_df = pd.read_csv(csv_path, sep='\t')
    # Build a set of already processed session keys (Year, Round)
    # Only consider a session processed if it already has qualifying times (avoid placeholders)
    def session_has_times(row):
        for col in ['Q1', 'Q1_sec', 'Q1_sec', 'Q2_sec', 'Q3_sec', 'best_qual_time', 'q1_sec', 'q2_sec', 'q3_sec']:
            if col in processed_df.columns and not processed_df[col].isna().all():
                # we'll detect per-row below
                break
    # Build processed_sessions by selecting rows where at least one qualifying time exists
    if not processed_df.empty:
        has_time_mask = (
            processed_df.get('q1_sec').notna().fillna(False)
            | processed_df.get('q2_sec').notna().fillna(False)
            | processed_df.get('q3_sec').notna().fillna(False)
            | processed_df.get('best_qual_time').notna().fillna(False)
        )
        processed_sessions = set(zip(processed_df.loc[has_time_mask, 'Year'], processed_df.loc[has_time_mask, 'Round']))
    else:
        processed_sessions = set()
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

        # Only process sessions more than 3 days in the future (allow upcoming sessions within 3 days)
        days_until = (qual_date - today).days
        print(f"Session {session_key} qualifying date: {qual_date} (days until: {days_until})")
        if days_until > 3:
            print(f"Skipping session: {session_key} (qualifying more than 3 days in the future)")
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
                qualifying_results['Round'] = round_number
                qualifying_results['Year'] = i
                qualifying_results['Event'] = qualifying.event['EventName']
                qualifying_results_list.append(qualifying_results)
        except Exception as e:
            print(f"Skipping round {round_number} for {i}: {e}")
            continue

# Combine all new qualifying results into a single DataFrame
if qualifying_results_list:
    new_results_df = pd.concat(qualifying_results_list, ignore_index=True)
    for col in ['Q1', 'Q2', 'Q3']:
        if col in new_results_df.columns:
            new_results_df[f'{col}_sec'] = pd.to_timedelta(new_results_df[col]).dt.total_seconds()
    # Combine with existing data and drop duplicates.
    # Prefer rows with non-null qualifying times and driverId when deduplicating
    if not processed_df.empty:
        combined_df = pd.concat([processed_df, new_results_df], ignore_index=True)
        # Group by the natural key and pick the first non-null value for each column when available.
        group_keys = ['Year', 'Round', 'DriverNumber']

        def _pick_first_non_null(g):
            # For each column in the group, prefer the first non-null value, otherwise fall back to the last value
            out = {}
            for c in g.columns:
                s = g[c]
                non_null = s.dropna()
                if not non_null.empty:
                    out[c] = non_null.iloc[0]
                else:
                    out[c] = s.iloc[-1]
            return pd.Series(out)

        # Use include_groups=False to avoid future pandas behavior where grouping columns
        # are excluded from the operation by default. This silences FutureWarning.
        combined_df = combined_df.groupby(group_keys, as_index=False).apply(_pick_first_non_null, include_groups=False)
        # groupby+apply produces a hierarchical index; reset it to a clean numeric index
        combined_df = combined_df.reset_index(drop=True)
    else:
        combined_df = new_results_df

    combined_df['Year'] = combined_df['Year'].astype(int)
    combined_df['Round'] = combined_df['Round'].astype(int)
    races['Year'] = races['Year'].astype(int)
    races['Round'] = races['Round'].astype(int)


    # Save the combined DataFrame to a CSV file
    # Merge raceId into your qualifying results DataFrame so new rows get raceId filled
    # Use a renamed column on the right to avoid duplicate-column suffix conflicts
    races_sub = races[['Year', 'Round', 'raceId']].rename(columns={'raceId': 'raceId_from_races'})
    results_with_raceId = combined_df.merge(
        races_sub,
        on=['Year', 'Round'],
        how='left'
    )
    # If a raceId column already exists, prefer the existing one and fill any missing values
    if 'raceId' in results_with_raceId.columns and 'raceId_from_races' in results_with_raceId.columns:
        results_with_raceId['raceId'] = results_with_raceId['raceId'].fillna(results_with_raceId['raceId_from_races'])
        results_with_raceId.drop(columns=['raceId_from_races'], inplace=True)
    elif 'raceId_from_races' in results_with_raceId.columns:
        results_with_raceId.rename(columns={'raceId_from_races': 'raceId'}, inplace=True)

    # Ensure we save the enriched DataFrame (with raceId) rather than the raw combined_df
    combined_df = results_with_raceId

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(csv_path, sep='\t', index=False)
else:
    print("No new qualifying results were found.")

qualifying = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')

qualifying['driverId'] = qualifying['driverId'].replace({'jos-verstappen': 'max-verstappen'})

##### why is this merged on abbreviation rather than driverID?

qualifying_with_driverId = pd.merge(
    qualifying,
    active_drivers,
    on='driverId',
    # left_on='Abbreviation', 
    # right_on='abbreviation',
    how='left',
    suffixes=('', '_drivers')
).drop_duplicates(subset=['Year', 'Round', 'DriverNumber'])

# print(qualifying_with_driverId.columns.to_list())
# qualifying_with_driverId['driverId'] = qualifying_with_driverId['driverId'].replace({'jos-verstappen': 'max-verstappen'})

qualifying_with_driverId = pd.merge(
    qualifying_with_driverId,
    constructors[['id', 'name']],
    left_on='constructorId',
    right_on='id',
    how='left',
    suffixes=('', '_constructor')
).drop_duplicates(subset=['Year', 'Round', 'DriverNumber'])

# Also, if there are duplicate columns in general, keep only the first occurrence of each
_, idx = np.unique(qualifying_with_driverId.columns, return_index=True)
qualifying_with_driverId = qualifying_with_driverId.iloc[:, idx]


# Example: Use Q3_sec if available, else Q2_sec or Q1_sec
if 'Q3_sec' in qualifying_with_driverId.columns and not qualifying_with_driverId['Q3_sec'].isnull().all():
    qual_col = 'Q3_sec'
elif 'Q2_sec' in qualifying_with_driverId.columns and not qualifying_with_driverId['Q2_sec'].isnull().all():
    qual_col = 'Q2_sec'
else:
    qual_col = 'Q1_sec'


qualifying_with_driverId.rename(columns={'Q1_sec': 'q1_sec', 'Q2_sec': 'q2_sec', 'Q3_sec': 'q3_sec'}, inplace=True)
# Create a column for each driver's best qualifying time (lowest non-null Q1/Q2/Q3)
qualifying_with_driverId['best_qual_time'] = qualifying_with_driverId[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)

# Create a stable grouping key for teammates. Prefer `constructorId` (stable),
# fall back to `constructorName` when `constructorId` is missing.
qualifying_with_driverId['constructor_group'] = qualifying_with_driverId['constructorId'].fillna(qualifying_with_driverId.get('constructorName'))

# initial teammate delta attempt (may be incomplete if constructorId/name missing)
qualifying_with_driverId = add_teammate_delta(
    qualifying_with_driverId,
    ['Year', 'Round', 'constructor_group'],
    'best_qual_time',
    'teammate_qual_delta'
)


# qualifying_with_driverId.rename(columns={'driverId_drivers': 'driverId'}, inplace=True)

# Merge raceId into the already-enriched qualifying_with_driverId (avoid duplicate-column merge errors)
# Rename the right-hand raceId column so we can safely fill missing values without creating conflicting suffixes
races_sub = races[['Year', 'Round', 'raceId']].rename(columns={'raceId': 'raceId_from_races'})
qualifying_with_driverId = qualifying_with_driverId.merge(
    races_sub,
    left_on=['Year', 'Round'],
    right_on=['Year', 'Round'],
    how='left'
)
# If a raceId column already exists, prefer the existing one and fill any missing values from the merged column
if 'raceId' in qualifying_with_driverId.columns and 'raceId_from_races' in qualifying_with_driverId.columns:
    qualifying_with_driverId['raceId'] = qualifying_with_driverId['raceId'].fillna(qualifying_with_driverId['raceId_from_races'])
    qualifying_with_driverId.drop(columns=['raceId_from_races'], inplace=True)
elif 'raceId_from_races' in qualifying_with_driverId.columns:
    qualifying_with_driverId.rename(columns={'raceId_from_races': 'raceId'}, inplace=True)

# Fill missing constructorId using race_results (if raceId and driverId match present in race_results)
if 'constructorId' in qualifying_with_driverId.columns:
    missing_ctor_mask = qualifying_with_driverId['constructorId'].isnull() & qualifying_with_driverId['raceId'].notna()
    if missing_ctor_mask.any():
        # build mapping from (raceId, driverId) -> constructorId
        rr_map = race_results[['raceId', 'driverId', 'constructorId']].dropna()
        rr_map = rr_map.astype({'raceId': 'int64'})
        rr_key = rr_map.set_index(['raceId', 'driverId'])['constructorId'].to_dict()

        def _infer_constructor(row):
            try:
                key = (int(row['raceId']), row['driverId'])
            except Exception:
                return row['constructorId']
            return rr_key.get(key, row['constructorId'])

        qualifying_with_driverId.loc[missing_ctor_mask, 'constructorId'] = qualifying_with_driverId.loc[missing_ctor_mask].apply(_infer_constructor, axis=1)

# Fill constructorName from constructors mapping when missing
ctor_map = constructors.set_index('id')['name'].to_dict()
if 'constructorName' in qualifying_with_driverId.columns:
    qualifying_with_driverId['constructorName'] = qualifying_with_driverId['constructorName'].fillna(qualifying_with_driverId['constructorId'].map(ctor_map))
else:
    qualifying_with_driverId['constructorName'] = qualifying_with_driverId['constructorId'].map(ctor_map)

# Recompute a stable constructor_group and re-run teammate-delta to catch rows previously missed
qualifying_with_driverId['constructor_group'] = qualifying_with_driverId['constructorId'].fillna(qualifying_with_driverId.get('constructorName'))
qualifying_with_driverId = add_teammate_delta(
    qualifying_with_driverId,
    ['Year', 'Round', 'constructor_group'],
    'best_qual_time',
    'teammate_qual_delta'
)

# Calculate position (rank) for Q1, Q2, Q3 times (lower time = better rank)
# Ensure numeric seconds columns
for _c in ['q1_sec', 'q2_sec', 'q3_sec']:
    if _c in qualifying_with_driverId.columns:
        # Handle possible duplicate column names (which yield a DataFrame slice)
        col = qualifying_with_driverId.loc[:, _c]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        qualifying_with_driverId[_c] = pd.to_numeric(col, errors='coerce')

for q in ['q1_sec', 'q2_sec', 'q3_sec']:
    if q in qualifying_with_driverId.columns:
        pos_col = q.lower().replace('_sec', '_pos')
        # Compute per-group ranks robustly even if selecting q yields a DataFrame
        def _rank_for_group(g):
            col = g[q]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            return col.rank(method='min', ascending=True)

        # Use groupby on the specific column to avoid applying on grouping columns
        # This prevents the FutureWarning about DataFrameGroupBy.apply operating on grouping cols.
        try:
            ranked = qualifying_with_driverId.groupby(['Year', 'Round'])[q].apply(lambda s: s.rank(method='min', ascending=True))
            # Align to original index by dropping the group index levels
            ranked = ranked.reset_index(level=[0, 1], drop=True)
            qualifying_with_driverId[pos_col] = ranked
        except Exception:
            # Fallback: use the earlier robust approach if selecting the column behaves unexpectedly
            # Use include_groups=False to avoid applying on the grouping columns in future pandas versions
            ranked = qualifying_with_driverId.groupby(['Year', 'Round']).apply(_rank_for_group, include_groups=False)
            if isinstance(ranked, pd.DataFrame):
                ranked = ranked.iloc[:, 0]
            ranked = ranked.reset_index(level=[0, 1], drop=True)
            qualifying_with_driverId[pos_col] = ranked

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