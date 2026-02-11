"""Check why lap-level columns are NaN after merge"""
import pandas as pd
from os import path

DATA_DIR = 'data_files'

# Load qualifying from CSV
qualifying_csv = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')
print(f"all_qualifying_races.csv: {len(qualifying_csv)} rows")
print(f"Columns: {qualifying_csv.columns.tolist()[:10]}")
print(f"\nSample data:")
print(qualifying_csv[['Year', 'Round', 'raceId', 'driverId', 'best_sector1_sec']].head())

# Check if Year/Round need to be used as merge keys
print(f"\nUnique Year/Round combinations: {len(qualifying_csv[['Year', 'Round']].drop_duplicates())}")
print(f"Unique raceId values: {len(qualifying_csv['raceId'].unique())}")
print(f"Sample raceId values: {qualifying_csv['raceId'].head().tolist()}")
print(f"raceId dtype: {qualifying_csv['raceId'].dtype}")

# Load races JSON to check raceId mapping
races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json'))
print(f"\nf1db-races.json: {len(races)} rows")
print(f"Sample races:")
print(races[['id', 'year', 'round']].head())
print(f"races.id dtype: {races['id'].dtype}")

# Check for matching raceIds
csv_race_ids = set(qualifying_csv['raceId'].dropna().astype(int))
json_race_ids = set(races['id'])
matching_ids = csv_race_ids & json_race_ids
print(f"\nMatching raceIds: {len(matching_ids)} / {len(csv_race_ids)}")
if len(matching_ids) < len(csv_race_ids):
    unmatched = csv_race_ids - json_race_ids
    print(f"Unmatched CSV raceIds (sample):{list(unmatched)[:5]}")
