import pandas as pd
from os import path

DATA_DIR = 'data_files'

# Load the CSV and rename columns
qual = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')
print(f'Loaded {len(qual)} qualifying rows')

# Apply column mapping
column_mapping = {
    'DriverId': 'driverId',
    'TeamId': 'constructorId',
    'TeamName': 'constructorName'
}
qual_renamed = qual.rename(columns=column_mapping)
print(f'\nAfter rename:')
print(f'  - driverId exists: {"driverId" in qual_renamed.columns}')
print(f'  - constructorId exists: {"constructorId" in qual_renamed.columns}')

# Load active_drivers
active_drivers = pd.read_csv(path.join(DATA_DIR, 'active_drivers.csv'), sep='\t')
print(f'\nLoaded {len(active_drivers)} active drivers')
print(f'Active drivers columns: {active_drivers.columns.tolist()[:5]}')

# Try the merge
merged = pd.merge(
    qual_renamed,
    active_drivers,
    on='driverId',
    how='left',
    suffixes=('', '_drivers')
)

print(f'\nMerged result:')
print(f'  - Total rows: {len(merged)}')
print(f'  - Rows with driver match: {merged["driverId"].notna().sum()}')
print(f'  - Unique drivers: {merged["driverId"].nunique()}')
print(f'\nSample merged data:')
print(merged[['Year', 'Round', 'FullName', 'driverId', 'Q1', 'Q2', 'Q3']].head(3))
