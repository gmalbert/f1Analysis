import pandas as pd
from os import path

DATA_DIR = 'data_files/'
qual_path = path.join(DATA_DIR, 'all_qualifying_races.csv')
print('Loading', qual_path)
df = pd.read_csv(qual_path, sep='\t')
print('rows before:', len(df))

races = pd.read_json(path.join(DATA_DIR, 'f1db-races.json'))
races = races.rename(columns={'year': 'Year', 'round': 'Round', 'id': 'raceId'})
races['Year'] = races['Year'].astype(int)
races['Round'] = races['Round'].astype(int)

# fill raceId where missing
if 'raceId' not in df.columns:
    df['raceId'] = pd.NA

before_missing = df['raceId'].isna().sum()
print('raceId missing before:', before_missing)

df = df.merge(races[['Year','Round','raceId']], on=['Year','Round'], how='left', suffixes=('','_r'))
# if raceId column existed, prefer existing non-null values
if 'raceId_r' in df.columns:
    df['raceId'] = df['raceId'].fillna(df['raceId_r'])
    df.drop(columns=['raceId_r'], inplace=True)

after_missing = df['raceId'].isna().sum()
print('raceId missing after:', after_missing)

# fill driverId via active_drivers by Abbreviation if driverId missing
active = pd.read_csv(path.join(DATA_DIR, 'active_drivers.csv'), sep='\t')
# mapping abbreviation -> driverId
abbr_map = dict(zip(active['abbreviation'], active['driverId']))

if 'Abbreviation' in df.columns:
    df['driverId'] = df.get('driverId')
    missing_driver_before = df['driverId'].isna().sum()
    print('driverId missing before:', missing_driver_before)
    df['driverId'] = df.apply(lambda r: abbr_map.get(r['Abbreviation']) if pd.isna(r.get('driverId')) else r.get('driverId'), axis=1)
    missing_driver_after = df['driverId'].isna().sum()
    print('driverId missing after:', missing_driver_after)
else:
    print('No Abbreviation column to map driverId')

# Save repaired CSV backup and overwrite
backup = qual_path + '.bak'
print('Writing backup to', backup)
df.to_csv(backup, sep='\t', index=False)
print('Writing repaired CSV to', qual_path)
df.to_csv(qual_path, sep='\t', index=False)
print('Done')
