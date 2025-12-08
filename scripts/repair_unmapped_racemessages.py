import pandas as pd
from os import path
import shutil
import datetime

DATA_DIR = 'data_files'
FNAME = path.join(DATA_DIR, 'all_race_control_messages.csv')
BACKUP = path.join(DATA_DIR, f"all_race_control_messages_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

if not path.exists(FNAME):
    print(f"File not found: {FNAME}")
    raise SystemExit(1)

print(f"Backing up {FNAME} -> {BACKUP}")
shutil.copy2(FNAME, BACKUP)

print(f"Reading {FNAME} ...")
df = pd.read_csv(FNAME, sep='\t')
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# helper
def _coalesce(df, target, candidates):
    for c in candidates:
        if c in df.columns:
            if target not in df.columns:
                df[target] = df[c]
            else:
                df[target] = df[target].fillna(df[c])

# Report before
print('Columns before:', df.columns.tolist())

# Coalesce into canonical columns
_coalesce(df, 'raceId', ['raceId', 'id', 'id_x', 'id_y'])
_coalesce(df, 'grandPrixId', ['grandPrixId', 'grandPrixId_x', 'grandPrixId_y', 'grand_prix_id', 'grand_prix'])
_coalesce(df, 'Round', ['Round', 'round', 'round_x', 'round_y'])
_coalesce(df, 'Year', ['Year', 'year', 'year_x', 'year_y'])

# Drop helper suffixed columns
for col in list(df.columns):
    if col.endswith('_x') or col.endswith('_y'):
        # drop common helper suffixed columns
        if any(base in col for base in ['id', 'round', 'year', 'grandPrixId', 'grand_prix']):
            try:
                df.drop(columns=[col], inplace=True)
            except Exception:
                pass

# Ensure types
if 'Year' in df.columns:
    try:
        df['Year'] = df['Year'].astype('Int64')
    except Exception:
        pass

# Report after
print('Columns after:', df.columns.tolist())

# Print counts by year before/after repair for missing
if 'Year' in df.columns:
    years = sorted(df['Year'].dropna().unique())
else:
    years = []

print('\nMissing identifier counts by year after repair:')
for y in years:
    subset = df[df['Year'] == y]
    missing_race = subset['raceId'].isna().sum() if 'raceId' in df.columns else len(subset)
    missing_gp = subset['grandPrixId'].isna().sum() if 'grandPrixId' in df.columns else len(subset)
    total = len(subset)
    print(f"Year {y}: total={total}, missing raceId={missing_race}, missing grandPrixId={missing_gp}")

# Overwrite the file
print(f"Writing repaired file back to {FNAME}")
df.to_csv(FNAME, sep='\t', index=False)
print('Done.')
