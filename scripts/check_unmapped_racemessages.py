import pandas as pd
from os import path

DATA_DIR = 'data_files'
FNAME = path.join(DATA_DIR, 'all_race_control_messages.csv')
OUT_SAMPLE = path.join(DATA_DIR, 'unmapped_race_messages_2025_sample.csv')

if not path.exists(FNAME):
    print(f"File not found: {FNAME}")
    raise SystemExit(1)

print(f"Reading {FNAME} ...")
df = pd.read_csv(FNAME, sep='\t')
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

# normalize column names (lowercase, strip)
df.columns = [c.strip() for c in df.columns]

cols = df.columns.tolist()
print('Detected columns:', cols)

# Which identifier columns exist?
has_raceId = 'raceId' in df.columns or 'raceid' in [c.lower() for c in df.columns]
has_gp = 'grandPrixId' in df.columns or any('grand' in c.lower() and 'prix' in c.lower() for c in df.columns)
print('Has raceId-like column:', has_raceId)
print('Has grandPrix-like column:', has_gp)

# Make canonical columns if present
if 'raceId' not in df.columns and 'id' in df.columns:
    df['raceId'] = df['id']

possible_gp_cols = [c for c in df.columns if 'grand' in c.lower() and 'prix' in c.lower()]
if 'grandPrixId' not in df.columns and possible_gp_cols:
    df['grandPrixId'] = df[possible_gp_cols[0]]

# Ensure Year column exists
if 'Year' not in df.columns and 'year' in [c.lower() for c in df.columns]:
    # find actual column name
    yr = [c for c in df.columns if c.lower() == 'year'][0]
    df['Year'] = df[yr]

# Report missing counts by year
if 'Year' in df.columns:
    years = sorted(df['Year'].dropna().unique())
else:
    years = ['unknown']

print('\nMissing identifier counts by year:')
for y in years:
    if y == 'unknown':
        mask_year = pd.Series([True]*len(df))
    else:
        mask_year = df['Year'] == y
    subset = df[mask_year]
    missing_race = subset['raceId'].isna().sum() if 'raceId' in df.columns else len(subset)
    missing_gp = subset['grandPrixId'].isna().sum() if 'grandPrixId' in df.columns else len(subset)
    total = len(subset)
    print(f"Year {y}: total={total}, missing raceId={missing_race}, missing grandPrixId={missing_gp}")

# Focus on 2025 if present
if 'Year' in df.columns and 2025 in df['Year'].values:
    mask2025 = df['Year'] == 2025
    df2025 = df[mask2025]
    missing_mask = pd.Series(False, index=df2025.index)
    if 'raceId' in df2025.columns:
        missing_mask = missing_mask | df2025['raceId'].isna()
    if 'grandPrixId' in df2025.columns:
        missing_mask = missing_mask | df2025['grandPrixId'].isna()
    missing_rows = df2025[missing_mask]
    print(f"\n2025 total rows: {len(df2025)}, rows with missing id/gp: {len(missing_rows)}")
    if len(missing_rows) > 0:
        sample = missing_rows.head(200)
        try:
            sample.to_csv(OUT_SAMPLE, sep='\t', index=False)
            print(f"Wrote sample of unmapped 2025 rows to: {OUT_SAMPLE}")
        except Exception as e:
            print('Could not write sample file:', e)
        print('\nSample rows (first 10):')
        print(sample.head(10).to_string())
else:
    print('\nNo 2025 rows found in the file to inspect.')

print('\nDone.')
