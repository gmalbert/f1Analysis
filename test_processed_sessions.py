import pandas as pd
from os import path

DATA_DIR = 'data_files'
csv_path = path.join(DATA_DIR, 'all_qualifying_races.csv')

# Load processed_df
processed_df = pd.read_csv(csv_path, sep='\t')
print(f'Loaded {len(processed_df)} rows')

# Build has_time_mask with both uppercase and lowercase column checks
has_time_mask = pd.Series([False] * len(processed_df), index=processed_df.index)
for col in ['q1_sec', 'q2_sec', 'q3_sec', 'best_qual_time', 'Q1_sec', 'Q2_sec', 'Q3_sec']:
    if col in processed_df.columns:
        print(f'  - Found column: {col}, non-null count: {processed_df[col].notna().sum()}')
        has_time_mask |= processed_df[col].notna()
    else:
        print(f'  - Missing column: {col}')

processed_sessions = set(zip(processed_df.loc[has_time_mask, 'Year'], processed_df.loc[has_time_mask, 'Round']))
print(f'\nProcessed sessions set: {len(processed_sessions)} unique (Year, Round) pairs')
print(f'Sample: {list(processed_sessions)[:5]}')
