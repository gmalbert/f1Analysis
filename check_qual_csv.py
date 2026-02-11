import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')
print(f'Total rows: {len(df)}')
print(f'Total columns: {len(df.columns)}')
print(f'\nAll columns:')
for i, col in enumerate(df.columns.tolist(), 1):
    print(f'{i:2d}. {col}')

print(f'\n2024 sessions: {len(df[df["Year"] == 2024])} rows')

print(f'\nNew lap-level fields (2024 only):')
df_2024 = df[df['Year'] == 2024]
for col in ['total_qualifying_laps', 'best_sector1_sec', 'theoretical_best_lap', 'theoretical_gap', 'primary_compound']:
    if col in df.columns:
        print(f'  {col}: {df_2024[col].notna().sum()} non-null out of {len(df_2024)}')
    else:
        print(f'  {col}: NOT FOUND in CSV')

print(f'\nSample 2024 data (first 3 rows):')
print(df_2024[['Year', 'Round', 'Event', 'FullName', 'best_sector1_sec', 'theoretical_best_lap']].head(3).to_string())
