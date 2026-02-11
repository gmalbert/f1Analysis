import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')
print(f'Shape: {df.shape}')
print(f'\nAll Columns ({len(df.columns)}):')
print(df.columns.tolist())

print('\n' + '='*80)
print('UPPERCASE Q columns check:')
for col in ['Q1', 'Q2', 'Q3']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f'  {col}: {non_null}/{len(df)} non-null values')
        if non_null > 0:
            print(f'    Sample values: {df[col].dropna().head(3).tolist()}')
    else:
        print(f'  {col}: NOT FOUND')

print('\n' + '='*80)
print('Lowercase q_sec columns check:')
for col in ['q1_sec', 'q2_sec', 'q3_sec']:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f'  {col}: {non_null}/{len(df)} non-null values')
        if non_null > 0:
            print(f'    Sample values: {df[col].dropna().head(3).tolist()}')
    else:
        print(f'  {col}: NOT FOUND')

print('\n' + '='*80)
print('First 5 rows sample:')
print(df[['Year', 'Round', 'FullName']].head())
