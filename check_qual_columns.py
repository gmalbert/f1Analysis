import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t', nrows=10)
print('Columns in all_qualifying_races.csv:')
print(df.columns.tolist())
print('\n' + '='*80 + '\n')
print('Columns with ANY data (first 10 rows):')
for col in df.columns:
    non_null = df[col].notna().sum()
    if non_null > 0:
        print(f'  {col}: {non_null}/10 non-null')

print('\n' + '='*80 + '\n')        
print('Columns that are 100% NULL (first 10 rows):')
for col in df.columns:
    non_null = df[col].notna().sum()
    if non_null == 0:
        print(f'  {col}: ALL NULL')
