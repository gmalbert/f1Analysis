import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')
print('Total rows', len(df))
print('Years in file:', sorted(df['Year'].unique()))
print('Counts by Round for 2025:')
print(df[df['Year']==2025].groupby('Round').size())
print('\nSample rows for Round 14 (2025):')
print(df[(df['Year']==2025) & (df['Round']==14)].head(20).to_string())
