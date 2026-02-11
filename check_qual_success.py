import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')
print(f'Total rows: {len(df)}')

print(f'\nData coverage by year:')
print(df.groupby('Year').size())

print(f'\nQualifying time columns populated:')
print(f'  q1_sec: {df["q1_sec"].notna().sum()} non-null ({df["q1_sec"].notna().sum()/len(df)*100:.1f}%)')
print(f'  q2_sec: {df["q2_sec"].notna().sum()} non-null ({df["q2_sec"].notna().sum()/len(df)*100:.1f}%)')
print(f'  q3_sec: {df["q3_sec"].notna().sum()} non-null ({df["q3_sec"].notna().sum()/len(df)*100:.1f}%)')
print(f'  best_qual_time: {df["best_qual_time"].notna().sum()} non-null ({df["best_qual_time"].notna().sum()/len(df)*100:.1f}%)')
print(f'  teammate_qual_delta: {df["teammate_qual_delta"].notna().sum()} non-null ({df["teammate_qual_delta"].notna().sum()/len(df)*100:.1f}%)')

print(f'\nSample of populated data (first 5 rows with qualifying times):')
sample = df[df['best_qual_time'].notna()].head(5)
if len(sample) > 0:
    print(sample[['Year', 'Round', 'FullName', 'q1_sec', 'q2_sec', 'q3_sec', 'best_qual_time']].to_string())
else:
    print("No qualifying times found!")

print(f'\nYear range: {df["Year"].min()} - {df["Year"].max()}')
