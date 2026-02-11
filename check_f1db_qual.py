import pandas as pd

df = pd.read_json('data_files/f1db-races-qualifying-results.json')
print(f'Total qualifying records: {len(df)}')
print(f'\nAll columns:')
for col in df.columns:
    print(f'  {col}')

print(f'\nData by year:')
print(df.groupby('year').size())

print(f'\nSample qualifying data:')
print(df[['year', 'round', 'driverId', 'positionNumber', 'q1', 'q2', 'q3', 'time']].head(10))

print(f'\nNull counts for qualifying times:')
print(f'  q1: {df["q1"].notna().sum()}/{len(df)} ({df["q1"].notna().sum()/len(df)*100:.1f}%)')
print(f'  q2: {df["q2"].notna().sum()}/{len(df)} ({df["q2"].notna().sum()/len(df)*100:.1f}%)')  
print(f'  q3: {df["q3"].notna().sum()}/{len(df)} ({df["q3"].notna().sum()/len(df)*100:.1f}%)')
