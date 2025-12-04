import pandas as pd
from pathlib import Path
p = Path('data_files/all_qualifying_races.csv')
if not p.exists():
    print('qualifying CSV not found at', p)
    raise SystemExit(1)
q = pd.read_csv(p, sep='\t', low_memory=False)
print('rows=', len(q))
print('\nColumns:', q.columns.tolist())
if 'Year' not in q.columns or 'Round' not in q.columns:
    print('ERROR: missing Year/Round columns')
    print(q.head())
    raise SystemExit(1)
grp = q.groupby(['Year', 'Round']).size().reset_index(name='count').sort_values(['Year', 'Round'])
print('\nYear/Round counts (first 30):')
print(grp.head(30).to_string(index=False))
print('\nYear/Round counts (last 30):')
print(grp.tail(30).to_string(index=False))
# missing best_qual_time by Year/Round
if 'best_qual_time' in q.columns:
    miss = q.groupby(['Year','Round'])['best_qual_time'].apply(lambda s: s.isnull().sum()).reset_index(name='missing_best')
    merged = grp.merge(miss, on=['Year','Round'])
    print('\nRounds with any missing best_qual_time (first 50 where missing>0):')
    print(merged[merged['missing_best']>0].head(50).to_string(index=False))
# show earliest and latest Year/Round present
first = grp.iloc[0].to_dict()
last = grp.iloc[-1].to_dict()
print('\nEarliest Year/Round present:', first)
print('Latest Year/Round present:', last)
print('\nSample rows for earliest Year/Round:')
y0 = int(first['Year']); r0 = int(first['Round'])
print(q[(q['Year']==y0)&(q['Round']==r0)].head(20).to_string(index=False))
print('\nSample rows for latest Year/Round:')
y1 = int(last['Year']); r1 = int(last['Round'])
print(q[(q['Year']==y1)&(q['Round']==r1)].head(50).to_string(index=False))
# Summary of missing driverId and raceId
for col in ['raceId','driverId','best_qual_time']:
    if col in q.columns:
        print(f"\nMissing counts for {col}: {q[col].isnull().sum()} / {len(q)}")
print('\nDone')
