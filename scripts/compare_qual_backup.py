import pandas as pd
from pathlib import Path
cur = Path('data_files/all_qualifying_races.csv')
bak = Path('data_files/all_qualifying_races.csv.bak')
if not cur.exists():
    print('Current qualifying CSV missing')
    raise SystemExit(1)
if not bak.exists():
    print('Backup qualifying CSV missing')
    raise SystemExit(1)
q = pd.read_csv(cur, sep='\t', low_memory=False)
qb = pd.read_csv(bak, sep='\t', low_memory=False)
print('rows cur=', len(q), 'bak=', len(qb))
for col in ['best_qual_time','raceId','driverId','q1_sec','q2_sec','q3_sec']:
    print(col, 'cur missing=', q.get(col).isnull().sum() if col in q.columns else 'absent', 'bak missing=', qb.get(col).isnull().sum() if col in qb.columns else 'absent')
# compare earliest/ latest
def yr_range(df):
    g=df.groupby(['Year','Round']).size().reset_index(name='count').sort_values(['Year','Round'])
    return g.iloc[0].to_dict(), g.iloc[-1].to_dict()
print('\ncur range:', yr_range(q))
print('\nbak range:', yr_range(qb))
# Compare per-Year missing fraction for best_qual_time
if 'best_qual_time' in q.columns and 'best_qual_time' in qb.columns:
    cur_by_year = q.groupby('Year')['best_qual_time'].apply(lambda s: s.isnull().sum()/len(s)).reset_index(name='frac_missing')
    bak_by_year = qb.groupby('Year')['best_qual_time'].apply(lambda s: s.isnull().sum()/len(s)).reset_index(name='frac_missing')
    print('\nfrac missing by year (current):')
    print(cur_by_year.to_string(index=False))
    print('\nfrac missing by year (backup):')
    print(bak_by_year.to_string(index=False))
else:
    print('best_qual_time missing in one of the files; skipping per-year comparison')
