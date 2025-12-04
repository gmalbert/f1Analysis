import os
import pandas as pd
# Prefer restored artifact if present, otherwise fall back to the canonical qualifying CSV
p_restored = 'data_files/all_qualifying_races.restored.csv'
p_main = 'data_files/all_qualifying_races.csv'
if os.path.exists(p_restored):
    p = p_restored
elif os.path.exists(p_main):
    p = p_main
else:
    raise FileNotFoundError(f'Neither {p_restored} nor {p_main} found')

print(f'Loading qualifying file: {p}')
df = pd.read_csv(p, sep='\t', low_memory=False)
print('rows',len(df))
for col in ['best_qual_time','driverId','q1_sec','q2_sec','q3_sec']:
    print(col,'missing=',df.get(col).isnull().sum() if col in df.columns else 'absent')
print('\nfrac missing by year:')
if 'best_qual_time' in df.columns:
    print(df.groupby('Year')['best_qual_time'].apply(lambda s: s.isnull().sum()/len(s)).reset_index().to_string(index=False))
