import pandas as pd
import numpy as np

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')
print('loaded rows', len(df))
qual = df.copy()
# simulate prior steps
_, idx = np.unique(qual.columns, return_index=True)
qual = qual.iloc[:, idx]
for col in ['q1_sec','q2_sec','q3_sec']:
    if col in qual.columns:
        print(col,'dtype',qual[col].dtype,'nulls',qual[col].isna().sum())

# compute best_qual_time
if 'q1_sec' in qual.columns:
    qual['best_qual_time'] = qual[['q1_sec','q2_sec','q3_sec']].min(axis=1)
else:
    print('no q1_sec present')

print('computing ranks')
for q in ['q1_sec','q2_sec','q3_sec']:
    if q in qual.columns:
        pos_col = q.lower().replace('_sec','_pos')
        print('group sizes sample:', qual.groupby(['Year','Round'])[q].size().head())
        ranks = qual.groupby(['Year','Round'])[q].transform(lambda s: s.rank(method='min', ascending=True))
        print('ranks len', len(ranks), 'df len', len(qual))
        qual[pos_col]=ranks
print('done')
print(qual[(qual['Year']==2025)&(qual['Round']==14)].head())
