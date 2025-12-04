import pandas as pd
import numpy as np

fn='data_files/f1ForAnalysis.csv'
print('Loading', fn)
df = pd.read_csv(fn, sep='\t', low_memory=False)
print('rows,cols:', df.shape)

# round nulls
if 'round' in df.columns:
    n_null = int(df['round'].isna().sum())
    print('round nulls:', n_null)
    if n_null > 0:
        print('\nSample rows with missing round:')
        print(df[df['round'].isna()][['raceId','resultsDriverId','short_date']].head(20).to_string(index=False))
else:
    print('round column not present')

# points_leader_gap checks
if 'points_leader_gap' in df.columns:
    s = df['points_leader_gap']
    print('\npoints_leader_gap: min', s.min(), 'max', s.max(), 'mean', s.mean())

    # choose grouping key
    if 'round' in df.columns and df['round'].notna().any():
        grp = df.groupby(['grandPrixYear','round'])
        key_name = "(grandPrixYear, round)"
    elif 'raceId' in df.columns:
        grp = df.groupby(['grandPrixYear','raceId'])
        key_name = "(grandPrixYear, raceId)"
    elif 'short_date' in df.columns:
        grp = df.groupby(['grandPrixYear','short_date'])
        key_name = "(grandPrixYear, short_date)"
    else:
        grp = None
        key_name = None

    if grp is not None:
        bad_groups = []
        missing_zero = []
        for name, g in grp:
            if len(g) == 0:
                continue
            min_gap = g['points_leader_gap'].min()
            if pd.isna(min_gap) or min_gap < 0:
                bad_groups.append((name, min_gap, len(g)))
            if (g['points_leader_gap'] == 0).sum() == 0:
                missing_zero.append(name)
        print('\nUsing grouping key:', key_name)
        print('groups with min gap < 0 or NaN:', len(bad_groups))
        if bad_groups:
            print('examples:', bad_groups[:5])
        print('groups with no zero-gap leader:', len(missing_zero))
        if missing_zero:
            print('examples (first 5):', missing_zero[:5])
else:
    print('points_leader_gap not present')
