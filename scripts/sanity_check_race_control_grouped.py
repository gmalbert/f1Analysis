import pandas as pd
from os import path
f = path.join('data_files', 'race_control_messages_grouped_with_dnf.csv')
if not path.exists(f):
    print('MISSING:', f)
    raise SystemExit(1)

df = pd.read_csv(f, sep='\t', low_memory=False)
print('File:', f)
print('Rows:', len(df))
print('Columns:', len(df), list(df.columns))
print('\nColumn dtypes:')
print(df.dtypes.to_string())
print('\nFirst 8 rows:\n')
print(df.head(8).to_string(index=False))

# Check for missing identifiers
for col in ['raceId','grandPrixId','Round','Year']:
    if col in df.columns:
        print(f"Missing {col}:", df[col].isna().sum())
    else:
        print(f"Column missing: {col}")

# DNF stats
if 'dnf_count' in df.columns:
    print('\nDNF count summary:')
    print(df['dnf_count'].describe())
else:
    print('\nNo dnf_count column present')

# SafetyCar / Flag sanity
for c in ['SafetyCarStatus','redFlag','yellowFlag','doubleYellowFlag']:
    if c in df.columns:
        print(f"{c}: min={df[c].min()}, max={df[c].max()}, missing={df[c].isna().sum()}")

print('\nUnique seasons:', df['Year'].sort_values().unique() if 'Year' in df.columns else 'Year missing')
print('Done.')
