import pandas as pd
from pathlib import Path

CUR = Path('data_files/all_qualifying_races.csv')
BAK = Path('data_files/all_qualifying_races.csv.bak')
if not CUR.exists():
    print('Current qualifying CSV missing:', CUR)
    raise SystemExit(1)
if not BAK.exists():
    print('Backup qualifying CSV missing:', BAK)
    raise SystemExit(1)

cur = pd.read_csv(CUR, sep='\t', low_memory=False)
bak = pd.read_csv(BAK, sep='\t', low_memory=False)

# Use Year+Round+DriverNumber as key to align rows
key_cols = ['Year', 'Round', 'DriverNumber']
for k in key_cols:
    if k not in cur.columns or k not in bak.columns:
        print(f'Missing key column: {k} in one of the files')
        raise SystemExit(1)

cur_indexed = cur.set_index(key_cols)
bak_indexed = bak.set_index(key_cols)

# Columns to restore from backup when current has nulls
restore_cols = []
for c in ['driverId', 'q1_sec', 'q2_sec', 'q3_sec', 'best_qual_time', 'constructorId', 'Abbreviation', 'FullName']:
    if c in bak_indexed.columns:
        restore_cols.append(c)

print('Columns considered for restore:', restore_cols)

restored = 0
for idx, row in cur_indexed.iterrows():
    if idx in bak_indexed.index:
        bak_row = bak_indexed.loc[idx]
        for col in restore_cols:
            try:
                cur_val = row.get(col) if col in row.index else None
            except Exception:
                cur_val = row[col] if col in row.index else None
            bak_val = bak_row.get(col) if col in bak_row.index else None
            if (pd.isna(cur_val) or (isinstance(cur_val, str) and cur_val.strip() == '')) and (not pd.isna(bak_val)):
                cur_indexed.at[idx, col] = bak_val
                restored += 1

# Write output to a new file and keep original as backup
OUT = CUR.with_suffix('.restored.csv')
cur_restored = cur_indexed.reset_index()
cur_restored.to_csv(OUT, sep='\t', index=False)
print(f'Restored {restored} values from backup. Wrote restored file to {OUT}')
print('If this looks good, replace the original file with the restored one.')
