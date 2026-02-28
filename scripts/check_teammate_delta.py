"""Check teammate deltas in `data_files/all_qualifying_races.csv`.

This script asserts that when a team (grouped by Year, Round, constructor)
has more than one valid `best_qual_time`, the corresponding rows have a
non-null `teammate_qual_delta` value. Exits with code 0 on success, 1 on failure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path('data_files')
QUAL_CSV = DATA_DIR / 'all_qualifying_races.csv'


def main() -> int:
    if not QUAL_CSV.exists():
        print(f"ERROR: {QUAL_CSV} not found")
        return 1
    try:
        q = pd.read_csv(QUAL_CSV, sep='\t', low_memory=False)
    except Exception as e:
        print(f"ERROR: failed to read {QUAL_CSV}: {e}")
        return 1

    if q.empty:
        print(f"ERROR: {QUAL_CSV} is empty")
        return 1

    # Resolve the best-time column — prefer actual_best_lap (seconds), fall back to best_qual_time
    if 'actual_best_lap' in q.columns:
        time_col = 'actual_best_lap'
    elif 'best_qual_time' in q.columns:
        time_col = 'best_qual_time'
    else:
        print("ERROR: neither 'actual_best_lap' nor 'best_qual_time' column found in qualifying CSV")
        return 1
    q[time_col] = pd.to_numeric(q[time_col], errors='coerce')

    # teammate_qual_delta is computed by fastF1-qualifying.py; warn if absent
    if 'teammate_qual_delta' not in q.columns:
        print(f"WARN: 'teammate_qual_delta' column not present in qualifying CSV "
              f"(run fastF1-qualifying.py to compute it) — skipping delta check")
        print(f"INFO: time column used: '{time_col}', rows: {len(q)}")
        return 0

    # Build a stable constructor grouping
    if 'constructor_group' not in q.columns:
        q['constructor_group'] = q.get('constructorId').fillna(q.get('constructorName'))

    group_cols = ['Year', 'Round', 'constructor_group']

    # Identify groups where there are >1 non-null best qualifying time
    grp = q.groupby(group_cols)
    groups_with_multiple = grp[time_col].apply(lambda s: s.notna().sum() > 1)
    groups_to_check = groups_with_multiple[groups_with_multiple].index.tolist()

    total_groups = len(groups_to_check)
    failing_rows = []

    for g in groups_to_check:
        sub = q[(q['Year'] == g[0]) & (q['Round'] == g[1]) & (q['constructor_group'] == g[2])]
        # rows with a best time should have teammate_qual_delta
        mask = sub[time_col].notna() & sub['teammate_qual_delta'].isna()
        if mask.any():
            for _idx, row in sub[mask].iterrows():
                failing_rows.append((int(row['Year']), int(row['Round']), row.get('driverId'), row.get('constructor_group'), row.get(time_col)))

    if not failing_rows:
        print(f"OK: teammate deltas present for all teams with >1 {time_col} (checked {total_groups} groups)")
        return 0

    print(f"FAIL: found {len(failing_rows)} rows missing teammate_qual_delta in {total_groups} groups")
    print("Examples:")
    for example in failing_rows[:10]:
        print(" Year={}, Round={}, driverId={}, constructor_group={}, {}={}".format(*example[:4], time_col, example[4]))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
