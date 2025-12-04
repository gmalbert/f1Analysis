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

    # Ensure best_qual_time is numeric
    if 'best_qual_time' in q.columns:
        q['best_qual_time'] = pd.to_numeric(q['best_qual_time'], errors='coerce')
    else:
        print("ERROR: 'best_qual_time' column missing from qualifying CSV")
        return 1

    # Build a stable constructor grouping
    if 'constructor_group' not in q.columns:
        q['constructor_group'] = q.get('constructorId').fillna(q.get('constructorName'))

    group_cols = ['Year', 'Round', 'constructor_group']

    # Identify groups where there are >1 non-null best_qual_time
    grp = q.groupby(group_cols)
    groups_with_multiple = grp['best_qual_time'].apply(lambda s: s.notna().sum() > 1)
    groups_to_check = groups_with_multiple[groups_with_multiple].index.tolist()

    total_groups = len(groups_to_check)
    failing_rows = []

    for g in groups_to_check:
        sub = q[(q['Year'] == g[0]) & (q['Round'] == g[1]) & (q['constructor_group'] == g[2])]
        # rows with a best time should have teammate_qual_delta
        mask = sub['best_qual_time'].notna() & sub['teammate_qual_delta'].isna()
        if mask.any():
            for _idx, row in sub[mask].iterrows():
                failing_rows.append((int(row['Year']), int(row['Round']), row.get('driverId'), row.get('constructor_group'), row.get('best_qual_time')))

    if not failing_rows:
        print(f"OK: teammate deltas present for all teams with >1 best_qual_time (checked {total_groups} groups)")
        return 0

    print(f"FAIL: found {len(failing_rows)} rows missing teammate_qual_delta in {total_groups} groups")
    print("Examples:")
    for example in failing_rows[:10]:
        print(" Year={}, Round={}, driverId={}, constructor_group={}, best_qual_time={}".format(*example))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
