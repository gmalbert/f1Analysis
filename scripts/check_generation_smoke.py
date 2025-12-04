"""
Smoke test to validate generation output after running `f1-generate-analysis.py`.

Checks:
- `data_files/f1ForAnalysis.csv` exists and has rows.
- `short_date.max()` from the analysis CSV is >= `date.max()` from `data_files/f1db-races.json` minus an allowed tolerance.
- Warn if a large fraction of `data_files/all_qualifying_races.csv` is missing `best_qual_time`.

Usage:
    python scripts/check_generation_smoke.py [--strict] [--qual-threshold 0.05] [--tolerance-days 0]

- `--strict`: exit non-zero when coverage check fails.
- `--qual-threshold`: fraction (0-1) above which the qualifying completeness check warns/fails.
- `--tolerance-days`: allow N days difference between expected max race date and analysis max date.

Returns exit code 0 on success (or warnings only), non-zero when `--strict` and checks fail.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path('data_files')
ANALYSIS_CSV = DATA_DIR / 'f1ForAnalysis.csv'
RACES_JSON = DATA_DIR / 'f1db-races.json'
QUAL_CSV = DATA_DIR / 'all_qualifying_races.csv'


def parse_args():
    p = argparse.ArgumentParser(description='Smoke test for generated F1 analysis files')
    p.add_argument('--strict', action='store_true', help='Exit non-zero when checks fail')
    p.add_argument('--qual-threshold', type=float, default=0.05, help='Max allowed fraction of missing qualifying best times before warning/fail (default 0.05)')
    p.add_argument('--tolerance-days', type=int, default=0, help='Allow this many days difference between expected max race date and analysis max date')
    return p.parse_args()


def read_analysis_max_date():
    if not ANALYSIS_CSV.exists():
        print(f"ERROR: {ANALYSIS_CSV} not found")
        return None, "missing"
    try:
        df = pd.read_csv(ANALYSIS_CSV, sep='\t', parse_dates=['short_date'], low_memory=False)
    except Exception as e:
        print(f"ERROR reading {ANALYSIS_CSV}: {e}")
        return None, "read_error"
    if df.shape[0] == 0:
        print(f"ERROR: {ANALYSIS_CSV} has zero rows")
        return None, "empty"
    max_date = df['short_date'].max()
    return pd.to_datetime(max_date), None


def read_races_max_date():
    if not RACES_JSON.exists():
        print(f"ERROR: {RACES_JSON} not found")
        return None, "missing"
    try:
        races = pd.read_json(RACES_JSON)
    except Exception as e:
        print(f"ERROR reading {RACES_JSON}: {e}")
        return None, "read_error"
    # Try a few common date fields
    date_cols = [c for c in races.columns if 'date' in c.lower()]
    if not date_cols:
        # attempt to look for 'short_date' created elsewhere
        print(f"ERROR: no date-like column found in {RACES_JSON}")
        return None, 'no_date_col'
    # coerce to datetimes and compute max
    for c in date_cols:
        try:
            races[c] = pd.to_datetime(races[c], errors='coerce')
        except Exception:
            races[c] = pd.to_datetime(races[c].astype(str), errors='coerce')
    max_date = races[date_cols].max(axis=1).max()
    return pd.to_datetime(max_date), None


def check_qualifying_completeness(threshold=0.05):
    if not QUAL_CSV.exists():
        print(f"WARN: {QUAL_CSV} not found — skipping qualifying completeness check")
        return True, 'missing_file'
    try:
        q = pd.read_csv(QUAL_CSV, sep='\t', low_memory=False)
    except Exception as e:
        print(f"WARN: could not read qualifying CSV: {e}")
        return False, 'read_error'
    total = len(q)
    if total == 0:
        print(f"WARN: {QUAL_CSV} is empty")
        return False, 'empty'
    missing_best = q['best_qual_time'].isnull().sum() if 'best_qual_time' in q.columns else total
    frac_missing = missing_best / total
    ok = frac_missing <= threshold
    return ok, {'total': total, 'missing_best': int(missing_best), 'frac_missing': frac_missing}


def main():
    args = parse_args()
    failures = []

    analysis_max, err = read_analysis_max_date()
    if err:
        failures.append(('analysis_read', err))
    else:
        print(f"Analysis max short_date: {analysis_max}")

    races_max, err = read_races_max_date()
    if err:
        failures.append(('races_read', err))
    else:
        print(f"Races max date: {races_max}")

    if analysis_max is not None and races_max is not None:
        # compare with tolerance
        delta_days = (pd.to_datetime(races_max) - pd.to_datetime(analysis_max)).days
        if delta_days > args.tolerance_days:
            msg = f"WARNING: analysis max date ({analysis_max.date()}) is behind races max date ({races_max.date()}) by {delta_days} days > tolerance {args.tolerance_days}"
            print(msg)
            failures.append(('date_coverage', {'delta_days': int(delta_days), 'analysis_max': str(analysis_max.date()), 'races_max': str(races_max.date())}))
        else:
            print(f"OK: analysis covers races up to {analysis_max.date()} (delta_days={delta_days})")

    # qualifying completeness
    ok, qual_info = check_qualifying_completeness(threshold=args.qual_threshold)
    if isinstance(qual_info, dict):
        print(f"Qualifying completeness: total={qual_info['total']}, missing_best={qual_info['missing_best']}, frac_missing={qual_info['frac_missing']:.3f}")
        if not ok:
            print(f"WARNING: fraction of missing qualifying best times {qual_info['frac_missing']:.3f} > threshold {args.qual_threshold}")
            failures.append(('qual_incomplete', qual_info))
        else:
            print("OK: qualifying completeness within threshold")
    else:
        # qual_info is a status string
        print(f"Qualifying check status: {qual_info}")

    # run teammate-delta check (scripts/check_teammate_delta.py)
    try:
        import subprocess
        res = subprocess.run(['python', 'scripts/check_teammate_delta.py'], check=False)
        if res.returncode != 0:
            print('WARNING: teammate-delta check failed')
            failures.append(('teammate_delta_check', {'returncode': res.returncode}))
        else:
            print('OK: teammate-delta check passed')
    except Exception as e:
        print(f'WARN: could not run teammate-delta check: {e}')
        failures.append(('teammate_delta_check', 'run_error'))

    if failures:
        print('\nSummary: FAILURES DETECTED')
        for t, info in failures:
            print(f" - {t}: {info}")
        if args.strict:
            print('\nExiting with non-zero due to --strict')
            sys.exit(2)
        else:
            print('\nRun with --strict to make this fail the build/test pipeline.')
            sys.exit(0)
    else:
        print('\nAll smoke checks passed')
        sys.exit(0)


if __name__ == '__main__':
    main()
