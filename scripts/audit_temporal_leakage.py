#!/usr/bin/env python3
"""Audit for temporal leakage in precomputed analysis CSVs.

Conservative checks performed:
- Verify `short_date` is not after the scheduled race date for any row.
- Flag suspicious column names that may indicate future-looking features.
- Detect columns that exactly equal `resultsFinalPositionNumber` for many rows (possible leakage).
- For scheduled/future races, flag any non-null practice/qualifying fields.
- Check `SafetyCar`-named columns for suspicious equality/correlation with the target.

This script prints a summary and exits 0; it errs on the side of caution and reports warnings.
"""
import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def find_date_column(df):
    candidates = ['date', 'race_date', 'short_date', 'start_date']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any datetime-like column
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    return None


def main():
    p = argparse.ArgumentParser(description='Audit temporal leakage in f1ForAnalysis.csv')
    p.add_argument('--data', default='data_files/f1ForAnalysis.csv')
    p.add_argument('--races', default='data_files/f1db-races.json')
    p.add_argument('--target', default='resultsFinalPositionNumber')
    args = p.parse_args()

    data_path = Path(args.data)
    races_path = Path(args.races)
    if not data_path.exists():
        print(f'ERROR: data file not found: {data_path}', file=sys.stderr)
        sys.exit(2)
    if not races_path.exists():
        print(f'ERROR: races file not found: {races_path}', file=sys.stderr)
        sys.exit(2)

    print(f'Loading data: {data_path}')
    df = pd.read_csv(data_path, sep='\t', low_memory=False)
    print(f'Loaded rows: {len(df)} columns: {len(df.columns)}')

    print(f'Loading races: {races_path}')
    races = pd.read_json(races_path)

    # Normalize date columns
    df_short_date_col = find_date_column(df)
    races_date_col = find_date_column(races)
    if df_short_date_col is None:
        print('WARNING: could not find a date-like column in analysis CSV (expected `short_date`)', file=sys.stderr)
    else:
        df[df_short_date_col] = pd.to_datetime(df[df_short_date_col], errors='coerce')

    if races_date_col is None:
        print('WARNING: could not find a date-like column in races JSON', file=sys.stderr)
    else:
        races[races_date_col] = pd.to_datetime(races[races_date_col], errors='coerce')

    # Merge on raceId -> races id column may be 'id'
    if 'raceId' in df.columns and 'id' in races.columns:
        merged = df.merge(races[['id', races_date_col]].rename(columns={'id': 'raceId', races_date_col: 'race_date'}), on='raceId', how='left')
    else:
        merged = df.copy()

    warnings = []

    # 1) short_date vs race_date
    if 'race_date' in merged.columns and df_short_date_col is not None:
        mask = merged[df_short_date_col].notna() & merged['race_date'].notna() & (merged[df_short_date_col] > merged['race_date'])
        n_bad = mask.sum()
        if n_bad:
            warnings.append(f'{n_bad} rows have analysis `short_date` after the scheduled race date (possible future-looking rows)')

    # 2) suspicious column names
    suspicious_keywords = ['future', 'next_', 'next', 'lead', 'ahead', 'shift', 'target', 'result_future']
    suspicious_cols = [c for c in df.columns if any(kw in c.lower() for kw in suspicious_keywords)]
    if suspicious_cols:
        warnings.append(f'Suspicious column names (possible future-looking): {suspicious_cols[:10]}{"..." if len(suspicious_cols)>10 else ""}')

    # 3) exact-equality leakage test vs target
    target = args.target
    if target in df.columns:
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in ('raceId', 'grandPrixYear') and c != target]
        equality_issues = []
        for c in numeric_cols:
            # compute fraction of rows where feature equals target (and not-null)
            valid = df[[c, target]].dropna()
            if len(valid) == 0:
                continue
            frac_eq = (valid[c] == valid[target]).mean()
            if frac_eq > 0.5:
                equality_issues.append((c, float(frac_eq), len(valid)))
        if equality_issues:
            fmt = ', '.join([f'{c} (frac_eq={f:.2f}, n={n})' for c, f, n in equality_issues])
            warnings.append('Possible exact-equality leakage detected for numeric columns: ' + fmt)

    # 4) future races should not have practice/qualifying present
    practice_qual_keywords = ['best_qual', 'qual', 'practice', 'FastestPractice', 'averagePractice']
    future_rows = None
    if 'race_date' in merged.columns:
        today = pd.Timestamp.now().normalize()
        future_rows = merged['race_date'] > today
        if future_rows.any():
            cols_with_data = []
            for c in df.columns:
                if any(kw.lower() in c.lower() for kw in practice_qual_keywords):
                    if merged.loc[future_rows, c].notna().any():
                        cols_with_data.append(c)
            if cols_with_data:
                warnings.append(f'Found practice/qualifying data in scheduled future races for columns: {cols_with_data[:10]}')

    # 5) SafetyCar columns check
    safety_cols = [c for c in df.columns if 'safetycar' in c.lower() or 'safety_car' in c.lower() or 'safety car' in c.lower()]
    if safety_cols and target in df.columns:
        sc_issues = []
        for c in safety_cols:
            valid = df[[c, target]].dropna()
            if len(valid) < 10:
                continue
            # fraction equal to target
            frac_eq = (valid[c] == valid[target]).mean()
            if frac_eq > 0.5:
                sc_issues.append((c, float(frac_eq)))
        if sc_issues:
            fmt = ', '.join([f'{c} (frac_eq={f:.2f})' for c, f in sc_issues])
            warnings.append('Suspicious SafetyCar-like columns with high equality to target: ' + fmt)

    # Print results
    print('\nTemporal Leakage Audit Summary')
    print('--------------------------------')
    if warnings:
        print('WARNINGS:')
        for w in warnings:
            print('-', w)
        print('\nPlease review the flagged columns/rows; these checks are conservative and intended to highlight likely issues.')
        sys.exit(0)
    else:
        print('No obvious temporal leakage detected by automated checks.')
        sys.exit(0)


if __name__ == '__main__':
    main()
