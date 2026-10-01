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


def run_audit(nrows=None):
    """Return the structured report expected by both application audit panels.

    The command-line entry point remains available. This callable performs
    the panel's name, equality, correlation, lag and future-date heuristics
    without changing the analysis dataset or fitting a prediction model.
    """
    if nrows is not None and (not isinstance(nrows, int) or nrows < 1):
        raise ValueError('Rows to read must be a positive integer or None.')
    data_dir = Path(__file__).resolve().parents[1] / 'data_files'
    df = pd.read_csv(data_dir / 'f1ForAnalysis.csv', sep='\t', low_memory=False, nrows=nrows)
    findings = []
    columns = ['feature', 'issue_type', 'target', 'metric', 'metric2', 'metric_name', 'explanation', 'diff', 'extra_info']

    def flag(feature, issue, target='', metric=None, metric2=None, name='', explanation='', diff=None, note=''):
        findings.append(dict(zip(columns, [feature, issue, target, metric, metric2, name, explanation, diff, note])))

    targets = [c for c in ['resultsFinalPositionNumber', 'DNF', 'SafetyCarStatus'] if c in df]
    patterns = ['post', 'after', 'final', 'result', 'total', 'future', 'next_', 'lead', 'target']
    for column in df:
        if column not in targets and any(pattern in column.lower() for pattern in patterns):
            flag(column, 'name_pattern', explanation='Name may describe a post-event result or accumulated statistic; review availability at prediction time.')

    numeric = df.select_dtypes(include=['number', 'bool']).astype(float)
    for target in targets:
        if target not in numeric or numeric[target].nunique() < 2:
            continue
        for column in numeric:
            if column == target:
                continue
            valid = numeric[[column, target]].dropna()
            if len(valid) < 10 or valid[column].nunique() < 2 or valid[target].nunique() < 2:
                continue
            correlation = valid[column].corr(valid[target])
            if pd.notna(correlation) and abs(correlation) >= .95:
                flag(column, 'high_correlation', target, float(correlation), name='pearson', explanation='Feature is very strongly correlated with the same-event target.', note=f'{len(valid)} non-missing pairs')
            equality = float((valid[column] == valid[target]).mean())
            if equality > .5:
                flag(column, 'exact_equality', target, equality, name='fraction_equal', explanation='Feature equals the target in more than half of non-missing pairs.', note=f'{len(valid)} pairs')

    driver = next((c for c in ['resultsDriverId', 'driverId', 'resultsDriverName'] if c in df), None)
    date_column = find_date_column(df)
    if driver and date_column:
        ordered = df.assign(_audit_date=pd.to_datetime(df[date_column], errors='coerce')).sort_values('_audit_date')
        for target in targets:
            next_target = pd.to_numeric(ordered.groupby(driver)[target].shift(-1), errors='coerce')
            current_target = pd.to_numeric(ordered[target], errors='coerce')
            for column in numeric:
                if column in targets:
                    continue
                feature = pd.to_numeric(ordered[column], errors='coerce')
                paired = pd.DataFrame({'feature': feature, 'next': next_target, 'current': current_target}).dropna()
                if len(paired) < 10 or any(paired[c].nunique() < 2 for c in paired):
                    continue
                future = paired['feature'].corr(paired['next'])
                current = paired['feature'].corr(paired['current'])
                if pd.notna(future) and pd.notna(current) and abs(future) >= .7 and abs(future) > abs(current) + .1:
                    flag(column, 'lagged_correlation', target, float(future), float(current), 'pearson_next_vs_current', 'Association with the next driver event is substantially stronger than with the current event.', float(abs(future) - abs(current)))

    races = pd.read_json(data_dir / 'f1db-races.json')
    race_date = find_date_column(races)
    if race_date and 'raceId' in df and 'id' in races:
        dates = pd.to_datetime(df['raceId'].map(races.set_index('id')[race_date]), errors='coerce')
        if date_column:
            after = pd.to_datetime(df[date_column], errors='coerce') > dates
            if after.any():
                flag(date_column, 'date_after_race', metric=int(after.sum()), name='rows', explanation='Analysis date occurs after the scheduled race date.')
        future = dates > pd.Timestamp.now().normalize()
        for column in df:
            if any(word in column.lower() for word in ['practice', 'qual', 'best_qual']):
                present = int(df.loc[future, column].notna().sum())
                if present:
                    flag(column, 'future_event_data', metric=present, name='rows', explanation='Practice or qualifying values are present for a future scheduled race.')
    return pd.DataFrame(findings, columns=columns)


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
