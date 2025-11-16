"""Audit script to detect potential temporal leakage in features.

Writes a diagnostics CSV to `data_files/leakage_audit_report.csv` and prints
a short summary. Heuristics used:
 - High Pearson correlation (abs > 0.95) with `resultsFinalPositionNumber` or `DNF`.
 - Feature name patterns suggesting post-race calculations (contains 'post','after','final','result','total').
 - For numeric features, compare corr(feature_t, result_t) vs corr(feature_t, result_{t+1})
   using per-driver ordering by `short_date`. If correlation with next-result is
   substantially larger, flag as suspicious (possible future-looking).

Usage:
    python audit_temporal_leakage.py

Set `DATA_DIR` or ensure `data_files/f1ForAnalysis.csv` exists.
"""

import pandas as pd
import numpy as np
from os import path

DATA_DIR = 'data_files/'
INPUT_FILE = path.join(DATA_DIR, 'f1ForAnalysis.csv')
OUTPUT_FILE = path.join(DATA_DIR, 'leakage_audit_report.csv')

THRESH_HIGH_CORR = 0.95
THRESH_LAG_CORR_DIFF = 0.1

# Friendly column name mapping for the output CSV
OUTPUT_COL_MAP = {
    'feature': 'feature',
    'issue_type': 'issue',
    'target': 'target',
    'metric': 'value',
    'metric2': 'value2',
    'metric_name': 'metric_name',
    'explanation': 'explanation',
    'diff': 'delta',
    'extra_info': 'note'
}

def load_data(nrows=None):
    if not path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Expected data file at {INPUT_FILE}")
    return pd.read_csv(INPUT_FILE, sep='\t', nrows=nrows)


def is_numeric_series(s):
    try:
        return pd.api.types.is_numeric_dtype(s)
    except Exception:
        return False


def high_corr_checks(df, target_col='resultsFinalPositionNumber'):
    results = []
    if target_col not in df.columns:
        return results
    y = df[target_col]
    for col in df.columns:
        if col == target_col:
            continue
        if not is_numeric_series(df[col]):
            continue
        x = df[col].astype(float)
        mask = x.notnull() & y.notnull()
        if mask.sum() < 10:
            continue
        try:
            corr = x[mask].corr(y[mask])
        except Exception:
            corr = np.nan
        if pd.notna(corr) and abs(corr) >= THRESH_HIGH_CORR:
            # Standardized report keys: feature, issue_type, target, metric, metric2, diff, extra_info
            metric_name = 'pearson_corr'
            explanation = f'corr={corr:.4f}' if pd.notna(corr) else ''
            results.append({'feature': col, 'issue_type': 'high_corr_with_target', 'target': target_col, 'metric': corr, 'metric2': np.nan, 'metric_name': metric_name, 'explanation': explanation, 'diff': np.nan, 'extra_info': ''})
    return results


def name_pattern_checks(df):
    patterns = ['post', 'after', 'final', 'result', 'total', 'avg_future', 'future']
    results = []
    for col in df.columns:
        lower = col.lower()
        for p in patterns:
            if p in lower:
                metric_name = 'name_pattern'
                explanation = f"matched_pattern={p}"
                results.append({'feature': col, 'issue_type': 'name_pattern', 'target': np.nan, 'metric': np.nan, 'metric2': np.nan, 'metric_name': metric_name, 'explanation': explanation, 'diff': np.nan, 'extra_info': p})
                break
    return results


def lagged_corr_checks(df, driver_id_col='resultsDriverId', date_col='short_date', target_col='resultsFinalPositionNumber'):
    # For each numeric feature, compute correlation with current target and next-race target
    results = []
    if driver_id_col not in df.columns or date_col not in df.columns or target_col not in df.columns:
        return results
    # Ensure date is datetime
    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        pass

    # Sort by driver and date
    df = df.sort_values([driver_id_col, date_col])

    # Build next-result column per driver
    df['_next_result'] = df.groupby(driver_id_col)[target_col].shift(-1)

    for col in df.columns:
        if col in [driver_id_col, date_col, target_col, '_next_result']:
            continue
        if not is_numeric_series(df[col]):
            continue
        series = df[col].astype(float)
        valid_mask = series.notnull() & df[target_col].notnull()
        if valid_mask.sum() < 20:
            continue
        try:
            corr_current = series[valid_mask].corr(df[target_col][valid_mask])
        except Exception:
            corr_current = np.nan
        valid_mask_next = series.notnull() & df['_next_result'].notnull()
        if valid_mask_next.sum() < 20:
            corr_next = np.nan
        else:
            try:
                corr_next = series[valid_mask_next].corr(df['_next_result'][valid_mask_next])
            except Exception:
                corr_next = np.nan
        if pd.notna(corr_next) and pd.notna(corr_current) and (corr_next - corr_current) >= THRESH_LAG_CORR_DIFF:
            metric_name = 'corr_current_vs_next'
            delta = corr_next - corr_current
            explanation = f'corr_current={corr_current:.4f},corr_next={corr_next:.4f},delta={delta:.4f}'
            results.append({'feature': col, 'issue_type': 'lagged_corr_suspicious', 'target': target_col, 'metric': corr_current, 'metric2': corr_next, 'metric_name': metric_name, 'explanation': explanation, 'diff': delta, 'extra_info': ''})
    return results


def safety_car_checks(df):
    results = []
    # Identify columns that look like safety car features
    candidates = [c for c in df.columns if 'safety' in c.lower() or 'safetycar' in c.lower() or 'safety_car' in c.lower()]
    for col in candidates:
        # Check correlation with target
        if is_numeric_series(df[col]) and 'resultsFinalPositionNumber' in df.columns:
            try:
                corr = df[col].astype(float).corr(df['resultsFinalPositionNumber'].astype(float))
            except Exception:
                corr = np.nan
        else:
            corr = np.nan
        metric_name = 'pearson_corr'
        explanation = f'corr={corr:.4f}' if pd.notna(corr) else ''
        results.append({'feature': col, 'issue_type': 'safetycar_candidate', 'target': 'resultsFinalPositionNumber', 'metric': corr, 'metric2': np.nan, 'metric_name': metric_name, 'explanation': explanation, 'diff': np.nan, 'extra_info': ''})
    return results


def run_audit(nrows=None):
    df = load_data(nrows=nrows)
    # Report how many rows were loaded so the caller can verify nrows behavior
    try:
        rows_loaded = len(df)
    except Exception:
        rows_loaded = None
    print(f'Loaded rows: {rows_loaded} (requested nrows={nrows})')
    report = []

    report.extend(high_corr_checks(df, target_col='resultsFinalPositionNumber'))
    report.extend(high_corr_checks(df, target_col='DNF'))
    report.extend(name_pattern_checks(df))
    report.extend(lagged_corr_checks(df))
    report.extend(safety_car_checks(df))

    # Create DataFrame with standardized columns so CSV headers are consistent
    cols = ['feature', 'issue_type', 'target', 'metric', 'metric2', 'metric_name', 'explanation', 'diff', 'extra_info']
    report_df = pd.DataFrame(report)
    # Ensure all expected columns exist (fill missing with NaN)
    for c in cols:
        if c not in report_df.columns:
            report_df[c] = np.nan
    # Reorder columns for readability
    report_df = report_df[cols]
    if report_df.empty:
        print('No suspicious features found by heuristics.')
    else:
        # Rename columns for a friendlier report CSV header
        out_df = report_df.rename(columns=OUTPUT_COL_MAP)
        out_df.to_csv(OUTPUT_FILE, index=False)
        print(f'Wrote leakage audit report to {OUTPUT_FILE} ({len(report_df)} items) — scanned {rows_loaded} rows')
        print('Report columns:', list(out_df.columns))

    # Also print a short summary (by issue_type)
    summary = report_df['issue_type'].value_counts().to_dict() if not report_df.empty else {}
    print('Summary:', summary)
    return report_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Audit F1 analysis features for temporal leakage (heuristics).')
    parser.add_argument('--nrows', type=int, default=None, help='Number of rows to read from CSV (for testing).')
    args = parser.parse_args()
    # Support legacy behavior where 0 is used to indicate 'all rows'
    nrows = args.nrows
    if nrows == 0:
        nrows = None
    run_audit(nrows=nrows)
