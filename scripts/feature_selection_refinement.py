#!/usr/bin/env python3
"""Feature selection refinement helper.

Runs conservative, fast analyses to help refine features:
- Boruta (if BorutaPy is installed) to propose relevant features
- SHAP importances from a quick XGBoost model (fast approximation)
- High-correlation pairs to flag redundant features

Outputs:
- scripts/output/boruta_selected.txt
- scripts/output/shap_ranking.txt
- scripts/output/correlated_pairs.csv
- scripts/output/feature_selection_report.txt

This is intended as an exploratory, reproducible helper — not a full hyperparameter/tuning job.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_output_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def try_import(name: str):
    try:
        mod = __import__(name)
        return mod
    except Exception:
        return None


def main():
    root = Path(__file__).resolve().parents[1]
    data_file = root / 'data_files' / 'f1ForAnalysis.csv'
    out_dir = Path(__file__).resolve().parent / 'output'
    ensure_output_dir(out_dir)

    if not data_file.exists():
        print(f'ERROR: expected data at {data_file} — run f1-generate-analysis.py first')
        sys.exit(2)

    logger.info('Loading data (this may take a few seconds)...')
    df = pd.read_csv(data_file, sep='\t', low_memory=False)

    # Basic cleaning: drop obviously non-feature columns
    drop_prefixes = ['id', 'raceId', 'grandPrixId', 'resultsDriverId', 'driverId', 'Driver', 'constructor']
    non_feature_cols = [c for c in df.columns if any(c.lower().startswith(p.lower()) for p in drop_prefixes)]
    # Also drop the main target if present — we'll use it explicitly
    target_col = 'resultsFinalPositionNumber'
    if target_col not in df.columns:
        # fallback target
        alt_targets = ['resultsFinalPosition', 'finalPosition']
        for t in alt_targets:
            if t in df.columns:
                target_col = t
                break

    logger.info('Target column: %s', target_col)

    feature_df = df.select_dtypes(include=[np.number]).copy()
    # remove id-like columns
    for c in non_feature_cols:
        if c in feature_df.columns:
            feature_df.drop(columns=[c], inplace=True)

    if target_col in feature_df.columns:
        y = feature_df[target_col].copy()
        X = feature_df.drop(columns=[target_col])
    else:
        print('WARNING: target not found in numeric columns; attempting to load target from main df')
        if target_col in df.columns:
            y = df[target_col]
            X = feature_df
        else:
            print('ERROR: no target found; aborting')
            sys.exit(2)

    # Simple imputation (median)
    X = X.fillna(X.median())

    # Ensure target is numeric and drop rows where target is missing or non-finite
    y = pd.to_numeric(y, errors='coerce')
    mask = y.notna() & np.isfinite(y)
    X = X.loc[mask]
    y = y.loc[mask]
    if X.shape[0] == 0:
        print('ERROR: no rows with valid target after dropping NaNs; aborting')
        sys.exit(2)

    report_lines = []
    report_lines.append(f'Input rows,cols: {df.shape}; numeric features: {X.shape[1]}')

    # 1) Correlation analysis — flag pairs with abs(corr) > 0.95
    logger.info('Computing correlation matrix...')
    corr = X.corr().abs()
    # extract upper triangle
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = (upper.stack().reset_index()
                 .rename(columns={'level_0': 'feature_a', 'level_1': 'feature_b', 0: 'abs_corr'})
                 .sort_values('abs_corr', ascending=False))
    high_corr = high_corr[high_corr['abs_corr'] > 0.95]
    high_corr_file = out_dir / 'correlated_pairs.csv'
    high_corr.to_csv(high_corr_file, index=False)
    report_lines.append(f'High-correlation pairs (>0.95): {len(high_corr)}; written to {high_corr_file}')

    # 2) Quick XGBoost + SHAP ranking
    xgb = try_import('xgboost')
    shap = try_import('shap')
    if xgb is None:
        report_lines.append('xgboost not installed — skipping SHAP ranking')
    else:
        logger.info('Training quick XGBoost model (for Boruta/feature checks)...')
        from xgboost import XGBRegressor
        # use a small model for speed; reduce verbosity
        model = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=4, random_state=42, verbosity=0)
        # drop constant columns if any
        nunique = X.nunique()
        const_cols = nunique[nunique <= 1].index.tolist()
        if const_cols:
            X = X.drop(columns=const_cols)
            report_lines.append(f'Dropped constant numeric cols: {len(const_cols)}')
        # Fit on a sample if dataset is large
        sample_frac = 1.0
        if X.shape[0] > 4000:
            sample_frac = 0.2
        X_train = X.sample(frac=sample_frac, random_state=42)
        y_train = y.loc[X_train.index]
        model.fit(X_train, y_train)

        # Use LightGBM for SHAP TreeExplainer if available — often more compatible
        lgb = try_import('lightgbm')
        if shap is None:
            report_lines.append('shap not installed — skipping SHAP analysis, but model trained')
        else:
            if lgb is None:
                report_lines.append('lightgbm not installed — attempting SHAP on XGBoost model (may fail)')
                try:
                    import shap as _shap
                        logger.info('Computing SHAP values with XGBoost TreeExplainer (may fail)...')
                    explainer = _shap.TreeExplainer(model)
                    X_shap = X_train.sample(n=min(500, len(X_train)), random_state=1)
                    shap_values = explainer.shap_values(X_shap)
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_ranking = pd.DataFrame({'feature': X_shap.columns, 'mean_abs_shap': mean_abs_shap})
                    shap_ranking = shap_ranking.sort_values('mean_abs_shap', ascending=False)
                    shap_file = out_dir / 'shap_ranking.txt'
                    shap_ranking.to_csv(shap_file, index=False)
                    report_lines.append(f'SHAP ranking written to {shap_file}; top features: {list(shap_ranking.head(10).feature)}')
                except Exception as e:
                    report_lines.append('SHAP on XGBoost failed: ' + str(e))
            else:
                try:
                    logger.info('Training LightGBM model for SHAP (fast settings)...')
                    from lightgbm import LGBMRegressor
                    # silence LightGBM output with verbose=-1
                    lgbm = LGBMRegressor(n_estimators=200, max_depth=6, n_jobs=4, random_state=42, verbose=-1)
                    lgbm.fit(X_train, y_train)
                    import shap as _shap
                    logger.info('Computing SHAP values with LightGBM TreeExplainer...')
                    explainer = _shap.TreeExplainer(lgbm)
                    X_shap = X_train.sample(n=min(500, len(X_train)), random_state=1)
                    shap_values = explainer.shap_values(X_shap)
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    shap_ranking = pd.DataFrame({'feature': X_shap.columns, 'mean_abs_shap': mean_abs_shap})
                    shap_ranking = shap_ranking.sort_values('mean_abs_shap', ascending=False)
                    shap_file = out_dir / 'shap_ranking.txt'
                    shap_ranking.to_csv(shap_file, index=False)
                    report_lines.append(f'SHAP ranking (LightGBM) written to {shap_file}; top features: {list(shap_ranking.head(10).feature)}')
                except Exception as e:
                    report_lines.append('SHAP with LightGBM failed: ' + str(e))

    # 3) Boruta selection (optional)
    boruta_mod = try_import('Boruta') or try_import('boruta')
    if boruta_mod is None:
        report_lines.append('Boruta not installed (BorutaPy) — skipping Boruta selection. To run, install boruta_py and scikit-learn.')
    else:
        try:
            print('Running Boruta feature selection (may take some time)...')
            # BorutaPy import paths vary
            try:
                from boruta import BorutaPy
            except Exception:
                from Boruta import BorutaPy
            from xgboost import XGBRegressor
            rf = XGBRegressor(n_estimators=200, max_depth=4, n_jobs=4, random_state=42)
            # Boruta expects numpy arrays
            X_np = X.values
            y_np = y.values
            boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42, verbose=0)
            boruta_selector.fit(X_np, y_np)
            support = boruta_selector.support_
            selected = X.columns[support].tolist()
            boruta_file = out_dir / 'boruta_selected.txt'
            with open(boruta_file, 'w') as f:
                for s in selected:
                    f.write(s + '\n')
            report_lines.append(f'Boruta selected {len(selected)} features; written to {boruta_file}')
        except Exception as e:
            report_lines.append('Boruta run failed: ' + str(e))

    # Write overall report
    rpt = out_dir / 'feature_selection_report.txt'
    with open(rpt, 'w') as f:
        for r in report_lines:
            f.write(r + '\n')

    print('\n'.join(report_lines))
    print('\nReport written to', rpt)


if __name__ == '__main__':
    main()
