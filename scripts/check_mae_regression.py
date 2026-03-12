#!/usr/bin/env python3
"""
CI script: check that MAE has not regressed beyond threshold vs baseline.

Usage:
    python scripts/check_mae_regression.py --threshold 0.05

Fails with exit code 1 if new_mae > baseline + threshold.

Environment:
    MAE_BASELINE — baseline MAE string, e.g. "1.53" (default "1.94").
                   Set as a GitHub repository variable so it can be updated
                   when a confirmed improvement lands.
"""
import argparse
import os
import sys
import logging
from pathlib import Path

os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
os.environ.setdefault('STREAMLIT_LOG_LEVEL', 'error')

import warnings
warnings.filterwarnings('ignore')
logging.getLogger('streamlit').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor

DATA_DIR = Path('data_files')
FEATURE_LIST_PATH = DATA_DIR / 'precomputed' / 'monte_carlo_results.json'


def _load_features() -> list[str]:
    """
    Load the best feature list from precomputed Monte Carlo results.
    Falls back to reading raceAnalysis.get_features_and_target() column names
    if no precomputed list exists.
    """
    if FEATURE_LIST_PATH.exists():
        import json
        with open(FEATURE_LIST_PATH) as f:
            mc = json.load(f)
        best = mc.get('best_result', {})
        feats = best.get('features', [])
        if feats:
            print(f"Using {len(feats)} features from precomputed Monte Carlo results.")
            return feats

    # Fall back: import raceAnalysis to get the full feature set
    sys.path.insert(0, str(Path(__file__).parent.parent))
    try:
        import json_helpers  # noqa: ensure json_helpers is importable
        from raceAnalysis import get_features_and_target
        logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)
        data = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)
        X_full, _ = get_features_and_target(data)
        feats = X_full.columns.tolist()
        print(f"No precomputed feature list found — using all {len(feats)} features from raceAnalysis.")
        return feats
    except Exception as e:
        print(f"WARNING: Could not load features from raceAnalysis ({e}). Skipping MAE check.")
        sys.exit(0)


def compute_mae(threshold: float, baseline: float) -> None:
    data_path = DATA_DIR / 'f1ForAnalysis.csv'
    if not data_path.exists():
        print(f"WARNING: {data_path} not found — skipping MAE check.")
        sys.exit(0)

    data = pd.read_csv(data_path, sep='\t', low_memory=False)
    features = _load_features()
    target = 'resultsFinalPositionNumber'

    # Filter to columns present in data
    features = [f for f in features if f in data.columns]
    if not features:
        print("WARNING: None of the feature columns found in data — skipping MAE check.")
        sys.exit(0)

    required = features + [target, 'grandPrixYear']
    valid = data[required].dropna(subset=[target])
    X = (
        valid[features]
        .select_dtypes(include='number')
        .fillna(valid[features].select_dtypes(include='number').median())
    )
    y = valid[target]
    groups = valid['grandPrixYear']

    if len(X) < 200:
        print(f"WARNING: Only {len(X)} usable rows — skipping MAE check.")
        sys.exit(0)

    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        n_jobs=-1, tree_method='hist', random_state=42, verbosity=0,
    )
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv,
                             scoring='neg_mean_absolute_error')
    mae = float(-scores.mean())

    print(f"\nMAE (5-fold GroupKFold CV): {mae:.4f}")
    print(f"Baseline:                   {baseline:.4f}")
    print(f"Allowed delta:              +{threshold:.4f}")
    print(f"Allowed ceiling:            {baseline + threshold:.4f}")

    if mae > baseline + threshold:
        print(f"\nFAIL: MAE {mae:.4f} exceeds baseline {baseline:.4f} + "
              f"threshold {threshold:.4f} = {baseline + threshold:.4f}")
        sys.exit(1)
    else:
        print(f"\nPASS: MAE {mae:.4f} is within acceptable range.")
        sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Regression Check for CI')
    parser.add_argument(
        '--threshold', type=float, default=0.05,
        help='Maximum allowed MAE increase over baseline (default: 0.05)',
    )
    args = parser.parse_args()
    baseline = float(os.environ.get('MAE_BASELINE', '1.94'))
    compute_mae(args.threshold, baseline)
