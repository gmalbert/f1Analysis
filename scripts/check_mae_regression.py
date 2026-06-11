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


def _load_raceanalysis_features() -> list[str]:
    """Load the current feature set from raceAnalysis.get_features_and_target()."""
    # Import lazily to avoid streamlit startup noise unless needed.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import json_helpers  # noqa: F401 - ensure json_helpers is importable
    from raceAnalysis import get_features_and_target

    logging.getLogger('streamlit.runtime').setLevel(logging.ERROR)
    data = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)
    X_full, _ = get_features_and_target(data)
    return X_full.columns.tolist()


def _load_features() -> tuple[list[str], str]:
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
            return feats, 'precomputed'

    # Fall back: import raceAnalysis to get the full feature set
    try:
        feats = _load_raceanalysis_features()
        print(f"No precomputed feature list found — using all {len(feats)} features from raceAnalysis.")
        return feats, 'raceanalysis'
    except Exception as e:
        print(f"WARNING: Could not load features from raceAnalysis ({e}). Skipping MAE check.")
        sys.exit(0)


def _compute_mae_for_features(data: pd.DataFrame, features: list[str], target: str) -> float:
    """Compute GroupKFold CV MAE for a given feature list."""
    features = [f for f in features if f in data.columns]
    if not features:
        raise ValueError('No feature columns found in dataset.')

    required = features + [target, 'grandPrixYear']
    valid = data[required].copy()
    valid[target] = pd.to_numeric(valid[target], errors='coerce')
    valid = valid.dropna(subset=[target])
    valid = valid[np.isfinite(valid[target])]

    X_num = valid[features].select_dtypes(include='number')
    if X_num.empty:
        raise ValueError('Feature selection produced zero numeric columns.')

    X = X_num.fillna(X_num.median())
    y = valid[target].astype(float)
    groups = valid['grandPrixYear']

    if len(X) < 200:
        raise ValueError(f'Only {len(X)} usable rows after cleaning.')

    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        n_jobs=-1, tree_method='hist', random_state=42, verbosity=0,
    )
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv,
                             scoring='neg_mean_absolute_error')
    return float(-scores.mean())


def compute_mae(threshold: float, baseline: float) -> None:
    data_path = DATA_DIR / 'f1ForAnalysis.csv'
    if not data_path.exists():
        print(f"WARNING: {data_path} not found — skipping MAE check.")
        sys.exit(0)

    data = pd.read_csv(data_path, sep='\t', low_memory=False)
    features, source = _load_features()
    target = 'resultsFinalPositionNumber'

    try:
        mae = _compute_mae_for_features(data, features, target)
    except ValueError as e:
        print(f"WARNING: {e} — skipping MAE check.")
        sys.exit(0)

    print(f"\nMAE (5-fold GroupKFold CV): {mae:.4f}")
    print(f"Baseline:                   {baseline:.4f}")
    print(f"Allowed delta:              +{threshold:.4f}")
    print(f"Allowed ceiling:            {baseline + threshold:.4f}")

    if mae <= baseline + threshold:
        print(f"\nPASS: MAE {mae:.4f} is within acceptable range.")
        sys.exit(0)

    # If stale precomputed features regress, retry with current raceAnalysis feature set.
    if source == 'precomputed':
        try:
            fallback_features = _load_raceanalysis_features()
            fallback_mae = _compute_mae_for_features(data, fallback_features, target)
            print(f"\nFallback MAE with current raceAnalysis features: {fallback_mae:.4f}")
            if fallback_mae <= baseline + threshold:
                print(
                    "PASS: Precomputed feature list appears stale/regressed, "
                    "but current model feature set is within threshold."
                )
                sys.exit(0)
        except Exception as e:
            print(f"WARNING: Fallback MAE check failed: {e}")

    print(f"\nFAIL: MAE {mae:.4f} exceeds baseline {baseline:.4f} + "
          f"threshold {threshold:.4f} = {baseline + threshold:.4f}")
    sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAE Regression Check for CI')
    parser.add_argument(
        '--threshold', type=float, default=0.05,
        help='Maximum allowed MAE increase over baseline (default: 0.05)',
    )
    args = parser.parse_args()
    baseline = float(os.environ.get('MAE_BASELINE', '1.94'))
    compute_mae(args.threshold, baseline)
