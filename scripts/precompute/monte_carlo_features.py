#!/usr/bin/env python3
"""
Precompute Monte Carlo feature selection results.
Run via GitHub Actions or locally for expensive feature subset search.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'  # Minimize Streamlit logging

import warnings
import logging
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# 4A: Production-equivalent model configs for feature selection (ROADMAP-4A)
# Using stronger models means selected features generalise to production.
# ---------------------------------------------------------------------------
MONTE_CARLO_MODEL_XGB = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    n_jobs=-1,
    tree_method='hist',
    random_state=42,
    verbosity=0,
)

MONTE_CARLO_MODEL_LGBM = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

# ---------------------------------------------------------------------------
# 4B: Tiered search config (ROADMAP-4B)
# Stage 1 wide-eliminates weak features; Stage 2 refines within survivors.
# ---------------------------------------------------------------------------
STAGE_1_CONFIG = {
    'min_features': 20,
    'max_features': 60,
    'n_trials': 500,
    'description': 'Wide search — eliminate weak features',
}

STAGE_2_CONFIG = {
    'min_features': 15,
    'max_features': 35,
    'n_trials': 500,
    'description': 'Narrow refinement — precision optimisation',
}

# helper for robust json serialization of numpy/pandas scalars used by precompute scripts
import json_helpers

def load_data():
    """Load the main analysis dataset."""
    data_path = Path('data_files/f1ForAnalysis.csv')
    data = pd.read_csv(data_path, sep='\t', low_memory=False)
    return data

def get_features_and_target(data):
    """Extract features and target from data."""
    from raceAnalysis import get_features_and_target as get_ft
    
    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)
    
    return get_ft(data)

def _prepare_subset(X, y, subset):
    """Prepare a feature subset for cross-validation. Returns (X_clean, y_clean) or (None, None)."""
    X_subset = X[list(subset)].copy()
    for col in X_subset.select_dtypes(include='object').columns:
        X_subset[col] = X_subset[col].astype('category').cat.codes
    for col in X_subset.select_dtypes(include='Int64').columns:
        X_subset[col] = X_subset[col].astype(float)
    X_subset = X_subset.fillna(X_subset.mean(numeric_only=True))
    mask = y.notnull() & np.isfinite(y)
    X_clean = X_subset[mask]
    y_clean = y[mask]
    if len(X_clean) < 100:
        return None, None
    return X_clean, y_clean


def _run_cv_trial(model, X_clean, y_clean, subset, cv, trial_idx, stage):
    """Run cross-validation for one feature subset. Returns a result dict or None."""
    try:
        mae_scores = cross_val_score(model, X_clean, y_clean, cv=cv,
                                     scoring='neg_mean_absolute_error')
        mae = float(-mae_scores.mean())
        mae_std = float(mae_scores.std())
        rmse_scores = cross_val_score(model, X_clean, y_clean, cv=cv,
                                      scoring='neg_root_mean_squared_error')
        rmse = float(-rmse_scores.mean())
        r2_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='r2')
        r2 = float(r2_scores.mean())
        return {
            'trial': trial_idx,
            'stage': stage,
            'features': list(subset),
            'n_features': len(subset),
            'mae': mae,
            'mae_std': mae_std,
            'rmse': rmse,
            'r2': r2,
        }
    except Exception as e:
        print(f"  Error with subset (trial {trial_idx}): {e}")
        return None


def monte_carlo_feature_selection(
    X, y, n_trials=1000, min_features=8, max_features=15, cv=10, random_state=42,
    model=None,
):
    """
    Run Monte Carlo feature subset search with cross-validation.

    4A: Uses a production-equivalent XGBoost model by default instead of the
        previous weak (n_estimators=100, depth=4) version.

    Returns sorted results by MAE (best first).
    """
    import random

    if model is None:
        import copy
        model = copy.deepcopy(MONTE_CARLO_MODEL_XGB)

    results = []
    feature_names = X.columns.tolist()
    rng = random.Random(random_state)
    tested_subsets = set()

    print(f"Starting Monte Carlo search: {n_trials} trials, {min_features}-{max_features} features")

    for i in range(n_trials):
        if (i + 1) % 100 == 0:
            print(f"  Trial {i + 1}/{n_trials}...")
        k = rng.randint(min_features, min(max_features, len(feature_names)))
        subset = tuple(sorted(rng.sample(feature_names, k=k)))
        if subset in tested_subsets:
            continue
        tested_subsets.add(subset)
        X_clean, y_clean = _prepare_subset(X, y, subset)
        if X_clean is None:
            continue
        result = _run_cv_trial(model, X_clean, y_clean, subset, cv, i, stage='single')
        if result:
            results.append(result)

    return sorted(results, key=lambda x: x['mae'])


def run_tiered_monte_carlo(X, y, n_trials_per_stage=500, cv=10, random_state=42):
    """
    4B: Two-stage tiered Monte Carlo feature selection (ROADMAP-4B).

    Stage 1 (wide): sample 20–60 features over n_trials_per_stage trials to
        eliminate clearly-weak features.
    Stage 2 (narrow): restrict to top-50% candidate pool from Stage 1 then
        refine with 15–35 features.

    Also runs a secondary LightGBM pass on the Stage-2 winner to check
    agreement (4A secondary run).

    Returns a dict with 'stage1_results', 'stage2_results', 'best_result',
    'candidate_pool', 'lgbm_agreement', and 'run_log' for the dashboard.
    """
    import random
    import copy

    rng = random.Random(random_state)
    xgb_model = copy.deepcopy(MONTE_CARLO_MODEL_XGB)
    lgbm_model = copy.deepcopy(MONTE_CARLO_MODEL_LGBM)
    feature_names = X.columns.tolist()

    all_trials = []  # for run_log

    # ---- Stage 1: wide search -----------------------------------------------
    print(f"\n[Stage 1] {STAGE_1_CONFIG['description']}")
    print(f"  {n_trials_per_stage} trials, "
          f"{STAGE_1_CONFIG['min_features']}–{STAGE_1_CONFIG['max_features']} features")
    stage1_results = []
    tested = set()
    for i in range(n_trials_per_stage):
        if (i + 1) % 100 == 0:
            print(f"  Stage 1 trial {i + 1}/{n_trials_per_stage}…")
        k = rng.randint(STAGE_1_CONFIG['min_features'],
                        min(STAGE_1_CONFIG['max_features'], len(feature_names)))
        subset = tuple(sorted(rng.sample(feature_names, k=k)))
        if subset in tested:
            continue
        tested.add(subset)
        X_clean, y_clean = _prepare_subset(X, y, subset)
        if X_clean is None:
            continue
        res = _run_cv_trial(xgb_model, X_clean, y_clean, subset, cv, i, stage='stage1')
        if res:
            stage1_results.append(res)
            all_trials.append({'trial': i, 'stage': 'stage1',
                                'n_features': res['n_features'], 'mae': res['mae']})

    stage1_results.sort(key=lambda x: x['mae'])
    # Build candidate pool: union of features from top-50% of Stage-1 runs
    cutoff = max(1, len(stage1_results) // 2)
    candidate_pool = list({f for r in stage1_results[:cutoff] for f in r['features']})
    print(f"\n[Stage 1 done] Best MAE: {stage1_results[0]['mae']:.4f} | "
          f"Candidate pool: {len(candidate_pool)} features")

    # ---- Stage 2: narrow refinement -----------------------------------------
    print(f"\n[Stage 2] {STAGE_2_CONFIG['description']}")
    print(f"  {n_trials_per_stage} trials, "
          f"{STAGE_2_CONFIG['min_features']}–{STAGE_2_CONFIG['max_features']} features")
    stage2_results = []
    tested2 = set()
    for i in range(n_trials_per_stage):
        if (i + 1) % 100 == 0:
            print(f"  Stage 2 trial {i + 1}/{n_trials_per_stage}…")
        k = rng.randint(STAGE_2_CONFIG['min_features'],
                        min(STAGE_2_CONFIG['max_features'], len(candidate_pool)))
        subset = tuple(sorted(rng.sample(candidate_pool, k=k)))
        if subset in tested2:
            continue
        tested2.add(subset)
        X_clean, y_clean = _prepare_subset(X, y, subset)
        if X_clean is None:
            continue
        res = _run_cv_trial(xgb_model, X_clean, y_clean, subset, cv,
                            n_trials_per_stage + i, stage='stage2')
        if res:
            stage2_results.append(res)
            all_trials.append({'trial': n_trials_per_stage + i, 'stage': 'stage2',
                                'n_features': res['n_features'], 'mae': res['mae']})

    stage2_results.sort(key=lambda x: x['mae'])
    best_xgb = stage2_results[0] if stage2_results else (stage1_results[0] if stage1_results else None)

    # ---- LightGBM agreement check on Stage-2 best subset -------------------
    lgbm_agreement = None
    if best_xgb:
        print(f"\n[LightGBM agreement] Re-evaluating Stage-2 best subset with LightGBM…")
        X_clean, y_clean = _prepare_subset(X, y, tuple(best_xgb['features']))
        if X_clean is not None:
            lgbm_res = _run_cv_trial(lgbm_model, X_clean, y_clean,
                                     tuple(best_xgb['features']), cv,
                                     trial_idx=-1, stage='lgbm_agreement')
            lgbm_agreement = lgbm_res
            if lgbm_res:
                print(f"  XGB MAE: {best_xgb['mae']:.4f}  |  LGBM MAE: {lgbm_res['mae']:.4f}")

    best_result = best_xgb
    print(f"\n[Tiered MC done] Best MAE: {best_result['mae'] if best_result else 'N/A':.4f}")

    return {
        'stage1_results': stage1_results,
        'stage2_results': stage2_results,
        'best_result': best_result,
        'candidate_pool': candidate_pool,
        'lgbm_agreement': lgbm_agreement,
        'run_log': {
            'trials': all_trials,
            'best_trial': best_result.get('trial') if best_result else None,
            'best_mae': best_result.get('mae') if best_result else None,
            'total_trials': len(all_trials),
        },
    }

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Feature Selection (Tiered, 4A+4B)')
    parser.add_argument('--n-trials', type=int, default=1000,
                        help='Total trials (split evenly across Stage 1 and Stage 2 for tiered mode)')
    parser.add_argument('--min-features', type=int, default=8,
                        help='Minimum features per trial (only used in --simple mode)')
    parser.add_argument('--max-features', type=int, default=15,
                        help='Maximum features per trial (only used in --simple mode)')
    parser.add_argument('--cv-folds', type=int, default=10)
    parser.add_argument('--output', type=str, default='data_files/precomputed/monte_carlo_results.json')
    parser.add_argument('--simple', action='store_true',
                        help='Run original single-stage search instead of tiered')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    data = load_data()
    X, y = get_features_and_target(data)
    print(f"Data shape: X={X.shape}, y={y.shape}")

    if args.simple:
        # ---- Legacy single-stage path ----------------------------------------
        results = monte_carlo_feature_selection(
            X, y,
            n_trials=args.n_trials,
            min_features=args.min_features,
            max_features=args.max_features,
            cv=args.cv_folds,
        )
        top_features = [f for r in results[:20] for f in r['features']]
        feature_counts = Counter(top_features)
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'mode': 'simple',
                'n_trials': args.n_trials,
                'min_features': args.min_features,
                'max_features': args.max_features,
                'cv_folds': args.cv_folds,
                'total_subsets_tested': len(results),
            },
            'best_result': results[0] if results else None,
            'top_20_results': results[:20],
            'feature_frequency_top_20': dict(feature_counts.most_common()),
            'all_results': results,
            'run_log': {
                'trials': [{'trial': i, 'stage': 'single',
                            'n_features': r['n_features'], 'mae': r['mae']}
                           for i, r in enumerate(results)],
                'best_trial': 0 if results else None,
                'best_mae': results[0]['mae'] if results else None,
                'total_trials': len(results),
            },
        }
    else:
        # ---- Tiered path (4B) -----------------------------------------------
        n_per_stage = max(100, args.n_trials // 2)
        tiered = run_tiered_monte_carlo(X, y, n_trials_per_stage=n_per_stage,
                                        cv=args.cv_folds)
        all_results = tiered['stage2_results'] or tiered['stage1_results']
        top_features = [f for r in all_results[:20] for f in r['features']]
        feature_counts = Counter(top_features)
        output = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'mode': 'tiered',
                'n_trials_per_stage': n_per_stage,
                'cv_folds': args.cv_folds,
                'stage1_config': STAGE_1_CONFIG,
                'stage2_config': STAGE_2_CONFIG,
                'stage1_tested': len(tiered['stage1_results']),
                'stage2_tested': len(tiered['stage2_results']),
                'candidate_pool_size': len(tiered['candidate_pool']),
            },
            'best_result': tiered['best_result'],
            'lgbm_agreement': tiered['lgbm_agreement'],
            'top_20_results': all_results[:20],
            'feature_frequency_top_20': dict(feature_counts.most_common()),
            'all_results': all_results,
            'run_log': tiered['run_log'],
        }

    json_helpers.safe_dump(output, output_path, indent=2)

    # Also write run_log to a dedicated file for the dashboard (4F)
    run_log_path = output_path.parent / 'monte_carlo_run_log.json'
    json_helpers.safe_dump(output.get('run_log', {}), run_log_path, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"Run log saved to {run_log_path}")
    if output.get('best_result'):
        best = output['best_result']
        print(f"Best MAE: {best['mae']:.4f}")
        print(f"Best features ({len(best['features'])}): {best['features']}")


if __name__ == '__main__':
    main()
