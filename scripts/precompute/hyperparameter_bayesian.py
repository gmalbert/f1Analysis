#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization using Optuna.
Precomputes optimal hyperparameters for XGBoost and LightGBM.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'  # Minimize Streamlit logging

import warnings
import logging
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def optimize_xgboost(X, y, season_groups, n_trials=100):
    """Optimize XGBoost hyperparameters using Optuna."""
    from raceAnalysis import get_preprocessor_position
    
    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)
    
    print(f"  Optimizing XGBoost ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        pipeline = Pipeline([
            ('preprocessor', get_preprocessor_position(X)),
            ('regressor', XGBRegressor(
                n_estimators=200,
                n_jobs=-1,
                tree_method='hist',
                random_state=42,
                **params
            ))
        ])
        
        if season_groups is not None:
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(pipeline, X, y, cv=cv, groups=season_groups, 
                                     scoring='neg_mean_absolute_error', n_jobs=1)
        else:
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
        
        return -scores.mean()
    
    study = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_mae': study.best_value,
        'n_trials': n_trials,
        'optimization_history': [
            {'trial': t.number, 'mae': t.value, 'params': t.params}
            for t in study.trials if t.value is not None
        ]
    }


def optimize_lightgbm(X, y, season_groups, n_trials=100):
    """Optimize LightGBM hyperparameters."""
    from lightgbm import LGBMRegressor
    from raceAnalysis import get_preprocessor_position
    
    print(f"  Optimizing LightGBM ({n_trials} trials)...")
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        pipeline = Pipeline([
            ('preprocessor', get_preprocessor_position(X)),
            ('regressor', LGBMRegressor(
                n_estimators=200,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
                **params
            ))
        ])
        
        if season_groups is not None:
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(pipeline, X, y, cv=cv, groups=season_groups,
                                     scoring='neg_mean_absolute_error', n_jobs=1)
        else:
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
        
        return -scores.mean()
    
    study = optuna.create_study(direction='minimize', study_name='lightgbm_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_mae': study.best_value,
        'n_trials': n_trials
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--output', type=str, default='data_files/precomputed/hyperparam_bayesian.json')
    parser.add_argument('--model-types', nargs='+', default=['xgboost', 'lightgbm'])
    args = parser.parse_args()
    
    print("=" * 60)
    print("Bayesian Hyperparameter Optimization")
    print("=" * 60)
    
    print("\nLoading data...")
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    
    from raceAnalysis import get_features_and_target
    X, y = get_features_and_target(data)
    
    # Clean data
    mask = y.notnull() & np.isfinite(y)
    X_clean, y_clean = X[mask], y[mask]
    
    # Get season groups for stratified CV
    season_groups = data.loc[y_clean.index, 'year'] if 'year' in data.columns else None
    
    print(f"Data shape: {X_clean.shape}")
    print(f"Trials per model: {args.n_trials}")
    
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_trials': args.n_trials,
            'data_rows': len(X_clean)
        },
        'optimizations': {}
    }
    
    if 'xgboost' in args.model_types:
        print("\n" + "-" * 60)
        print("XGBoost Optimization")
        print("-" * 60)
        results['optimizations']['xgboost'] = optimize_xgboost(X_clean, y_clean, season_groups, args.n_trials)
        print(f"  [OK] Best MAE: {results['optimizations']['xgboost']['best_mae']:.4f}")
    
    if 'lightgbm' in args.model_types:
        print("\n" + "-" * 60)
        print("LightGBM Optimization")
        print("-" * 60)
        results['optimizations']['lightgbm'] = optimize_lightgbm(X_clean, y_clean, season_groups, args.n_trials)
        print(f"  [OK] Best MAE: {results['optimizations']['lightgbm']['best_mae']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
