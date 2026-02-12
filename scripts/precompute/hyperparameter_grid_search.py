#!/usr/bin/env python3
"""
Grid search hyperparameter optimization.
Precomputes optimal hyperparameters using exhaustive grid search.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data_files/precomputed/hyperparam_grid.json')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Grid Search Hyperparameter Optimization")
    print("=" * 60)
    
    print("\nLoading data...")
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    
    from raceAnalysis import get_features_and_target, get_preprocessor_position
    X, y = get_features_and_target(data)
    
    # Clean data
    mask = y.notnull() & np.isfinite(y)
    X_clean, y_clean = X[mask], y[mask]
    
    # Get season groups for stratified CV
    season_groups = data.loc[y_clean.index, 'year'] if 'year' in data.columns else None
    
    print(f"Data shape: {X_clean.shape}")
    
    # Define parameter grid
    param_grid = {
        'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 4, 5, 6, 7],
        'regressor__min_child_weight': [1, 3, 5, 7],
    }
    
    print("\nRunning grid search...")
    print(f"Parameters: {param_grid}")
    
    pipeline = Pipeline([
        ('preprocessor', get_preprocessor_position(X_clean)),
        ('regressor', XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, tree_method='hist'))
    ])
    
    # Use GroupKFold if season data available, else regular CV
    if season_groups is not None:
        cv = GroupKFold(n_splits=5)
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, groups=season_groups, 
                                    scoring='neg_mean_absolute_error', n_jobs=1, verbose=2)
    else:
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, 
                                    scoring='neg_mean_absolute_error', n_jobs=1, verbose=2)
    
    grid_search.fit(X_clean, y_clean)
    
    # Extract results
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'data_rows': len(X_clean),
            'n_combinations': len(grid_search.cv_results_['params'])
        },
        'best_params': grid_search.best_params_,
        'best_mae': float(-grid_search.best_score_),
        'all_results': [
            {
                'params': params,
                'mae': float(-score),
                'rank': int(rank)
            }
            for params, score, rank in zip(
                grid_search.cv_results_['params'],
                grid_search.cv_results_['mean_test_score'],
                grid_search.cv_results_['rank_test_score']
            )
        ]
    }
    
    # Sort by MAE
    results['all_results'].sort(key=lambda x: x['mae'])
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Grid search complete!")
    print(f"Best MAE: {results['best_mae']:.4f}")
    print(f"Best params: {results['best_params']}")
    print(f"Results saved to {output_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
