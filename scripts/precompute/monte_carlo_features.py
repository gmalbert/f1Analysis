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

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_data():
    """Load the main analysis dataset."""
    data_path = Path('data_files/f1ForAnalysis.csv')
    data = pd.read_csv(data_path, sep='\t', low_memory=False)
    return data

def get_features_and_target(data):
    """Extract features and target from data."""
    from raceAnalysis import get_features_and_target as get_ft
    return get_ft(data)

def monte_carlo_feature_selection(
    X, y, n_trials=1000, min_features=8, max_features=15, cv=10, random_state=42
):
    """
    Run Monte Carlo feature subset search with cross-validation.
    Returns sorted results by MAE (best first).
    """
    import random
    
    results = []
    feature_names = X.columns.tolist()
    rng = random.Random(random_state)
    tested_subsets = set()
    
    print(f"Starting Monte Carlo search: {n_trials} trials, {min_features}-{max_features} features")
    
    for i in range(n_trials):
        if (i + 1) % 100 == 0:
            print(f"  Trial {i + 1}/{n_trials}...")
        
        # Generate random feature subset
        k = rng.randint(min_features, max_features)
        subset = tuple(sorted(rng.sample(feature_names, k=k)))
        
        if subset in tested_subsets:
            continue
        tested_subsets.add(subset)
        
        # Prepare data
        X_subset = X[list(subset)].copy()
        
        # Convert object columns to category codes
        for col in X_subset.select_dtypes(include='object').columns:
            X_subset[col] = X_subset[col].astype('category').cat.codes
        for col in X_subset.select_dtypes(include='Int64').columns:
            X_subset[col] = X_subset[col].astype(float)
        X_subset = X_subset.fillna(X_subset.mean(numeric_only=True))
        
        # Clean y
        mask = y.notnull() & np.isfinite(y)
        X_clean = X_subset[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 100:
            continue
        
        # Cross-validation
        model = XGBRegressor(
            n_estimators=100, max_depth=4, n_jobs=-1, 
            tree_method='hist', random_state=42
        )
        
        try:
            mae_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='neg_mean_absolute_error')
            mae = -mae_scores.mean()
            mae_std = mae_scores.std()
            
            rmse_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='neg_root_mean_squared_error')
            rmse = -rmse_scores.mean()
            
            r2_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring='r2')
            r2 = r2_scores.mean()
            
            results.append({
                'features': list(subset),
                'n_features': len(subset),
                'mae': float(mae),
                'mae_std': float(mae_std),
                'rmse': float(rmse),
                'r2': float(r2)
            })
        except Exception as e:
            print(f"  Error with subset: {e}")
            continue
    
    # Sort by MAE
    results = sorted(results, key=lambda x: x['mae'])
    return results

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Feature Selection')
    parser.add_argument('--n-trials', type=int, default=1000)
    parser.add_argument('--min-features', type=int, default=8)
    parser.add_argument('--max-features', type=int, default=15)
    parser.add_argument('--cv-folds', type=int, default=10)
    parser.add_argument('--output', type=str, default='data_files/precomputed/monte_carlo_results.json')
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    data = load_data()
    X, y = get_features_and_target(data)
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    
    results = monte_carlo_feature_selection(
        X, y,
        n_trials=args.n_trials,
        min_features=args.min_features,
        max_features=args.max_features,
        cv=args.cv_folds
    )
    
    # Compute feature frequency in top 20
    top_features = [f for r in results[:20] for f in r['features']]
    feature_counts = Counter(top_features)
    
    # Build output
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_trials': args.n_trials,
            'min_features': args.min_features,
            'max_features': args.max_features,
            'cv_folds': args.cv_folds,
            'total_subsets_tested': len(results)
        },
        'best_result': results[0] if results else None,
        'top_20_results': results[:20],
        'feature_frequency_top_20': dict(feature_counts.most_common()),
        'all_results': results
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    if output['best_result']:
        print(f"Best MAE: {output['best_result']['mae']:.4f}")
        print(f"Best features ({len(output['best_result']['features'])}): {output['best_result']['features']}")

if __name__ == '__main__':
    main()
