#!/usr/bin/env python3
"""
Precompute Permutation Importance feature analysis.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'  # Minimize Streamlit logging

import warnings
import logging
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# helper for robust json serialization of numpy/pandas scalars used by precompute scripts
import json_helpers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-repeats', type=int, default=10)
    parser.add_argument('--output', type=str, default='data_files/precomputed/permutation_results.json')
    args = parser.parse_args()
    
    print("Loading data...")
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    
    from raceAnalysis import get_features_and_target, get_preprocessor_position
    
    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)
    
    X, y = get_features_and_target(data)
    
    # Clean data
    mask = y.notnull() & np.isfinite(y)
    X_clean, y_clean = X[mask], y[mask]
    
    print("Training model...")
    preprocessor = get_preprocessor_position(X_clean)
    X_prep = preprocessor.fit_transform(X_clean)
    
    model = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
    model.fit(X_prep, y_clean)
    
    print(f"Computing permutation importance (n_repeats={args.n_repeats})...")
    result = permutation_importance(model, X_prep, y_clean, n_repeats=args.n_repeats, random_state=42)
    
    importances = result.importances_mean
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
    
    # Create feature importance list
    feature_importance = [
        {
            'feature': feat,
            'importance': float(imp),
            'std': float(std)
        }
        for feat, imp, std in zip(feature_names, importances, result.importances_std)
    ]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_repeats': args.n_repeats,
            'total_features': len(feature_names)
        },
        'feature_importance': feature_importance,
        'top_20': feature_importance[:20]
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_helpers.safe_dump(output, output_path, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Top 5 features by permutation importance:")
    for i, item in enumerate(feature_importance[:5], 1):
        print(f"  {i}. {item['feature']}: {item['importance']:.4f}")

if __name__ == '__main__':
    main()
