#!/usr/bin/env python3
"""
Precompute RFE (Recursive Feature Elimination) feature selection results.
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
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# helper for robust json serialization of numpy/pandas scalars used by precompute scripts
import json_helpers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-features', type=int, default=15)
    parser.add_argument('--output', type=str, default='data_files/precomputed/rfe_results.json')
    args = parser.parse_args()
    
    print("Loading data...")
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    
    from raceAnalysis import get_features_and_target
    
    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)
    
    X, y = get_features_and_target(data)
    
    # Clean data
    mask = y.notnull() & np.isfinite(y)
    X_clean, y_clean = X[mask], y[mask]
    
    # Convert object columns
    for col in X_clean.select_dtypes(include='object').columns:
        X_clean[col] = X_clean[col].astype('category').cat.codes
    
    print(f"Running RFE with {args.n_features} features...")
    
    estimator = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
    rfe = RFE(estimator, n_features_to_select=args.n_features, step=1)
    rfe.fit(X_clean, y_clean)
    
    selected_features = X_clean.columns[rfe.support_].tolist()
    ranking = rfe.ranking_.tolist()
    
    # Create feature ranking dataframe
    feature_ranking = [
        {'feature': feat, 'rank': int(rank), 'selected': rank == 1}
        for feat, rank in zip(X_clean.columns, ranking)
    ]
    feature_ranking.sort(key=lambda x: x['rank'])
    
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_features_requested': args.n_features,
            'total_features': len(X_clean.columns)
        },
        'selected_features': selected_features,
        'feature_ranking': feature_ranking
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_helpers.safe_dump(output, output_path, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Selected {len(selected_features)} features: {selected_features}")

if __name__ == '__main__':
    main()
