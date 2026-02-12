#!/usr/bin/env python3
"""
Precompute Boruta feature selection results.
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
from boruta import BorutaPy
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-iter', type=int, default=200)
    parser.add_argument('--output', type=str, default='data_files/precomputed/boruta_results.json')
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
    for col in X_clean.select_dtypes(include='Int64').columns:
        X_clean[col] = X_clean[col].astype(float)
    
    X_clean = X_clean.fillna(X_clean.mean(numeric_only=True))
    y_clean = y_clean.fillna(y_clean.mean())
    
    # Ensure y is 1D array
    if isinstance(y_clean, pd.DataFrame):
        y_clean = y_clean.iloc[:, 0]
    y_clean = np.asarray(y_clean).ravel()
    
    print(f"Running Boruta with max_iter={args.max_iter}...")
    
    estimator = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42)
    boruta_selector = BorutaPy(estimator, n_estimators='auto', verbose=0, random_state=42, max_iter=args.max_iter)
    boruta_selector.fit(X_clean.values, y_clean)
    
    selected_features = X_clean.columns[boruta_selector.support_].tolist()
    ranking = boruta_selector.ranking_.tolist()
    
    # Create feature ranking dataframe
    feature_ranking = [
        {'feature': feat, 'rank': int(rank), 'selected': select}
        for feat, rank, select in zip(X_clean.columns, ranking, boruta_selector.support_)
    ]
    feature_ranking.sort(key=lambda x: x['rank'])
    
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'max_iter': args.max_iter,
            'total_features': len(X_clean.columns),
            'n_selected': len(selected_features)
        },
        'selected_features': selected_features,
        'feature_ranking': feature_ranking
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Selected {len(selected_features)} features")

if __name__ == '__main__':
    main()
