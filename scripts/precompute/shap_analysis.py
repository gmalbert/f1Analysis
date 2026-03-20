#!/usr/bin/env python3
"""
Precompute SHAP (SHapley Additive exPlanations) analysis.
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
import shap
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# helper for robust json serialization of numpy/pandas scalars used by precompute scripts
import json_helpers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-samples', type=int, default=1000, help='Max samples for SHAP calculation')
    parser.add_argument('--output', type=str, default='data_files/precomputed/shap_results.json')
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
    
    # Explicitly set base_score=0.5 to prevent XGBoost 2.x from storing it in
    # the '[9.xxE0]' bracketed string format that SHAP's TreeExplainer cannot parse.
    model = XGBRegressor(n_estimators=100, max_depth=4, n_jobs=-1, tree_method='hist', random_state=42, base_score=0.5)
    model.fit(X_prep, y_clean)
    
    # Densify sparse matrices (produced by OneHotEncoder) before SHAP — TreeExplainer
    # requires dense numpy arrays in most SHAP versions.
    import scipy.sparse
    if scipy.sparse.issparse(X_prep):
        X_prep = X_prep.toarray()

    # Limit samples for SHAP if dataset is large
    if len(X_prep) > args.max_samples:
        print(f"Sampling {args.max_samples} rows for SHAP analysis...")
        sample_idx = np.random.choice(len(X_prep), args.max_samples, replace=False)
        X_shap = X_prep[sample_idx]
    else:
        X_shap = X_prep
    
    print("Computing SHAP values...")
    # XGBoost 2.x may still store base_score as '[9.xxE0]' in the booster JSON
    # config even when base_score was explicitly set.  Patch the config to
    # normalise it before handing off to SHAP's TreeExplainer.
    import json as _json
    import re as _re
    try:
        booster = model.get_booster()
        config = _json.loads(booster.save_config())
        lmp = config.get('learner', {}).get('learner_model_param', {})
        raw_bs = lmp.get('base_score', '')
        if isinstance(raw_bs, str):
            # Strip any surrounding brackets/whitespace and parse the numeric token
            cleaned = raw_bs.strip()
            m = _re.search(r'[\[\(]?([0-9eE.+\-]+)[\]\)]?', cleaned)
            if m:
                try:
                    lmp['base_score'] = str(float(m.group(1)))
                    booster.load_config(_json.dumps(config))
                except ValueError:
                    pass  # leave as-is; TreeExplainer may still work
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_shap)
    except Exception as e1:
        print(f"  TreeExplainer (patched booster) failed: {e1}")
        try:
            # Second fallback: pass raw XGBRegressor with check_additivity off
            explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
            shap_values = explainer.shap_values(X_shap, check_additivity=False)
        except Exception as e2:
            print(f"  TreeExplainer fallback also failed: {e2}")
            # Final fallback: model-agnostic permutation explainer (slow but always works)
            background = X_shap[:min(50, len(X_shap))]
            explainer = shap.Explainer(model.predict, shap.maskers.Independent(background, max_samples=50))
            shap_values = explainer(X_shap[:min(200, len(X_shap))]).values
    
    # Get feature names
    feature_names = preprocessor.get_feature_names_out()
    feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
    
    # Calculate mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create feature importance list
    feature_importance = [
        {
            'feature': feat,
            'mean_abs_shap': float(shap_val)
        }
        for feat, shap_val in zip(feature_names, mean_abs_shap)
    ]
    feature_importance.sort(key=lambda x: x['mean_abs_shap'], reverse=True)
    
    output = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_samples_analyzed': len(X_shap),
            'total_features': len(feature_names)
        },
        'feature_importance': feature_importance,
        'top_20': feature_importance[:20]
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    json_helpers.safe_dump(output, output_path, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print(f"Top 5 features by SHAP importance:")
    for i, item in enumerate(feature_importance[:5], 1):
        print(f"  {i}. {item['feature']}: {item['mean_abs_shap']:.4f}")

if __name__ == '__main__':
    main()
