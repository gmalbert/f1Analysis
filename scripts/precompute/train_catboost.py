#!/usr/bin/env python3
"""
Train CatBoost models for position prediction.
Run via GitHub Actions or locally for model artifact generation.
"""
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'  # Minimize Streamlit logging

import warnings
import logging
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def train_catboost_models():
    """Train all CatBoost models and save to data_files/models/catboost/"""
    
    print("=" * 60)
    print("CatBoost Model Training")
    print("=" * 60)
    
    output_dir = Path("data_files/models/catboost")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from raceAnalysis import (
        load_data,
        CACHE_VERSION,
        train_and_evaluate_model
    )
    
    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)
    
    print(f"\nLoading data (CACHE_VERSION={CACHE_VERSION})...")
    data, _ = load_data(10000, CACHE_VERSION)
    
    # Apply column renaming
    if 'constructorName_results_with_qualifying' in data.columns:
        data.rename(columns={'constructorName_results_with_qualifying': 'constructorName'}, inplace=True)
    elif 'constructorName_qualifying' in data.columns:
        data.rename(columns={'constructorName_qualifying': 'constructorName'}, inplace=True)
    
    print(f"Data loaded: {data.shape}")
    
    # Train position prediction model with CatBoost
    print("\n" + "-" * 60)
    print("Training Position Prediction Model (CatBoost)")
    print("-" * 60)
    
    model, mse, r2, mae, mean_err, evals_result, preprocessor = train_and_evaluate_model(
        data, 
        early_stopping_rounds=20,
        model_type="CatBoost"
    )
    
    position_artifact = {
        'model': model,
        'preprocessor': preprocessor,
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'mean_err': mean_err,
        'evals_result': evals_result,
        'cache_version': CACHE_VERSION,
        'model_type': 'CatBoost',
        'trained_at': datetime.now().isoformat()
    }
    
    with open(output_dir / 'position_model.pkl', 'wb') as f:
        pickle.dump(position_artifact, f)
    print(f"[OK] Position model saved (MAE: {mae:.4f})")
    
    # Save metadata
    metadata = {
        'cache_version': CACHE_VERSION,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'CatBoost',
        'data_rows': len(data),
        'models': {
            'position': {
                'mae': float(mae),
                'mse': float(mse),
                'r2': float(r2)
            }
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("CatBoost Training Complete!")
    print("=" * 60)
    print(f"Position MAE: {mae:.4f}")
    
    return metadata

if __name__ == '__main__':
    try:
        metadata = train_catboost_models()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
