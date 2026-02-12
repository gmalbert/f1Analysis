#!/usr/bin/env python3
"""
Train XGBoost models for position, DNF, and safety car prediction.
Run via GitHub Actions or locally for model artifact generation.
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

# Suppress Streamlit warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import warnings
warnings.filterwarnings("ignore", message=".*No runtime found.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def train_xgboost_models():
    """Train all XGBoost models and save to data_files/models/xgboost/"""
    
    print("=" * 60)
    print("XGBoost Model Training")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("data_files/models/xgboost")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Import training functions
    from raceAnalysis import (
        load_data,
        CACHE_VERSION,
        train_and_evaluate_model,
        train_and_evaluate_dnf_model,
        train_and_evaluate_safetycar_model
    )
    
    print(f"\nLoading data (CACHE_VERSION={CACHE_VERSION})...")
    data, _ = load_data(10000, CACHE_VERSION)
    
    # Apply column renaming logic
    print("Applying column renaming...")
    if 'constructorName_results_with_qualifying' in data.columns:
        data.rename(columns={'constructorName_results_with_qualifying': 'constructorName'}, inplace=True)
    elif 'constructorName_qualifying' in data.columns:
        data.rename(columns={'constructorName_qualifying': 'constructorName'}, inplace=True)
    
    if 'best_qual_time_results_with_qualifying' in data.columns:
        data.rename(columns={'best_qual_time_results_with_qualifying': 'best_qual_time'}, inplace=True)
    elif 'best_qual_time_qualifying' in data.columns:
        data.rename(columns={'best_qual_time_qualifying': 'best_qual_time'}, inplace=True)
    
    if 'teammate_qual_delta_results_with_qualifying' in data.columns:
        data.rename(columns={'teammate_qual_delta_results_with_qualifying': 'teammate_qual_delta'}, inplace=True)
    elif 'teammate_qual_delta_qualifying' in data.columns:
        data.rename(columns={'teammate_qual_delta_qualifying': 'teammate_qual_delta'}, inplace=True)
    
    print(f"Data loaded: {data.shape}")
    
    # Train position prediction model
    print("\n" + "-" * 60)
    print("1. Training Position Prediction Model (XGBoost)")
    print("-" * 60)
    
    model, mse, r2, mae, mean_err, evals_result, preprocessor = train_and_evaluate_model(
        data, 
        early_stopping_rounds=20,
        model_type="XGBoost"
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
        'model_type': 'XGBoost',
        'trained_at': datetime.now().isoformat()
    }
    
    with open(output_dir / 'position_model.pkl', 'wb') as f:
        pickle.dump(position_artifact, f)
    print(f"✓ Position model saved (MAE: {mae:.4f})")
    
    # Train DNF prediction model
    print("\n" + "-" * 60)
    print("2. Training DNF Prediction Model")
    print("-" * 60)
    
    dnf_model = train_and_evaluate_dnf_model(data, CACHE_VERSION)
    
    dnf_artifact = {
        'model': dnf_model,
        'cache_version': CACHE_VERSION,
        'trained_at': datetime.now().isoformat()
    }
    
    with open(output_dir / 'dnf_model.pkl', 'wb') as f:
        pickle.dump(dnf_artifact, f)
    print("✓ DNF model saved")
    
    # Train safety car prediction model
    print("\n" + "-" * 60)
    print("3. Training Safety Car Prediction Model")
    print("-" * 60)
    
    safety_cars_file = Path('data_files/f1SafetyCarFeatures.csv')
    if safety_cars_file.exists():
        safety_cars = pd.read_csv(safety_cars_file, sep='\t')
        safetycar_model = train_and_evaluate_safetycar_model(safety_cars, CACHE_VERSION)
        
        safetycar_artifact = {
            'model': safetycar_model,
            'cache_version': CACHE_VERSION,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(output_dir / 'safetycar_model.pkl', 'wb') as f:
            pickle.dump(safetycar_artifact, f)
        print("✓ Safety car model saved")
    else:
        print("⚠ Safety car data not found - skipping")
    
    # Save metadata
    metadata = {
        'cache_version': CACHE_VERSION,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'XGBoost',
        'data_rows': len(data),
        'models': {
            'position': {
                'mae': float(mae),
                'mse': float(mse),
                'r2': float(r2)
            },
            'dnf': {'trained': True},
            'safetycar': {'trained': safety_cars_file.exists()}
        }
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("XGBoost Training Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Position MAE: {mae:.4f}")
    print(f"Files created: {len(list(output_dir.glob('*.pkl')))} model files + metadata")
    
    return metadata

if __name__ == '__main__':
    try:
        metadata = train_xgboost_models()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
