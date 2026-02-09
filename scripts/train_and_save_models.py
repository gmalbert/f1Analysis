"""
Pre-train all models and save them for fast loading in Streamlit app.
This script should be run in GitHub Actions or locally to generate model artifacts.
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path

# Suppress Streamlit warnings when running outside of Streamlit context
import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

import logging

# Custom logging filter to suppress specific Streamlit warnings
class StreamlitWarningFilter(logging.Filter):
    def filter(self, record):
        # Suppress these specific messages
        suppressed_messages = [
            "missing ScriptRunContext",
            "No runtime found, using MemoryCacheStorageManager",
            "Session state does not function",
            "to view this Streamlit app on a browser"
        ]
        return not any(msg in record.getMessage() for msg in suppressed_messages)

# Apply the filter to all Streamlit loggers
for logger_name in ['streamlit', 'streamlit.runtime.scriptrunner_utils.script_run_context', 
                     'streamlit.runtime.caching.cache_data_api', 'streamlit.runtime']:
    logger = logging.getLogger(logger_name)
    logger.addFilter(StreamlitWarningFilter())
    logger.setLevel(logging.ERROR)

# Suppress specific warnings from warnings module
warnings.filterwarnings("ignore", message=".*No runtime found, using MemoryCacheStorageManager.*")
warnings.filterwarnings("ignore", message=".*Thread 'MainThread': missing ScriptRunContext.*")
warnings.filterwarnings("ignore", message=".*Session state does not function when running a script without.*")
warnings.filterwarnings("ignore", message=".*Warning: to view this Streamlit app on a browser.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext! This warning can be ignored when running in bare mode.*")
warnings.filterwarnings("ignore", message=".*No runtime found, using MemoryCacheStorageManager*")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import functions from raceAnalysis
# We'll use a minimal import approach to avoid loading the full Streamlit app
def train_all_models():
    """Train all models and save them to data_files/models/"""
    
    print("Starting model training...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("data_files/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Import necessary modules (avoid importing streamlit)
    os.environ['STREAMLIT_SERVER_HEADLESS'] = '1'  # Prevent Streamlit from starting
    
    # Import training functions
    from raceAnalysis import (
        load_data,
        CACHE_VERSION,
        train_and_evaluate_model,
        train_and_evaluate_dnf_model,
        train_and_evaluate_safetycar_model,
        get_features_and_target_safety_car,
    )
    
    print(f"Loading data with CACHE_VERSION={CACHE_VERSION}...")
    data, _ = load_data(10000, CACHE_VERSION)
    
    # Apply the same column renaming logic as in raceAnalysis.py
    print("Applying column renaming logic...")
    if 'constructorName_results_with_qualifying' in data.columns:
        data.rename(columns={'constructorName_results_with_qualifying': 'constructorName'}, inplace=True)
        print("   Renamed constructorName_results_with_qualifying to constructorName")
    elif 'constructorName_qualifying' in data.columns:
        data.rename(columns={'constructorName_qualifying': 'constructorName'}, inplace=True)
        print("   Renamed constructorName_qualifying to constructorName")
    
    if 'best_qual_time_results_with_qualifying' in data.columns:
        data.rename(columns={'best_qual_time_results_with_qualifying': 'best_qual_time'}, inplace=True)
    elif 'best_qual_time_qualifying' in data.columns:
        data.rename(columns={'best_qual_time_qualifying': 'best_qual_time'}, inplace=True)
    
    if 'teammate_qual_delta_results_with_qualifying' in data.columns:
        data.rename(columns={'teammate_qual_delta_results_with_qualifying': 'teammate_qual_delta'}, inplace=True)
    elif 'teammate_qual_delta_qualifying' in data.columns:
        data.rename(columns={'teammate_qual_delta_qualifying': 'teammate_qual_delta'}, inplace=True)
    
    print(f"Data loaded successfully. Shape: {data.shape}")
    print(f"Has constructorName: {'constructorName' in data.columns}")
    
    # Load safety car data
    safety_cars_file = Path('data_files/f1SafetyCarFeatures.csv')
    if safety_cars_file.exists():
        print("Loading safety car data...")
        safety_cars = pd.read_csv(safety_cars_file, sep='\t')
    else:
        print("Warning: Safety car data file not found. Skipping safety car model.")
        safety_cars = None
    
    # Train position prediction model (main model)
    print("\n1. Training position prediction model (XGBoost)...")
    model, mse, r2, mae, mean_err, evals_result = train_and_evaluate_model(
        data, 
        early_stopping_rounds=20,
        model_type="XGBoost"
    )
    
    # Save model and metrics
    model_artifact = {
        'model': model,
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'mean_err': mean_err,
        'evals_result': evals_result,
        'cache_version': CACHE_VERSION,
        'model_type': 'XGBoost'
    }
    
    with open(models_dir / 'position_model.pkl', 'wb') as f:
        pickle.dump(model_artifact, f)
    print(f"   [OK] Saved position model (MAE: {mae:.3f})")
    
    # Train DNF prediction model
    print("\n2. Training DNF prediction model...")
    dnf_model = train_and_evaluate_dnf_model(data, CACHE_VERSION)
    
    with open(models_dir / 'dnf_model.pkl', 'wb') as f:
        pickle.dump({'model': dnf_model, 'cache_version': CACHE_VERSION}, f)
    print("   [OK] Saved DNF model")
    
    # Train safety car prediction model
    if safety_cars is not None:
        print("\n3. Training safety car prediction model...")
        safetycar_model = train_and_evaluate_safetycar_model(safety_cars, CACHE_VERSION)
        
        with open(models_dir / 'safetycar_model.pkl', 'wb') as f:
            pickle.dump({'model': safetycar_model, 'cache_version': CACHE_VERSION}, f)
        print("   [OK] Saved safety car model")
    
    # Save metadata
    metadata = {
        'cache_version': CACHE_VERSION,
        'trained_at': pd.Timestamp.now().isoformat(),
        'data_rows': len(data),
        'models': ['position_model', 'dnf_model', 'safetycar_model' if safety_cars is not None else None]
    }
    
    with open(models_dir / 'metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n[OK] All models trained and saved to {models_dir}")
    print(f"   Total models: {len([m for m in metadata['models'] if m])}")
    print(f"   CACHE_VERSION: {CACHE_VERSION}")
    print(f"   Trained at: {metadata['trained_at']}")
    
    return metadata

if __name__ == '__main__':
    try:
        metadata = train_all_models()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Error training models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
