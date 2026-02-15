#!/usr/bin/env python3
"""
Generate predictions for the next race using all trained models.
Includes position predictions, DNF probabilities, and Monte Carlo rookie simulations.
"""
import argparse
import json
import os
import pickle
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
from scipy.stats import truncnorm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# helper for robust json serialization of numpy/pandas scalars used by precompute scripts
import json_helpers


def load_models(models_dir: Path):
    """Load all pre-trained models."""
    models = {}
    
    for model_type in ['xgboost', 'lightgbm', 'catboost', 'ensemble']:
        model_path = models_dir / model_type / 'position_model.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[model_type] = pickle.load(f)
            print(f"  [OK] Loaded {model_type} model")
    
    # Also load DNF model
    dnf_path = models_dir / 'xgboost' / 'dnf_model.pkl'
    if dnf_path.exists():
        with open(dnf_path, 'rb') as f:
            models['dnf'] = pickle.load(f)
        print("  [OK] Loaded DNF model")
    
    return models


def get_next_race_info(race_schedule):
    """Determine the next race from the schedule."""
    today = pd.Timestamp.now().normalize()
    
    # Parse race dates
    race_schedule['date_parsed'] = pd.to_datetime(race_schedule['date'], errors='coerce')
    future_races = race_schedule[race_schedule['date_parsed'] >= today].sort_values('date_parsed')
    
    if future_races.empty:
        print("No upcoming races found!")
        return None
    
    next_race = future_races.iloc[0]
    return next_race


def generate_predictions(data, models, next_race, output_dir: Path):
    """Generate predictions for all drivers for the next race."""
    
    from raceAnalysis import get_features_and_target
    
    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)
    
    features, _ = get_features_and_target(data)
    feature_names = features.columns.tolist()
    
    # Filter to active drivers for next race
    current_year = next_race.get('year', datetime.now().year)
    
    # Get latest data for each driver (use last year as template)
    input_data = data[data['grandPrixYear'] == current_year - 1].copy()
    
    if input_data.empty:
        print(f"Warning: No data for year {current_year - 1}, using most recent year")
        max_year = data['grandPrixYear'].max()
        input_data = data[data['grandPrixYear'] == max_year].copy()
    
    predictions = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'next_race': {
                'name': next_race.get('fullName', 'Unknown'),
                'date': str(next_race.get('date', '')),
                'round': int(next_race.get('round', 0)) if pd.notna(next_race.get('round')) else 0
            }
        },
        'predictions_by_model': {}
    }
    
    # Generate predictions for each model type
    for model_type, model_data in models.items():
        if model_type == 'dnf':
            continue  # Handle separately
        
        model = model_data.get('model')
        preprocessor = model_data.get('preprocessor')
        
        if model is None or preprocessor is None:
            print(f"  Skipping {model_type}: missing model or preprocessor")
            continue
        
        try:
            # Prepare features
            available_features = [f for f in feature_names if f in input_data.columns]
            X_predict = input_data[available_features].copy()
            X_predict = X_predict.fillna(X_predict.mean(numeric_only=True))
            
            # Transform
            X_prep = preprocessor.transform(X_predict)
            
            # Predict
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_prep)
            else:
                import xgboost as xgb
                y_pred = model.predict(xgb.DMatrix(X_prep))
            
            # Store predictions
            driver_predictions = []
            for i, (_, row) in enumerate(input_data.iterrows()):
                driver_predictions.append({
                    'driverId': int(row.get('resultsDriverId', 0)) if pd.notna(row.get('resultsDriverId')) else 0,
                    'driverName': str(row.get('resultsDriverName', 'Unknown')),
                    'abbreviation': str(row.get('Abbreviation', '')),
                    'constructor': str(row.get('constructorName', '')),
                    'predicted_position': float(y_pred[i]),
                    'mae': float(model_data.get('mae', 0))
                })
            
            # Sort by predicted position
            driver_predictions.sort(key=lambda x: x['predicted_position'])
            
            # Add predicted rank
            for rank, pred in enumerate(driver_predictions, 1):
                pred['predicted_rank'] = rank
            
            predictions['predictions_by_model'][model_type] = {
                'model_mae': float(model_data.get('mae', 0)),
                'predictions': driver_predictions
            }
            
            print(f"  [OK] Generated {len(driver_predictions)} predictions with {model_type} (MAE: {model_data.get('mae', 0):.4f})")
            
        except Exception as e:
            print(f"  ✗ Error generating predictions with {model_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    race_name = next_race.get('fullName', 'unknown').replace(' ', '_').lower()
    output_file = output_dir / f'predictions_{race_name}_{current_year}.json'
    
    json_helpers.safe_dump(predictions, output_file, indent=2)
    
    print(f"\n[OK] Predictions saved to {output_file}")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data_files/precomputed/predictions/')
    parser.add_argument('--all-models', action='store_true', help='Generate predictions with all model types')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    models_dir = Path('data_files/models')
    
    print("=" * 60)
    print("Race Predictions Generator")
    print("=" * 60)
    
    print("\nLoading data...")
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    race_schedule = pd.read_json('data_files/f1db-races.json')
    
    print("Loading models...")
    models = load_models(models_dir)
    
    if not models:
        print("[ERROR] No models found! Run model training first.")
        sys.exit(1)
    
    print(f"\nFound {len(models)} models")
    
    print("\nDetermining next race...")
    next_race = get_next_race_info(race_schedule)
    
    if next_race is None:
        print("No upcoming races - exiting")
        sys.exit(0)
    
    print(f"Next race: {next_race.get('fullName')} on {next_race.get('date')}")
    
    print("\nGenerating predictions...")
    generate_predictions(data, models, next_race, output_dir)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
