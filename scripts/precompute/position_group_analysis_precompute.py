#!/usr/bin/env python3
"""
Precompute detailed position-specific MAE analysis.
Generates metrics for podium, points, midfield, and backmarker positions.
"""
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'  # Minimize Streamlit logging

import warnings
import logging
warnings.filterwarnings("ignore")
# Suppress specific Streamlit runtime warnings
logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)
logging.getLogger("streamlit.runtime").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# helper for robust json serialization of numpy/pandas scalars used by precompute scripts
import json_helpers


def compute_position_specific_mae(y_true, y_pred):
    """Compute MAE for different position groups."""
    results = {}
    
    # Create DataFrame for analysis
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'error': np.abs(y_true - y_pred)
    })
    
    # Position groups
    groups = {
        'winners': df['actual'] == 1,
        'podium': df['actual'] <= 3,
        'top_5': df['actual'] <= 5,
        'points': df['actual'] <= 10,
        'midfield': (df['actual'] > 10) & (df['actual'] <= 15),
        'backmarkers': df['actual'] > 15,
        'overall': pd.Series([True] * len(df))
    }
    
    for group_name, mask in groups.items():
        group_df = df[mask]
        if len(group_df) > 0:
            results[group_name] = {
                'mae': float(group_df['error'].mean()),
                'median_error': float(group_df['error'].median()),
                'std_error': float(group_df['error'].std()),
                'max_error': float(group_df['error'].max()),
                'count': int(len(group_df)),
                'percentile_25': float(group_df['error'].quantile(0.25)),
                'percentile_75': float(group_df['error'].quantile(0.75)),
                'percentile_90': float(group_df['error'].quantile(0.90))
            }
    
    return results


def compute_driver_specific_mae(data, y_true, y_pred):
    """Compute MAE breakdown by driver."""
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'driver': data.loc[y_true.index, 'resultsDriverName'].values if 'resultsDriverName' in data.columns else ['Unknown'] * len(y_true)
    })
    df['error'] = np.abs(df['actual'] - df['predicted'])
    
    driver_stats = df.groupby('driver').agg(
        mae=('error', 'mean'),
        count=('error', 'count'),
        std=('error', 'std')
    ).round(3).to_dict('index')
    
    return driver_stats


def compute_constructor_specific_mae(data, y_true, y_pred):
    """Compute MAE breakdown by constructor."""
    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'constructor': data.loc[y_true.index, 'constructorName'].values if 'constructorName' in data.columns else ['Unknown'] * len(y_true)
    })
    df['error'] = np.abs(df['actual'] - df['predicted'])
    
    constructor_stats = df.groupby('constructor').agg(
        mae=('error', 'mean'),
        count=('error', 'count'),
        std=('error', 'std')
    ).round(3).to_dict('index')
    
    return constructor_stats


def main():
    print("=" * 60)
    print("Position-Specific MAE Analysis")
    print("=" * 60)
    
    print("\nLoading data and models...")
    
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    
    # Get features and target first (needed for both loading and training)
    from raceAnalysis import get_features_and_target, _build_advanced_preprocessor
    
    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)
    
    # Get features and target
    X, y = get_features_and_target(data)
    
    # Clean data: Drop all-NaN columns, then drop rows where label is NaN/inf
    X = X.dropna(axis=1, how='all')
    valid_idx = y.replace([np.inf, -np.inf], np.nan).dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx].astype(np.float64)
    
    # Do train/test split ONCE for entire script (same as UI)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load or train model
    model_path = Path('data_files/models/position_model.pkl')
    if not model_path.exists():
        model_path = Path('data_files/models/xgboost/position_model.pkl')
    
    if not model_path.exists():
        print("[INFO] No model found. Training fresh model...")
        import xgboost as xgb
        
        preprocessor = _build_advanced_preprocessor(X_train)
        X_train_prep = preprocessor.fit_transform(X_train, y_train)
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_prep, y_train)
        
        # Compute metrics on test set
        X_test_prep_fresh = preprocessor.transform(X_test)
        y_pred_fresh = model.predict(X_test_prep_fresh)
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae_fresh = mean_absolute_error(y_test, y_pred_fresh)
        mse_fresh = mean_squared_error(y_test, y_pred_fresh)
        r2_fresh = r2_score(y_test, y_pred_fresh)
        mean_err_fresh = np.mean(y_test - y_pred_fresh)  # Mean error (not absolute)
        
        # Save the model for future runs
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': model,
            'preprocessor': preprocessor,
            'mae': mae_fresh,
            'mse': mse_fresh,
            'r2': r2_fresh,
            'mean_err': mean_err_fresh,
            'evals_result': {},  # No early stopping in precompute, so empty dict
            'timestamp': datetime.now().isoformat(),
            'cache_version': 'v3.3'  # Match raceAnalysis.py CACHE_VERSION
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"[INFO] Fresh model trained and saved to {model_path} (MAE={mae_fresh:.3f}, MSE={mse_fresh:.3f}, R²={r2_fresh:.3f})")
    else:
        # Load model with encoding fallback for cross-platform compatibility
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        except UnicodeDecodeError:
            # Fallback for Windows-generated pickles loaded on Linux (or vice versa)
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f, encoding='latin1')

        model = model_data['model']
        preprocessor = model_data['preprocessor']
        
        # Detect preprocessor/feature-set mismatch (stale pickle) and retrain inline
        _expected = set(getattr(preprocessor, 'feature_names_in_', []))
        _actual   = set(X_train.columns)
        _missing  = _expected - _actual
        _extra    = _actual  - _expected
        if _missing or _extra:
            print(f"[WARN] Stale preprocessor detected "
                  f"({len(_missing)} missing cols, {len(_extra)} extra cols). "
                  f"Retraining inline on current data …")
            import xgboost as xgb
            
            preprocessor = _build_advanced_preprocessor(X_train)
            X_train_prep = preprocessor.fit_transform(X_train, y_train)
            
            model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X_train_prep, y_train)
            
            # Compute metrics on test set
            X_test_prep_retrain = preprocessor.transform(X_test)
            y_pred_retrain = model.predict(X_test_prep_retrain)
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            mae_retrain = mean_absolute_error(y_test, y_pred_retrain)
            mse_retrain = mean_squared_error(y_test, y_pred_retrain)
            r2_retrain = r2_score(y_test, y_pred_retrain)
            mean_err_retrain = np.mean(y_test - y_pred_retrain)
            
            # Save retrained model
            try:
                fresh = dict(model_data)
                fresh['model'] = model
                fresh['preprocessor'] = preprocessor
                fresh['mae'] = mae_retrain
                fresh['mse'] = mse_retrain
                fresh['r2'] = r2_retrain
                fresh['mean_err'] = mean_err_retrain
                fresh['evals_result'] = {}
                fresh['cache_version'] = 'v3.3'  # Match raceAnalysis.py CACHE_VERSION
                with open(model_path, 'wb') as _f:
                    pickle.dump(fresh, _f)
                print(f"[INFO] Retrained model saved to {model_path} (MAE={mae_retrain:.3f}, MSE={mse_retrain:.3f}, R²={r2_retrain:.3f})")
            except Exception as _e:
                print(f"[WARN] Could not persist retrained model: {_e}")

    # Transform and predict (X_train and X_test already created above)
    print("Generating predictions...")
    X_test_prep = preprocessor.transform(X_test)
    
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test_prep)
    else:
        import xgboost as xgb
        y_pred = model.predict(xgb.DMatrix(X_test_prep))

    # Persist the row-level holdout predictions for the reporting script.  The
    # per-race predictions_*.csv files only cover races for which an export was
    # made (often just the current season); this evaluation set carries the
    # historical season attached to every model residual.
    evaluation = data.loc[y_test.index].copy()
    season_col = next(
        (column for column in ('resultsYear', 'grandPrixYear', 'season', 'year')
         if column in evaluation.columns),
        None,
    )
    evaluation_out = pd.DataFrame({
        'season': pd.to_numeric(evaluation[season_col], errors='coerce') if season_col else np.nan,
        'actual': y_test.to_numpy(),
        'predicted': np.asarray(y_pred),
    }, index=y_test.index)
    for column in (
        'resultsDriverName', 'DriverId', 'driverId',
        'constructorName', 'constructorId',
        'grandPrixName', 'circuitName', 'shortCircuitName',
    ):
        if column in evaluation.columns:
            evaluation_out[column] = evaluation[column]
    evaluation_out['residual'] = evaluation_out['predicted'] - evaluation_out['actual']
    evaluation_out = evaluation_out.dropna(subset=['season', 'actual', 'predicted'])
    evaluation_out['season'] = evaluation_out['season'].astype(int)
    residuals_path = Path('data_files/precomputed/position_evaluation_residuals.csv')
    residuals_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_out.to_csv(residuals_path, index=False)
    print(f"Wrote {residuals_path} ({evaluation_out['season'].nunique()} seasons)")
    
    print("Computing position-specific MAE...")
    position_mae = compute_position_specific_mae(y_test.values, y_pred)
    
    print("Computing driver-specific MAE...")
    driver_mae = compute_driver_specific_mae(data, y_test, y_pred)
    
    print("Computing constructor-specific MAE...")
    constructor_mae = compute_constructor_specific_mae(data, y_test, y_pred)
    
    # Compile results
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'model_mae': float(model_data.get('mae', 0)),
            'test_set_size': len(y_test),
            'seasons': sorted(evaluation_out['season'].unique().astype(int).tolist()),
        },
        'position_groups': position_mae,
        'by_driver': driver_mae,
        'by_constructor': constructor_mae
    }
    
    # Save results
    output_dir = Path('data_files/precomputed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'position_mae_detailed.json'
    json_helpers.safe_dump(results, output_path, indent=2)
    
    print("\n" + "=" * 60)
    print(f"Results saved to {output_path}")
    print("\nPosition Group MAE Summary:")
    print("-" * 60)
    for group, stats in position_mae.items():
        print(f"  {group:15} MAE={stats['mae']:6.3f}  (n={stats['count']:4})")
    print("=" * 60)


if __name__ == '__main__':
    main()
