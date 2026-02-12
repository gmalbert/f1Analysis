# F1 Analysis Precomputation Strategy

## Overview

This document outlines a comprehensive strategy to offload runtime analyses from the Streamlit UI to GitHub Actions precomputation. By precomputing expensive ML operations, we can dramatically improve user experience with instant page loads.

**Current Pain Points:**
- Model training takes 30-60 seconds on first page load
- Feature selection (Monte Carlo, Boruta, RFE) takes 2-15 minutes per run
- SHAP analysis can take 5-10 minutes
- Hyperparameter tuning takes 10-30+ minutes
- Rookie Monte Carlo simulations add ~5 seconds per prediction

**Goal:** Page load under 3 seconds with all analyses pre-computed and cached.

---

## Phase 1: Model Training Pipeline (Already Partially Implemented)

### Current State
The `train-models.yml` workflow trains and saves models, but only covers:
- Position prediction model (XGBoost)
- DNF model
- Safety car model

### Enhancements Needed

#### 1a. Multi-Model Training Workflow
**File:** `.github/workflows/train-all-models.yml`

```yaml
name: Train All Models (Multi-Type)

on:
  workflow_run:
    workflows: ["Generate F1 Analysis Data"]
    types: [completed]
  workflow_dispatch:
    inputs:
      force_retrain:
        description: 'Force full retrain even if models exist'
        type: boolean
        default: false
  schedule:
    - cron: '0 3 * * 0'  # Sundays at 3 AM UTC

env:
  PYTHON_VERSION: '3.11'

jobs:
  train-xgboost:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Train XGBoost Models
        run: python scripts/precompute/train_xgboost.py
        env:
          MODEL_TYPE: XGBoost
      - uses: actions/upload-artifact@v4
        with:
          name: xgboost-models
          path: data_files/models/xgboost/
          retention-days: 90

  train-lightgbm:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Train LightGBM Models
        run: python scripts/precompute/train_lightgbm.py
        env:
          MODEL_TYPE: LightGBM
      - uses: actions/upload-artifact@v4
        with:
          name: lightgbm-models
          path: data_files/models/lightgbm/
          retention-days: 90

  train-catboost:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Train CatBoost Models
        run: python scripts/precompute/train_catboost.py
        env:
          MODEL_TYPE: CatBoost
      - uses: actions/upload-artifact@v4
        with:
          name: catboost-models
          path: data_files/models/catboost/
          retention-days: 90

  train-ensemble:
    needs: [train-xgboost, train-lightgbm, train-catboost]
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - uses: actions/download-artifact@v4
        with:
          pattern: '*-models'
          path: data_files/models/
          merge-multiple: true
      - run: pip install -r requirements.txt
      - name: Train Ensemble Model
        run: python scripts/precompute/train_ensemble.py
      - uses: actions/upload-artifact@v4
        with:
          name: ensemble-models
          path: data_files/models/ensemble/
          retention-days: 90

  commit-models:
    needs: [train-ensemble]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          pattern: '*-models'
          path: data_files/models/
          merge-multiple: true
      - name: Commit all models
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data_files/models/
          if git diff --staged --quiet; then
            echo "No changes to commit"
          else
            git commit -m "chore: update pre-trained models (all types) [skip ci]"
            git push
          fi
```

---

## Phase 2: Feature Selection Precomputation

### 2a. Monte Carlo Feature Selection
**File:** `.github/workflows/feature-selection-monte-carlo.yml`

This is one of the most expensive operations. Break it into configurable chunks.

```yaml
name: Monte Carlo Feature Selection

on:
  workflow_dispatch:
    inputs:
      n_trials:
        description: 'Number of trials'
        type: number
        default: 1000
      min_features:
        description: 'Minimum features per trial'
        type: number
        default: 8
      max_features:
        description: 'Maximum features per trial'
        type: number
        default: 15
  schedule:
    - cron: '0 4 * * 1'  # Mondays at 4 AM UTC (weekly)

jobs:
  monte-carlo-search:
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      
      - name: Run Monte Carlo Feature Selection
        run: |
          python scripts/precompute/monte_carlo_features.py \
            --n-trials ${{ inputs.n_trials || 1000 }} \
            --min-features ${{ inputs.min_features || 8 }} \
            --max-features ${{ inputs.max_features || 15 }} \
            --cv-folds 10 \
            --output data_files/precomputed/monte_carlo_results.json
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      
      - uses: actions/upload-artifact@v4
        with:
          name: monte-carlo-results
          path: data_files/precomputed/monte_carlo_results.json
      
      - name: Commit results
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data_files/precomputed/
          if git diff --staged --quiet; then
            echo "No changes"
          else
            git commit -m "chore: update Monte Carlo feature selection results [skip ci]"
            git push
          fi
```

**Script:** `scripts/precompute/monte_carlo_features.py`

```python
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

# Suppress Streamlit warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def load_data():
    """Load the main analysis dataset."""
    data_path = Path('data_files/f1ForAnalysis.csv')
    data = pd.read_csv(data_path, sep='\t', low_memory=False)
    return data

def get_features_and_target(data):
    """Extract features and target from data."""
    # Import the feature list from raceAnalysis
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
    
    # Progress tracking
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
    print(f"Best MAE: {output['best_result']['mae']:.4f}")
    print(f"Best features: {output['best_result']['features']}")

if __name__ == '__main__':
    main()
```

### 2b. SHAP, RFE, and Boruta Combined Workflow
**File:** `.github/workflows/feature-selection-suite.yml`

```yaml
name: Feature Selection Suite

on:
  workflow_dispatch:
  schedule:
    - cron: '0 5 * * 1'  # Mondays at 5 AM UTC

jobs:
  rfe-selection:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Run RFE
        run: python scripts/precompute/rfe_features.py --output data_files/precomputed/rfe_results.json
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      - uses: actions/upload-artifact@v4
        with:
          name: rfe-results
          path: data_files/precomputed/rfe_results.json

  boruta-selection:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Run Boruta
        run: python scripts/precompute/boruta_features.py --max-iter 200 --output data_files/precomputed/boruta_results.json
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      - uses: actions/upload-artifact@v4
        with:
          name: boruta-results
          path: data_files/precomputed/boruta_results.json

  shap-analysis:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Run SHAP Analysis
        run: python scripts/precompute/shap_analysis.py --output data_files/precomputed/shap_results.json
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      - uses: actions/upload-artifact@v4
        with:
          name: shap-results
          path: data_files/precomputed/shap_results.json

  permutation-importance:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Run Permutation Importance
        run: python scripts/precompute/permutation_importance.py --output data_files/precomputed/permutation_results.json
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      - uses: actions/upload-artifact@v4
        with:
          name: permutation-results
          path: data_files/precomputed/permutation_results.json

  commit-all:
    needs: [rfe-selection, boruta-selection, shap-analysis, permutation-importance]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          pattern: '*-results'
          path: data_files/precomputed/
          merge-multiple: true
      - name: Commit feature selection results
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data_files/precomputed/
          if git diff --staged --quiet; then
            echo "No changes"
          else
            git commit -m "chore: update feature selection results [skip ci]"
            git push
          fi
```

---

## Phase 3: Predictions Precomputation

### 3a. Next Race Predictions
**File:** `.github/workflows/precompute-predictions.yml`

```yaml
name: Precompute Race Predictions

on:
  workflow_run:
    workflows: ["Train All Models (Multi-Type)"]
    types: [completed]
  workflow_dispatch:
  schedule:
    - cron: '0 6 * * 4'  # Thursdays at 6 AM UTC (before race weekends)

jobs:
  generate-predictions:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      
      - name: Generate predictions for next race
        run: |
          python scripts/precompute/generate_race_predictions.py \
            --output data_files/precomputed/predictions/ \
            --all-models
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      
      - uses: actions/upload-artifact@v4
        with:
          name: race-predictions
          path: data_files/precomputed/predictions/
      
      - name: Commit predictions
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data_files/precomputed/predictions/
          if git diff --staged --quiet; then
            echo "No changes"
          else
            git commit -m "chore: update race predictions [skip ci]"
            git push
          fi
```

**Script:** `scripts/precompute/generate_race_predictions.py`

```python
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

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_models(models_dir: Path):
    """Load all pre-trained models."""
    models = {}
    
    for model_type in ['xgboost', 'lightgbm', 'catboost', 'ensemble']:
        model_path = models_dir / model_type / 'position_model.pkl'
        if model_path.exists():
            with open(model_path, 'rb') as f:
                models[model_type] = pickle.load(f)
            print(f"  Loaded {model_type} model")
    
    # Also load DNF model
    dnf_path = models_dir / 'dnf_model.pkl'
    if dnf_path.exists():
        with open(dnf_path, 'rb') as f:
            models['dnf'] = pickle.load(f)
        print("  Loaded DNF model")
    
    return models


def get_next_race_info(data, race_schedule):
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


def monte_carlo_rookie_simulation(
    rookie_data, historical_data, n_simulations=1000
):
    """
    Run Monte Carlo simulations for rookie predictions.
    Returns adjusted predictions with uncertainty estimates.
    """
    results = []
    
    for idx, rookie in rookie_data.iterrows():
        # Get historical rookie performances
        hist_positions = historical_data[
            historical_data['yearsActive'] <= 1
        ]['resultsFinalPositionNumber'].dropna()
        
        if len(hist_positions) < 10:
            hist_positions = historical_data['resultsFinalPositionNumber'].dropna()
        
        mu, sigma = hist_positions.mean(), hist_positions.std()
        a, b = (1 - mu) / sigma, (20 - mu) / sigma
        
        # Sample positions
        sampled = truncnorm.rvs(a, b, loc=mu, scale=sigma, size=n_simulations)
        
        # Adjust by constructor strength
        constructor_rank = rookie.get('constructorRank', 10)
        constructor_adj = np.clip(1 + (constructor_rank - 10) * 0.2, 0.7, 1.3)
        
        # Adjust by practice position
        practice_adj = 1.0
        if pd.notna(rookie.get('averagePracticePosition')):
            practice_adj = np.clip(rookie['averagePracticePosition'] / 10, 0.7, 1.3)
        
        simulated_positions = sampled * constructor_adj * practice_adj
        
        results.append({
            'driverId': rookie.get('resultsDriverId'),
            'driverName': rookie.get('resultsDriverName', 'Unknown'),
            'predicted_position': float(np.median(simulated_positions)),
            'predicted_position_std': float(np.std(simulated_positions)),
            'predicted_position_25pct': float(np.percentile(simulated_positions, 25)),
            'predicted_position_75pct': float(np.percentile(simulated_positions, 75)),
            'is_rookie': True
        })
    
    return results


def generate_predictions(data, models, next_race, output_dir: Path):
    """Generate predictions for all drivers for the next race."""
    
    # Get features
    from raceAnalysis import get_features_and_target, get_preprocessor_position
    
    features, _ = get_features_and_target(data)
    feature_names = features.columns.tolist()
    
    # Filter to active drivers for next race
    current_year = next_race.get('year', datetime.now().year)
    
    # Get latest data for each driver
    input_data = data[data['grandPrixYear'] == current_year - 1].copy()  # Use last year as baseline
    
    predictions = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'next_race': {
                'name': next_race.get('fullName', 'Unknown'),
                'date': str(next_race.get('date', '')),
                'round': int(next_race.get('round', 0))
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
            X_predict = input_data[feature_names].copy()
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
                    'driverId': row.get('resultsDriverId'),
                    'driverName': row.get('resultsDriverName', 'Unknown'),
                    'abbreviation': row.get('Abbreviation', ''),
                    'constructor': row.get('constructorName', ''),
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
            
            print(f"  Generated {len(driver_predictions)} predictions with {model_type}")
            
        except Exception as e:
            print(f"  Error generating predictions with {model_type}: {e}")
    
    # Save predictions
    output_dir.mkdir(parents=True, exist_ok=True)
    race_name = next_race.get('fullName', 'unknown').replace(' ', '_').lower()
    output_file = output_dir / f'predictions_{race_name}_{current_year}.json'
    
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"\nPredictions saved to {output_file}")
    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='data_files/precomputed/predictions/')
    parser.add_argument('--all-models', action='store_true', help='Generate predictions with all model types')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    models_dir = Path('data_files/models')
    
    print("Loading data...")
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    race_schedule = pd.read_json('data_files/f1db-races.json')
    
    print("Loading models...")
    models = load_models(models_dir)
    
    if not models:
        print("No models found! Run model training first.")
        sys.exit(1)
    
    print("Determining next race...")
    next_race = get_next_race_info(data, race_schedule)
    
    if next_race is None:
        print("No upcoming races - exiting")
        sys.exit(0)
    
    print(f"Next race: {next_race.get('fullName')} on {next_race.get('date')}")
    
    print("\nGenerating predictions...")
    generate_predictions(data, models, next_race, output_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
```

---

## Phase 4: Hyperparameter Optimization

### 4a. Scheduled Hyperparameter Tuning
**File:** `.github/workflows/hyperparameter-optimization.yml`

```yaml
name: Hyperparameter Optimization

on:
  workflow_dispatch:
    inputs:
      method:
        description: 'Optimization method'
        type: choice
        options:
          - grid_search
          - bayesian
          - both
        default: bayesian
      n_trials:
        description: 'Number of Optuna trials (bayesian only)'
        type: number
        default: 100
  schedule:
    - cron: '0 2 1 * *'  # First day of each month at 2 AM

jobs:
  grid-search:
    if: ${{ inputs.method == 'grid_search' || inputs.method == 'both' || github.event_name == 'schedule' }}
    runs-on: ubuntu-latest
    timeout-minutes: 120
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Run Grid Search
        run: python scripts/precompute/hyperparameter_grid_search.py --output data_files/precomputed/hyperparam_grid.json
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      - uses: actions/upload-artifact@v4
        with:
          name: grid-search-results
          path: data_files/precomputed/hyperparam_grid.json

  bayesian-optimization:
    if: ${{ inputs.method == 'bayesian' || inputs.method == 'both' || github.event_name == 'schedule' }}
    runs-on: ubuntu-latest
    timeout-minutes: 180
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      - name: Run Bayesian Optimization
        run: |
          python scripts/precompute/hyperparameter_bayesian.py \
            --n-trials ${{ inputs.n_trials || 100 }} \
            --output data_files/precomputed/hyperparam_bayesian.json
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      - uses: actions/upload-artifact@v4
        with:
          name: bayesian-results
          path: data_files/precomputed/hyperparam_bayesian.json

  commit-results:
    needs: [grid-search, bayesian-optimization]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          path: data_files/precomputed/
          merge-multiple: true
      - name: Commit hyperparameter results
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add data_files/precomputed/
          if git diff --staged --quiet; then
            echo "No changes"
          else
            git commit -m "chore: update hyperparameter optimization results [skip ci]"
            git push
          fi
```

**Script:** `scripts/precompute/hyperparameter_bayesian.py`

```python
#!/usr/bin/env python3
"""
Bayesian hyperparameter optimization using Optuna.
Precomputes optimal hyperparameters for all model types.
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def optimize_xgboost(X, y, season_groups, n_trials=100):
    """Optimize XGBoost hyperparameters using Optuna."""
    from raceAnalysis import get_preprocessor_position
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        pipeline = Pipeline([
            ('preprocessor', get_preprocessor_position(X)),
            ('regressor', XGBRegressor(
                n_estimators=200,
                n_jobs=-1,
                tree_method='hist',
                random_state=42,
                **params
            ))
        ])
        
        if season_groups is not None:
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(pipeline, X, y, cv=cv, groups=season_groups, 
                                     scoring='neg_mean_absolute_error')
        else:
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        return -scores.mean()
    
    study = optuna.create_study(direction='minimize', study_name='xgboost_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_mae': study.best_value,
        'n_trials': n_trials,
        'optimization_history': [
            {'trial': t.number, 'mae': t.value, 'params': t.params}
            for t in study.trials if t.value is not None
        ]
    }


def optimize_lightgbm(X, y, season_groups, n_trials=100):
    """Optimize LightGBM hyperparameters."""
    from lightgbm import LGBMRegressor
    from raceAnalysis import get_preprocessor_position
    
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        pipeline = Pipeline([
            ('preprocessor', get_preprocessor_position(X)),
            ('regressor', LGBMRegressor(
                n_estimators=200,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
                **params
            ))
        ])
        
        if season_groups is not None:
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(pipeline, X, y, cv=cv, groups=season_groups,
                                     scoring='neg_mean_absolute_error')
        else:
            scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        return -scores.mean()
    
    study = optuna.create_study(direction='minimize', study_name='lightgbm_optimization')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return {
        'best_params': study.best_params,
        'best_mae': study.best_value,
        'n_trials': n_trials
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=100)
    parser.add_argument('--output', type=str, default='data_files/precomputed/hyperparam_bayesian.json')
    parser.add_argument('--model-types', nargs='+', default=['xgboost', 'lightgbm'])
    args = parser.parse_args()
    
    print("Loading data...")
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    
    from raceAnalysis import get_features_and_target
    X, y = get_features_and_target(data)
    
    # Clean data
    mask = y.notnull() & np.isfinite(y)
    X_clean, y_clean = X[mask], y[mask]
    
    # Get season groups for stratified CV
    season_groups = data.loc[y_clean.index, 'year'] if 'year' in data.columns else None
    
    print(f"Data shape: {X_clean.shape}")
    
    results = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'n_trials': args.n_trials,
            'data_rows': len(X_clean)
        },
        'optimizations': {}
    }
    
    if 'xgboost' in args.model_types:
        print("\nOptimizing XGBoost...")
        results['optimizations']['xgboost'] = optimize_xgboost(X_clean, y_clean, season_groups, args.n_trials)
        print(f"  Best MAE: {results['optimizations']['xgboost']['best_mae']:.4f}")
    
    if 'lightgbm' in args.model_types:
        print("\nOptimizing LightGBM...")
        results['optimizations']['lightgbm'] = optimize_lightgbm(X_clean, y_clean, season_groups, args.n_trials)
        print(f"  Best MAE: {results['optimizations']['lightgbm']['best_mae']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
```

---

## Phase 5: Position-Specific Analysis

### 5a. MAE by Position Group Analysis
**File:** `.github/workflows/position-analysis.yml`

```yaml
name: Position Group Analysis

on:
  workflow_run:
    workflows: ["Train All Models (Multi-Type)"]
    types: [completed]
  workflow_dispatch:
  schedule:
    - cron: '0 7 * * 0'  # Sundays at 7 AM

jobs:
  position-analysis:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'
      - run: pip install -r requirements.txt
      
      - name: Run Position Group Analysis
        run: python scripts/precompute/position_group_analysis_precompute.py
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      
      - name: Generate MAE by Season Trends
        run: python scripts/position_group_analysis.py
        env:
          STREAMLIT_SERVER_HEADLESS: '1'
      
      - uses: actions/upload-artifact@v4
        with:
          name: position-analysis
          path: |
            scripts/output/mae_by_season.csv
            scripts/output/mae_trends.png
            scripts/output/confid_int_*.csv
            scripts/output/heatmap_*.png
            data_files/precomputed/position_mae_detailed.json
      
      - name: Commit analysis results
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          git add scripts/output/ data_files/precomputed/
          if git diff --staged --quiet; then
            echo "No changes"
          else
            git commit -m "chore: update position group analysis [skip ci]"
            git push
          fi
```

**Script:** `scripts/precompute/position_group_analysis_precompute.py`

```python
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

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
    print("Loading data and models...")
    
    data = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
    
    # Load model
    model_path = Path('data_files/models/position_model.pkl')
    if not model_path.exists():
        model_path = Path('data_files/models/xgboost/position_model.pkl')
    
    if not model_path.exists():
        print("No model found! Run model training first.")
        sys.exit(1)
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    preprocessor = model_data['preprocessor']
    
    # Get features and target
    from raceAnalysis import get_features_and_target
    X, y = get_features_and_target(data)
    
    # Train/test split (same as in app)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Transform and predict
    X_test_prep = preprocessor.transform(X_test)
    
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test_prep)
    else:
        import xgboost as xgb
        y_pred = model.predict(xgb.DMatrix(X_test_prep))
    
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
            'test_set_size': len(y_test)
        },
        'position_groups': position_mae,
        'by_driver': driver_mae,
        'by_constructor': constructor_mae
    }
    
    # Save results
    output_dir = Path('data_files/precomputed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'position_mae_detailed.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    print("\nPosition Group MAE Summary:")
    for group, stats in position_mae.items():
        print(f"  {group}: MAE={stats['mae']:.3f} (n={stats['count']})")


if __name__ == '__main__':
    main()
```

---

## Phase 6: Master Orchestration Workflow

### 6a. Weekly Full Precomputation
**File:** `.github/workflows/weekly-precompute-all.yml`

```yaml
name: Weekly Full Precomputation

on:
  schedule:
    - cron: '0 1 * * 0'  # Sundays at 1 AM UTC
  workflow_dispatch:
    inputs:
      skip_data_generation:
        description: 'Skip data generation (use existing data)'
        type: boolean
        default: false

jobs:
  data-generation:
    if: ${{ !inputs.skip_data_generation }}
    uses: ./.github/workflows/generate-data.yml  # Reference existing workflow
  
  train-models:
    needs: [data-generation]
    if: always() && (needs.data-generation.result == 'success' || inputs.skip_data_generation)
    uses: ./.github/workflows/train-all-models.yml
  
  feature-selection:
    needs: [train-models]
    if: always() && needs.train-models.result == 'success'
    uses: ./.github/workflows/feature-selection-suite.yml
  
  monte-carlo-features:
    needs: [train-models]
    if: always() && needs.train-models.result == 'success'
    uses: ./.github/workflows/feature-selection-monte-carlo.yml
    with:
      n_trials: 500  # Weekly: moderate
  
  predictions:
    needs: [train-models]
    if: always() && needs.train-models.result == 'success'
    uses: ./.github/workflows/precompute-predictions.yml
  
  position-analysis:
    needs: [train-models]
    if: always() && needs.train-models.result == 'success'
    uses: ./.github/workflows/position-analysis.yml
  
  summary:
    needs: [feature-selection, monte-carlo-features, predictions, position-analysis]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - name: Summary
        run: |
          echo "## Weekly Precomputation Summary" >> $GITHUB_STEP_SUMMARY
          echo "" >> $GITHUB_STEP_SUMMARY
          echo "| Job | Status |" >> $GITHUB_STEP_SUMMARY
          echo "|-----|--------|" >> $GITHUB_STEP_SUMMARY
          echo "| Data Generation | ${{ needs.data-generation.result || 'skipped' }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Model Training | ${{ needs.train-models.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Feature Selection | ${{ needs.feature-selection.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Monte Carlo | ${{ needs.monte-carlo-features.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Predictions | ${{ needs.predictions.result }} |" >> $GITHUB_STEP_SUMMARY
          echo "| Position Analysis | ${{ needs.position-analysis.result }} |" >> $GITHUB_STEP_SUMMARY
```

---

## Phase 7: UI Changes to Load Precomputed Data

### Key Changes to `raceAnalysis.py`

Add functions to load precomputed data instead of computing on-demand:

```python
# Add near the top of raceAnalysis.py (after imports)

PRECOMPUTED_DIR = Path('data_files/precomputed')

@st.cache_data
def load_precomputed_monte_carlo(CACHE_VERSION):
    """Load precomputed Monte Carlo feature selection results."""
    path = PRECOMPUTED_DIR / 'monte_carlo_results.json'
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_precomputed_shap(CACHE_VERSION):
    """Load precomputed SHAP analysis results."""
    path = PRECOMPUTED_DIR / 'shap_results.json'
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_precomputed_hyperparams(CACHE_VERSION):
    """Load precomputed hyperparameter optimization results."""
    bayesian_path = PRECOMPUTED_DIR / 'hyperparam_bayesian.json'
    grid_path = PRECOMPUTED_DIR / 'hyperparam_grid.json'
    
    results = {}
    if bayesian_path.exists():
        with open(bayesian_path, 'r') as f:
            results['bayesian'] = json.load(f)
    if grid_path.exists():
        with open(grid_path, 'r') as f:
            results['grid'] = json.load(f)
    
    return results if results else None

@st.cache_data
def load_precomputed_position_mae(CACHE_VERSION):
    """Load precomputed position-specific MAE analysis."""
    path = PRECOMPUTED_DIR / 'position_mae_detailed.json'
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_precomputed_predictions(race_name, CACHE_VERSION):
    """Load precomputed predictions for a specific race."""
    predictions_dir = PRECOMPUTED_DIR / 'predictions'
    if predictions_dir.exists():
        # Find matching prediction file
        for f in predictions_dir.glob('predictions_*.json'):
            with open(f, 'r') as fp:
                data = json.load(fp)
                if race_name.lower() in f.stem.lower():
                    return data
    return None


# Then in tab5 (Feature Selection section), replace the compute-on-demand code:

with tab_select:
    st.subheader("Feature Selection Tools")
    
    # Check for precomputed results first
    monte_carlo_results = load_precomputed_monte_carlo(CACHE_VERSION)
    
    if monte_carlo_results:
        st.success(f"✓ Using precomputed Monte Carlo results (generated {monte_carlo_results['metadata']['generated_at'][:10]})")
        
        best = monte_carlo_results['best_result']
        st.write("### Best Feature Subset (Precomputed)")
        st.write(f"**Best MAE:** {best['mae']:.4f}")
        st.write(f"**Features ({len(best['features'])}):**")
        st.code(", ".join([f"'{f}'" for f in best['features']]))
        
        # Show top 20 results
        st.subheader("Top 20 Feature Subsets")
        top_df = pd.DataFrame(monte_carlo_results['top_20_results'])
        st.dataframe(top_df, hide_index=True)
        
        # Feature frequency
        st.subheader("Feature Frequency in Top 20")
        freq_df = pd.DataFrame(
            monte_carlo_results['feature_frequency_top_20'].items(),
            columns=['Feature', 'Appearances']
        ).sort_values('Appearances', ascending=False)
        st.dataframe(freq_df, hide_index=True)
    else:
        st.info("No precomputed Monte Carlo results found. Run manually below or wait for scheduled GitHub Action.")
    
    # Still allow manual runs, but show they're optional
    with st.expander("🔄 Run Manual Monte Carlo Search (Optional)", expanded=False):
        # ... existing manual run code ...
```

---

## Directory Structure

After implementing all phases, your precomputed data will be organized as:

```
data_files/
├── models/
│   ├── xgboost/
│   │   ├── position_model.pkl
│   │   ├── dnf_model.pkl
│   │   └── metadata.json
│   ├── lightgbm/
│   │   └── ...
│   ├── catboost/
│   │   └── ...
│   └── ensemble/
│       └── ...
├── precomputed/
│   ├── monte_carlo_results.json
│   ├── shap_results.json
│   ├── rfe_results.json
│   ├── boruta_results.json
│   ├── permutation_results.json
│   ├── hyperparam_bayesian.json
│   ├── hyperparam_grid.json
│   ├── position_mae_detailed.json
│   └── predictions/
│       ├── predictions_australian_gp_2026.json
│       └── ...
scripts/
├── precompute/
│   ├── __init__.py
│   ├── train_xgboost.py
│   ├── train_lightgbm.py
│   ├── train_catboost.py
│   ├── train_ensemble.py
│   ├── monte_carlo_features.py
│   ├── shap_analysis.py
│   ├── rfe_features.py
│   ├── boruta_features.py
│   ├── permutation_importance.py
│   ├── hyperparameter_bayesian.py
│   ├── hyperparameter_grid_search.py
│   ├── generate_race_predictions.py
│   └── position_group_analysis_precompute.py
```

---

## Implementation Checklist

### Phase 1: Model Training (Priority: HIGH)
- [ ] Create `scripts/precompute/` directory
- [ ] Split `train_and_save_models.py` into model-specific scripts
- [ ] Create `.github/workflows/train-all-models.yml`
- [ ] Test parallel model training

### Phase 2: Feature Selection (Priority: HIGH)
- [ ] Create `monte_carlo_features.py`
- [ ] Create `shap_analysis.py`  
- [ ] Create `rfe_features.py`
- [ ] Create `boruta_features.py`
- [ ] Create `.github/workflows/feature-selection-suite.yml`
- [ ] Create `.github/workflows/feature-selection-monte-carlo.yml`

### Phase 3: Predictions (Priority: HIGH)
- [ ] Create `generate_race_predictions.py`
- [ ] Create `.github/workflows/precompute-predictions.yml`
- [ ] Add Monte Carlo rookie simulation to predictions

### Phase 4: Hyperparameters (Priority: MEDIUM)
- [ ] Create `hyperparameter_bayesian.py`
- [ ] Create `hyperparameter_grid_search.py`
- [ ] Create `.github/workflows/hyperparameter-optimization.yml`

### Phase 5: Position Analysis (Priority: MEDIUM)
- [ ] Create `position_group_analysis_precompute.py`
- [ ] Create `.github/workflows/position-analysis.yml`

### Phase 6: Orchestration (Priority: LOW)
- [ ] Create `.github/workflows/weekly-precompute-all.yml`
- [ ] Configure workflow dependencies

### Phase 7: UI Integration (Priority: HIGH)
- [ ] Add `load_precomputed_*` functions to `raceAnalysis.py`
- [ ] Update Tab 4 (Next Race) to use precomputed predictions
- [ ] Update Tab 5 (Predictive Models) to use precomputed analyses
- [ ] Add fallback to compute-on-demand if precomputed data missing
- [ ] Add "Last Updated" timestamps in UI

---

## Timeout Mitigation Strategies

1. **Parallel Jobs**: Split independent tasks into parallel jobs (model training, feature selection methods)

2. **Chunked Processing**: For Monte Carlo, split into multiple runs:
   ```yaml
   strategy:
     matrix:
       chunk: [1, 2, 3, 4, 5]
   ```
   Each chunk runs 200 trials, results merged in final job.

3. **Timeout Settings**: Set conservative timeouts per job:
   - Model training: 30-45 minutes each
   - Monte Carlo (1000 trials): 120 minutes
   - SHAP analysis: 60 minutes
   - Bayesian optimization: 180 minutes

4. **Early Stopping**: Add checkpointing to long-running scripts:
   ```python
   # Save intermediate results every N trials
   if trial_num % 100 == 0:
       save_checkpoint(results, checkpoint_path)
   ```

5. **Artifact Passing**: Use GitHub Actions artifacts to pass data between jobs without re-loading from scratch.

---

## Expected Performance Improvements

| Operation | Current Runtime | After Precomputation |
|-----------|----------------|---------------------|
| Page Load | 30-60s | <3s |
| Monte Carlo (1000 trials) | 5-15 min | Instant (precomputed) |
| SHAP Analysis | 5-10 min | Instant |
| Boruta Selection | 3-8 min | Instant |
| Hyperparameter Tuning | 10-30 min | Instant |
| Next Race Predictions | 10-20s | Instant |
| Position MAE Analysis | 15-30s | Instant |

**Total estimated runtime savings per user session: 20-60+ minutes**
