# Roadmap Part 3: Model Architecture Improvements for MAE ≤ 1.5

**Baseline after ROADMAP-1: 1.69 (80/20) / 1.80 (GroupKFold) → Target: ≤ 1.5 | Estimated impact of this section: 0.15–0.25**

---

## ⏳ 3A. Position-Specific Sub-Models (HIGH IMPACT) — pending

The current single-model approach tries to minimize average error across all positions. But predicting position 1 (winner) is a very different problem from predicting positions 10–15. Position-specific models allow the algorithm to learn different signal patterns for each outcome segment.

### Architecture

```
Race → 3 parallel models merge ranked predictions:
  - Model A: "Podium contenders" (positions 1–3) → LightGBM with class_weight emphasis
  - Model B: "Points scorers" (positions 4–10) → CatBoost (handles dense, competitive range)
  - Model C: "Outside points + DNFs" (positions 11–20+) → XGBoost with DNF flag
  - Ensemble blender: Weighted stacking using GroupKFold CV predictions
```

### Implementation in `raceAnalysis.py`

Add a new function `train_position_group_model()` and integrate with the existing model training flow (around line ~3782 in the Tab5 section):

```python
def train_position_group_models(df_train, features, CACHE_VERSION):
    """Train three position-specific sub-models and return a blended predictor."""
    from sklearn.base import BaseEstimator, RegressorMixin

    X = df_train[features].copy()
    y = df_train['resultsFinalPositionNumber'].copy()

    preprocessor = get_preprocessor_position(X)
    X_prep = preprocessor.fit_transform(X)

    # Define splits
    podium_mask = y <= 3
    points_mask = (y > 3) & (y <= 10)
    outside_mask = y > 10

    models = {}
    from lightgbm import LGBMRegressor
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor

    models['podium'] = LGBMRegressor(
        n_estimators=300, learning_rate=0.05, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    ).fit(X_prep[podium_mask], y[podium_mask])

    models['points'] = CatBoostRegressor(
        iterations=300, learning_rate=0.05, depth=6,
        random_seed=42, verbose=0
    ).fit(X_prep[points_mask], y[points_mask])

    models['outside'] = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    ).fit(X_prep[outside_mask], y[outside_mask])

    return models, preprocessor

def predict_with_position_models(models, preprocessor, X_pred):
    """Blend predictions from all three position-specific sub-models."""
    X_prep = preprocessor.transform(X_pred)
    preds = np.column_stack([
        models['podium'].predict(X_prep),
        models['points'].predict(X_prep),
        models['outside'].predict(X_prep),
    ])
    # Simple equal-weight blend for now; adjust via CV
    return preds.mean(axis=1)
```

**Estimated MAE impact:** 0.06–0.10  
**Effort:** 4–6 hrs

---

## ⏳ 3B. IterativeImputer for Missing Features (vs. Median Fill) — pending

Currently, all NaN values are filled with column medians or zeros before model training. `IterativeImputer` uses other features to predict missing values, which can recover significant information from sparse features like `tire_management_score` or `wet_race_vs_quali_delta`.

**File:** `raceAnalysis.py` — modify `get_preprocessor_position()` (~line 2100):

```python
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

# Replace SimpleImputer with IterativeImputer for numeric features:
numeric_transformer = Pipeline(steps=[
    ('imputer', IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy='median',
        min_value=0,         # Positions are non-negative
        skip_complete=True,  # Skip columns with no NaNs for speed
    )),
    ('scaler', StandardScaler()),
])
```

**Note:** IterativeImputer adds ~2–5 sec overhead per training call. Cache the preprocessor when model type doesn't change.

**Estimated MAE impact:** 0.02–0.04  
**Effort:** 1–2 hrs

---

## ⏳ 3C. Target Encoding for Categorical Features — pending

`constructorId`, `circuitId`, `driverId` are currently one-hot encoded, producing sparse, high-cardinality matrices. Target encoding replaces each category with the mean target value (with shrinkage toward the global mean), which is significantly more information-dense and reduces model complexity.

**File:** `raceAnalysis.py` — modify `get_preprocessor_position()`:

```python
from sklearn.preprocessing import TargetEncoder  # sklearn >= 1.3

# Replace OneHotEncoder for high-cardinality fields:
high_card_transformer = Pipeline(steps=[
    ('target_enc', TargetEncoder(
        target_type='continuous',
        smooth='auto',         # Bayesian shrinkage
        random_state=42,
    )),
])

# In ColumnTransformer, replace categorical_cols entry:
categorical_low = ['is_wet_race', 'had_grid_penalty', ...]   # Still one-hot
categorical_high = ['constructorId', 'circuitId', 'resultsDriverName']  # Target-encode

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat_low', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_low),
    ('cat_high', high_card_transformer, categorical_high),
])
```

**Estimated MAE impact:** 0.02–0.04  
**Effort:** 2–3 hrs

---

## ⏳ 3D. Outlier Capping & Robust Normalization — pending

Extreme outlier positions (DNFs coded as 20+, retirements in lap 1) skew the distribution and pull gradient-boosted trees toward incorrect split points. Capping at the 95th percentile + applying RobustScaler for position-adjacent features improves stability.

**File:** `raceAnalysis.py` — add a preprocessing step before feature engineering:

```python
from sklearn.preprocessing import RobustScaler

# Cap target variable
y_capped = y.clip(upper=20)  # Cap DNF at 20 (max finishers)

# For gradients, use RobustScaler instead of StandardScaler on a subset of features:
#   - 'resultsFinalPositionNumber' (target)
#   - Any feature that includes position numbers in its derivation
robust_cols = [c for c in X.columns if 'position' in c.lower() or 'rank' in c.lower()]

# In ColumnTransformer, add robust_scaler transformer for robust_cols:
('pos_robust', RobustScaler(), robust_cols),
```

**Estimated MAE impact:** 0.01–0.02  
**Effort:** 1 hr

---

## ⏳ 3E. Track-Type Ensemble Weights — pending

Some models perform better on specific circuit archetypes. Calibrating ensemble weights per track type (street circuit vs. high-speed vs. technical) can shave a couple tenths off MAE.

### Circuit type classification

```python
# Add to data_files/circuit_altitude.csv (new column):
# circuit_type: 'street' | 'high_speed' | 'technical' | 'mixed'

CIRCUIT_TYPES = {
    'monaco': 'street',
    'baku': 'street',
    'jeddah': 'street',
    'albert_park': 'street',  # partially
    'spa': 'high_speed',
    'monza': 'high_speed',
    'suzuka': 'technical',
    'silverstone': 'high_speed',
    'red_bull_ring': 'technical',
    'americas': 'mixed',
}
```

### Dynamic ensemble weighting

```python
# In raceAnalysis.py, before final prediction blending:
CIRCUIT_ENSEMBLE_WEIGHTS = {
    'street':      {'xgb': 0.30, 'lgbm': 0.25, 'cat': 0.45},  # CatBoost better on street
    'high_speed':  {'xgb': 0.45, 'lgbm': 0.35, 'cat': 0.20},  # XGBoost better on fast tracks
    'technical':   {'xgb': 0.35, 'lgbm': 0.40, 'cat': 0.25},  # LightGBM better on complex
    'mixed':       {'xgb': 0.33, 'lgbm': 0.33, 'cat': 0.34},  # Equal weight default
}

circuit_type = get_circuit_type(current_circuit_ref)  # helper lookup
weights = CIRCUIT_ENSEMBLE_WEIGHTS.get(circuit_type, CIRCUIT_ENSEMBLE_WEIGHTS['mixed'])

final_pred = (
    weights['xgb'] * pred_xgb +
    weights['lgbm'] * pred_lgbm +
    weights['cat'] * pred_cat
)
```

**Calibrate weights using GroupKFold CV where groups = circuit_type.**

**Estimated MAE impact:** 0.02–0.04  
**Effort:** 2–3 hrs

---

## ⏳ 3F. Optuna Monthly Hyperparameter Refresh — pending

The current Optuna HPO runs only in the weekly Monte Carlo workflow (feature selection mode). A separate monthly hyperparameter refresh with a full Optuna study on the final feature set will keep model params optimal as new race data accumulates.

### New GitHub Actions workflow: `.github/workflows/monthly-hpo.yml`

```yaml
name: Monthly Hyperparameter Optimization

on:
  schedule:
    - cron: '0 2 1 * *'   # 1st of every month at 2 AM UTC
  workflow_dispatch:

jobs:
  hpo:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install -r requirements.txt
      - name: Run monthly HPO
        run: python scripts/precompute/monthly_hpo.py --trials 200 --cv-folds 5
      - name: Commit updated params
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: 'chore: monthly hyperparameter refresh [skip ci]'
          file_pattern: 'data_files/best_hyperparams_*.json'
```

### New script: `scripts/precompute/monthly_hpo.py`

```python
#!/usr/bin/env python3
"""Monthly Optuna hyperparameter optimization.
Runs full trials on all four model types, writes best params to JSON.
"""
import optuna
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

DATA_DIR = Path('data_files')

def load_data():
    df = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)
    # Load best feature set from Monte Carlo output
    feat_file = DATA_DIR / 'precomputed_features' / 'best_features.txt'
    if feat_file.exists():
        features = feat_file.read_text().strip().split('\n')
    else:
        raise FileNotFoundError("Run Monte Carlo feature selection first.")
    target = 'resultsFinalPositionNumber'
    valid = df[features + [target, 'grandPrixYear']].dropna(subset=[target])
    return valid[features], valid[target], valid['grandPrixYear']

def objective_xgb(trial, X, y, groups):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 800),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42, 'verbosity': 0,
    }
    model = XGBRegressor(**params)
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='neg_mean_absolute_error')
    return -scores.mean()

def run_hpo(model_type, n_trials):
    X, y, seasons = load_data()
    X_num = X.select_dtypes(include='number').fillna(X.median())
    groups = seasons.values

    objectives = {'xgb': objective_xgb}  # Add lgbm, cat similarly
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objectives[model_type](t, X_num, y, groups), n_trials=n_trials)

    best = study.best_params
    out_file = DATA_DIR / f'best_hyperparams_{model_type}.json'
    json.dump(best, out_file.open('w'), indent=2)
    print(f"Best {model_type} MAE: {study.best_value:.4f}")
    print(f"Params saved to {out_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--cv-folds', type=int, default=5)
    args = parser.parse_args()
    for mt in ['xgb']:  # Expand to ['xgb', 'lgbm', 'cat'] as needed
        run_hpo(mt, args.trials)
```

**Estimated MAE impact:** 0.02–0.04 (from optimized hyperparameters)  
**Effort:** 3–4 hrs

---

## Summary

| Improvement | Est. MAE Impact | Effort | Priority | Status |
|-------------|----------------|--------|----------|--------|
| Position-specific sub-models | 0.06–0.10 | 4–6 hrs | **P1** | ⏳ pending |
| IterativeImputer (vs median) | 0.02–0.04 | 1–2 hrs | **P1** | ⏳ pending |
| Target encoding (high-cardinality) | 0.02–0.04 | 2–3 hrs | **P2** | ⏳ pending |
| Outlier capping + RobustScaler | 0.01–0.02 | 1 hr | **P2** | ⏳ pending |
| Track-type ensemble weighting | 0.02–0.04 | 2–3 hrs | **P2** | ⏳ pending |
| Monthly Optuna HPO refresh | 0.02–0.04 | 3–4 hrs | **P2** | ⏳ pending |
| **Total estimated** | **0.15–0.28** | | |

**Combined with ROADMAP_1 + ROADMAP_2: MAE → ~1.40–1.65**  
**Confidence in hitting ≤1.5: ~70% if all three roadmaps are fully implemented**
