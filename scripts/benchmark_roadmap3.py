#!/usr/bin/env python3
"""ROADMAP-3 benchmark: compare old SimpleImputer+OHE preprocessing vs
new IterativeImputer+TargetEncoder+RobustScaler preprocessing.

Measures MAE using 5-fold season-stratified GroupKFold CV on XGBoost
so results are directly comparable to the project baseline.

Usage:
    python scripts/benchmark_roadmap3.py
"""
import warnings, time, json, sys, pathlib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, TargetEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

DATA_DIR = pathlib.Path('data_files')
TARGET = 'resultsFinalPositionNumber'
N_SPLITS = 5
SEED = 42

# ─── Load data ──────────────────────────────────────────────────────────────
print("Loading f1ForAnalysis.csv …")
df = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)
print(f"  Rows: {len(df):,}  Cols: {len(df.columns):,}")

df = df.dropna(subset=[TARGET])
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df = df.dropna(subset=[TARGET])

# ─── Feature selection (mirror raceAnalysis.py logic) ───────────────────────
HIGH_CARD = ['constructorId', 'circuitId', 'resultsDriverName']
LOW_CARD  = ['is_wet_race', 'had_grid_penalty', 'SafetyCarStatus']

NUMERIC_CANDIDATES = [
    'resultsGridPositionNumber', 'qualifyingPositionNumber',
    'constructor_recent_form_3_races', 'driver_recent_form_3_races',
    'practice_improvement', 'podium_potential', 'track_experience',
    'driverDNFCount', 'recent_dnf_rate_3_races', 'grid_x_avg_pit_time',
    'qualPos_x_last_practicePos', 'recent_form_x_qual',
    'numberOfStops', 'averageStopTime',
    'average_temp', 'total_precipitation', 'average_wind_speed',
    'best_s1_sec', 'best_s2_sec', 'best_s3_sec',
    'teammate_qual_delta', 'points_leader_gap',
    'tire_management_score', 'wet_race_vs_quali_delta',
]

# Keep only columns that actually exist
numeric_cols  = [c for c in NUMERIC_CANDIDATES if c in df.columns]
low_card_cols = [c for c in LOW_CARD  if c in df.columns]
high_card_cols= [c for c in HIGH_CARD if c in df.columns]

feature_cols  = numeric_cols + low_card_cols + high_card_cols
feature_cols  = [c for c in feature_cols if c in df.columns]
print(f"  Features used: {len(feature_cols)}  (numeric={len(numeric_cols)}, "
      f"low_card={len(low_card_cols)}, high_card={len(high_card_cols)})")

X = df[feature_cols].copy()
y = df[TARGET].astype(float).values
groups = df['grandPrixYear'].fillna(0).astype(int).values

# ─── XGBoost base estimator ─────────────────────────────────────────────────
def make_xgb():
    return XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbosity=0, n_jobs=-1,
    )

cv = GroupKFold(n_splits=N_SPLITS)

# ─── OLD preprocessor (SimpleImputer + OHE) ─────────────────────────────────
print("\n[OLD] Building SimpleImputer + OneHotEncoder pipeline …")
old_num = Pipeline([
    ('imp',    SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
old_cat_low = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
old_cat_high = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
old_pre = ColumnTransformer([
    ('num',      old_num,      numeric_cols),
    ('cat_low',  old_cat_low,  low_card_cols),
    ('cat_high', old_cat_high, high_card_cols),
], remainder='drop')

old_pipe = Pipeline([('pre', old_pre), ('xgb', make_xgb())])

t0 = time.perf_counter()
old_scores = cross_val_score(old_pipe, X, y, groups=groups, cv=cv,
                              scoring='neg_mean_absolute_error', n_jobs=1)
old_time = time.perf_counter() - t0
old_mae = -old_scores.mean()
old_std = old_scores.std()
print(f"  OLD MAE: {old_mae:.4f} ± {old_std:.4f}  ({old_time:.1f}s)")

# ─── NEW preprocessor (IterativeImputer + TargetEncoder + RobustScaler) ─────
print("\n[NEW] Building IterativeImputer + TargetEncoder + RobustScaler pipeline …")

robust_cols = [c for c in numeric_cols if 'position' in c.lower() or 'rank' in c.lower()]
plain_cols  = [c for c in numeric_cols if c not in robust_cols]

new_plain = Pipeline([
    ('imp',    IterativeImputer(max_iter=10, random_state=SEED,
                                initial_strategy='median',
                                min_value=0, skip_complete=True)),
    ('scaler', StandardScaler()),
])
new_robust = Pipeline([
    ('imp',    SimpleImputer(strategy='median')),  # IterativeImputer on sub-slice
    ('scaler', RobustScaler()),
])
new_cat_low = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
])
new_cat_high = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),   # TargetEncoder handles its own missings
    ('te',  TargetEncoder(target_type='continuous', smooth='auto', random_state=SEED)),
])

transformers = [('cat_low', new_cat_low, low_card_cols),
                ('cat_high', new_cat_high, high_card_cols)]
if plain_cols:
    transformers.insert(0, ('num', new_plain, plain_cols))
if robust_cols:
    transformers.insert(1, ('pos', new_robust, robust_cols))

new_pre = ColumnTransformer(transformers, remainder='drop')
new_pipe = Pipeline([('pre', new_pre), ('xgb', make_xgb())])

t0 = time.perf_counter()
new_scores = cross_val_score(new_pipe, X, y, groups=groups, cv=cv,
                              scoring='neg_mean_absolute_error', n_jobs=1)
new_time = time.perf_counter() - t0
new_mae = -new_scores.mean()
new_std = new_scores.std()
print(f"  NEW MAE: {new_mae:.4f} ± {new_std:.4f}  ({new_time:.1f}s)")

# ─── Summary ─────────────────────────────────────────────────────────────────
delta   = old_mae - new_mae
pct     = delta / old_mae * 100
outcome = "IMPROVEMENT ✅" if delta > 0 else "REGRESSION ❌"

print("\n" + "="*55)
print(f"  OLD MAE  : {old_mae:.4f}")
print(f"  NEW MAE  : {new_mae:.4f}")
print(f"  Delta    : {delta:+.4f}  ({pct:+.2f}%)  — {outcome}")
print(f"  Folds    : {old_scores.tolist()}")
print(f"  New folds: {new_scores.tolist()}")
print("="*55)

# ─── Persist results ─────────────────────────────────────────────────────────
result = {
    'old_mae': round(float(old_mae), 4),
    'old_std': round(float(old_std), 4),
    'new_mae': round(float(new_mae), 4),
    'new_std': round(float(new_std), 4),
    'delta':   round(float(delta), 4),
    'pct_improvement': round(float(pct), 2),
    'n_features': len(feature_cols),
    'cv_folds': N_SPLITS,
    'old_fold_maes': [round(float(-x), 4) for x in old_scores],
    'new_fold_maes': [round(float(-x), 4) for x in new_scores],
}
out_path = DATA_DIR / 'roadmap3_benchmark_results.json'
json.dump(result, out_path.open('w'), indent=2)
print(f"\nResults saved → {out_path}")
