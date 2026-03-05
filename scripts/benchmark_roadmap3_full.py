#!/usr/bin/env python3
"""ROADMAP-3 benchmark — full feature set version.

Uses the same feature engineering as raceAnalysis.py (all numeric
non-bin columns with <50% NaN), to get a realistic MAE comparison.
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
TARGET   = 'resultsFinalPositionNumber'
N_SPLITS = 5
SEED     = 42
MAX_NAN_RATE = 0.50

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading f1ForAnalysis.csv …")
df = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)
print(f"  Rows: {len(df):,}   Cols: {len(df.columns):,}")

df = df.dropna(subset=[TARGET])
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df = df.dropna(subset=[TARGET])
print(f"  After target dropna: {len(df):,}")

# ─── Build feature sets ──────────────────────────────────────────────────────
# Exclude known-bad / target-leaking / identifier columns
EXCLUDE = {
    TARGET, 'grandPrixYear', 'round', 'raceId', 'raceId_results',
    'driverId', 'constructorId', 'grandPrixId',
    'resultsYear', 'positionsGained',   # pos gained = post-race info
    'finishingTime', 'timeMillis_results', 'LapTime_sec',
}
# High-cardinality categoricals → target-encode
HIGH_CARD = [c for c in ['constructorId', 'circuitId', 'resultsDriverName',
                          'grandPrixName', 'circuitRef']
             if c in df.columns]
# Low-cardinality booleans / flags → OHE
LOW_CARD  = [c for c in ['is_wet_race', 'had_grid_penalty', 'SafetyCarStatus',
                          'tyre_compound', 'circuit_type']
             if c in df.columns]

# All numeric non-bin columns with <50% missing, excluding the above lists
all_cols  = df.columns.tolist()
bin_like  = {c for c in all_cols if c.endswith('_bin') or '_bin.' in c}
cat_set   = set(HIGH_CARD + LOW_CARD)
null_rate = df.isnull().mean()

numeric_cols = [
    c for c in df.select_dtypes(include='number').columns
    if c not in EXCLUDE
    and c not in bin_like
    and c not in cat_set
    and null_rate[c] < MAX_NAN_RATE
    and not c.startswith('Unnamed')
]

print(f"  Numeric features: {len(numeric_cols)}")
print(f"  High-card cat   : {len(HIGH_CARD)}")
print(f"  Low-card cat    : {len(LOW_CARD)}")

X = df[numeric_cols + HIGH_CARD + LOW_CARD].copy()
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
print("\n[OLD] SimpleImputer + OneHotEncoder …")
old_num = Pipeline([
    ('imp',    SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])
old_cat_low  = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50)),
])
old_cat_high = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50)),
])
old_pre = ColumnTransformer([
    ('num',      old_num,      numeric_cols),
    ('cat_low',  old_cat_low,  LOW_CARD),
    ('cat_high', old_cat_high, HIGH_CARD),
], remainder='drop')

old_pipe = Pipeline([('pre', old_pre), ('xgb', make_xgb())])

t0 = time.perf_counter()
old_scores = cross_val_score(old_pipe, X, y, groups=groups, cv=cv,
                              scoring='neg_mean_absolute_error', n_jobs=1)
old_time = time.perf_counter() - t0
old_mae  = -old_scores.mean()
old_std  = old_scores.std()
print(f"  MAE: {old_mae:.4f} ± {old_std:.4f}  ({old_time:.1f}s)")

# ─── NEW preprocessor (IterativeImputer + TargetEncoder + RobustScaler) ─────
print("\n[NEW] IterativeImputer + TargetEncoder + RobustScaler …")

robust_cols = [c for c in numeric_cols
               if 'position' in c.lower() or 'rank' in c.lower() or 'pos' in c.lower()]
plain_cols  = [c for c in numeric_cols if c not in robust_cols]

new_plain = Pipeline([
    ('imp',    IterativeImputer(max_iter=10, random_state=SEED,
                                initial_strategy='median', skip_complete=True)),
    ('scaler', StandardScaler()),
])
new_robust = Pipeline([
    ('imp',    SimpleImputer(strategy='median')),
    ('scaler', RobustScaler()),
])
new_cat_low = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50)),
])
new_cat_high = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('te',  TargetEncoder(target_type='continuous', smooth='auto', random_state=SEED)),
])

transformers = [('cat_low', new_cat_low, LOW_CARD),
                ('cat_high', new_cat_high, HIGH_CARD)]
if plain_cols:
    transformers.insert(0, ('num', new_plain, plain_cols))
if robust_cols:
    transformers.insert(1, ('pos', new_robust, robust_cols))

new_pre  = ColumnTransformer(transformers, remainder='drop')
new_pipe = Pipeline([('pre', new_pre), ('xgb', make_xgb())])

t0 = time.perf_counter()
new_scores = cross_val_score(new_pipe, X, y, groups=groups, cv=cv,
                              scoring='neg_mean_absolute_error', n_jobs=1)
new_time = time.perf_counter() - t0
new_mae  = -new_scores.mean()
new_std  = new_scores.std()
print(f"  MAE: {new_mae:.4f} ± {new_std:.4f}  ({new_time:.1f}s)")

# ─── Position Group (3A): LGB podium + CAT points + XGB outside ─────────────
print("\n[POS-GROUP] Position-specific sub-models (LGB + CAT + XGB) …")

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin

class _PGE(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        pm = (y <= 3); pts = (y > 3) & (y <= 10); out = (y > 10)
        self.m1_ = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                  num_leaves=63, random_state=SEED,
                                  verbose=-1, n_jobs=-1).fit(X[pm],  y[pm])
        self.m2_ = CatBoostRegressor(iterations=300, learning_rate=0.05,
                                      depth=6, random_seed=SEED,
                                      verbose=0).fit(X[pts], y[pts])
        self.m3_ = XGBRegressor(n_estimators=300, learning_rate=0.05,
                                  max_depth=6, random_state=SEED,
                                  verbosity=0, n_jobs=-1).fit(X[out], y[out])
        return self
    def predict(self, X):
        return (self.m1_.predict(X) + self.m2_.predict(X) + self.m3_.predict(X)) / 3.0

pg_pipe = Pipeline([('pre', new_pre), ('pge', _PGE())])

t0 = time.perf_counter()
pg_scores = cross_val_score(pg_pipe, X, y, groups=groups, cv=cv,
                             scoring='neg_mean_absolute_error', n_jobs=1)
pg_time = time.perf_counter() - t0
pg_mae  = -pg_scores.mean()
pg_std  = pg_scores.std()
print(f"  MAE: {pg_mae:.4f} ± {pg_std:.4f}  ({pg_time:.1f}s)")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  Baseline OLD  (SimpleImputer + OHE)  : {old_mae:.4f} ± {old_std:.4f}")
print(f"  New preprocessor (Iter + TargetEnc)  : {new_mae:.4f} ± {new_std:.4f}   Δ={old_mae-new_mae:+.4f}")
print(f"  Position Group  (LGB+CAT+XGB blend)  : {pg_mae:.4f}  ± {pg_std:.4f}   Δ={old_mae-pg_mae:+.4f}")
print("="*60)

result = {
    'old_mae':     round(float(old_mae), 4),
    'new_pre_mae': round(float(new_mae), 4),
    'pos_group_mae': round(float(pg_mae), 4),
    'delta_new_pre': round(float(old_mae - new_mae), 4),
    'delta_pos_group': round(float(old_mae - pg_mae), 4),
    'n_numeric_features': len(numeric_cols),
    'n_high_card': len(HIGH_CARD),
    'n_low_card':  len(LOW_CARD),
    'cv_folds': N_SPLITS,
    'old_fold_maes':  [round(float(-x), 4) for x in old_scores],
    'new_fold_maes':  [round(float(-x), 4) for x in new_scores],
    'pg_fold_maes':   [round(float(-x), 4) for x in pg_scores],
}
out_path = DATA_DIR / 'roadmap3_benchmark_full.json'
json.dump(result, out_path.open('w'), indent=2)
print(f"\nResults → {out_path}")
