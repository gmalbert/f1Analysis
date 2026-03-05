#!/usr/bin/env python3
"""ROADMAP-3 final benchmark — scoped IterativeImputer + Position Group.

Fast version: 100 estimators, 3-fold CV for speed.
Compares:
  A) OLD: SimpleImputer + OneHotEncoder
  B) FIXED NEW: IterativeImputer on sparse-only + TargetEncoder + RobustScaler
  C) FIXED NEW + Position Group (LGB podium / CAT points / XGB outside)
"""
import warnings, time, json, pathlib, sys
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
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

DATA_DIR = pathlib.Path('data_files')
TARGET   = 'resultsFinalPositionNumber'
N_SPLITS = 3     # 3-fold for speed
N_TREES  = 100   # smaller for speed
SEED = 42

# ─── Load ───────────────────────────────────────────────────────────────────
print("Loading …", flush=True)
df = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)
df = df.dropna(subset=[TARGET])
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df = df.dropna(subset=[TARGET])
print(f"  {len(df):,} rows")

EXCLUDE = {
    TARGET, 'grandPrixYear', 'round', 'raceId', 'raceId_results',
    'driverId', 'constructorId', 'grandPrixId', 'resultsYear',
    'positionsGained', 'finishingTime', 'timeMillis_results', 'LapTime_sec',
}
HIGH_CARD = [c for c in ['constructorId', 'circuitId', 'resultsDriverName',
                          'grandPrixName', 'circuitRef'] if c in df.columns]
LOW_CARD  = [c for c in ['is_wet_race', 'had_grid_penalty', 'SafetyCarStatus',
                          'tyre_compound', 'circuit_type'] if c in df.columns]
bin_like  = {c for c in df.columns if c.endswith('_bin') or '_bin.' in c}
cat_set   = set(HIGH_CARD + LOW_CARD)
null_rate = df.isnull().mean()

numeric_cols = [
    c for c in df.select_dtypes(include='number').columns
    if c not in EXCLUDE and c not in bin_like and c not in cat_set
    and null_rate[c] < 0.50 and not c.startswith('Unnamed')
]
print(f"  {len(numeric_cols)} numeric, {len(HIGH_CARD)} high-card, {len(LOW_CARD)} low-card")

X = df[numeric_cols + HIGH_CARD + LOW_CARD].copy()
y = df[TARGET].astype(float).values
groups = df['grandPrixYear'].fillna(0).astype(int).values

cv = GroupKFold(n_splits=N_SPLITS)

# ─── Helper: build OLD preprocessor ─────────────────────────────────────────
def make_old_pre():
    return ColumnTransformer([
        ('num',  Pipeline([('imp', SimpleImputer(strategy='median')),
                           ('sc', StandardScaler())]), numeric_cols),
        ('clow', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                           ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                  sparse_output=False,
                                                  max_categories=50))]), LOW_CARD),
        ('chi',  Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                           ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                  sparse_output=False,
                                                  max_categories=50))]), HIGH_CARD),
    ], remainder='drop')

# ─── Helper: build FIXED NEW preprocessor ───────────────────────────────────
def make_new_pre(X_ref):
    nr = X_ref.isnull().mean()
    _pos_kw = ('position', 'rank', 'grid', 'pos')
    pos_num  = [c for c in numeric_cols if any(kw in c.lower() for kw in _pos_kw)]
    reg_num  = [c for c in numeric_cols if c not in pos_num]

    sparse  = lambda cols: [c for c in cols if 0.05 <= nr.get(c, 0) <= 0.50]
    dense   = lambda cols: [c for c in cols if c not in sparse(cols)]

    reg_s, reg_d = sparse(reg_num), dense(reg_num)
    pos_s, pos_d = sparse(pos_num), dense(pos_num)

    print(f"    sparse cols (IterativeImputer): {len(reg_s)+len(pos_s)} "
          f"  dense (SimpleImputer): {len(reg_d)+len(pos_d)}")

    trf = []
    if reg_d: trf.append(('reg_d', Pipeline([('i', SimpleImputer(strategy='median')),
                                              ('s', StandardScaler())]), reg_d))
    if reg_s: trf.append(('reg_s', Pipeline([('i', IterativeImputer(max_iter=10,
                                                random_state=SEED, initial_strategy='median',
                                                min_value=0, skip_complete=True)),
                                              ('s', StandardScaler())]), reg_s))
    if pos_d: trf.append(('pos_d', Pipeline([('i', SimpleImputer(strategy='median')),
                                              ('s', RobustScaler())]), pos_d))
    if pos_s: trf.append(('pos_s', Pipeline([('i', IterativeImputer(max_iter=10,
                                                random_state=SEED, initial_strategy='median',
                                                min_value=0, skip_complete=True)),
                                              ('s', RobustScaler())]), pos_s))
    trf.append(('clow', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                                   ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                          sparse_output=False,
                                                          max_categories=50))]), LOW_CARD))
    trf.append(('chi',  Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                                   ('te', TargetEncoder(target_type='continuous',
                                                         smooth='auto',
                                                         random_state=SEED))]), HIGH_CARD))
    return ColumnTransformer(trf, remainder='drop')

def make_xgb(): return XGBRegressor(n_estimators=N_TREES, max_depth=5,
                                      learning_rate=0.1, random_state=SEED,
                                      verbosity=0, n_jobs=-1)

# ─── A) OLD ──────────────────────────────────────────────────────────────────
print("\n[A] OLD: SimpleImputer + OHE", flush=True)
t0 = time.perf_counter()
old_scores = cross_val_score(Pipeline([('p', make_old_pre()), ('m', make_xgb())]),
                              X, y, groups=groups, cv=cv,
                              scoring='neg_mean_absolute_error', n_jobs=1)
old_mae, old_t = -old_scores.mean(), time.perf_counter()-t0
print(f"  MAE: {old_mae:.4f} ± {old_scores.std():.4f}  ({old_t:.1f}s)")

# ─── B) FIXED NEW pre + XGB ───────────────────────────────────────────────────
print("\n[B] NEW: Scoped IterativeImputer + TargetEncoder + RobustScaler + XGB", flush=True)
new_pre = make_new_pre(X)
t0 = time.perf_counter()
new_scores = cross_val_score(Pipeline([('p', new_pre), ('m', make_xgb())]),
                              X, y, groups=groups, cv=cv,
                              scoring='neg_mean_absolute_error', n_jobs=1)
new_mae, new_t = -new_scores.mean(), time.perf_counter()-t0
print(f"  MAE: {new_mae:.4f} ± {new_scores.std():.4f}  ({new_t:.1f}s)")

# ─── C) FIXED NEW pre + Position Group ───────────────────────────────────────
print("\n[C] Position Group (LGB podium + CAT points + XGB outside)", flush=True)

class PGE(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        pm  = (y <= 3);  pts = (y > 3) & (y <= 10);  out = (y > 10)
        LR  = LGBMRegressor(n_estimators=N_TREES, learning_rate=0.1, num_leaves=31,
                             random_state=SEED, verbose=-1, n_jobs=-1)
        CR  = CatBoostRegressor(iterations=N_TREES, learning_rate=0.1, depth=5,
                                 random_seed=SEED, verbose=0)
        XR  = XGBRegressor(n_estimators=N_TREES, learning_rate=0.1, max_depth=5,
                             random_state=SEED, verbosity=0, n_jobs=-1)
        # Fallback: if a group is empty use all data
        self.m1_ = LR.fit(X[pm]  if pm.sum()  > 10 else X, y[pm]  if pm.sum()  > 10 else y)
        self.m2_ = CR.fit(X[pts] if pts.sum() > 10 else X, y[pts] if pts.sum() > 10 else y)
        self.m3_ = XR.fit(X[out] if out.sum() > 10 else X, y[out] if out.sum() > 10 else y)
        return self
    def predict(self, X):
        return (self.m1_.predict(X) + self.m2_.predict(X) + self.m3_.predict(X)) / 3.0

pg_pre = make_new_pre(X)
t0 = time.perf_counter()
pg_scores = cross_val_score(Pipeline([('p', pg_pre), ('m', PGE())]),
                             X, y, groups=groups, cv=cv,
                             scoring='neg_mean_absolute_error', n_jobs=1)
pg_mae, pg_t = -pg_scores.mean(), time.perf_counter()-t0
print(f"  MAE: {pg_mae:.4f} ± {pg_scores.std():.4f}  ({pg_t:.1f}s)")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print(f"  [A] OLD  SimpleImputer + OHE                  : {old_mae:.4f}")
print(f"  [B] NEW  Scoped Iter + TargetEnc + Robust     : {new_mae:.4f}   Δ={old_mae-new_mae:+.4f}")
print(f"  [C] POS-GROUP  (LGB+CAT+XGB, new pre)         : {pg_mae:.4f}   Δ={old_mae-pg_mae:+.4f}")
print("="*65)

result = dict(
    old_mae=round(float(old_mae),4),
    new_pre_mae=round(float(new_mae),4),
    pos_group_mae=round(float(pg_mae),4),
    delta_new_pre=round(float(old_mae-new_mae),4),
    delta_pos_group=round(float(old_mae-pg_mae),4),
    n_numeric=len(numeric_cols), cv_folds=N_SPLITS, n_trees=N_TREES,
    old_folds=[round(float(-x),4) for x in old_scores],
    new_folds=[round(float(-x),4) for x in new_scores],
    pg_folds=[round(float(-x),4) for x in pg_scores],
)
out = DATA_DIR / 'roadmap3_benchmark_final.json'
json.dump(result, open(out,'w'), indent=2)
print(f"\nSaved → {out}")
