#!/usr/bin/env python3
"""ROADMAP-3 corrected benchmark.
Tests only what survives benchmarking:
  A) OLD: SimpleImputer + OHE (baseline)
  B) 3C+3D: SimpleImputer + TargetEncoder + RobustScaler (no IterativeImputer)
  C) 3A+3C+3D: Position Group with router + TargetEncoder + RobustScaler
"""
import warnings, time, json, pathlib
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, TargetEncoder
from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

DATA_DIR = pathlib.Path('data_files')
TARGET   = 'resultsFinalPositionNumber'
N_SPLITS = 5
N_TREES  = 200
SEED     = 42

print("Loading …", flush=True)
df = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)
df = df.dropna(subset=[TARGET])
df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
df = df.dropna(subset=[TARGET])
print(f"  {len(df):,} rows")

EXCLUDE = {TARGET,'grandPrixYear','round','raceId','raceId_results','driverId',
           'constructorId','grandPrixId','resultsYear','positionsGained',
           'finishingTime','timeMillis_results','LapTime_sec'}
HIGH_CARD = [c for c in ['constructorId','circuitId','resultsDriverName',
                          'grandPrixName','circuitRef'] if c in df.columns]
LOW_CARD  = [c for c in ['is_wet_race','had_grid_penalty','SafetyCarStatus',
                          'tyre_compound'] if c in df.columns]
bin_like  = {c for c in df.columns if c.endswith('_bin') or '_bin.' in c}
cat_set   = set(HIGH_CARD + LOW_CARD)
null_rate = df.isnull().mean()
pos_kw    = ('position','rank','grid','pos')

numeric_cols = [
    c for c in df.select_dtypes(include='number').columns
    if c not in EXCLUDE and c not in bin_like and c not in cat_set
    and null_rate[c] < 0.50 and not c.startswith('Unnamed')
]
pos_num  = [c for c in numeric_cols if any(kw in c.lower() for kw in pos_kw)]
reg_num  = [c for c in numeric_cols if c not in pos_num]
print(f"  {len(numeric_cols)} numeric ({len(pos_num)} pos/rank), "
      f"{len(HIGH_CARD)} high-card, {len(LOW_CARD)} low-card")

X = df[numeric_cols + HIGH_CARD + LOW_CARD].copy()
y = df[TARGET].astype(float).values
groups = df['grandPrixYear'].fillna(0).astype(int).values
cv = GroupKFold(n_splits=N_SPLITS)

# ─── A) OLD ──────────────────────────────────────────────────────────────────
def make_old_pre():
    return ColumnTransformer([
        ('num',  Pipeline([('i', SimpleImputer(strategy='median')),
                           ('s', StandardScaler())]), numeric_cols),
        ('clow', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                           ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                  sparse_output=False,
                                                  max_categories=50))]), LOW_CARD),
        ('chi',  Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                           ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                  sparse_output=False,
                                                  max_categories=50))]), HIGH_CARD),
    ], remainder='drop')

def make_xgb(): return XGBRegressor(n_estimators=N_TREES, max_depth=5,
                                     learning_rate=0.05, random_state=SEED,
                                     verbosity=0, n_jobs=-1)

print("\n[A] OLD: SimpleImputer + OHE", flush=True)
t0 = time.perf_counter()
sr = cross_val_score(Pipeline([('p', make_old_pre()), ('m', make_xgb())]),
                     X, y, groups=groups, cv=cv,
                     scoring='neg_mean_absolute_error', n_jobs=1)
old_mae, old_t = -sr.mean(), time.perf_counter()-t0
print(f"  MAE: {old_mae:.4f} ± {sr.std():.4f}  ({old_t:.1f}s)  folds={[-round(float(x),4) for x in sr]}")

# ─── B) 3C+3D: SimpleImputer + TargetEncoder + RobustScaler ──────────────────
def make_new_pre():
    return ColumnTransformer([
        ('reg',  Pipeline([('i', SimpleImputer(strategy='median')),
                           ('s', StandardScaler())]), reg_num),
        ('pos',  Pipeline([('i', SimpleImputer(strategy='median')),
                           ('s', RobustScaler())]), pos_num),        # 3D
        ('clow', Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                           ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                  sparse_output=False,
                                                  max_categories=50))]), LOW_CARD),
        ('chi',  Pipeline([('i', SimpleImputer(strategy='most_frequent')),
                           ('te', TargetEncoder(target_type='continuous',  # 3C
                                                smooth='auto', random_state=SEED))]),
         HIGH_CARD),
    ], remainder='drop')

print("\n[B] 3C+3D: SimpleImputer + TargetEncoder + RobustScaler", flush=True)
t0 = time.perf_counter()
sr = cross_val_score(Pipeline([('p', make_new_pre()), ('m', make_xgb())]),
                     X, y, groups=groups, cv=cv,
                     scoring='neg_mean_absolute_error', n_jobs=1)
new_mae, new_t = -sr.mean(), time.perf_counter()-t0
print(f"  MAE: {new_mae:.4f} ± {sr.std():.4f}  ({new_t:.1f}s)  folds={[-round(float(x),4) for x in sr]}")

# ─── C) 3A+3C+3D: Position Group with Router ────────────────────────────────
CENTRES = (2.0, 7.0, 15.0)

class PGE_Fixed(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        # Router: trained on ALL data for coarse position estimate
        self.router_ = XGBRegressor(n_estimators=150, learning_rate=0.1,
                                     max_depth=5, random_state=SEED,
                                     verbosity=0, n_jobs=-1).fit(X, y)
        # Sub-models: trained on ALL data but with range-emphasized weights.
        # This avoids the "biased training set" problem where a sub-model
        # trained only on pos 1-3 rows never sees backmarker features
        # and extrapolates wildly for mid-field drivers.
        n = 150
        w_pod = np.where(y <= 3,             8.0, 0.5)  # 16x emphasis on podium
        w_pts = np.where((y>=4)&(y<=10),    5.0, 0.5)  # 10x emphasis on points
        w_out = np.where(y > 10,             5.0, 0.5)  # 10x emphasis on outside
        self.m1_ = LGBMRegressor(n_estimators=n, learning_rate=0.1, num_leaves=31,
                                   random_state=SEED, verbose=-1, n_jobs=-1
                                   ).fit(X, y, sample_weight=w_pod)
        self.m2_ = CatBoostRegressor(iterations=n, learning_rate=0.1, depth=5,
                                      random_seed=SEED, verbose=0
                                      ).fit(X, y, sample_weight=w_pts)
        self.m3_ = XGBRegressor(n_estimators=n, learning_rate=0.1, max_depth=5,
                                  random_state=SEED, verbosity=0, n_jobs=-1
                                  ).fit(X, y, sample_weight=w_out)
        return self

    def predict(self, X):
        routing = self.router_.predict(X)
        p1, p2, p3 = self.m1_.predict(X), self.m2_.predict(X), self.m3_.predict(X)
        c1, c2, c3 = CENTRES
        w1 = 1.0 / (np.abs(routing - c1) + 1.0)
        w2 = 1.0 / (np.abs(routing - c2) + 1.0)
        w3 = 1.0 / (np.abs(routing - c3) + 1.0)
        tot = w1 + w2 + w3
        return (w1 * p1 + w2 * p2 + w3 * p3) / tot

print("\n[C] 3A+3C+3D: Position Group (router) + TargetEncoder + RobustScaler", flush=True)
t0 = time.perf_counter()
sr = cross_val_score(Pipeline([('p', make_new_pre()), ('m', PGE_Fixed())]),
                     X, y, groups=groups, cv=cv,
                     scoring='neg_mean_absolute_error', n_jobs=1)
pg_mae, pg_t = -sr.mean(), time.perf_counter()-t0
print(f"  MAE: {pg_mae:.4f} ± {sr.std():.4f}  ({pg_t:.1f}s)  folds={[-round(float(x),4) for x in sr]}")

# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"  [A] OLD (SimpleImputer + OHE)            : {old_mae:.4f}")
print(f"  [B] 3C+3D (TargetEnc + RobustScaler)     : {new_mae:.4f}   Δ={old_mae-new_mae:+.4f}")
print(f"  [C] 3A+3C+3D (PGE router + TargetEnc)   : {pg_mae:.4f}   Δ={old_mae-pg_mae:+.4f}")
print("="*70)

result = dict(
    old_mae=round(float(old_mae),4),
    new_pre_mae=round(float(new_mae),4),
    pos_group_mae=round(float(pg_mae),4),
    delta_B=round(float(old_mae-new_mae),4),
    delta_C=round(float(old_mae-pg_mae),4),
    cv_folds=N_SPLITS, n_trees=N_TREES,
)
out = DATA_DIR / 'roadmap3_corrected_benchmark.json'
json.dump(result, open(out,'w'), indent=2)
print(f"\nSaved → {out}")
