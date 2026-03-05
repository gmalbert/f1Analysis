#!/usr/bin/env python3
"""
calibrate_circuit_weights.py
============================
Fit per-circuit-type blend weights for TrackWeightedEnsemble (ROADMAP-3E)
using out-of-fold predictions from XGBoost, LightGBM, and CatBoost.

Algorithm
---------
1. Load f1ForAnalysis.csv and build the same feature set used in training.
2. Collect out-of-fold (OOF) predictions from XGB / LGBM / CAT via
   GroupKFold(n_splits=5, group=grandPrixYear).
3. For each circuit type (street / high_speed / technical / mixed + individual
   circuit overrides), solve:

       min || y_true - (w_xgb*p_xgb + w_lgbm*p_lgbm + w_cat*p_cat) ||_2
   s.t. w_i >= 0

   using scipy.optimize.nnls, then L1-normalise so the weights sum to 1.

4. Validate: compare OOF MAE of calibrated weights vs the hardcoded defaults.
5. Write data_files/circuit_ensemble_weights.json  (format matches
   CIRCUIT_ENSEMBLE_WEIGHTS in raceAnalysis.py so the app can load it at
   startup).

Usage
-----
    python scripts/calibrate_circuit_weights.py

Options (env-vars)
------------------
    N_SPLITS   – number of CV folds  (default 5)
    N_TREES    – estimator count      (default 300)
    SEED       – random seed          (default 42)
"""

import json
import os
import pathlib
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, TargetEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path("data_files")
OUTPUT   = DATA_DIR / "circuit_ensemble_weights.json"
TARGET   = "resultsFinalPositionNumber"
N_SPLITS = int(os.environ.get("N_SPLITS", 5))
N_TREES  = int(os.environ.get("N_TREES",  300))
SEED     = int(os.environ.get("SEED",      42))

# Circuit-type map — kept identical to CIRCUIT_TYPES in raceAnalysis.py
CIRCUIT_TYPES: dict[str, str] = {
    "monaco":        "street",
    "baku":          "street",
    "jeddah":        "street",
    "albert_park":   "street",
    "las_vegas":     "street",
    "miami":         "street",
    "singapore":     "street",
    "spa":           "high_speed",
    "monza":         "high_speed",
    "silverstone":   "high_speed",
    "bahrain":       "high_speed",
    "interlagos":    "high_speed",
    "yas_marina":    "high_speed",
    "suzuka":        "technical",
    "red_bull_ring": "technical",
    "hungaroring":   "technical",
    "zandvoort":     "technical",
    "imola":         "technical",
    "barcelona":     "technical",
    "americas":      "mixed",
    "mexico_city":   "mixed",
    "shanghai":      "mixed",
    "sochi":         "mixed",
}

# Melbourne circuitId alias to match CIRCUIT_TYPES (albert_park)
CIRCUIT_ID_ALIASES = {
    "melbourne": "albert_park",
    "austin":    "americas",
}

# Hardcoded defaults (baseline) — from raceAnalysis.py CIRCUIT_ENSEMBLE_WEIGHTS
DEFAULT_WEIGHTS: dict[str, dict] = {
    "street":     {"xgb": 0.30, "lgbm": 0.25, "cat": 0.45},
    "high_speed": {"xgb": 0.45, "lgbm": 0.35, "cat": 0.20},
    "technical":  {"xgb": 0.35, "lgbm": 0.40, "cat": 0.25},
    "mixed":      {"xgb": 0.33, "lgbm": 0.33, "cat": 0.34},
}

# ── Data loading ─────────────────────────────────────────────────────────────
print("Loading f1ForAnalysis.csv …", flush=True)
df = pd.read_csv(DATA_DIR / "f1ForAnalysis.csv", sep="\t", low_memory=False)
df = df.dropna(subset=[TARGET])
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
print(f"  {len(df):,} rows after dropping missing target")

# Attach circuit_type per row
def _circuit_type(cid: str) -> str:
    if not isinstance(cid, str):
        return "mixed"
    cid = cid.lower().replace(" ", "_").replace("-", "_")
    cid = CIRCUIT_ID_ALIASES.get(cid, cid)
    return CIRCUIT_TYPES.get(cid, "mixed")

df["_circuit_type"] = df["circuitId"].apply(_circuit_type)
print("Circuit type distribution:")
print(df.groupby("_circuit_type")["grandPrixYear"].nunique().rename("seasons").to_string())
print()

# ── Feature selection (mirrors benchmark_roadmap3_corrected.py) ───────────────
EXCLUDE = {
    TARGET, "grandPrixYear", "round", "raceId", "raceId_results", "driverId",
    "constructorId", "grandPrixId", "resultsYear", "positionsGained",
    "finishingTime", "timeMillis_results", "LapTime_sec", "_circuit_type",
    "circuitId",
}
HIGH_CARD = [c for c in ["constructorId", "circuitId", "resultsDriverName",
                          "grandPrixName", "circuitRef"] if c in df.columns and c not in EXCLUDE]
LOW_CARD  = [c for c in ["is_wet_race", "had_grid_penalty", "SafetyCarStatus",
                          "tyre_compound"] if c in df.columns]
bin_like  = {c for c in df.columns if c.endswith("_bin") or "_bin." in c}
cat_set   = set(HIGH_CARD + LOW_CARD)
null_rate = df.isnull().mean()
pos_kw    = ("position", "rank", "grid", "pos")

numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in EXCLUDE and c not in bin_like and c not in cat_set
    and null_rate[c] < 0.50 and not c.startswith("Unnamed")
]
pos_num  = [c for c in numeric_cols if any(kw in c.lower() for kw in pos_kw)]
reg_num  = [c for c in numeric_cols if c not in pos_num]

print(f"Features: {len(numeric_cols)} numeric ({len(pos_num)} pos/rank), "
      f"{len(HIGH_CARD)} high-card, {len(LOW_CARD)} low-card")

feature_cols = numeric_cols + HIGH_CARD + LOW_CARD
X = df[feature_cols].copy()
y = df[TARGET].astype(float).values
groups = df["grandPrixYear"].fillna(0).astype(int).values
circuit_types = df["_circuit_type"].values


# ── Preprocessor ─────────────────────────────────────────────────────────────
def build_preprocessor():
    transformers = []
    if reg_num:
        transformers.append(("num", Pipeline([
            ("i", SimpleImputer(strategy="median")),
            ("s", StandardScaler()),
        ]), reg_num))
    if pos_num:
        transformers.append(("pos", Pipeline([
            ("i", SimpleImputer(strategy="median")),
            ("s", RobustScaler()),
        ]), pos_num))
    if LOW_CARD:
        transformers.append(("lo", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                   max_categories=50)),
        ]), LOW_CARD))
    if HIGH_CARD:
        transformers.append(("hi", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("te", TargetEncoder(target_type="continuous", smooth="auto",
                                  random_state=SEED)),
        ]), HIGH_CARD))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ── OOF prediction collection ─────────────────────────────────────────────────
def collect_oof_predictions():
    """Return (oof_xgb, oof_lgbm, oof_cat, y_oof, ct_oof) arrays."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    n   = len(y)
    oof_xgb  = np.full(n, np.nan)
    oof_lgbm = np.full(n, np.nan)
    oof_cat  = np.full(n, np.nan)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), 1):
        t0 = time.perf_counter()
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx],       y[va_idx]

        pre = build_preprocessor()
        X_tr_t = pre.fit_transform(X_tr, y_tr)
        X_va_t = pre.transform(X_va)

        # XGBoost
        xgb_m = XGBRegressor(n_estimators=N_TREES, max_depth=5, learning_rate=0.05,
                              random_state=SEED, verbosity=0, n_jobs=-1)
        xgb_m.fit(X_tr_t, y_tr)
        oof_xgb[va_idx] = xgb_m.predict(X_va_t)

        # LightGBM
        lgbm_m = LGBMRegressor(n_estimators=N_TREES, num_leaves=63, learning_rate=0.05,
                                random_state=SEED, verbose=-1, n_jobs=-1)
        lgbm_m.fit(X_tr_t, y_tr)
        oof_lgbm[va_idx] = lgbm_m.predict(X_va_t)

        # CatBoost
        cat_m = CatBoostRegressor(iterations=N_TREES, learning_rate=0.05, depth=5,
                                   random_seed=SEED, verbose=0)
        cat_m.fit(X_tr_t, y_tr)
        oof_cat[va_idx] = cat_m.predict(X_va_t)

        elapsed = time.perf_counter() - t0
        # Per-fold MAEs
        m_xgb  = np.mean(np.abs(y_va - oof_xgb[va_idx]))
        m_lgbm = np.mean(np.abs(y_va - oof_lgbm[va_idx]))
        m_cat  = np.mean(np.abs(y_va - oof_cat[va_idx]))
        print(f"  fold {fold}/{N_SPLITS}  XGB={m_xgb:.4f}  LGBM={m_lgbm:.4f}  "
              f"CAT={m_cat:.4f}  ({elapsed:.1f}s)", flush=True)

    valid = ~(np.isnan(oof_xgb) | np.isnan(oof_lgbm) | np.isnan(oof_cat))
    return (oof_xgb[valid], oof_lgbm[valid], oof_cat[valid],
            y[valid], circuit_types[valid])


print(f"Collecting OOF predictions ({N_SPLITS} folds, {N_TREES} trees each) …")
t_start = time.perf_counter()
p_xgb, p_lgbm, p_cat, y_oof, ct_oof = collect_oof_predictions()
print(f"OOF collection done in {time.perf_counter()-t_start:.1f}s\n")


# ── NNLS weight calibration per circuit type ─────────────────────────────────
def calibrate_weights(mask: np.ndarray) -> dict:
    """Solve NNLS for rows where mask==True; return normalised weight dict."""
    A = np.column_stack([p_xgb[mask], p_lgbm[mask], p_cat[mask]])
    b = y_oof[mask]
    w, _ = nnls(A, b)
    total = w.sum()
    if total < 1e-9:
        w = np.array([1/3, 1/3, 1/3])
    else:
        w = w / total
    return {"xgb": round(float(w[0]), 4),
            "lgbm": round(float(w[1]), 4),
            "cat":  round(float(w[2]), 4)}


def weighted_mae(weights: dict, mask: np.ndarray) -> float:
    blend = (weights["xgb"]  * p_xgb[mask] +
             weights["lgbm"] * p_lgbm[mask] +
             weights["cat"]  * p_cat[mask])
    return float(np.mean(np.abs(y_oof[mask] - blend)))


print("=" * 60)
print("NNLS weight calibration per circuit type")
print("=" * 60)

calibrated: dict[str, dict] = {}
all_types = sorted(set(ct_oof))

for ct in all_types:
    mask = ct_oof == ct
    n_rows = mask.sum()
    if n_rows < 30:
        print(f"  {ct:12s}  SKIP (only {n_rows} OOF rows) → using 'mixed' fallback")
        calibrated[ct] = calibrate_weights(ct_oof == "mixed") if any(ct_oof == "mixed") else {"xgb": 1/3, "lgbm": 1/3, "cat": 1/3}
        continue

    cal_w   = calibrate_weights(mask)
    def_w   = DEFAULT_WEIGHTS.get(ct, DEFAULT_WEIGHTS["mixed"])
    mae_cal = weighted_mae(cal_w, mask)
    mae_def = weighted_mae(def_w, mask)
    delta   = mae_def - mae_cal  # positive = calibrated is better

    print(f"  {ct:12s}  n={n_rows:5d}"
          f"  calibrated={cal_w}"
          f"  default={def_w}"
          f"  MAE: cal={mae_cal:.4f}  def={mae_def:.4f}  d={delta:+.4f}")
    calibrated[ct] = cal_w

# Ensure all four canonical types are present
for ct in ("street", "high_speed", "technical", "mixed"):
    if ct not in calibrated:
        print(f"  {ct:12s}  FALLBACK — no OOF rows found, using default hardcoded weights")
        calibrated[ct] = DEFAULT_WEIGHTS[ct]

# ── Overall OOF MAE comparison ────────────────────────────────────────────────
def overall_mae(weight_map: dict) -> float:
    blend = np.zeros(len(y_oof))
    for ct in set(ct_oof):
        m = ct_oof == ct
        w = weight_map.get(ct, weight_map.get("mixed", DEFAULT_WEIGHTS["mixed"]))
        blend[m] = (w["xgb"]  * p_xgb[m] +
                    w["lgbm"] * p_lgbm[m] +
                    w["cat"]  * p_cat[m])
    return float(np.mean(np.abs(y_oof - blend)))

mae_cal  = overall_mae(calibrated)
mae_def  = overall_mae(DEFAULT_WEIGHTS)
mae_xgb  = float(np.mean(np.abs(y_oof - p_xgb)))
mae_lgbm = float(np.mean(np.abs(y_oof - p_lgbm)))
mae_cat  = float(np.mean(np.abs(y_oof - p_cat)))

print()
print("=" * 60)
print("  Overall OOF MAE comparison")
print("=" * 60)
print(f"  XGB alone:           {mae_xgb:.4f}")
print(f"  LGBM alone:          {mae_lgbm:.4f}")
print(f"  CAT alone:           {mae_cat:.4f}")
print(f"  Default weights:     {mae_def:.4f}")
print(f"  Calibrated weights:  {mae_cal:.4f}  d={mae_def-mae_cal:+.4f}")
print("=" * 60)


# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "_meta": {
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_splits": N_SPLITS,
        "n_trees":  N_TREES,
        "n_rows":   int(len(y_oof)),
        "overall_mae_default":    round(mae_def,  4),
        "overall_mae_calibrated": round(mae_cal,  4),
        "delta_mae":              round(mae_def - mae_cal, 4),
    },
    **calibrated,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"\nSaved to: {OUTPUT}")
print("Done.")
