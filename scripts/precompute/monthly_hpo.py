#!/usr/bin/env python3
"""Monthly Optuna hyperparameter optimisation — ROADMAP-3F.

Runs a full Optuna study for XGBoost, LightGBM, and CatBoost against the
ROADMAP-3 advanced preprocessor (_build_advanced_preprocessor) and writes
best params to ``data_files/best_hyperparams_<model>.json``.

Usage
-----
    python scripts/precompute/monthly_hpo.py [--trials N] [--cv-folds K] [--models xgb,lgbm,cat]

Triggered automatically by ``.github/workflows/monthly-hpo.yml`` on the 1st
of every month.  Run locally at any time to refresh hyperparameters after
collecting several new races of data.
"""

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# ── Environment setup (must happen before any Streamlit import) ──────────────
os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
os.environ.setdefault('STREAMLIT_LOG_LEVEL', 'error')

warnings.filterwarnings("ignore")

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
DATA_DIR = ROOT / 'data_files'

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO,
    datefmt='%H:%M:%S',
)
log = logging.getLogger('monthly_hpo')


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Return (X, y, season_groups) ready for GroupKFold cross-validation."""
    # Lazy import so raceAnalysis Streamlit side-effects run only here
    from raceAnalysis import (  # noqa: PLC0415
        load_data as ra_load_data,
        get_features_and_target,
        _build_advanced_preprocessor,
        CACHE_VERSION,
    )
    import logging as _log

    # Suppress Streamlit noise that fires after the import
    for _lname in (
        'streamlit.runtime.scriptrunner_utils.script_run_context',
        'streamlit.runtime.caching.cache_data_api',
        'streamlit',
        'streamlit.runtime.state.session_state_proxy',
    ):
        _log.getLogger(_lname).setLevel(_log.ERROR)

    csv_path = DATA_DIR / 'f1ForAnalysis.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Main CSV not found: {csv_path}")

    log.info("Loading f1ForAnalysis.csv …")
    data, _ = ra_load_data(
        10000,
        CACHE_VERSION,
        csv_path.stat().st_mtime,
    )

    X, y = get_features_and_target(data)
    y = y.fillna(y.mean()).clip(upper=20)
    season_groups = data['grandPrixYear'].values[y.index]

    return X, y, season_groups, _build_advanced_preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Objective functions
# ─────────────────────────────────────────────────────────────────────────────

def _cv_mae(pipeline, X_num, y, groups, n_splits: int) -> float:
    from sklearn.model_selection import cross_val_score, GroupKFold
    cv = GroupKFold(n_splits=n_splits)
    scores = cross_val_score(
        pipeline, X_num, y,
        cv=cv, groups=groups,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
    )
    return float(-scores.mean())


def objective_xgb(trial, X_num, y, groups, n_splits: int):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBRegressor

    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 800),
        'max_depth':         trial.suggest_int('max_depth', 3, 8),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'verbosity': 0, 'n_jobs': 1,
    }
    pipe = Pipeline([
        ('imp',   SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('reg',   XGBRegressor(**params)),
    ])
    return _cv_mae(pipe, X_num, y, groups, n_splits)


def objective_lgbm(trial, X_num, y, groups, n_splits: int):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMRegressor

    params = {
        'n_estimators':    trial.suggest_int('n_estimators', 200, 800),
        'max_depth':       trial.suggest_int('max_depth', 3, 15),
        'learning_rate':   trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'num_leaves':      trial.suggest_int('num_leaves', 20, 300),
        'subsample':       trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha':       trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda':      trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': 42, 'verbose': -1, 'n_jobs': 1,
    }
    pipe = Pipeline([
        ('imp',   SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('reg',   LGBMRegressor(**params)),
    ])
    return _cv_mae(pipe, X_num, y, groups, n_splits)


def objective_cat(trial, X_num, y, groups, n_splits: int):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from catboost import CatBoostRegressor

    params = {
        'iterations':    trial.suggest_int('iterations', 200, 800),
        'depth':         trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        'l2_leaf_reg':   trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'random_seed': 42, 'verbose': 0,
    }
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('reg', CatBoostRegressor(**params)),
    ])
    return _cv_mae(pipe, X_num, y, groups, n_splits)


_OBJECTIVES = {
    'xgb':  objective_xgb,
    'lgbm': objective_lgbm,
    'cat':  objective_cat,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def run_hpo(model_key: str, X, y, groups, n_trials: int, cv_folds: int):
    """Run an Optuna study for *model_key* and persist best params to JSON."""
    obj_fn = _OBJECTIVES[model_key]

    # Use only numeric columns for this simplified HPO pipeline
    X_num = X.select_dtypes(include='number').fillna(X.select_dtypes(include='number').median())

    log.info(f"[{model_key.upper()}] starting {n_trials}-trial study "
             f"(GroupKFold cv={cv_folds}, {X_num.shape[1]} numeric features) …")

    study = optuna.create_study(
        direction='minimize',
        study_name=f'monthly_hpo_{model_key}',
    )
    study.optimize(
        lambda t: obj_fn(t, X_num, y, groups, cv_folds),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_mae   = study.best_value
    best_params = study.best_params

    log.info(f"[{model_key.upper()}] best MAE = {best_mae:.4f}")
    log.info(f"[{model_key.upper()}] best params: {best_params}")

    out_file = DATA_DIR / f'best_hyperparams_{model_key}.json'
    payload = {
        'model_key':          model_key,
        'best_mae':           best_mae,
        'best_params':        best_params,
        'n_trials':           n_trials,
        'cv_folds':           cv_folds,
        'optimised_at':       datetime.utcnow().isoformat() + 'Z',
        'optimization_history': [
            {'trial': t.number, 'mae': t.value, 'params': t.params}
            for t in study.trials
            if t.value is not None
        ],
    }
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    log.info(f"[{model_key.upper()}] params saved → {out_file.relative_to(ROOT)}")
    return best_mae


def main():
    parser = argparse.ArgumentParser(description='Monthly Optuna HPO refresh (ROADMAP-3F)')
    parser.add_argument('--trials',  type=int, default=200,          help='Optuna trials per model (default: 200)')
    parser.add_argument('--cv-folds', type=int, default=5,           help='GroupKFold folds (default: 5)')
    parser.add_argument('--models',  type=str, default='xgb,lgbm,cat',
                        help='Comma-separated list of models to optimise (default: xgb,lgbm,cat)')
    args = parser.parse_args()

    models_to_run = [m.strip().lower() for m in args.models.split(',')]
    unknown = [m for m in models_to_run if m not in _OBJECTIVES]
    if unknown:
        parser.error(f"Unknown model keys: {unknown}. Valid: {list(_OBJECTIVES)}")

    X, y, groups, _ = load_data()

    results = {}
    for model_key in models_to_run:
        best_mae = run_hpo(model_key, X, y, groups, args.trials, args.cv_folds)
        results[model_key] = best_mae

    log.info("=" * 55)
    log.info("MONTHLY HPO COMPLETE")
    for k, v in results.items():
        log.info(f"  {k.upper():<10} best MAE = {v:.4f}")
    log.info("=" * 55)


if __name__ == '__main__':
    main()
