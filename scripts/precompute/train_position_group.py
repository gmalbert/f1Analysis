#!/usr/bin/env python3
"""
Train Position Group ensemble model (ROADMAP-3A).

Uses 5-fold GroupKFold CV (grouped by season) to compute an honest OOF MAE,
then refits the final model on all data.  Saved artifact is loaded by the
Streamlit UI via load_pretrained_model() at startup — no live retraining needed.

Run via GitHub Actions or locally:
    python scripts/precompute/train_position_group.py
"""
import os
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'

import warnings
warnings.filterwarnings("ignore", message=".*No runtime found.*")
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

import pandas as pd
import numpy as np

DATA_DIR = 'data_files/'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json_helpers


def train_position_group():
    print("=" * 60)
    print("Position Group Ensemble Training (ROADMAP-3A)")
    print("=" * 60)

    output_dir = Path("data_files/models/position_group")
    output_dir.mkdir(parents=True, exist_ok=True)

    from raceAnalysis import (
        load_data,
        CACHE_VERSION,
        train_and_evaluate_model,
    )

    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)

    print(f"\nLoading data (CACHE_VERSION={CACHE_VERSION})...")
    data, _ = load_data(
        10000,
        CACHE_VERSION,
        os.path.getmtime(os.path.join(DATA_DIR, 'f1ForAnalysis.csv'))
    )

    # Apply standard column renames (matches other train scripts)
    for src, dst in [
        ('constructorName_results_with_qualifying', 'constructorName'),
        ('constructorName_qualifying',              'constructorName'),
        ('best_qual_time_results_with_qualifying',  'best_qual_time'),
        ('best_qual_time_qualifying',               'best_qual_time'),
        ('teammate_qual_delta_results_with_qualifying', 'teammate_qual_delta'),
        ('teammate_qual_delta_qualifying',              'teammate_qual_delta'),
    ]:
        if src in data.columns and dst not in data.columns:
            data.rename(columns={src: dst}, inplace=True)

    print(f"Data loaded: {data.shape}")

    print("\n" + "-" * 60)
    print("Training Position Group Ensemble")
    print("  • 5-fold GroupKFold CV  (grouped by grandPrixYear)")
    print("  • 20 sub-model fits during CV + 4 final fits on ALL data")
    print("  • Expected runtime: 5-10 min on GitHub-hosted runner")
    print("-" * 60)

    model, mse, r2, mae, mean_err, evals_result, preprocessor = train_and_evaluate_model(
        data,
        early_stopping_rounds=20,
        model_type="Position Group",
    )

    artifact = {
        'model':        model,
        'preprocessor': preprocessor,
        'mse':          float(mse),
        'r2':           float(r2),
        'mae':          float(mae),
        'mean_err':     float(mean_err),
        'evals_result': evals_result,
        'cache_version': CACHE_VERSION,
        'model_type':   'Position Group',
        'trained_at':   datetime.now().isoformat(),
    }

    pkl_path = output_dir / 'position_model.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(artifact, f)
    print(f"[OK] Position Group model saved → {pkl_path}  (OOF MAE: {mae:.4f})")

    metadata = {
        'cache_version': CACHE_VERSION,
        'trained_at':    datetime.now().isoformat(),
        'model_type':    'Position Group',
        'data_rows':     len(data),
        'models': {
            'position': {'mae': float(mae), 'mse': float(mse), 'r2': float(r2)},
        },
    }
    json_helpers.safe_dump(metadata, output_dir / 'metadata.json', indent=2)

    print("\n" + "=" * 60)
    print("Position Group Training Complete!")
    print(f"  OOF MAE  : {mae:.4f}")
    print(f"  R²       : {r2:.4f}")
    print(f"  Output   : {output_dir.absolute()}")
    print("=" * 60)
    return metadata


if __name__ == '__main__':
    try:
        train_position_group()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
