#!/usr/bin/env python3
"""
Train Track-Weighted Ensemble model for position prediction.
Run via GitHub Actions or locally for model artifact generation.

Saves a pre-trained pkl so Streamlit Cloud loads from file instead of
triggering expensive live training that can cause OOM on the Community tier.
"""
import os
import sys
import pickle
from pathlib import Path
from datetime import datetime

os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
os.environ['STREAMLIT_LOG_LEVEL'] = 'error'

import warnings
import logging
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json_helpers
from model_artifacts import build_data_fingerprint, stamp_artifact
from f1bet.artifacts import save_training_manifest

DATA_DIR = 'data_files/'


def train_track_weighted_models():
    """Train Track-Weighted Ensemble and save to data_files/models/track_weighted/"""

    print("=" * 60)
    print("Track-Weighted Ensemble Model Training")
    print("=" * 60)

    output_dir = Path("data_files/models/track_weighted")
    output_dir.mkdir(parents=True, exist_ok=True)

    from raceAnalysis import (
        load_data,
        CACHE_VERSION,
        train_and_evaluate_model,
    )

    # Suppress Streamlit headless mode warnings AFTER streamlit is imported
    logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
    logging.getLogger('streamlit').setLevel(logging.ERROR)
    logging.getLogger('streamlit.runtime.state.session_state_proxy').setLevel(logging.ERROR)

    print(f"\nLoading data (CACHE_VERSION={CACHE_VERSION})...")
    analysis_fingerprint = build_data_fingerprint(Path(DATA_DIR) / 'f1ForAnalysis.csv')
    data, _ = load_data(
        10000,
        CACHE_VERSION,
        analysis_fingerprint['data_sha256']
    )

    # Apply column renaming logic (mirrors other train_*.py scripts; guarded so an
    # already-present short name is never clobbered into a duplicate column)
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
    print("Training Track-Weighted Ensemble Position Model")
    print("-" * 60)

    model, mse, r2, mae, mean_err, evals_result, preprocessor = train_and_evaluate_model(
        data,
        early_stopping_rounds=20,
        model_type="Track-Weighted Ensemble",
    )

    if model is None:
        print("[ERROR] Training returned None — LightGBM may not be available in this environment.")
        return None

    position_artifact = stamp_artifact({
        'model': model,
        'preprocessor': preprocessor,
        'mse': float(mse) if mse is not None else None,
        'r2': float(r2) if r2 is not None else None,
        'mae': float(mae) if mae is not None else None,
        'mean_err': float(mean_err) if mean_err is not None else None,
        'evals_result': evals_result,
        'cache_version': CACHE_VERSION,
        'model_type': 'Track-Weighted Ensemble',
        'trained_at': datetime.now().isoformat(),
    }, analysis_fingerprint)

    with open(output_dir / 'position_model.pkl', 'wb') as f:
        pickle.dump(position_artifact, f)
    print(f"[OK] Position model saved (MAE: {mae:.4f})")

    metadata = stamp_artifact({
        'cache_version': CACHE_VERSION,
        'trained_at': datetime.now().isoformat(),
        'model_type': 'Track-Weighted Ensemble',
        'data_rows': len(data),
        'models': {
            'position': {
                'mae': float(mae) if mae is not None else None,
                'mse': float(mse) if mse is not None else None,
                'r2': float(r2) if r2 is not None else None,
            }
        }
    }, analysis_fingerprint)

    json_helpers.safe_dump(metadata, output_dir / 'metadata.json', indent=2)
    save_training_manifest(
        destination=output_dir / 'manifest.json',
        model_name='position', model_version=f'track-weighted-{CACHE_VERSION}', estimator='Track-Weighted Ensemble',
        preprocessor=preprocessor, training_frame=data,
        data_path=Path(DATA_DIR) / 'f1ForAnalysis.csv',
        metrics={'mae': float(mae), 'mse': float(mse), 'r2': float(r2)},
    )

    print("\n" + "=" * 60)
    print("Track-Weighted Ensemble Training Complete!")
    print("=" * 60)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Position MAE: {mae:.4f}")

    return metadata


if __name__ == '__main__':
    try:
        metadata = train_track_weighted_models()
        sys.exit(0 if metadata is not None else 1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
