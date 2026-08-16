#!/usr/bin/env python3
"""Precompute cross-validation metrics and holdout predictions for Streamlit."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["STREAMLIT_SERVER_HEADLESS"] = "1"
os.environ["STREAMLIT_LOG_LEVEL"] = "error"

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import json_helpers
from f1bet.validation import sklearn_expanding_window_cv


def _clean_target(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    valid = y.notnull() & np.isfinite(y)
    return X.loc[valid], y.loc[valid]


def _position_mae(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    values = pd.DataFrame({"actual": y_true.to_numpy(), "predicted": y_pred})
    groups = {
        "Podium (1-3)": values["actual"] <= 3,
        "Winners": values["actual"] == 1,
        "Points (1-10)": values["actual"] <= 10,
    }
    return {
        label: float(mean_absolute_error(values.loc[mask, "actual"], values.loc[mask, "predicted"]))
        for label, mask in groups.items()
        if mask.any()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="data_files/precomputed/historical_validation.json",
    )
    args = parser.parse_args()

    from raceAnalysis import (
        CACHE_VERSION,
        _build_advanced_preprocessor,
        data,
        get_data_fingerprint,
        get_dnf_model,
        get_features_and_target,
        get_features_and_target_dnf,
        get_features_and_target_safety_car,
        get_preprocessor_position,
        get_safetycar_model,
        safety_cars,
    )

    for logger_name in (
        "streamlit",
        "streamlit.runtime",
        "streamlit.runtime.scriptrunner_utils.script_run_context",
    ):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    print("Computing position-model cross-validation...")
    X_position, y_position = _clean_target(*get_features_and_target(data))
    position_pipeline = Pipeline(
        [
            ("pre", get_preprocessor_position(X_position)),
            (
                "model",
                XGBRegressor(
                    n_estimators=100,
                    max_depth=4,
                    n_jobs=-1,
                    tree_method="hist",
                    random_state=42,
                ),
            ),
        ]
    )
    position_cv = sklearn_expanding_window_cv(
        data.loc[X_position.index], n_splits=5, embargo_events=1
    )
    position_scores = cross_val_score(
        position_pipeline,
        X_position,
        y_position,
        cv=position_cv,
        scoring="neg_mean_squared_error",
    )

    print("Computing DNF validation...")
    X_dnf, y_dnf = _clean_target(*get_features_and_target_dnf(data))
    dnf_model = get_dnf_model(CACHE_VERSION)
    dnf_cv = sklearn_expanding_window_cv(
        data.loc[X_dnf.index], n_splits=5, embargo_events=1
    )
    dnf_train, dnf_test = dnf_cv[-1]
    X_train_dnf, X_test_dnf = X_dnf.iloc[dnf_train], X_dnf.iloc[dnf_test]
    y_train_dnf, y_test_dnf = y_dnf.iloc[dnf_train], y_dnf.iloc[dnf_test]
    dnf_holdout_model = clone(dnf_model).fit(X_train_dnf, y_train_dnf)
    dnf_test_probabilities = dnf_holdout_model.predict_proba(X_test_dnf)[:, 1]
    dnf_scores = cross_val_score(dnf_model, X_dnf, y_dnf, cv=dnf_cv, scoring="roc_auc")

    print("Computing safety-car validation...")
    X_safety, y_safety = _clean_target(
        *get_features_and_target_safety_car(safety_cars)
    )
    safety_model = get_safetycar_model(CACHE_VERSION)
    safety_cv = sklearn_expanding_window_cv(
        safety_cars.loc[X_safety.index], n_splits=5, embargo_events=1
    )
    safety_scores = cross_val_score(
        safety_model,
        X_safety,
        y_safety,
        cv=safety_cv,
        scoring="roc_auc",
    )

    print("Generating position-model holdout predictions...")
    train_index, test_index = position_cv[-1]
    X_train, X_test = X_position.iloc[train_index], X_position.iloc[test_index]
    y_train, y_test = y_position.iloc[train_index], y_position.iloc[test_index]
    holdout_pipeline = Pipeline(
        [
            ("pre", _build_advanced_preprocessor(X_position)),
            (
                "model",
                XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    holdout_pipeline.fit(X_train, y_train)
    predictions = np.asarray(holdout_pipeline.predict(X_test))

    source_rows = data.loc[y_test.index]
    holdout_rows = pd.DataFrame(
        {
            "constructorName": source_rows["constructorName"].astype(str),
            "resultsDriverName": source_rows["resultsDriverName"].astype(str),
            "ActualFinalPosition": y_test.to_numpy(),
            "PredictedFinalPosition": predictions,
        }
    )
    holdout_rows["Error"] = (
        holdout_rows["ActualFinalPosition"]
        - holdout_rows["PredictedFinalPosition"]
    )

    output = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_type": "XGBoost",
            "folds": 5,
            "validation": "race-grouped expanding window with one-event embargo",
            "data_fingerprint": get_data_fingerprint(),
        },
        "position_cv": {
            "mse_mean": float(-position_scores.mean()),
            "mse_std": float(position_scores.std()),
            "sample_count": int(len(y_position)),
        },
        "dnf_validation": {
            "test_mae": float(mean_absolute_error(y_test_dnf, dnf_test_probabilities)),
            "roc_auc_mean": float(dnf_scores.mean()),
            "roc_auc_std": float(dnf_scores.std()),
            "sample_count": int(len(y_dnf)),
        },
        "safety_car_validation": {
            "roc_auc_mean": float(safety_scores.mean()),
            "roc_auc_std": float(safety_scores.std()),
            "sample_count": int(len(y_safety)),
        },
        "holdout": {
            "metrics": {
                "mse": float(mean_squared_error(y_test, predictions)),
                "r2": float(r2_score(y_test, predictions)),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "mean_error": float(np.mean(y_test.to_numpy() - predictions)),
            },
            "position_mae": _position_mae(y_test, predictions),
            "rows": holdout_rows.to_dict(orient="records"),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    json_helpers.safe_dump(output, output_path, indent=2)
    print(f"Historical validation saved to {output_path}")


if __name__ == "__main__":
    main()
