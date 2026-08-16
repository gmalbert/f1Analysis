"""Calibration diagnostics designed for small, time-ordered sports samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


EPSILON = 1e-12


def _arrays(probabilities: Iterable[float], outcomes: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(list(probabilities), dtype=float)
    y = np.asarray(list(outcomes), dtype=float)
    if p.shape != y.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("probabilities and outcomes must be equally sized non-empty vectors")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if np.any(~np.isin(y, [0.0, 1.0])):
        raise ValueError("outcomes must be binary")
    return p, y


def brier_score(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p, y = _arrays(probabilities, outcomes)
    return float(np.mean((p - y) ** 2))


def logarithmic_loss(probabilities: Iterable[float], outcomes: Iterable[int]) -> float:
    p, y = _arrays(probabilities, outcomes)
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def calibration_table(
    probabilities: Iterable[float], outcomes: Iterable[int], n_bins: int = 10
) -> pd.DataFrame:
    """Return equal-frequency reliability bins to reduce sparse-bin noise."""
    p, y = _arrays(probabilities, outcomes)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    frame = pd.DataFrame({"probability": p, "outcome": y})
    effective_bins = min(n_bins, len(frame))
    ranked = frame["probability"].rank(method="first")
    frame["bin"] = pd.qcut(ranked, q=effective_bins, labels=False, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(
            count=("outcome", "size"),
            mean_probability=("probability", "mean"),
            observed_rate=("outcome", "mean"),
            min_probability=("probability", "min"),
            max_probability=("probability", "max"),
        )
        .reset_index()
    )
    table["absolute_gap"] = (table["mean_probability"] - table["observed_rate"]).abs()
    return table


def expected_calibration_error(
    probabilities: Iterable[float], outcomes: Iterable[int], n_bins: int = 10
) -> float:
    table = calibration_table(probabilities, outcomes, n_bins=n_bins)
    return float((table["count"] * table["absolute_gap"]).sum() / table["count"].sum())


def calibration_slope_intercept(
    probabilities: Iterable[float], outcomes: Iterable[int]
) -> tuple[float, float]:
    p, y = _arrays(probabilities, outcomes)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    logits = np.log(np.clip(p, EPSILON, 1 - EPSILON) / np.clip(1 - p, EPSILON, 1))
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(logits.reshape(-1, 1), y.astype(int))
    return float(model.coef_[0, 0]), float(model.intercept_[0])


@dataclass(slots=True)
class IsotonicProbabilityCalibrator:
    """Thin serializable wrapper that fails closed on insufficient samples."""

    min_samples: int = 50
    model: IsotonicRegression | None = None

    def fit(self, probabilities: Iterable[float], outcomes: Iterable[int]) -> "IsotonicProbabilityCalibrator":
        p, y = _arrays(probabilities, outcomes)
        if len(p) < self.min_samples or np.unique(y).size < 2:
            raise ValueError("insufficient calibration data")
        self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model.fit(p, y)
        return self

    def predict(self, probabilities: Iterable[float]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        p = np.asarray(list(probabilities), dtype=float)
        return np.asarray(self.model.predict(np.clip(p, 0.0, 1.0)), dtype=float)


@dataclass(slots=True)
class SigmoidProbabilityCalibrator:
    """Platt-style calibrator for samples too sparse for isotonic fitting."""

    min_samples: int = 20
    model: LogisticRegression | None = None

    def fit(self, probabilities: Iterable[float], outcomes: Iterable[int]) -> "SigmoidProbabilityCalibrator":
        p, y = _arrays(probabilities, outcomes)
        if len(p) < self.min_samples or np.unique(y).size < 2:
            raise ValueError("insufficient calibration data")
        logits = np.log(np.clip(p, EPSILON, 1 - EPSILON) / np.clip(1 - p, EPSILON, 1))
        self.model = LogisticRegression(C=1e6, solver="lbfgs")
        self.model.fit(logits.reshape(-1, 1), y.astype(int))
        return self

    def predict(self, probabilities: Iterable[float]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        p = np.asarray(list(probabilities), dtype=float)
        if np.any(~np.isfinite(p)):
            raise ValueError("probabilities must be finite")
        logits = np.log(np.clip(p, EPSILON, 1 - EPSILON) / np.clip(1 - p, EPSILON, 1))
        return self.model.predict_proba(logits.reshape(-1, 1))[:, 1]


def probability_metrics(
    probabilities: Iterable[float], outcomes: Iterable[int], n_bins: int = 10
) -> dict[str, float]:
    p, y = _arrays(probabilities, outcomes)
    slope, intercept = calibration_slope_intercept(p, y)
    roc_auc = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return {
        "n": float(len(p)),
        "brier": brier_score(p, y),
        "log_loss": logarithmic_loss(p, y),
        "ece": expected_calibration_error(p, y, n_bins=n_bins),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_probability": float(np.mean(p)),
        "base_rate": float(np.mean(y)),
        "roc_auc": roc_auc,
        "reliability_bins": float(min(n_bins, len(p))),
    }
