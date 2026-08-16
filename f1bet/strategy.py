"""Interpretable tyre/fuel/pit strategy simulation with common random numbers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class TyreCurve:
    compound: str
    intercept: float
    linear_deg: float
    quadratic_deg: float = 0.0
    warmup_penalty: float = 0.0

    def __post_init__(self) -> None:
        if not self.compound.strip():
            raise ValueError("compound is required")
        coefficients = (self.intercept, self.linear_deg, self.quadratic_deg, self.warmup_penalty)
        if any(not np.isfinite(value) for value in coefficients):
            raise ValueError("tyre curve coefficients must be finite")

    def loss(self, tyre_age: np.ndarray | float) -> np.ndarray:
        age = np.asarray(tyre_age, dtype=float)
        if np.any(age < 0) or np.any(~np.isfinite(age)):
            raise ValueError("tyre_age must be finite and non-negative")
        warmup = np.where(age <= 2, self.warmup_penalty * (3 - age), 0.0)
        return self.intercept + self.linear_deg * age + self.quadratic_deg * age**2 + warmup


@dataclass(frozen=True, slots=True)
class Stint:
    compound: str
    start_lap: int
    end_lap: int
    starting_age: int = 0

    def __post_init__(self) -> None:
        if self.start_lap < 1 or self.end_lap < self.start_lap or self.starting_age < 0:
            raise ValueError("invalid stint bounds")


@dataclass(frozen=True, slots=True)
class StrategyPlan:
    name: str
    stints: tuple[Stint, ...]

    def validate(self, race_laps: int) -> None:
        if not self.name.strip() or not self.stints:
            raise ValueError("strategy name and at least one stint are required")
        expected = 1
        previous_compound: str | None = None
        for stint in self.stints:
            if stint.start_lap != expected:
                raise ValueError(f"strategy {self.name!r} has a lap gap or overlap at {expected}")
            compound = stint.compound.upper()
            if previous_compound == compound:
                raise ValueError(f"strategy {self.name!r} repeats {compound} in adjacent stints")
            previous_compound = compound
            expected = stint.end_lap + 1
        if expected != race_laps + 1:
            raise ValueError(f"strategy {self.name!r} does not cover all {race_laps} laps")


def fit_tyre_curve(
    observations: pd.DataFrame,
    *,
    compound: str,
    tyre_age_col: str = "tyre_age",
    corrected_lap_col: str = "fuel_corrected_lap_time",
) -> TyreCurve:
    """Fit a robust quadratic curve after trimming gross lap-time outliers."""
    subset = observations[observations["compound"].astype(str).str.upper() == compound.upper()].copy()
    subset[tyre_age_col] = pd.to_numeric(subset[tyre_age_col], errors="coerce")
    subset[corrected_lap_col] = pd.to_numeric(subset[corrected_lap_col], errors="coerce")
    subset = subset.dropna(subset=[tyre_age_col, corrected_lap_col])
    if len(subset) < 8:
        raise ValueError(f"at least 8 clean {compound} laps are required")
    median = subset[corrected_lap_col].median()
    mad = (subset[corrected_lap_col] - median).abs().median()
    if mad > 0:
        subset = subset[(subset[corrected_lap_col] - median).abs() <= 4.5 * mad]
    x = subset[tyre_age_col].to_numpy(dtype=float)
    y = subset[corrected_lap_col].to_numpy(dtype=float)
    # The simulator supplies base pace separately. Fit degradation as loss
    # relative to the clean-lap floor so the intercept is not double counted.
    y = y - np.quantile(y, 0.05)
    coefficient = np.polyfit(x, y, deg=2)
    return TyreCurve(
        compound=compound.upper(),
        intercept=float(coefficient[2]),
        linear_deg=float(coefficient[1]),
        quadratic_deg=float(coefficient[0]),
    )


def adaptive_degradation_residuals(
    observations: Iterable[float],
    *,
    process_variance: float = 0.01,
    observation_variance: float = 0.09,
    initial_variance: float = 1.0,
) -> pd.DataFrame:
    """Estimate a drifting lap-time residual with a one-state Kalman filter."""

    values = np.asarray(list(observations), dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("observations must be a non-empty finite vector")
    if min(process_variance, observation_variance, initial_variance) <= 0:
        raise ValueError("state-space variances must be positive")
    mean = 0.0
    variance = float(initial_variance)
    rows: list[dict[str, float | int]] = []
    for lap, observed in enumerate(values, start=1):
        predicted_variance = variance + process_variance
        gain = predicted_variance / (predicted_variance + observation_variance)
        innovation = float(observed - mean)
        mean = mean + gain * innovation
        variance = (1.0 - gain) * predicted_variance
        rows.append(
            {
                "lap": lap,
                "observed_residual": float(observed),
                "innovation": innovation,
                "posterior_mean": float(mean),
                "posterior_std": float(np.sqrt(variance)),
                "kalman_gain": float(gain),
            }
        )
    return pd.DataFrame(rows)


def compare_strategies(
    plans: Iterable[StrategyPlan],
    curves: dict[str, TyreCurve],
    *,
    race_laps: int,
    base_lap_time: float,
    fuel_gain_per_lap: float,
    pit_loss: float,
    safety_car_probability_by_lap: np.ndarray | None = None,
    n_simulations: int = 5_000,
    random_seed: int = 42,
    lap_noise: float = 0.25,
    safety_car_pit_loss_multiplier: float = 0.55,
    traffic_loss_by_strategy: dict[str, np.ndarray] | None = None,
    track_evolution_by_lap: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare plans under identical random draws so deltas have low variance."""
    strategy_list = tuple(plans)
    if not strategy_list:
        raise ValueError("at least one strategy is required")
    if race_laps <= 0 or n_simulations < 100 or min(base_lap_time, pit_loss, lap_noise) < 0:
        raise ValueError("invalid simulation inputs")
    if not 0 <= safety_car_pit_loss_multiplier <= 1:
        raise ValueError("safety_car_pit_loss_multiplier must be in [0, 1]")
    for plan in strategy_list:
        plan.validate(race_laps)
    sc_prob = (
        np.zeros(race_laps)
        if safety_car_probability_by_lap is None
        else np.asarray(safety_car_probability_by_lap, dtype=float)
    )
    if sc_prob.shape != (race_laps,) or np.any((sc_prob < 0) | (sc_prob > 1)):
        raise ValueError("safety_car_probability_by_lap must be a probability per lap")
    track_evolution = (
        np.zeros(race_laps)
        if track_evolution_by_lap is None
        else np.asarray(track_evolution_by_lap, dtype=float)
    )
    if track_evolution.shape != (race_laps,) or np.any(~np.isfinite(track_evolution)):
        raise ValueError("track_evolution_by_lap must be finite with one value per lap")
    rng = np.random.default_rng(random_seed)
    common_lap_noise = rng.normal(0, lap_noise, size=(n_simulations, race_laps))
    common_sc = rng.random((n_simulations, race_laps)) < sc_prob[None, :]
    totals = np.empty((n_simulations, len(strategy_list)), dtype=float)
    fuel = -fuel_gain_per_lap * np.arange(race_laps)

    for plan_index, plan in enumerate(strategy_list):
        deterministic = np.full(race_laps, base_lap_time, dtype=float) + fuel + track_evolution
        traffic = np.zeros(race_laps)
        if traffic_loss_by_strategy is not None and plan.name in traffic_loss_by_strategy:
            traffic = np.asarray(traffic_loss_by_strategy[plan.name], dtype=float)
            if traffic.shape != (race_laps,) or np.any(~np.isfinite(traffic)) or np.any(traffic < 0):
                raise ValueError(f"traffic loss for {plan.name!r} must be finite, non-negative, and lap-sized")
        deterministic += traffic
        pit_laps: list[int] = []
        for stint_index, stint in enumerate(plan.stints):
            curve = curves.get(stint.compound.upper())
            if curve is None:
                raise KeyError(f"missing tyre curve for {stint.compound!r}")
            indices = np.arange(stint.start_lap - 1, stint.end_lap)
            ages = stint.starting_age + np.arange(1, len(indices) + 1)
            deterministic[indices] += curve.loss(ages)
            if stint_index > 0:
                pit_laps.append(stint.start_lap - 1)
        total = deterministic.sum() + common_lap_noise.sum(axis=1)
        for pit_lap in pit_laps:
            # A stop under SC/VSC is cheaper but not free; 55% is an explicit scenario assumption.
            total += np.where(
                common_sc[:, pit_lap - 1],
                pit_loss * safety_car_pit_loss_multiplier,
                pit_loss,
            )
        totals[:, plan_index] = total

    best = np.argmin(totals, axis=1)
    rows = []
    benchmark = totals[:, 0]
    for index, plan in enumerate(strategy_list):
        values = totals[:, index]
        rows.append(
            {
                "strategy": plan.name,
                "mean_race_time": float(values.mean()),
                "p10_race_time": float(np.quantile(values, 0.10)),
                "p50_race_time": float(np.quantile(values, 0.50)),
                "p90_race_time": float(np.quantile(values, 0.90)),
                "mean_delta_to_first_plan": float((values - benchmark).mean()),
                "probability_fastest": float((best == index).mean()),
                "mean_regret": float((values - totals.min(axis=1)).mean()),
                "p90_regret": float(np.quantile(values - totals.min(axis=1), 0.90)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_race_time")
