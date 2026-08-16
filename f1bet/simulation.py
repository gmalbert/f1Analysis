"""Coherent field-level race simulation with correlated uncertainty and DNF risk."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .domain import MarketType


@dataclass(frozen=True, slots=True)
class RaceEntry:
    driver_id: str
    constructor_id: str
    pace_score: float
    dnf_probability: float = 0.05
    uncertainty: float = 1.0
    race_sensitivity: float = 1.0

    def __post_init__(self) -> None:
        if not self.driver_id or not self.constructor_id:
            raise ValueError("driver_id and constructor_id are required")
        if (
            not np.isfinite(self.pace_score)
            or not np.isfinite(self.uncertainty)
            or self.uncertainty < 0
            or not np.isfinite(self.race_sensitivity)
        ):
            raise ValueError("pace_score/race_sensitivity must be finite and uncertainty non-negative")
        if not 0 <= self.dnf_probability <= 1:
            raise ValueError("dnf_probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    n_simulations: int = 10_000
    random_seed: int = 42
    constructor_shock: float = 0.45
    race_shock: float = 0.15
    dnf_position_penalty: float = 50.0

    def __post_init__(self) -> None:
        if self.n_simulations < 100:
            raise ValueError("n_simulations must be at least 100")
        values = (self.constructor_shock, self.race_shock, self.dnf_position_penalty)
        if any(not np.isfinite(value) for value in values) or min(values) < 0:
            raise ValueError("shock scales and penalties must be non-negative")


@dataclass(slots=True)
class SimulationResult:
    driver_ids: tuple[str, ...]
    positions: np.ndarray
    dnf: np.ndarray

    def __post_init__(self) -> None:
        if self.positions.ndim != 2 or self.dnf.shape != self.positions.shape:
            raise ValueError("positions and dnf must be equally shaped matrices")
        if self.positions.shape[1] != len(self.driver_ids):
            raise ValueError("driver_ids must match simulation columns")
        if len(set(self.driver_ids)) != len(self.driver_ids):
            raise ValueError("driver_ids must be unique")
        expected = np.arange(1, len(self.driver_ids) + 1)
        if not np.all(np.sort(self.positions, axis=1) == expected):
            raise ValueError("every simulation must contain one unique full-field ranking")

    def probability(self, driver_id: str, market: MarketType, opponent_id: str | None = None) -> float:
        index = self.driver_ids.index(driver_id)
        if market is MarketType.DNF:
            return float(self.dnf[:, index].mean())
        if market is MarketType.HEAD_TO_HEAD:
            if opponent_id is None:
                raise ValueError("head-to-head probability requires opponent_id")
            opponent = self.driver_ids.index(opponent_id)
            return float((self.positions[:, index] < self.positions[:, opponent]).mean())
        cutoff = {
            MarketType.WIN: 1,
            MarketType.PODIUM: 3,
            MarketType.TOP_6: 6,
            MarketType.TOP_10: 10,
        }.get(market)
        if cutoff is None:
            raise ValueError(f"market {market.value} is not produced by finishing simulation")
        return float((self.positions[:, index] <= cutoff).mean())

    def market_table(self) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for index, driver in enumerate(self.driver_ids):
            values = self.positions[:, index]
            rows.append(
                {
                    "driver_id": driver,
                    "expected_position": float(values.mean()),
                    "position_p10": float(np.quantile(values, 0.10)),
                    "position_p50": float(np.quantile(values, 0.50)),
                    "position_p90": float(np.quantile(values, 0.90)),
                    "win_probability": self.probability(driver, MarketType.WIN),
                    "podium_probability": self.probability(driver, MarketType.PODIUM),
                    "top_6_probability": self.probability(driver, MarketType.TOP_6),
                    "top_10_probability": self.probability(driver, MarketType.TOP_10),
                    "dnf_probability": self.probability(driver, MarketType.DNF),
                }
            )
        return pd.DataFrame(rows).sort_values("expected_position")


def simulate_race(
    entries: Iterable[RaceEntry], config: SimulationConfig | None = None
) -> SimulationResult:
    """Simulate unique field orderings; lower pace scores are faster."""
    cfg = config or SimulationConfig()
    field = tuple(entries)
    if len(field) < 2:
        raise ValueError("a race simulation requires at least two entries")
    if len({entry.driver_id for entry in field}) != len(field):
        raise ValueError("driver_id values must be unique")
    rng = np.random.default_rng(cfg.random_seed)
    n, drivers = cfg.n_simulations, len(field)
    constructors = sorted({entry.constructor_id for entry in field})
    constructor_index = {constructor: index for index, constructor in enumerate(constructors)}
    constructor_draws = rng.normal(0, cfg.constructor_shock, size=(n, len(constructors)))
    race_draws = rng.normal(0, cfg.race_shock, size=(n, 1))
    individual = rng.normal(
        0,
        np.asarray([entry.uncertainty for entry in field]),
        size=(n, drivers),
    )
    base = np.asarray([entry.pace_score for entry in field])[None, :]
    team = np.column_stack(
        [constructor_draws[:, constructor_index[entry.constructor_id]] for entry in field]
    )
    base_dnf_probability = np.asarray([entry.dnf_probability for entry in field])[None, :]
    clipped_dnf = np.clip(base_dnf_probability, 1e-9, 1 - 1e-9)
    dnf_logit = np.log(clipped_dnf / (1 - clipped_dnf)) + race_draws
    dnf_probability = 1 / (1 + np.exp(-dnf_logit))
    dnf_probability = np.where(base_dnf_probability == 0, 0.0, dnf_probability)
    dnf_probability = np.where(base_dnf_probability == 1, 1.0, dnf_probability)
    dnf = rng.random((n, drivers)) < dnf_probability
    # Random failure timing makes DNF ordering coherent without pretending it is known.
    failure_timing = rng.uniform(0, 1, size=(n, drivers))
    race_loading = np.asarray([entry.race_sensitivity for entry in field])[None, :]
    performance = base + race_draws * race_loading + team + individual
    # Finishers always classify ahead of DNFs regardless of the arbitrary pace
    # score scale. Among DNFs, a later failure receives the better classification.
    ranking_score = np.where(dnf, cfg.dnf_position_penalty - failure_timing, performance)
    order = np.lexsort((ranking_score, dnf.astype(np.int8)), axis=1)
    positions = np.empty_like(order)
    positions[np.arange(n)[:, None], order] = np.arange(1, drivers + 1)
    return SimulationResult(tuple(entry.driver_id for entry in field), positions, dnf)


def plackett_luce_simulation(
    strengths: dict[str, float], *, n_simulations: int = 10_000, random_seed: int = 42
) -> SimulationResult:
    """Sample rankings using the Gumbel-max representation of Plackett-Luce."""
    if n_simulations < 100:
        raise ValueError("n_simulations must be at least 100")
    if len(strengths) < 2 or any(value <= 0 or not np.isfinite(value) for value in strengths.values()):
        raise ValueError("strengths require at least two positive finite values")
    rng = np.random.default_rng(random_seed)
    drivers = tuple(strengths)
    utilities = np.log(np.asarray(list(strengths.values())))[None, :] + rng.gumbel(
        size=(n_simulations, len(drivers))
    )
    order = np.argsort(-utilities, axis=1, kind="stable")
    positions = np.empty_like(order)
    positions[np.arange(n_simulations)[:, None], order] = np.arange(1, len(drivers) + 1)
    return SimulationResult(drivers, positions, np.zeros_like(positions, dtype=bool))
