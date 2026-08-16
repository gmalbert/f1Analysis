from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1bet.domain import MarketType
from f1bet.markets import probability_coherence_issues
from f1bet.simulation import (
    RaceEntry,
    SimulationConfig,
    plackett_luce_simulation,
    simulate_race,
)
from f1bet.strategy import (
    Stint,
    StrategyPlan,
    TyreCurve,
    adaptive_degradation_residuals,
    compare_strategies,
    fit_tyre_curve,
)


def entries() -> list[RaceEntry]:
    return [
        RaceEntry("a", "team-1", 1.0, 0.02, 0.5),
        RaceEntry("b", "team-1", 1.5, 0.05, 0.6),
        RaceEntry("c", "team-2", 2.5, 0.15, 0.8),
        RaceEntry("d", "team-2", 3.0, 0.20, 0.8),
    ]


def test_race_simulation_is_seeded_unique_and_coherent() -> None:
    config = SimulationConfig(n_simulations=2_000, random_seed=7)
    first = simulate_race(entries(), config)
    second = simulate_race(entries(), config)
    assert np.array_equal(first.positions, second.positions)
    assert np.array_equal(np.sort(first.positions, axis=1), np.tile([1, 2, 3, 4], (2_000, 1)))
    table = first.market_table()
    assert table.win_probability.sum() == pytest.approx(1.0)
    mapped = {
        row.driver_id: {
            MarketType.WIN: row.win_probability,
            MarketType.PODIUM: row.podium_probability,
            MarketType.TOP_6: row.top_6_probability,
            MarketType.TOP_10: row.top_10_probability,
        }
        for row in table.itertuples(index=False)
    }
    assert probability_coherence_issues(mapped) == []
    assert first.probability("a", MarketType.HEAD_TO_HEAD, "c") > 0.5


def test_plackett_luce_strength_controls_win_rate() -> None:
    result = plackett_luce_simulation({"fast": 5.0, "slow": 1.0}, n_simulations=5_000, random_seed=3)
    assert result.probability("fast", MarketType.WIN) > 0.75


def test_strategy_comparison_uses_common_random_numbers() -> None:
    curves = {
        "S": TyreCurve("S", 0.0, 0.08, 0.004),
        "M": TyreCurve("M", 0.2, 0.04, 0.002),
        "H": TyreCurve("H", 0.5, 0.02, 0.001),
    }
    plans = [
        StrategyPlan("one-stop", (Stint("M", 1, 15), Stint("H", 16, 30))),
        StrategyPlan("two-stop", (Stint("S", 1, 10), Stint("M", 11, 20), Stint("S", 21, 30))),
    ]
    first = compare_strategies(
        plans,
        curves,
        race_laps=30,
        base_lap_time=80,
        fuel_gain_per_lap=0.05,
        pit_loss=18,
        n_simulations=1_000,
        random_seed=9,
    )
    second = compare_strategies(
        plans,
        curves,
        race_laps=30,
        base_lap_time=80,
        fuel_gain_per_lap=0.05,
        pit_loss=18,
        n_simulations=1_000,
        random_seed=9,
    )
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert first.probability_fastest.sum() == pytest.approx(1.0)


def test_tyre_curve_fit_recovers_positive_degradation() -> None:
    age = np.arange(1, 25)
    observations = pd.DataFrame(
        {
            "compound": "M",
            "tyre_age": age,
            "fuel_corrected_lap_time": 80 + 0.04 * age + 0.003 * age**2,
        }
    )
    curve = fit_tyre_curve(observations, compound="M")
    assert curve.linear_deg > 0
    assert curve.quadratic_deg > 0


def test_adaptive_degradation_tracks_a_changing_residual() -> None:
    posterior = adaptive_degradation_residuals([0.0, 0.1, 0.3, 0.6])
    assert posterior.posterior_mean.iloc[-1] > posterior.posterior_mean.iloc[0]
    assert (posterior.posterior_std > 0).all()
