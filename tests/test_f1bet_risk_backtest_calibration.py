from __future__ import annotations

import pandas as pd
import pytest

from f1bet.backtest import run_backtest
from f1bet.calibration import (
    IsotonicProbabilityCalibrator,
    brier_score,
    calibration_table,
    expected_calibration_error,
    logarithmic_loss,
    probability_metrics,
)
from f1bet.risk import PortfolioState, RiskPolicy, full_kelly_fraction, propose_stake


def test_probability_metrics_reward_calibrated_forecasts() -> None:
    outcomes = [0, 0, 1, 1]
    good = [0.1, 0.2, 0.8, 0.9]
    bad = [0.9, 0.8, 0.2, 0.1]
    assert brier_score(good, outcomes) < brier_score(bad, outcomes)
    assert logarithmic_loss(good, outcomes) < logarithmic_loss(bad, outcomes)
    assert len(calibration_table(good, outcomes, n_bins=2)) == 2
    assert expected_calibration_error(good, outcomes, n_bins=2) < 0.2
    assert {"brier", "log_loss", "ece"} <= probability_metrics(good, outcomes, 2).keys()


def test_isotonic_calibrator_requires_data_and_bounds_outputs() -> None:
    calibrator = IsotonicProbabilityCalibrator(min_samples=4)
    calibrator.fit([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    predictions = calibrator.predict([-1, 0.5, 2])
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_fractional_kelly_is_capped_and_drawdown_can_pause() -> None:
    assert full_kelly_fraction(0.60, 2.0) == pytest.approx(0.20)
    policy = RiskPolicy(max_bet_fraction=0.01, min_stake=0.01)
    state = PortfolioState(10_000)
    proposal = propose_stake(
        event_id="e",
        selection_id="a",
        probability=0.60,
        decimal_odds=2.0,
        uncertainty=0.0,
        market_probability=0.50,
        state=state,
        policy=policy,
    )
    assert 0 < proposal.stake <= 100
    paused = PortfolioState(7_000, peak_bankroll=10_000)
    assert propose_stake(
        event_id="e",
        selection_id="a",
        probability=0.70,
        decimal_odds=2.0,
        uncertainty=0.0,
        market_probability=0.50,
        state=paused,
        policy=policy,
    ).reason_code == "drawdown_pause"


def backtest_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "2025-R01-R",
                "selection_id": "a",
                "market": "head_to_head",
                "forecast_at": "2025-03-01T10:05:00Z",
                "quote_at": "2025-03-01T10:00:00Z",
                "event_start_at": "2025-03-02T12:00:00Z",
                "probability": 0.65,
                "uncertainty": 0.01,
                "fair_market_probability": 0.48,
                "decimal_odds": 2.10,
                "closing_odds": 1.95,
                "outcome": 1,
            },
            {
                "event_id": "2025-R02-R",
                "selection_id": "a",
                "market": "head_to_head",
                "forecast_at": "2025-03-07T10:05:00Z",
                "quote_at": "2025-03-07T10:00:00Z",
                "event_start_at": "2025-03-08T12:00:00Z",
                "probability": 0.62,
                "uncertainty": 0.01,
                "fair_market_probability": 0.49,
                "decimal_odds": 2.05,
                "closing_odds": 1.98,
                "outcome": 0,
            },
            {
                "event_id": "2025-R03-R",
                "selection_id": "a",
                "market": "head_to_head",
                "forecast_at": "2025-03-10T10:00:00Z",
                "quote_at": "2025-03-10T10:05:00Z",
                "event_start_at": "2025-03-11T12:00:00Z",
                "probability": 0.80,
                "uncertainty": 0.0,
                "fair_market_probability": 0.45,
                "decimal_odds": 2.20,
                "closing_odds": 2.0,
                "outcome": 1,
            },
        ]
    )


def test_backtest_requires_frozen_prices_and_reports_clv() -> None:
    result = run_backtest(
        backtest_records(),
        policy=RiskPolicy(min_stake=0.01, minimum_edge=0.01, minimum_ev=0.01),
    )
    assert result.summary.bets == 2
    assert result.summary.rejected_lookahead == 1
    assert result.summary.mean_clv is not None and result.summary.mean_clv > 0
    assert set(result.ledger.status) == {"won", "lost"}


def test_backtest_refuses_model_implied_odds_only() -> None:
    with pytest.raises(KeyError, match="decimal_odds"):
        run_backtest(backtest_records().drop(columns="decimal_odds"))
