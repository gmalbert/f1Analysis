"""Conservative paper-betting sizing and portfolio exposure controls."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

from .odds import expected_value


def full_kelly_fraction(probability: float, decimal_odds: float) -> float:
    if not 0 <= probability <= 1 or decimal_odds <= 1:
        raise ValueError("invalid probability or odds")
    net = decimal_odds - 1.0
    return max(0.0, (probability * decimal_odds - 1.0) / net)


def conservative_probability(
    probability: float,
    uncertainty: float,
    *,
    z_score: float = 1.0,
    market_probability: float | None = None,
    market_blend: float = 0.0,
) -> float:
    """Haircut a forecast and optionally shrink it toward market consensus."""
    if not 0 <= probability <= 1 or uncertainty < 0 or not 0 <= market_blend <= 1:
        raise ValueError("invalid probability, uncertainty, or market_blend")
    adjusted = max(0.0, min(1.0, probability - z_score * uncertainty))
    if market_probability is not None:
        if not 0 <= market_probability <= 1:
            raise ValueError("market_probability must be in [0, 1]")
        adjusted = (1 - market_blend) * adjusted + market_blend * market_probability
    return adjusted


@dataclass(frozen=True, slots=True)
class RiskPolicy:
    staking_mode: Literal["fractional_kelly", "flat"] = "fractional_kelly"
    kelly_fraction: float = 0.25
    flat_stake_fraction: float = 0.01
    max_bet_fraction: float = 0.01
    max_event_fraction: float = 0.03
    max_selection_fraction: float = 0.015
    minimum_edge: float = 0.02
    minimum_ev: float = 0.02
    drawdown_pause: float = 0.20
    min_stake: float = 1.0
    uncertainty_z_score: float = 1.0
    market_blend: float = 0.10

    def __post_init__(self) -> None:
        fractions = (
            self.kelly_fraction,
            self.flat_stake_fraction,
            self.max_bet_fraction,
            self.max_event_fraction,
            self.max_selection_fraction,
            self.drawdown_pause,
            self.market_blend,
        )
        if any(value < 0 or value > 1 for value in fractions):
            raise ValueError("risk fractions must be in [0, 1]")
        if self.minimum_edge < 0 or self.minimum_ev < 0 or self.min_stake < 0 or self.uncertainty_z_score < 0:
            raise ValueError("risk thresholds must be non-negative")
        if self.staking_mode not in {"fractional_kelly", "flat"}:
            raise ValueError("staking_mode must be fractional_kelly or flat")


@dataclass(slots=True)
class PortfolioState:
    bankroll: float
    peak_bankroll: float | None = None
    event_exposure: dict[str, float] | None = None
    selection_exposure: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.bankroll < 0:
            raise ValueError("bankroll must be non-negative")
        self.peak_bankroll = max(self.bankroll, self.peak_bankroll or self.bankroll)
        self.event_exposure = dict(self.event_exposure or {})
        self.selection_exposure = dict(self.selection_exposure or {})

    @property
    def drawdown(self) -> float:
        if not self.peak_bankroll:
            return 0.0
        return max(0.0, 1.0 - self.bankroll / self.peak_bankroll)

    def settle(self, profit: float) -> None:
        self.bankroll += profit
        self.peak_bankroll = max(self.peak_bankroll or 0.0, self.bankroll)


@dataclass(frozen=True, slots=True)
class StakeProposal:
    stake: float
    adjusted_probability: float
    kelly: float
    expected_value: float
    reason_code: str


def propose_stake(
    *,
    event_id: str,
    selection_id: str,
    probability: float,
    decimal_odds: float,
    uncertainty: float,
    market_probability: float,
    state: PortfolioState,
    policy: RiskPolicy,
) -> StakeProposal:
    if state.drawdown >= policy.drawdown_pause:
        return StakeProposal(0.0, probability, 0.0, expected_value(probability, decimal_odds), "drawdown_pause")
    adjusted = conservative_probability(
        probability,
        uncertainty,
        z_score=policy.uncertainty_z_score,
        market_probability=market_probability,
        market_blend=policy.market_blend,
    )
    edge = adjusted - market_probability
    ev = expected_value(adjusted, decimal_odds)
    if edge < policy.minimum_edge:
        return StakeProposal(0.0, adjusted, 0.0, ev, "edge_below_threshold")
    if ev < policy.minimum_ev:
        return StakeProposal(0.0, adjusted, 0.0, ev, "ev_below_threshold")
    kelly = full_kelly_fraction(adjusted, decimal_odds) * policy.kelly_fraction
    base_cap = state.bankroll * policy.max_bet_fraction
    event_remaining = max(
        0.0,
        state.bankroll * policy.max_event_fraction - (state.event_exposure or {}).get(event_id, 0.0),
    )
    selection_remaining = max(
        0.0,
        state.bankroll * policy.max_selection_fraction
        - (state.selection_exposure or {}).get(selection_id, 0.0),
    )
    uncapped = (
        state.bankroll * policy.flat_stake_fraction
        if policy.staking_mode == "flat"
        else state.bankroll * kelly
    )
    stake = min(uncapped, base_cap, event_remaining, selection_remaining)
    if not math.isfinite(stake) or stake < policy.min_stake:
        return StakeProposal(0.0, adjusted, kelly, ev, "stake_below_minimum")
    return StakeProposal(round(stake, 2), adjusted, kelly, ev, "paper_bet")


def record_exposure(state: PortfolioState, event_id: str, selection_id: str, stake: float) -> None:
    if stake < 0:
        raise ValueError("stake must be non-negative")
    assert state.event_exposure is not None and state.selection_exposure is not None
    state.event_exposure[event_id] = state.event_exposure.get(event_id, 0.0) + stake
    state.selection_exposure[selection_id] = state.selection_exposure.get(selection_id, 0.0) + stake
