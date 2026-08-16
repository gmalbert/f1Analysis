"""Odds conversion, overround removal, value, and closing-line utilities."""

from __future__ import annotations

from collections.abc import Iterable
from fractions import Fraction
import math


def decimal_to_implied_probability(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    return 1.0 / float(decimal_odds)


def american_to_decimal(american_odds: float) -> float:
    if american_odds == 0:
        raise ValueError("American odds cannot be zero")
    return 1.0 + (american_odds / 100.0 if american_odds > 0 else 100.0 / abs(american_odds))


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    return (decimal_odds - 1.0) * 100.0 if decimal_odds >= 2.0 else -100.0 / (decimal_odds - 1.0)


def fractional_to_decimal(value: str) -> float:
    fraction = Fraction(value.strip())
    if fraction < 0:
        raise ValueError("fractional odds must be non-negative")
    return 1.0 + float(fraction)


def overround(decimal_odds: Iterable[float]) -> float:
    odds = list(decimal_odds)
    if len(odds) < 2:
        raise ValueError("at least two mutually exclusive outcomes are required")
    return sum(decimal_to_implied_probability(price) for price in odds) - 1.0


def _validate_raw_probabilities(raw: Iterable[float]) -> list[float]:
    values = [float(value) for value in raw]
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("raw probabilities must contain at least two positive finite values")
    return values


def devig_probabilities(raw: Iterable[float], method: str = "multiplicative") -> list[float]:
    """Remove bookmaker margin from a complete set of exclusive outcomes.

    ``multiplicative`` is stable for general markets. ``additive`` subtracts
    equal margin and then renormalizes. ``power`` solves for an exponent whose
    transformed probabilities sum to one and is useful when favorite-longshot
    bias is suspected.
    """
    values = _validate_raw_probabilities(raw)
    method = method.lower().strip()
    if method == "multiplicative":
        total = sum(values)
        return [value / total for value in values]
    if method == "additive":
        deduction = (sum(values) - 1.0) / len(values)
        adjusted = [max(value - deduction, 1e-12) for value in values]
        total = sum(adjusted)
        return [value / total for value in adjusted]
    if method == "power":
        low, high = 0.01, 20.0
        for _ in range(100):
            exponent = (low + high) / 2.0
            total = sum(value**exponent for value in values)
            if total > 1.0:
                low = exponent
            else:
                high = exponent
        exponent = (low + high) / 2.0
        adjusted = [value**exponent for value in values]
        total = sum(adjusted)
        return [value / total for value in adjusted]
    raise ValueError(f"unsupported devig method: {method}")


def devig_decimal_odds(decimal_odds: Iterable[float], method: str = "multiplicative") -> list[float]:
    return devig_probabilities(
        [decimal_to_implied_probability(price) for price in decimal_odds], method=method
    )


def expected_value(probability: float, decimal_odds: float) -> float:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if decimal_odds <= 1.0:
        raise ValueError("decimal odds must be greater than 1")
    return probability * decimal_odds - 1.0


def probability_edge(model_probability: float, fair_market_probability: float) -> float:
    if not 0 <= model_probability <= 1 or not 0 <= fair_market_probability <= 1:
        raise ValueError("probabilities must be in [0, 1]")
    return model_probability - fair_market_probability


def closing_line_value(taken_odds: float, closing_odds: float) -> float:
    """Return price-based CLV; positive means a better price than the close."""
    if min(taken_odds, closing_odds) <= 1.0:
        raise ValueError("odds must be greater than 1")
    return taken_odds / closing_odds - 1.0
