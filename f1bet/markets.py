"""Market grouping, quote selection, coherence checks, and settlement rules."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping

import numpy as np

from .domain import MarketQuote, MarketType
from .odds import decimal_to_implied_probability, devig_probabilities


@dataclass(frozen=True, slots=True)
class RaceResult:
    positions: Mapping[str, int]
    dnf: frozenset[str] = frozenset()
    fastest_lap_driver: str | None = None
    safety_car: bool | None = None


@dataclass(frozen=True, slots=True)
class SettlementRuleSet:
    """Versioned bookmaker interpretation for the supported standard markets."""

    bookmaker: str
    rule_version: str
    market_policies: Mapping[MarketType, str]

    def __post_init__(self) -> None:
        if not self.bookmaker.strip() or not self.rule_version.strip():
            raise ValueError("bookmaker and rule_version are required")
        unsupported = set(self.market_policies) - set(MarketType)
        if unsupported:
            raise ValueError(f"unsupported settlement markets: {sorted(map(str, unsupported))}")
        invalid = {
            market: policy
            for market, policy in self.market_policies.items()
            if policy not in {"standard_classification", "void"}
        }
        if invalid:
            raise ValueError(f"unsupported settlement policies: {invalid}")


@dataclass(frozen=True, slots=True)
class RuledOutcome:
    outcome: bool | None
    bookmaker: str
    rule_version: str


def quote_market_key(quote: MarketQuote) -> tuple[str, MarketType, str | None, str, object, float | None]:
    """Key for mutually exclusive outcomes at one bookmaker and timestamp."""
    pair = None
    if quote.market is MarketType.HEAD_TO_HEAD:
        pair = "|".join(sorted([quote.selection_id, str(quote.opponent_id)]))
    return quote.event_id, quote.market, pair, quote.bookmaker, quote.captured_at, quote.line


def fair_probabilities_for_quotes(
    quotes: Iterable[MarketQuote],
    *,
    method: str = "multiplicative",
    expected_outcomes: int | Mapping[tuple[str, MarketType, str | None, str, object, float | None], int] | None = None,
) -> dict[str, float]:
    """De-vig only market snapshots known to contain every exclusive outcome.

    Head-to-head snapshots are intrinsically complete when both sides are
    present.  Outright boards require an explicit expected outcome count.
    Podium/top-N/DNF selections are not mutually exclusive across drivers and
    are therefore never normalized together here.
    """
    grouped: dict[tuple[str, MarketType, str | None, str, object, float | None], list[MarketQuote]] = defaultdict(list)
    for quote in quotes:
        grouped[quote_market_key(quote)].append(quote)
    fair: dict[str, float] = {}
    for key, market_quotes in grouped.items():
        market = key[1]
        if market in {MarketType.PODIUM, MarketType.TOP_6, MarketType.TOP_10, MarketType.DNF}:
            continue
        required = 2 if market is MarketType.HEAD_TO_HEAD else None
        if isinstance(expected_outcomes, int):
            required = expected_outcomes
        elif expected_outcomes is not None:
            required = expected_outcomes.get(key)
        if required is None or len(market_quotes) != required:
            continue
        if len({quote.selection_id for quote in market_quotes}) != required:
            continue
        raw = [decimal_to_implied_probability(quote.decimal_odds) for quote in market_quotes]
        adjusted = devig_probabilities(raw, method=method)
        fair.update({quote.quote_id: probability for quote, probability in zip(market_quotes, adjusted)})
    return fair


def best_available_quotes(
    quotes: Iterable[MarketQuote], *, available_at: datetime | None = None
) -> list[MarketQuote]:
    """Keep the highest decimal price per event/market/selection/opponent."""
    if available_at is not None and (available_at.tzinfo is None or available_at.utcoffset() is None):
        raise ValueError("available_at must be timezone-aware")
    best: dict[tuple[str, MarketType, str, str | None, float | None], MarketQuote] = {}
    for quote in quotes:
        if available_at is not None and quote.captured_at > available_at:
            continue
        key = (quote.event_id, quote.market, quote.selection_id, quote.opponent_id, quote.line)
        incumbent = best.get(key)
        if incumbent is None or quote.decimal_odds > incumbent.decimal_odds:
            best[key] = quote
    return sorted(best.values(), key=lambda quote: (quote.event_id, quote.market.value, quote.selection_id))


def market_consensus(
    quotes: Iterable[MarketQuote],
    *,
    method: str = "multiplicative",
    expected_outcomes: int | Mapping[tuple[str, MarketType, str | None, str, object, float | None], int] | None = None,
    available_at: datetime | None = None,
) -> dict[tuple[str, MarketType, str], float]:
    """Median of each book's latest complete de-vigged snapshot."""
    if available_at is not None and (available_at.tzinfo is None or available_at.utcoffset() is None):
        raise ValueError("available_at must be timezone-aware")
    quote_list = [
        quote for quote in quotes if available_at is None or quote.captured_at <= available_at
    ]
    fair = fair_probabilities_for_quotes(
        quote_list, method=method, expected_outcomes=expected_outcomes
    )
    latest: dict[tuple[str, MarketType, str, str], tuple[datetime, float]] = {}
    for quote in quote_list:
        if quote.quote_id in fair:
            key = (quote.event_id, quote.market, quote.selection_id, quote.bookmaker)
            incumbent = latest.get(key)
            if incumbent is None or quote.captured_at > incumbent[0]:
                latest[key] = (quote.captured_at, fair[quote.quote_id])
    values: dict[tuple[str, MarketType, str], list[float]] = defaultdict(list)
    for (event_id, market, selection_id, _), (_, probability) in latest.items():
        values[(event_id, market, selection_id)].append(probability)
    return {key: float(np.median(items)) for key, items in values.items()}


def settle_market(
    market: MarketType,
    selection_id: str,
    result: RaceResult,
    *,
    opponent_id: str | None = None,
) -> bool | None:
    """Settle common F1 markets; ``None`` means the result must be voided."""
    position = result.positions.get(selection_id)
    if market is MarketType.SAFETY_CAR:
        return result.safety_car
    if market is MarketType.FASTEST_LAP:
        return None if result.fastest_lap_driver is None else result.fastest_lap_driver == selection_id
    if market is MarketType.DNF:
        if selection_id not in result.positions and selection_id not in result.dnf:
            return None
        return selection_id in result.dnf
    if market is MarketType.HEAD_TO_HEAD:
        if opponent_id is None:
            raise ValueError("head-to-head settlement requires opponent_id")
        opponent_position = result.positions.get(opponent_id)
        if position is None or opponent_position is None or position == opponent_position:
            return None
        return position < opponent_position
    if position is None:
        return None
    cutoff = {
        MarketType.WIN: 1,
        MarketType.PODIUM: 3,
        MarketType.TOP_6: 6,
        MarketType.TOP_10: 10,
    }.get(market)
    if cutoff is None:
        raise ValueError(f"unsupported market: {market}")
    return position <= cutoff


def settle_market_with_rules(
    market: MarketType,
    selection_id: str,
    result: RaceResult,
    rules: SettlementRuleSet,
    *,
    opponent_id: str | None = None,
) -> RuledOutcome:
    """Settle only when the bookmaker's versioned policy covers the market."""

    policy = rules.market_policies.get(market)
    if policy is None:
        raise KeyError(f"{rules.bookmaker} rules {rules.rule_version} do not cover {market.value}")
    outcome = None if policy == "void" else settle_market(
        market,
        selection_id,
        result,
        opponent_id=opponent_id,
    )
    return RuledOutcome(outcome, rules.bookmaker, rules.rule_version)


def probability_coherence_issues(
    probabilities: Mapping[str, Mapping[MarketType, float]], tolerance: float = 1e-9
) -> list[str]:
    """Check nested finish-market probabilities for impossible ordering."""
    issues: list[str] = []
    for driver, markets in probabilities.items():
        ordered = [
            markets.get(MarketType.WIN),
            markets.get(MarketType.PODIUM),
            markets.get(MarketType.TOP_6),
            markets.get(MarketType.TOP_10),
        ]
        known = [value for value in ordered if value is not None]
        if any(value < -tolerance or value > 1 + tolerance for value in known):
            issues.append(f"{driver}: probability outside [0, 1]")
        if any(left > right + tolerance for left, right in zip(known, known[1:])):
            issues.append(f"{driver}: win/podium/top-6/top-10 probabilities are not nested")
    win_total = sum(markets.get(MarketType.WIN, 0.0) for markets in probabilities.values())
    if probabilities and abs(win_total - 1.0) > 0.02:
        issues.append(f"field: win probabilities sum to {win_total:.4f}, expected 1")
    return issues
