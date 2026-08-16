from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from f1bet.contracts import (
    FORECAST_LEDGER_CONTRACT,
    ODDS_LEDGER_CONTRACT,
    add_event_identity,
)
from f1bet.domain import EventKey, Forecast, MarketQuote, MarketType, SessionStage
from f1bet.markets import (
    RaceResult,
    SettlementRuleSet,
    fair_probabilities_for_quotes,
    settle_market,
    settle_market_with_rules,
)
from f1bet.odds import (
    american_to_decimal,
    closing_line_value,
    devig_decimal_odds,
    expected_value,
    fractional_to_decimal,
    overround,
)


NOW = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)


def test_event_and_records_are_stable_and_timezone_safe() -> None:
    event = EventKey(2026, 12, "R", "belgium")
    assert event.event_id == "2026-belgium-R"
    quote = MarketQuote(
        event.event_id,
        MarketType.WIN,
        "driver-a",
        "book",
        NOW,
        4.0,
        NOW + timedelta(days=1),
    )
    forecast = Forecast(
        event.event_id,
        MarketType.WIN,
        "driver-a",
        0.30,
        NOW,
        SessionStage.PRE_RACE,
        "model-v1",
    )
    assert quote.quote_id == quote.quote_id
    assert forecast.forecast_id == forecast.forecast_id
    assert ODDS_LEDGER_CONTRACT.validate(pd.DataFrame([quote.as_record()])).valid
    assert FORECAST_LEDGER_CONTRACT.validate(pd.DataFrame([forecast.as_record()])).valid


def test_naive_timestamps_and_invalid_odds_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        MarketQuote("event", MarketType.WIN, "d", "b", datetime.now(), 2.0, NOW)
    with pytest.raises(ValueError, match="greater than 1"):
        MarketQuote("event", MarketType.WIN, "d", "b", NOW, 1.0, NOW)


def test_odds_conversion_devig_and_value() -> None:
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)
    assert fractional_to_decimal("3/1") == pytest.approx(4.0)
    fair = devig_decimal_odds([1.91, 1.91])
    assert fair == pytest.approx([0.5, 0.5])
    assert overround([1.91, 1.91]) > 0
    assert expected_value(0.60, 2.0) == pytest.approx(0.20)
    assert closing_line_value(2.2, 2.0) == pytest.approx(0.10)


@pytest.mark.parametrize("method", ["multiplicative", "additive", "power"])
def test_all_devig_methods_sum_to_one(method: str) -> None:
    assert sum(devig_decimal_odds([2.4, 3.2, 3.4], method=method)) == pytest.approx(1.0)


def test_complete_quote_set_gets_fair_probabilities() -> None:
    quotes = [
        MarketQuote("e", MarketType.HEAD_TO_HEAD, "a", "book", NOW, 1.91, NOW + timedelta(days=1), opponent_id="b"),
        MarketQuote("e", MarketType.HEAD_TO_HEAD, "b", "book", NOW, 1.91, NOW + timedelta(days=1), opponent_id="a"),
    ]
    fair = fair_probabilities_for_quotes(quotes)
    assert set(fair) == {quote.quote_id for quote in quotes}
    assert list(fair.values()) == pytest.approx([0.5, 0.5])


def test_market_settlement_is_rule_aware() -> None:
    result = RaceResult(
        positions={"a": 1, "b": 2, "c": 11},
        dnf=frozenset({"c"}),
        fastest_lap_driver="b",
        safety_car=True,
    )
    assert settle_market(MarketType.WIN, "a", result)
    assert settle_market(MarketType.PODIUM, "b", result)
    assert not settle_market(MarketType.TOP_10, "c", result)
    assert settle_market(MarketType.DNF, "c", result)
    assert settle_market(MarketType.HEAD_TO_HEAD, "a", result, opponent_id="b")
    assert settle_market(MarketType.FASTEST_LAP, "b", result)
    assert settle_market(MarketType.SAFETY_CAR, "ignored", result)


def test_book_specific_settlement_requires_versioned_market_policy() -> None:
    result = RaceResult({"a": 1, "b": 2})
    rules = SettlementRuleSet(
        bookmaker="example-book",
        rule_version="2026-01",
        market_policies={MarketType.HEAD_TO_HEAD: "standard_classification"},
    )
    ruled = settle_market_with_rules(
        MarketType.HEAD_TO_HEAD, "a", result, rules, opponent_id="b"
    )
    assert ruled.outcome is True and ruled.rule_version == "2026-01"
    with pytest.raises(KeyError, match="do not cover"):
        settle_market_with_rules(MarketType.WIN, "a", result, rules)


def test_contract_reports_duplicate_quote_ids_and_range_errors() -> None:
    row = {
        "quote_id": "same",
        "event_id": "e",
        "market": "win",
        "selection_id": "a",
        "opponent_id": None,
        "bookmaker": "book",
        "captured_at": NOW.isoformat(),
        "event_start_at": (NOW + timedelta(days=1)).isoformat(),
        "decimal_odds": 0.5,
        "line": None,
    }
    report = ODDS_LEDGER_CONTRACT.validate(pd.DataFrame([row, row]))
    codes = {issue.code for issue in report.issues}
    assert {"below_minimum", "duplicate_key"} <= codes
    assert not report.valid


def test_legacy_event_identity_is_deterministic() -> None:
    frame = pd.DataFrame({"grandPrixYear": [2025, 2025], "round": [1, 2]})
    migrated = add_event_identity(frame)
    assert migrated["event_id"].tolist() == ["2025-R01-R", "2025-R02-R"]
