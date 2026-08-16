"""Canonical domain objects for race forecasts and market research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum
from hashlib import sha256
from typing import Any, Mapping


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


def stable_id(*parts: object) -> str:
    """Return a compact deterministic identifier without exposing raw values."""
    payload = "|".join(str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()[:20]


def _required_text(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


class SessionStage(IntEnum):
    """Information set available when a feature or forecast was created."""

    PRE_WEEKEND = 0
    POST_FP1 = 10
    POST_FP2 = 20
    POST_FP3 = 30
    POST_SPRINT = 35
    POST_QUALIFYING = 40
    PRE_RACE = 50
    LIVE = 60
    POST_RACE = 100


class MarketType(str, Enum):
    WIN = "win"
    PODIUM = "podium"
    TOP_6 = "top_6"
    TOP_10 = "top_10"
    HEAD_TO_HEAD = "head_to_head"
    DNF = "dnf"
    FASTEST_LAP = "fastest_lap"
    SAFETY_CAR = "safety_car"


class BetStatus(str, Enum):
    PAPER = "paper"
    WON = "won"
    LOST = "lost"
    VOID = "void"


class DecisionStatus(str, Enum):
    PLACED = "placed"
    ABSTAINED = "abstained"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class EventKey:
    season: int
    round_number: int
    session: str = "R"
    race_id: str | None = None

    def __post_init__(self) -> None:
        if self.season < 1950 or self.season > 2200:
            raise ValueError("season is outside the supported range")
        if self.round_number < 0 or self.round_number > 99:
            raise ValueError("round_number must be between 0 and 99")
        _required_text(self.session, "session")
        if self.race_id is not None:
            _required_text(self.race_id, "race_id")

    @property
    def event_id(self) -> str:
        suffix = self.race_id or f"R{self.round_number:02d}"
        return f"{self.season}-{suffix}-{self.session.upper()}"


@dataclass(frozen=True, slots=True)
class Event:
    """Canonical event dimension row."""

    event_id: str
    season: int
    round_number: int
    session: str
    scheduled_start_at: datetime
    circuit_id: str
    regulation_era: str

    def __post_init__(self) -> None:
        _required_text(self.event_id, "event_id")
        EventKey(self.season, self.round_number, self.session)
        _required_text(self.circuit_id, "circuit_id")
        _required_text(self.regulation_era, "regulation_era")
        object.__setattr__(self, "scheduled_start_at", _utc(self.scheduled_start_at))

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["scheduled_start_at"] = self.scheduled_start_at.isoformat()
        return record


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    """Lineage for one driver/event/stage feature observation."""

    event_id: str
    driver_id: str
    constructor_id: str | None
    feature_as_of: datetime
    feature_stage: SessionStage
    schema_version: str
    source_manifest_id: str
    snapshot_version: str = "1"

    def __post_init__(self) -> None:
        for value, name in (
            (self.event_id, "event_id"),
            (self.driver_id, "driver_id"),
            (self.schema_version, "schema_version"),
            (self.source_manifest_id, "source_manifest_id"),
            (self.snapshot_version, "snapshot_version"),
        ):
            _required_text(value, name)
        object.__setattr__(self, "feature_as_of", _utc(self.feature_as_of))

    @property
    def snapshot_id(self) -> str:
        return stable_id(
            self.event_id,
            self.driver_id,
            self.feature_stage.name,
            self.feature_as_of.isoformat(),
            self.schema_version,
            self.snapshot_version,
        )

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["feature_stage"] = self.feature_stage.name
        record["feature_as_of"] = self.feature_as_of.isoformat()
        record["snapshot_id"] = self.snapshot_id
        return record


@dataclass(frozen=True, slots=True)
class MarketQuote:
    event_id: str
    market: MarketType
    selection_id: str
    bookmaker: str
    captured_at: datetime
    decimal_odds: float
    event_start_at: datetime
    opponent_id: str | None = None
    line: float | None = None
    source_quote_id: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.event_id, "event_id"),
            (self.selection_id, "selection_id"),
            (self.bookmaker, "bookmaker"),
        ):
            _required_text(value, name)
        object.__setattr__(self, "captured_at", _utc(self.captured_at))
        object.__setattr__(self, "event_start_at", _utc(self.event_start_at))
        if self.decimal_odds <= 1.0:
            raise ValueError("decimal_odds must be greater than 1")
        if self.captured_at > self.event_start_at:
            raise ValueError("pre-race quote cannot be captured after event start")
        if self.market is MarketType.HEAD_TO_HEAD and not self.opponent_id:
            raise ValueError("head-to-head quotes require opponent_id")

    @property
    def quote_id(self) -> str:
        return self.source_quote_id or stable_id(
            self.event_id,
            self.market.value,
            self.selection_id,
            self.opponent_id,
            self.bookmaker,
            self.captured_at.isoformat(),
            self.line,
        )

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["market"] = self.market.value
        record["captured_at"] = self.captured_at.isoformat()
        record["event_start_at"] = self.event_start_at.isoformat()
        record["quote_id"] = self.quote_id
        return record


@dataclass(frozen=True, slots=True)
class Forecast:
    event_id: str
    market: MarketType
    selection_id: str
    probability: float
    generated_at: datetime
    stage: SessionStage
    model_version: str
    opponent_id: str | None = None
    uncertainty: float = 0.0
    feature_snapshot_id: str | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.event_id, "event_id"),
            (self.selection_id, "selection_id"),
            (self.model_version, "model_version"),
        ):
            _required_text(value, name)
        object.__setattr__(self, "generated_at", _utc(self.generated_at))
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("uncertainty must be in [0, 1]")
        if self.market is MarketType.HEAD_TO_HEAD and not self.opponent_id:
            raise ValueError("head-to-head forecasts require opponent_id")

    @property
    def forecast_id(self) -> str:
        return stable_id(
            self.event_id,
            self.market.value,
            self.selection_id,
            self.opponent_id,
            self.model_version,
            self.generated_at.isoformat(),
        )

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["market"] = self.market.value
        record["stage"] = self.stage.name
        record["generated_at"] = self.generated_at.isoformat()
        record["forecast_id"] = self.forecast_id
        return record


@dataclass(frozen=True, slots=True)
class BetDecision:
    quote_id: str
    forecast_id: str
    event_id: str
    market: MarketType
    selection_id: str
    decided_at: datetime
    model_probability: float
    fair_market_probability: float
    decimal_odds: float
    edge: float
    expected_value: float
    stake: float
    bankroll_before: float
    reason_code: str
    opponent_id: str | None = None
    status: DecisionStatus = DecisionStatus.PLACED

    def __post_init__(self) -> None:
        for value, name in (
            (self.quote_id, "quote_id"),
            (self.forecast_id, "forecast_id"),
            (self.event_id, "event_id"),
            (self.selection_id, "selection_id"),
            (self.reason_code, "reason_code"),
        ):
            _required_text(value, name)
        object.__setattr__(self, "decided_at", _utc(self.decided_at))
        for value, name in (
            (self.model_probability, "model_probability"),
            (self.fair_market_probability, "fair_market_probability"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.decimal_odds <= 1.0:
            raise ValueError("decimal_odds must be greater than 1")
        if self.stake < 0 or self.bankroll_before < 0:
            raise ValueError("stake and bankroll must be non-negative")
        if self.stake > self.bankroll_before:
            raise ValueError("stake cannot exceed bankroll")

    @property
    def bet_id(self) -> str:
        return stable_id(self.quote_id, self.forecast_id, self.decided_at.isoformat())

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["market"] = self.market.value
        record["status"] = self.status.value
        record["decided_at"] = self.decided_at.isoformat()
        record["bet_id"] = self.bet_id
        return record


@dataclass(frozen=True, slots=True)
class Settlement:
    """Versioned settlement fact for a paper decision."""

    bet_id: str
    status: BetStatus
    profit: float
    settled_at: datetime
    rule_version: str
    closing_odds: float | None = None
    closing_line_value: float | None = None
    settlement_version: int = 1
    supersedes_settlement_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.bet_id, "bet_id")
        _required_text(self.rule_version, "rule_version")
        object.__setattr__(self, "settled_at", _utc(self.settled_at))
        if self.status is BetStatus.PAPER:
            raise ValueError("a settlement status must be won, lost, or void")
        if self.closing_odds is not None and self.closing_odds <= 1.0:
            raise ValueError("closing_odds must be greater than 1")
        if self.settlement_version < 1:
            raise ValueError("settlement_version must be positive")

    @property
    def settlement_id(self) -> str:
        return stable_id(self.bet_id, self.settlement_version, self.rule_version)

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["status"] = self.status.value
        record["settled_at"] = self.settled_at.isoformat()
        record["settlement_id"] = self.settlement_id
        return record


@dataclass(frozen=True, slots=True)
class ModelVersion:
    """Minimal normalized model-version dimension used by forecast ledgers."""

    model_version: str
    data_sha256: str
    schema_version: str
    code_revision: str
    training_start_event: str
    training_end_event: str
    calibration_method: str | None
    feature_names: tuple[str, ...]
    dependency_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        for value, name in (
            (self.model_version, "model_version"),
            (self.data_sha256, "data_sha256"),
            (self.schema_version, "schema_version"),
            (self.code_revision, "code_revision"),
            (self.training_start_event, "training_start_event"),
            (self.training_end_event, "training_end_event"),
        ):
            _required_text(value, name)
        if len(self.data_sha256) != 64 or any(character not in "0123456789abcdef" for character in self.data_sha256.lower()):
            raise ValueError("data_sha256 must be a 64-character hexadecimal digest")
        if not self.feature_names:
            raise ValueError("feature_names cannot be empty")
