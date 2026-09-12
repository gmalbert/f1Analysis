from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FilterSpec(BaseModel):
    column: str
    kind: Literal["range", "date_range", "exact", "boolean"]
    value: Any


class QueryRequest(BaseModel):
    filters: list[FilterSpec] = Field(default_factory=list)
    columns: list[str] | None = None
    sort: list[str] = Field(default_factory=list)
    descending: bool = False
    offset: int = 0
    limit: int = Field(default=200, ge=1, le=5000)


class AnalyticsRequest(BaseModel):
    filters: list[FilterSpec] = Field(default_factory=list)
    max_rows: int = Field(default=5000, ge=100, le=50000)


class BettingValueRequest(BaseModel):
    model_probability: float = Field(0.25, gt=0, lt=1)
    decimal_odds: float = Field(2.10, gt=1)
    opposing_odds: float = Field(1.80, gt=1)
    uncertainty: float = Field(0.02, ge=0, le=0.5)
    devig_method: Literal["multiplicative", "additive", "power"] = "multiplicative"
    bankroll: float = Field(10000, gt=0)


class SimulationEntry(BaseModel):
    driver_id: str
    constructor_id: str
    pace_score: float
    dnf_probability: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0)
    race_sensitivity: float = 1.0


class SimulationRequest(BaseModel):
    entries: list[SimulationEntry]
    simulations: int = Field(10000, ge=1000, le=50000)
    seed: int = 42


class RowsPayload(BaseModel):
    rows: list[dict[str, Any]]


class ToolRunRequest(BaseModel):
    tool: str
    args: list[str] = Field(default_factory=list)
