"""Idempotent migrations from legacy prediction/odds exports to v2 ledgers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

import numpy as np
import pandas as pd

from .domain import MarketType, SessionStage, stable_id


LEGACY_PROBABILITY_COLUMNS: Mapping[str, MarketType] = {
    "win_probability": MarketType.WIN,
    "podium_probability": MarketType.PODIUM,
    "top_6_probability": MarketType.TOP_6,
    "top_10_probability": MarketType.TOP_10,
    "dnf_probability": MarketType.DNF,
}


def migrate_prediction_wide_to_forecasts(
    frame: pd.DataFrame,
    *,
    event_id: str,
    model_version: str,
    generated_at: datetime,
    selection_col: str = "driver_id",
    stage: SessionStage = SessionStage.PRE_RACE,
    invalid_policy: str = "raise",
) -> pd.DataFrame:
    if generated_at.tzinfo is None:
        raise ValueError("generated_at must be timezone-aware")
    if selection_col not in frame:
        raise KeyError(selection_col)
    if invalid_policy not in {"raise", "skip"}:
        raise ValueError("invalid_policy must be raise or skip")
    rows: list[dict[str, object]] = []
    for source_column, market in LEGACY_PROBABILITY_COLUMNS.items():
        if source_column not in frame:
            continue
        for record in frame[[selection_col, source_column]].itertuples(index=False, name=None):
            selection, value = record
            if pd.isna(value):
                continue
            probability = float(value)
            if not 0 <= probability <= 1:
                if invalid_policy == "raise":
                    raise ValueError(f"{source_column} contains probability outside [0, 1]: {value!r}")
                continue
            forecast_id = stable_id(event_id, market.value, selection, model_version, generated_at.isoformat())
            rows.append(
                {
                    "forecast_id": forecast_id,
                    "event_id": event_id,
                    "market": market.value,
                    "selection_id": str(selection),
                    "opponent_id": None,
                    "probability": probability,
                    "uncertainty": 0.0,
                    "generated_at": generated_at.astimezone(timezone.utc).isoformat(),
                    "stage": stage.name,
                    "model_version": model_version,
                    "feature_snapshot_id": None,
                }
            )
    return pd.DataFrame(rows)


def migrate_legacy_odds(
    frame: pd.DataFrame,
    *,
    event_id: str,
    event_start_at: datetime,
    captured_at: datetime,
    bookmaker: str,
    selection_col: str = "driver",
    odds_columns: Mapping[str, MarketType] | None = None,
    invalid_policy: str = "raise",
) -> pd.DataFrame:
    if event_start_at.tzinfo is None or captured_at.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    if captured_at > event_start_at:
        raise ValueError("captured_at cannot be later than event_start_at")
    if selection_col not in frame:
        raise KeyError(selection_col)
    if invalid_policy not in {"raise", "skip"}:
        raise ValueError("invalid_policy must be raise or skip")
    mapping = odds_columns or {
        "win_odds_decimal": MarketType.WIN,
        "podium_odds_decimal": MarketType.PODIUM,
    }
    rows: list[dict[str, object]] = []
    for column, market in mapping.items():
        if column not in frame:
            continue
        for selection, odds in frame[[selection_col, column]].itertuples(index=False, name=None):
            if pd.isna(odds):
                continue
            if float(odds) <= 1:
                if invalid_policy == "raise":
                    raise ValueError(f"{column} contains decimal odds at or below 1: {odds!r}")
                continue
            quote_id = stable_id(
                event_id,
                market.value,
                selection,
                None,
                bookmaker,
                captured_at.isoformat(),
                None,
            )
            rows.append(
                {
                    "quote_id": quote_id,
                    "event_id": event_id,
                    "market": market.value,
                    "selection_id": str(selection),
                    "opponent_id": None,
                    "bookmaker": bookmaker,
                    "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
                    "event_start_at": event_start_at.astimezone(timezone.utc).isoformat(),
                    "decimal_odds": float(odds),
                    "line": np.nan,
                }
            )
    return pd.DataFrame(rows)
