"""Race-grouped expanding-window validation with optional embargo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    train_events: tuple[str, ...]
    test_events: tuple[str, ...]
    embargoed_events: tuple[str, ...]


def event_order(
    frame: pd.DataFrame,
    *,
    event_col: str = "event_id",
    season_col: str = "grandPrixYear",
    round_col: str = "round",
) -> list[str]:
    missing = {event_col, season_col, round_col} - set(frame.columns)
    if missing:
        raise KeyError(f"missing event ordering columns: {sorted(missing)}")
    events = frame[[event_col, season_col, round_col]].drop_duplicates().copy()
    duplicates = events.groupby(event_col).size()
    if (duplicates > 1).any():
        raise ValueError("an event_id maps to multiple season/round pairs")
    events[season_col] = pd.to_numeric(events[season_col], errors="raise")
    events[round_col] = pd.to_numeric(events[round_col], errors="raise")
    return (
        events.sort_values([season_col, round_col, event_col], kind="stable")[event_col]
        .astype(str)
        .tolist()
    )


def expanding_window_splits(
    frame: pd.DataFrame,
    *,
    min_train_events: int = 20,
    test_events: int = 1,
    step_events: int = 1,
    embargo_events: int = 0,
    event_col: str = "event_id",
    season_col: str = "grandPrixYear",
    round_col: str = "round",
) -> Iterator[WalkForwardFold]:
    """Yield chronological folds while keeping every driver in a race together."""
    if min(min_train_events, test_events, step_events) < 1 or embargo_events < 0:
        raise ValueError("split sizes must be positive and embargo non-negative")
    ordered = event_order(
        frame,
        event_col=event_col,
        season_col=season_col,
        round_col=round_col,
    )
    event_values = frame[event_col].astype(str)
    fold_number = 0
    test_start = min_train_events + embargo_events
    while test_start + test_events <= len(ordered):
        train_end = test_start - embargo_events
        train_ids = tuple(ordered[:train_end])
        embargoed = tuple(ordered[train_end:test_start])
        test_ids = tuple(ordered[test_start : test_start + test_events])
        train_idx = np.flatnonzero(event_values.isin(train_ids).to_numpy())
        test_idx = np.flatnonzero(event_values.isin(test_ids).to_numpy())
        if np.intersect1d(train_idx, test_idx).size:
            raise AssertionError("train/test row overlap detected")
        yield WalkForwardFold(
            fold=fold_number,
            train_index=train_idx,
            test_index=test_idx,
            train_events=train_ids,
            test_events=test_ids,
            embargoed_events=embargoed,
        )
        fold_number += 1
        test_start += step_events


def assert_strictly_future(
    frame: pd.DataFrame,
    fold: WalkForwardFold,
    *,
    season_col: str = "grandPrixYear",
    round_col: str = "round",
) -> None:
    train_pairs = frame.iloc[fold.train_index][[season_col, round_col]].drop_duplicates().copy()
    test_pairs = frame.iloc[fold.test_index][[season_col, round_col]].drop_duplicates().copy()
    if train_pairs.empty or test_pairs.empty:
        raise ValueError("fold contains an empty partition")
    for pairs in (train_pairs, test_pairs):
        pairs[season_col] = pd.to_numeric(pairs[season_col], errors="raise")
        pairs[round_col] = pd.to_numeric(pairs[round_col], errors="raise")
    train_max = max(map(tuple, train_pairs.to_numpy().tolist()))
    test_min = min(map(tuple, test_pairs.to_numpy().tolist()))
    if train_max >= test_min:
        raise AssertionError(f"non-future test fold: train max {train_max}, test min {test_min}")


def leave_season_forward_splits(
    frame: pd.DataFrame,
    *,
    season_col: str = "grandPrixYear",
    min_train_seasons: int = 2,
) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
    numeric_seasons = pd.to_numeric(frame[season_col], errors="raise")
    seasons: Sequence[int] = sorted(int(value) for value in numeric_seasons.dropna().unique())
    for position in range(min_train_seasons, len(seasons)):
        test_season = seasons[position]
        train = np.flatnonzero((numeric_seasons < test_season).to_numpy())
        test = np.flatnonzero((numeric_seasons == test_season).to_numpy())
        if len(train) and len(test):
            yield train, test, test_season


def final_season_holdout_indices(
    frame: pd.DataFrame,
    *,
    season_col: str = "grandPrixYear",
    event_col: str = "event_id",
    embargo_events: int = 1,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], int]:
    """Reserve the latest complete season for a single untouched final test.

    The returned development rows contain only earlier seasons.  An optional
    event embargo is removed from the end of development, never from the final
    test season, so feature/model selection cannot inspect the final block.
    """

    if embargo_events < 0:
        raise ValueError("embargo_events must be non-negative")
    working = with_event_identity(frame)
    numeric_seasons = pd.to_numeric(working[season_col], errors="raise")
    seasons = sorted(int(value) for value in numeric_seasons.dropna().unique())
    if len(seasons) < 2:
        raise ValueError("at least two seasons are required for a final-season holdout")
    final_season = seasons[-1]
    ordered = event_order(working)
    final_events = tuple(
        working.loc[numeric_seasons.eq(final_season), event_col].astype(str).drop_duplicates()
    )
    development_events = [event for event in ordered if event not in set(final_events)]
    if len(development_events) <= embargo_events:
        raise ValueError("insufficient development events after final-season embargo")
    embargoed = tuple(development_events[-embargo_events:]) if embargo_events else ()
    allowed_development = development_events[:-embargo_events] if embargo_events else development_events
    event_values = working[event_col].astype(str)
    development_index = np.flatnonzero(event_values.isin(allowed_development).to_numpy())
    final_index = np.flatnonzero(event_values.isin(final_events).to_numpy())
    if not len(development_index) or not len(final_index):
        raise ValueError("final-season holdout produced an empty partition")
    return development_index, final_index, embargoed, final_season


def grouped_bootstrap_interval(
    frame: pd.DataFrame,
    *,
    value_col: str,
    event_col: str = "event_id",
    statistic: str = "mean",
    n_bootstrap: int = 2_000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a statistic by complete event rather than dependent driver rows."""

    if value_col not in frame or event_col not in frame:
        raise KeyError(f"{value_col!r} and {event_col!r} are required")
    if n_bootstrap < 100 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap must be >= 100 and confidence in (0, 1)")
    clean = frame[[event_col, value_col]].copy()
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna()
    events = clean[event_col].drop_duplicates().to_numpy()
    if len(events) < 2:
        raise ValueError("at least two complete events are required")
    grouped = {event: clean.loc[clean[event_col] == event, value_col].to_numpy() for event in events}
    functions = {
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
    }
    if statistic not in functions:
        raise ValueError("statistic must be mean, median, or sum")
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sampled = rng.choice(events, size=len(events), replace=True)
        values = np.concatenate([grouped[event] for event in sampled])
        estimates[index] = functions[statistic](values)
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def with_event_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a position-preserving frame with event_id and numeric round."""

    working = frame.copy()
    if "round" not in working:
        race_col = next(
            (column for column in ("raceId_results", "raceId", "grandPrixId") if column in working),
            None,
        )
        if race_col is None or "grandPrixYear" not in working:
            raise KeyError("temporal validation requires round or a race identifier with grandPrixYear")
        working["_f1bet_row_order"] = np.arange(len(working))
        race_order = (
            working[["grandPrixYear", race_col]]
            .drop_duplicates()
            .sort_values(["grandPrixYear", race_col], kind="stable")
        )
        race_order["round"] = race_order.groupby("grandPrixYear", observed=True).cumcount() + 1
        working = (
            working.merge(
                race_order,
                on=["grandPrixYear", race_col],
                how="left",
                validate="many_to_one",
                sort=False,
            )
            .sort_values("_f1bet_row_order", kind="stable")
            .drop(columns="_f1bet_row_order")
        )
    if "event_id" not in working:
        from .contracts import add_event_identity

        working = add_event_identity(working)
    return working


def final_event_holdout_indices(
    frame: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    embargo_events: int = 1,
    minimum_events: int = 5,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Return a final chronological event holdout and its embargoed event IDs."""

    if not 0 < test_fraction < 1 or embargo_events < 0 or minimum_events < 3:
        raise ValueError("invalid temporal holdout configuration")
    working = with_event_identity(frame)
    ordered = event_order(working)
    if len(ordered) < minimum_events:
        raise ValueError(f"at least {minimum_events} complete events are required for temporal holdout")
    test_count = max(1, int(np.ceil(len(ordered) * test_fraction)))
    test_start = len(ordered) - test_count
    train_end = test_start - embargo_events
    if train_end < 2:
        raise ValueError("insufficient pre-holdout events after embargo")
    event_values = working["event_id"].astype(str)
    train_index = np.flatnonzero(event_values.isin(ordered[:train_end]).to_numpy())
    test_index = np.flatnonzero(event_values.isin(ordered[test_start:]).to_numpy())
    if not len(train_index) or not len(test_index):
        raise ValueError("temporal holdout produced an empty partition")
    return train_index, test_index, tuple(ordered[train_end:test_start])


def sklearn_expanding_window_cv(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    embargo_events: int = 1,
    minimum_train_fraction: float = 0.5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return sklearn-compatible chronological race-event CV indexes."""

    if n_splits < 2 or not 0 < minimum_train_fraction < 1:
        raise ValueError("n_splits must be >= 2 and minimum_train_fraction in (0, 1)")
    working = with_event_identity(frame)
    events = event_order(working)
    minimum_train = max(2, int(len(events) * minimum_train_fraction))
    first_test = minimum_train + embargo_events
    available = len(events) - first_test
    if available < n_splits:
        raise ValueError("insufficient events for requested temporal folds")
    blocks = np.array_split(np.arange(first_test, len(events)), n_splits)
    event_values = working["event_id"].astype(str)
    indexes: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        test_start = int(block[0])
        train_end = test_start - embargo_events
        train_ids = events[:train_end]
        test_ids = [events[int(position)] for position in block]
        train_index = np.flatnonzero(event_values.isin(train_ids).to_numpy())
        test_index = np.flatnonzero(event_values.isin(test_ids).to_numpy())
        indexes.append((train_index, test_index))
    return indexes


def sklearn_model_selection_cv(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    embargo_events: int = 1,
    minimum_train_fraction: float = 0.5,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, int]:
    """Create development-only temporal CV and reserve the latest season.

    CV indexes are expressed in the original frame's row positions.  The
    returned final index is never present in any search fold.
    """

    development, final_index, _, final_season = final_season_holdout_indices(
        frame,
        embargo_events=embargo_events,
    )
    development_frame = frame.iloc[development]
    local_folds = sklearn_expanding_window_cv(
        development_frame,
        n_splits=n_splits,
        embargo_events=embargo_events,
        minimum_train_fraction=minimum_train_fraction,
    )
    folds = [(development[train], development[test]) for train, test in local_folds]
    if any(np.intersect1d(index, final_index).size for fold in folds for index in fold):
        raise AssertionError("final season leaked into model-selection folds")
    return folds, final_index, final_season
