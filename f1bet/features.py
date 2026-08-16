"""Feature registry and leakage-safe temporal feature builders."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from .domain import SessionStage


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    sources: tuple[str, ...]
    available_at: SessionStage
    lookback_events: int | None = None
    description: str = ""
    leakage_risk: str = "low"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.sources or any(not source.strip() for source in self.sources):
            raise ValueError("feature name and at least one non-empty source are required")
        if self.lookback_events is not None and self.lookback_events < 1:
            raise ValueError("lookback_events must be positive when provided")
        if self.leakage_risk not in {"low", "medium", "high", "target"}:
            raise ValueError("leakage_risk must be low, medium, high, or target")
        if not self.description.strip():
            object.__setattr__(
                self,
                "description",
                self.name.replace("_", " ").capitalize() + ".",
            )


@dataclass(slots=True)
class FeatureRegistry:
    definitions: dict[str, FeatureDefinition] = field(default_factory=dict)

    def register(self, definition: FeatureDefinition) -> None:
        if definition.name in self.definitions:
            raise ValueError(f"feature {definition.name!r} is already registered")
        self.definitions[definition.name] = definition

    def available(self, stage: SessionStage) -> list[str]:
        return sorted(
            name
            for name, definition in self.definitions.items()
            if definition.available_at <= stage
        )

    def assert_available(self, features: Iterable[str], stage: SessionStage) -> None:
        unknown = set(features) - set(self.definitions)
        if unknown:
            raise KeyError(f"unregistered features: {sorted(unknown)}")
        late = [
            feature
            for feature in features
            if self.definitions[feature].available_at > stage
        ]
        if late:
            raise ValueError(f"features unavailable at {stage.name}: {sorted(late)}")

    def manifest(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for definition in sorted(self.definitions.values(), key=lambda item: item.name):
            row = asdict(definition)
            row["available_at"] = definition.available_at.name
            rows.append(row)
        return rows


def leakage_safe_rolling(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_cols: str | list[str],
    order_cols: list[str],
    windows: Iterable[int] = (3, 5),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Add shifted rolling mean/std so the current event never trains on itself."""
    result = frame.copy()
    groups = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    required = set(groups + order_cols + [value_col])
    missing = required - set(result.columns)
    if missing:
        raise KeyError(f"missing rolling feature columns: {sorted(missing)}")
    result["__source_order"] = np.arange(len(result))
    result = result.sort_values(groups + order_cols, kind="stable")
    shifted = result.groupby(groups, sort=False, observed=True)[value_col].shift(1)
    stem = prefix or value_col
    for window in windows:
        if window < 2:
            raise ValueError("rolling windows must be at least 2")
        grouped = shifted.groupby([result[column] for column in groups], sort=False)
        result[f"{stem}_mean_{window}r"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        result[f"{stem}_std_{window}r"] = grouped.transform(
            lambda values: values.rolling(window, min_periods=2).std()
        )
    return result.sort_values("__source_order").drop(columns="__source_order")


def leakage_safe_event_rolling(
    frame: pd.DataFrame,
    *,
    value_col: str,
    group_cols: str | list[str],
    event_cols: list[str],
    order_cols: list[str],
    windows: Iterable[int] = (3, 5),
    prefix: str | None = None,
    event_aggregation: str = "mean",
) -> pd.DataFrame:
    """Build prior-event rolling features without leaking between rows in one event.

    The legacy table may contain multiple session rows per driver and naturally
    contains two drivers per constructor.  Rolling directly over rows lets an
    earlier row from the current Grand Prix influence a later row from that same
    Grand Prix.  This helper first collapses to the declared event grain, shifts
    whole events, and joins the resulting features back to every source row.
    """

    result = frame.copy()
    groups = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    keys = groups + event_cols
    required = set(keys + order_cols + [value_col])
    missing = required - set(result.columns)
    if missing:
        raise KeyError(f"missing event rolling feature columns: {sorted(missing)}")
    window_values = tuple(int(window) for window in windows)
    if not window_values or any(window < 2 for window in window_values):
        raise ValueError("rolling windows must be at least 2")

    numeric = pd.to_numeric(result[value_col], errors="coerce")
    working_columns = list(dict.fromkeys(keys + order_cols))
    working = result[working_columns].copy()
    working["__value"] = numeric
    aggregation = {"mean": "mean", "max": "max", "min": "min"}.get(event_aggregation)
    if aggregation is None:
        raise ValueError("event_aggregation must be mean, min, or max")
    order_aggregations = {
        column: (column, "first") for column in order_cols if column not in keys
    }
    event_frame = (
        working.groupby(keys, dropna=False, observed=True, as_index=False)
        .agg(
            __value=("__value", aggregation),
            **order_aggregations,
        )
        .sort_values(groups + order_cols + event_cols, kind="stable")
    )
    shifted = event_frame.groupby(groups, dropna=False, sort=False, observed=True)["__value"].shift(1)
    stem = prefix or value_col
    generated: list[str] = []
    groupers = [event_frame[column] for column in groups]
    for window in window_values:
        grouped = shifted.groupby(groupers, dropna=False, sort=False)
        mean_name = f"{stem}_mean_{window}r"
        std_name = f"{stem}_std_{window}r"
        event_frame[mean_name] = grouped.transform(
            lambda values: values.rolling(window, min_periods=1).mean()
        )
        event_frame[std_name] = grouped.transform(
            lambda values: values.rolling(window, min_periods=2).std()
        )
        generated.extend([mean_name, std_name])

    return result.merge(
        event_frame[keys + generated],
        on=keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )


def add_pre_race_form_features(
    frame: pd.DataFrame,
    *,
    driver_col: str = "resultsDriverId",
    constructor_col: str = "constructorName",
    target_col: str = "resultsFinalPositionNumber",
    dnf_col: str = "DNF",
    season_col: str = "grandPrixYear",
    round_col: str = "round",
) -> pd.DataFrame:
    """Build a compact, auditable form family from only prior race results."""
    event_cols = ["event_id"] if "event_id" in frame else [season_col, round_col]
    result = leakage_safe_event_rolling(
        frame,
        value_col=target_col,
        group_cols=driver_col,
        event_cols=event_cols,
        order_cols=[season_col, round_col],
        windows=(3, 5, 10),
        prefix="driver_finish",
    )
    result = leakage_safe_event_rolling(
        result,
        value_col=target_col,
        group_cols=constructor_col,
        event_cols=event_cols,
        order_cols=[season_col, round_col],
        windows=(3, 5),
        prefix="constructor_finish",
    )
    if dnf_col in result:
        result = leakage_safe_event_rolling(
            result,
            value_col=dnf_col,
            group_cols=driver_col,
            event_cols=event_cols,
            order_cols=[season_col, round_col],
            windows=(5, 10),
            prefix="driver_dnf",
            event_aggregation="max",
        )
    event_keys = [driver_col] + event_cols
    unique_starts = (
        result[event_keys + [season_col, round_col]]
        .drop_duplicates(event_keys)
        .sort_values([driver_col, season_col, round_col] + event_cols, kind="stable")
    )
    unique_starts["prior_career_starts"] = unique_starts.groupby(
        driver_col, dropna=False, observed=True
    ).cumcount()
    result = result.merge(
        unique_starts[event_keys + ["prior_career_starts"]],
        on=event_keys,
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result["experience_log"] = np.log1p(result["prior_career_starts"])
    result["season_progress"] = pd.to_numeric(result[round_col], errors="coerce") / 30.0
    result["regulation_era"] = pd.cut(
        pd.to_numeric(result[season_col], errors="coerce"),
        bins=[1949, 2008, 2013, 2016, 2021, 2025, 2200],
        labels=["pre_2009", "v8", "early_hybrid", "wide_car", "ground_effect", "2026_rules"],
    ).astype("string")
    return result


def build_head_to_head_deltas(
    frame: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    event_col: str = "event_id",
    driver_col: str = "resultsDriverId",
) -> pd.DataFrame:
    """Create ordered pairwise deltas at one driver-event row per selection."""

    features = list(dict.fromkeys(feature_columns))
    required = {event_col, driver_col, *features}
    missing = required - set(frame)
    if missing:
        raise KeyError(f"head-to-head input missing columns: {sorted(missing)}")
    if frame.duplicated([event_col, driver_col]).any():
        raise ValueError("head-to-head features require one row per driver and event")
    selection = frame[[event_col, driver_col, *features]].copy()
    opponent = selection.rename(
        columns={
            driver_col: "opponent_id",
            **{column: f"__opponent_{column}" for column in features},
        }
    )
    pairs = selection.merge(opponent, on=event_col, how="inner", validate="many_to_many")
    pairs = pairs.loc[pairs[driver_col].astype(str) != pairs["opponent_id"].astype(str)].copy()
    pairs = pairs.rename(columns={driver_col: "selection_id"})
    for column in features:
        selected = pd.to_numeric(pairs[column], errors="coerce")
        opposing = pd.to_numeric(pairs[f"__opponent_{column}"], errors="coerce")
        pairs[f"{column}_delta"] = selected - opposing
    return pairs[
        [event_col, "selection_id", "opponent_id"]
        + [f"{column}_delta" for column in features]
    ].reset_index(drop=True)


def add_prior_dnf_hazard_features(
    frame: pd.DataFrame,
    *,
    cause_col: str = "dnf_cause",
    driver_col: str = "resultsDriverId",
    season_col: str = "grandPrixYear",
    round_col: str = "round",
) -> pd.DataFrame:
    """Separate shifted mechanical, collision, and non-classification hazards."""

    if cause_col not in frame:
        raise KeyError(cause_col)
    result = frame.copy()
    normalized = result[cause_col].astype("string").str.lower().fillna("")
    mechanical_tokens = r"engine|gearbox|hydraulic|electrical|power unit|mechanical|brake|suspension"
    collision_tokens = r"collision|accident|crash|damage"
    nonclassification_tokens = r"not classified|disqualified|excluded|did not start|dns"
    result["__dnf_mechanical"] = normalized.str.contains(mechanical_tokens, regex=True).astype(int)
    result["__dnf_collision"] = normalized.str.contains(collision_tokens, regex=True).astype(int)
    result["__dnf_nonclassification"] = normalized.str.contains(
        nonclassification_tokens, regex=True
    ).astype(int)
    event_cols = ["event_id"] if "event_id" in result else [season_col, round_col]
    for indicator, prefix in (
        ("__dnf_mechanical", "mechanical_dnf"),
        ("__dnf_collision", "collision_dnf"),
        ("__dnf_nonclassification", "nonclassification"),
    ):
        result = leakage_safe_event_rolling(
            result,
            value_col=indicator,
            group_cols=driver_col,
            event_cols=event_cols,
            order_cols=[season_col, round_col],
            windows=(5, 10),
            prefix=prefix,
            event_aggregation="max",
        )
    return result.drop(
        columns=["__dnf_mechanical", "__dnf_collision", "__dnf_nonclassification"]
    )


def default_registry() -> FeatureRegistry:
    registry = FeatureRegistry()
    definitions = (
        FeatureDefinition("driver_finish_mean_3r", ("race_results",), SessionStage.PRE_WEEKEND, 3),
        FeatureDefinition("driver_finish_std_3r", ("race_results",), SessionStage.PRE_WEEKEND, 3),
        FeatureDefinition("driver_finish_mean_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("driver_finish_std_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("driver_finish_mean_10r", ("race_results",), SessionStage.PRE_WEEKEND, 10),
        FeatureDefinition("driver_finish_std_10r", ("race_results",), SessionStage.PRE_WEEKEND, 10),
        FeatureDefinition("constructor_finish_mean_3r", ("race_results",), SessionStage.PRE_WEEKEND, 3),
        FeatureDefinition("constructor_finish_std_3r", ("race_results",), SessionStage.PRE_WEEKEND, 3),
        FeatureDefinition("constructor_finish_mean_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("constructor_finish_std_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("driver_dnf_mean_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("driver_dnf_std_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("driver_dnf_mean_10r", ("race_results",), SessionStage.PRE_WEEKEND, 10),
        FeatureDefinition("driver_dnf_std_10r", ("race_results",), SessionStage.PRE_WEEKEND, 10),
        FeatureDefinition("prior_career_starts", ("race_results",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("experience_log", ("race_results",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("season_progress", ("schedule",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("regulation_era", ("schedule",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("mechanical_dnf_mean_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("mechanical_dnf_mean_10r", ("race_results",), SessionStage.PRE_WEEKEND, 10),
        FeatureDefinition("collision_dnf_mean_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("collision_dnf_mean_10r", ("race_results",), SessionStage.PRE_WEEKEND, 10),
        FeatureDefinition("nonclassification_mean_5r", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("teammate_relative_prior_pace", ("race_results",), SessionStage.PRE_WEEKEND, 5),
        FeatureDefinition("track_archetype", ("circuit",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("overtaking_difficulty", ("circuit", "race_results"), SessionStage.PRE_WEEKEND),
        FeatureDefinition("similar_circuit_form", ("race_results", "circuit"), SessionStage.PRE_WEEKEND, 10),
        FeatureDefinition("announced_upgrade_effect", ("team_updates",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("historical_weather_regime", ("historical_weather",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("travel_timezone_change", ("schedule", "circuit"), SessionStage.PRE_WEEKEND),
        FeatureDefinition("race_turnaround_days", ("schedule",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("tyre_allocation", ("fia_documents",), SessionStage.PRE_WEEKEND),
        FeatureDefinition("circuit_pit_loss_distribution", ("pit_stops", "circuit"), SessionStage.PRE_WEEKEND),
        FeatureDefinition("fp1_long_run_pace", ("fastf1_laps",), SessionStage.POST_FP1),
        FeatureDefinition("fp1_sector_delta", ("fastf1_laps",), SessionStage.POST_FP1),
        FeatureDefinition("fp1_track_evolution", ("fastf1_laps",), SessionStage.POST_FP1),
        FeatureDefinition("fp1_run_plan_lap_count", ("fastf1_laps",), SessionStage.POST_FP1),
        FeatureDefinition("fp2_long_run_pace", ("fastf1_laps",), SessionStage.POST_FP2),
        FeatureDefinition("fp2_fuel_corrected_long_run_delta", ("fastf1_laps", "traffic_estimates"), SessionStage.POST_FP2),
        FeatureDefinition("fp2_degradation_posterior", ("fastf1_laps",), SessionStage.POST_FP2),
        FeatureDefinition("fp2_clean_air_delta", ("fastf1_laps", "traffic_estimates"), SessionStage.POST_FP2),
        FeatureDefinition("lap_residual_skew", ("fastf1_laps",), SessionStage.POST_FP2),
        FeatureDefinition("lap_residual_tail_weight", ("fastf1_laps",), SessionStage.POST_FP2),
        FeatureDefinition("track_evolution_rate", ("fastf1_laps",), SessionStage.POST_FP2),
        FeatureDefinition("traffic_clean_air_split", ("fastf1_laps", "traffic_estimates"), SessionStage.POST_FP2),
        FeatureDefinition("speed_trap_aero_match", ("fastf1_laps", "circuit"), SessionStage.POST_FP2),
        FeatureDefinition("practice_weather_change", ("weather", "fastf1_laps"), SessionStage.POST_FP2),
        FeatureDefinition("comparable_stint_teammate_delta", ("fastf1_laps",), SessionStage.POST_FP2),
        FeatureDefinition("fp3_sector_delta", ("fastf1_laps",), SessionStage.POST_FP3),
        FeatureDefinition("qualifying_rank", ("qualifying",), SessionStage.POST_QUALIFYING),
        FeatureDefinition("qualifying_lap_distribution", ("qualifying",), SessionStage.POST_QUALIFYING),
        FeatureDefinition("qualifying_teammate_delta", ("qualifying",), SessionStage.POST_QUALIFYING),
        FeatureDefinition("start_position_incident_exposure", ("race_control", "qualifying"), SessionStage.POST_QUALIFYING),
        FeatureDefinition("grid_penalty", ("fia_documents",), SessionStage.PRE_RACE),
        FeatureDefinition("confirmed_grid_position", ("fia_documents", "schedule"), SessionStage.PRE_RACE),
        FeatureDefinition("forecast_rain_probability", ("weather",), SessionStage.PRE_RACE),
        FeatureDefinition("forecast_wind_speed", ("weather",), SessionStage.PRE_RACE),
        FeatureDefinition("forecast_track_temperature", ("weather",), SessionStage.PRE_RACE),
        FeatureDefinition("market_consensus_probability", ("odds_quotes",), SessionStage.PRE_RACE),
        FeatureDefinition("starting_tyre_scenario", ("tyre_history", "fia_documents"), SessionStage.PRE_RACE),
        FeatureDefinition("strategy_scenario_probability", ("strategy_simulation",), SessionStage.PRE_RACE),
        FeatureDefinition("critical_source_freshness", ("source_manifest",), SessionStage.PRE_RACE),
        FeatureDefinition("critical_source_missingness", ("source_manifest",), SessionStage.PRE_RACE),
        FeatureDefinition("closing_market_probability", ("odds_quotes",), SessionStage.POST_RACE, leakage_risk="high"),
        FeatureDefinition("first_lap_position", ("race_laps",), SessionStage.LIVE, leakage_risk="high"),
        FeatureDefinition("live_gap_to_ahead", ("race_laps",), SessionStage.LIVE),
        FeatureDefinition("live_traffic_train_size", ("race_laps",), SessionStage.LIVE),
        FeatureDefinition("live_tyre_compound", ("race_laps",), SessionStage.LIVE),
        FeatureDefinition("live_tyre_age", ("race_laps",), SessionStage.LIVE),
        FeatureDefinition("live_track_status", ("race_control",), SessionStage.LIVE),
        FeatureDefinition("live_incident_state", ("race_control",), SessionStage.LIVE),
        FeatureDefinition("live_degradation_residual", ("race_laps",), SessionStage.LIVE),
        FeatureDefinition("live_pit_window_state", ("race_laps", "strategy_simulation"), SessionStage.LIVE),
        FeatureDefinition("live_undercut_overcut_state", ("race_laps", "strategy_simulation"), SessionStage.LIVE),
        FeatureDefinition("final_position", ("race_results",), SessionStage.POST_RACE, leakage_risk="target"),
    )
    for definition in definitions:
        registry.register(definition)
    return registry
