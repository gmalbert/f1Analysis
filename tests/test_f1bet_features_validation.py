from __future__ import annotations

import pandas as pd
import pytest

from f1bet.domain import SessionStage
from f1bet.features import (
    add_pre_race_form_features,
    add_prior_dnf_hazard_features,
    build_head_to_head_deltas,
    default_registry,
)
from f1bet.validation import (
    assert_strictly_future,
    expanding_window_splits,
    final_season_holdout_indices,
    final_event_holdout_indices,
    leave_season_forward_splits,
    sklearn_expanding_window_cv,
    sklearn_model_selection_cv,
)


def race_frame(events: int = 8, drivers: int = 3) -> pd.DataFrame:
    rows = []
    for event in range(events):
        season = 2024 + event // 4
        round_number = event % 4 + 1
        for driver in range(drivers):
            rows.append(
                {
                    "event_id": f"{season}-R{round_number:02d}-R",
                    "grandPrixYear": season,
                    "round": round_number,
                    "resultsDriverId": f"d{driver}",
                    "constructorName": f"c{driver // 2}",
                    "resultsFinalPositionNumber": driver + 1 + event,
                    "DNF": int(driver == 2 and event % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_walk_forward_keeps_races_together_and_future_only() -> None:
    frame = race_frame()
    folds = list(
        expanding_window_splits(
            frame,
            min_train_events=3,
            test_events=1,
            embargo_events=1,
        )
    )
    assert len(folds) == 4
    first = folds[0]
    assert len(first.train_events) == 3
    assert len(first.embargoed_events) == 1
    assert len(first.test_index) == 3
    assert_strictly_future(frame, first)
    assert set(frame.iloc[first.train_index].event_id).isdisjoint(frame.iloc[first.test_index].event_id)


def test_leave_season_forward_never_trains_on_test_season() -> None:
    frame = race_frame(12)
    splits = list(leave_season_forward_splits(frame, min_train_seasons=1))
    assert splits
    for train, test, season in splits:
        assert (frame.iloc[train].grandPrixYear < season).all()
        assert (frame.iloc[test].grandPrixYear == season).all()


def test_temporal_helpers_derive_round_without_reordering_rows() -> None:
    frame = race_frame(events=10, drivers=2).drop(columns=["event_id", "round"])
    frame["raceId_results"] = [100 + index // 2 for index in range(len(frame))]
    train, test, embargoed = final_event_holdout_indices(frame, test_fraction=0.2, embargo_events=1)
    assert len(embargoed) == 1
    assert set(frame.iloc[train].raceId_results).isdisjoint(frame.iloc[test].raceId_results)
    assert frame.iloc[train].raceId_results.max() < frame.iloc[test].raceId_results.min()
    folds = sklearn_expanding_window_cv(frame, n_splits=3, embargo_events=1)
    assert len(folds) == 3
    assert max(folds[-1][1]) == len(frame) - 1


def test_model_selection_cv_never_observes_final_season() -> None:
    frame = race_frame(events=12, drivers=2)
    development, final, embargoed, season = final_season_holdout_indices(frame)
    assert season == 2026
    assert len(embargoed) == 1
    assert (frame.iloc[development].grandPrixYear < season).all()
    assert (frame.iloc[final].grandPrixYear == season).all()
    folds, final_again, final_season = sklearn_model_selection_cv(frame, n_splits=3)
    assert final_season == season
    assert set(final_again) == set(final)
    assert all(set(train).isdisjoint(final) and set(test).isdisjoint(final) for train, test in folds)


def test_rolling_features_shift_current_outcome() -> None:
    frame = race_frame(events=4, drivers=2)
    built = add_pre_race_form_features(frame)
    driver = built[built.resultsDriverId == "d0"].sort_values(["grandPrixYear", "round"])
    assert pd.isna(driver.iloc[0]["driver_finish_mean_3r"])
    assert driver.iloc[1]["driver_finish_mean_3r"] == pytest.approx(driver.iloc[0].resultsFinalPositionNumber)
    # Changing the current target must not change the same row's historical feature.
    changed = frame.copy()
    changed.loc[changed.event_id == frame.event_id.unique()[2], "resultsFinalPositionNumber"] = 99
    rebuilt = add_pre_race_form_features(changed)
    row_key = (frame.event_id.unique()[2], "d0")
    before = built[(built.event_id == row_key[0]) & (built.resultsDriverId == row_key[1])].iloc[0]
    after = rebuilt[(rebuilt.event_id == row_key[0]) & (rebuilt.resultsDriverId == row_key[1])].iloc[0]
    assert before.driver_finish_mean_3r == pytest.approx(after.driver_finish_mean_3r)


def test_feature_registry_blocks_post_race_leakage() -> None:
    registry = default_registry()
    registry.assert_available(["driver_finish_mean_3r", "qualifying_rank"], SessionStage.PRE_RACE)
    with pytest.raises(ValueError, match="unavailable"):
        registry.assert_available(["first_lap_position"], SessionStage.PRE_RACE)
    with pytest.raises(ValueError, match="unavailable"):
        registry.assert_available(["closing_market_probability"], SessionStage.PRE_RACE)
    assert len(registry.manifest()) >= 70
    assert all(row["description"] for row in registry.manifest())


def test_head_to_head_deltas_are_ordered_and_antisymmetric() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["e", "e"],
            "resultsDriverId": ["a", "b"],
            "grid": [2, 5],
            "practice": [1.2, 1.8],
        }
    )
    pairs = build_head_to_head_deltas(frame, feature_columns=["grid", "practice"])
    assert len(pairs) == 2
    assert pairs.loc[pairs.selection_id.eq("a"), "grid_delta"].iloc[0] == -3
    assert pairs.loc[pairs.selection_id.eq("b"), "grid_delta"].iloc[0] == 3


def test_separate_dnf_hazards_are_shifted_by_complete_event() -> None:
    frame = race_frame(events=3, drivers=1)
    frame["dnf_cause"] = ["Engine", "Collision", "Finished"]
    built = add_prior_dnf_hazard_features(frame)
    ordered = built.sort_values(["grandPrixYear", "round"])
    assert pd.isna(ordered.iloc[0].mechanical_dnf_mean_5r)
    assert ordered.iloc[1].mechanical_dnf_mean_5r == 1
    assert ordered.iloc[2].collision_dnf_mean_5r == 0.5
