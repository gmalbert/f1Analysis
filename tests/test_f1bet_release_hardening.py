from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json

import numpy as np
import pandas as pd
import pytest

from f1bet.artifacts import ModelManifest
from f1bet.backtest import run_backtest, run_risk_sensitivity
from f1bet.contracts import (
    DECISION_LEDGER_CONTRACT,
    FEATURE_SNAPSHOT_CONTRACT,
    ODDS_LEDGER_CONTRACT,
    RACE_MODEL_CONTRACT,
    SCHEMA_VERSION,
)
from f1bet.domain import MarketQuote, MarketType, SessionStage
from f1bet.evaluation import AblationCoverage, SliceEvaluation, probability_slice_report
from f1bet.features import add_pre_race_form_features
from f1bet.identity import IdentityResolver
from f1bet.markets import best_available_quotes, fair_probabilities_for_quotes, market_consensus
from f1bet.monitoring import evaluate_abstention
from f1bet.pipeline import build_v2_event_snapshot
from f1bet.release import (
    GateStatus,
    data_integrity_gate,
    drift_abstention_gate,
    market_replay_gate,
    probability_quality_gate,
    reproducibility_gate,
    software_release_gate,
    temporal_validation_gate,
)
from f1bet.risk import RiskPolicy
from f1bet.simulation import RaceEntry, SimulationConfig, simulate_race
from f1bet.sources import persist_raw_snapshot
from f1bet.validation import expanding_window_splits


NOW = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)


def test_dataframe_contract_rejects_naive_and_post_start_quotes() -> None:
    row = {
        "quote_id": "q",
        "event_id": "2026-test-R",
        "market": "win",
        "selection_id": "a",
        "opponent_id": None,
        "bookmaker": "book",
        "captured_at": "2026-08-10T12:00:00",
        "event_start_at": "2026-08-10T11:00:00Z",
        "decimal_odds": 2.0,
        "line": None,
    }
    report = ODDS_LEDGER_CONTRACT.validate(pd.DataFrame([row]))
    codes = {issue.code for issue in report.issues}
    assert {"invalid_type", "quote_after_event_start"} <= codes


def test_race_contract_enforces_information_stage() -> None:
    row = {
        "event_id": "2026-test-R",
        "grandPrixYear": 2026,
        "round": 1,
        "resultsDriverId": "a",
        "constructorName": "team",
        "feature_as_of": NOW.isoformat(),
        "feature_stage": "PRE_RACE",
        "schema_version": SCHEMA_VERSION,
        "resultsStartingGridPositionNumber": 1,
        "resultsFinalPositionNumber": 1,
    }
    report = RACE_MODEL_CONTRACT.validate(pd.DataFrame([row]))
    assert any(issue.code == "feature_unavailable_at_stage" for issue in report.issues)


def test_quote_identity_and_devig_keep_snapshots_separate() -> None:
    first = MarketQuote("e", MarketType.HEAD_TO_HEAD, "a", "book", NOW, 1.9, NOW + timedelta(days=1), opponent_id="b", line=0)
    changed_line = MarketQuote("e", MarketType.HEAD_TO_HEAD, "a", "book", NOW, 1.9, NOW + timedelta(days=1), opponent_id="b", line=1)
    assert first.quote_id != changed_line.quote_id
    quotes = [
        first,
        MarketQuote("e", MarketType.HEAD_TO_HEAD, "b", "book", NOW, 1.9, NOW + timedelta(days=1), opponent_id="a", line=0),
        MarketQuote("e", MarketType.HEAD_TO_HEAD, "a", "book", NOW + timedelta(minutes=1), 1.8, NOW + timedelta(days=1), opponent_id="b", line=0),
    ]
    fair = fair_probabilities_for_quotes(quotes)
    assert len(fair) == 2
    with pytest.raises(ValueError, match="timezone-aware"):
        best_available_quotes(quotes, available_at=datetime(2026, 8, 10, 12))


def test_consensus_uses_each_books_latest_complete_snapshot() -> None:
    quotes = []
    for book, minute, odds_a, odds_b in (
        ("one", 0, 1.8, 2.1),
        ("one", 5, 1.6, 2.4),
        ("two", 2, 2.0, 1.9),
    ):
        captured = NOW + timedelta(minutes=minute)
        quotes.extend(
            [
                MarketQuote("e", MarketType.HEAD_TO_HEAD, "a", book, captured, odds_a, NOW + timedelta(days=1), opponent_id="b"),
                MarketQuote("e", MarketType.HEAD_TO_HEAD, "b", book, captured, odds_b, NOW + timedelta(days=1), opponent_id="a"),
            ]
        )
    consensus = market_consensus(quotes, available_at=NOW + timedelta(minutes=10))
    assert set(consensus) == {
        ("e", MarketType.HEAD_TO_HEAD, "a"),
        ("e", MarketType.HEAD_TO_HEAD, "b"),
    }
    assert sum(consensus.values()) == pytest.approx(1.0)


def test_event_rolling_does_not_leak_between_teammates_or_session_rows() -> None:
    rows = []
    for round_number in (1, 2):
        for driver, finish in (("a", 1), ("b", 10)):
            for session in ("fp1", "fp2"):
                rows.append(
                    {
                        "event_id": f"2026-R{round_number:02d}-R",
                        "grandPrixYear": 2026,
                        "round": round_number,
                        "resultsDriverId": driver,
                        "constructorName": "team",
                        "resultsFinalPositionNumber": finish + round_number,
                        "DNF": 0,
                        "session": session,
                    }
                )
    built = add_pre_race_form_features(pd.DataFrame(rows))
    first_event = built[built["round"] == 1]
    assert first_event["driver_finish_mean_3r"].isna().all()
    assert first_event["constructor_finish_mean_3r"].isna().all()
    second = built[built["round"] == 2]
    assert second["constructor_finish_mean_3r"].nunique() == 1
    assert second["prior_career_starts"].eq(1).all()


def _same_event_replay() -> pd.DataFrame:
    common = {
        "event_id": "2025-R01-R",
        "market": "head_to_head",
        "forecast_at": "2025-03-01T10:05:00Z",
        "quote_at": "2025-03-01T10:00:00Z",
        "event_start_at": "2025-03-02T12:00:00Z",
        "probability": 0.70,
        "uncertainty": 0.0,
        "fair_market_probability": 0.48,
        "decimal_odds": 2.10,
        "closing_odds": 1.95,
    }
    return pd.DataFrame(
        [
            {**common, "selection_id": "a", "outcome": 1},
            {**common, "selection_id": "b", "outcome": 0},
        ]
    )


def test_replay_sizes_whole_event_before_settlement_and_retains_decisions() -> None:
    result = run_backtest(
        _same_event_replay(),
        policy=RiskPolicy(min_stake=0.01, minimum_edge=0.0, minimum_ev=0.0),
    )
    assert result.ledger["stake"].tolist() == [100.0, 100.0]
    assert result.ledger["bankroll_before"].nunique() == 1
    assert len(result.decisions) == 2
    assert set(result.decisions.status) == {"placed"}
    assert DECISION_LEDGER_CONTRACT.validate(result.decisions).valid
    assert set(run_risk_sensitivity(_same_event_replay()).scenario) == {
        "flat_1pct",
        "kelly_0.10",
        "kelly_0.25",
        "kelly_0.50",
    }


def test_replay_rejects_ambiguous_outcomes() -> None:
    records = _same_event_replay()
    records["outcome"] = records["outcome"].astype(object)
    records.loc[0, "outcome"] = "maybe"
    with pytest.raises(ValueError, match="outcome must be binary"):
        run_backtest(records)


def test_strict_data_and_reproducibility_gates_can_pass_complete_evidence() -> None:
    feature_as_of = "2026-08-10T09:00:00Z"
    generated_at = "2026-08-10T10:05:00Z"
    event_start = "2026-08-10T12:00:00Z"
    race = pd.DataFrame(
        [
            {
                "event_id": "2026-test-R",
                "grandPrixYear": 2026,
                "round": 1,
                "resultsDriverId": "a",
                "constructorName": "team",
                "feature_as_of": feature_as_of,
                "feature_stage": "PRE_RACE",
                "schema_version": SCHEMA_VERSION,
                "resultsStartingGridPositionNumber": np.nan,
                "resultsFinalPositionNumber": np.nan,
                "snapshot_id": "snapshot-a",
                "driver_id": "a",
                "constructor_id": "team",
                "source_manifest_id": "source-1",
                "snapshot_version": SCHEMA_VERSION,
            }
        ]
    )
    odds = pd.DataFrame(
        [
            {
                "quote_id": "quote-a",
                "event_id": "2026-test-R",
                "market": "win",
                "selection_id": "a",
                "opponent_id": None,
                "bookmaker": "book",
                "captured_at": "2026-08-10T10:00:00Z",
                "event_start_at": event_start,
                "decimal_odds": 2.1,
                "line": np.nan,
            }
        ]
    )
    forecasts = pd.DataFrame(
        [
            {
                "forecast_id": "forecast-a",
                "event_id": "2026-test-R",
                "market": "win",
                "selection_id": "a",
                "opponent_id": None,
                "probability": 0.65,
                "uncertainty": 0.05,
                "generated_at": generated_at,
                "stage": "PRE_RACE",
                "model_version": "test-v1",
                "feature_snapshot_id": "snapshot-a",
            }
        ]
    )
    decisions = pd.DataFrame(
        [
            {
                "bet_id": "bet-a",
                "quote_id": "quote-a",
                "forecast_id": "forecast-a",
                "event_id": "2026-test-R",
                "market": "win",
                "selection_id": "a",
                "opponent_id": None,
                "decided_at": generated_at,
                "model_probability": 0.65,
                "fair_market_probability": 0.48,
                "decimal_odds": 2.1,
                "edge": 0.17,
                "expected_value": 0.365,
                "stake": 100.0,
                "bankroll_before": 10_000.0,
                "reason_code": "paper_bet",
                "status": "placed",
            }
        ]
    )
    integrity = data_integrity_gate(
        race_snapshot=race,
        odds_ledger=odds,
        forecast_ledger=forecasts,
        decision_ledger=decisions,
        identity_coverage=1.0,
        source_coverage={"jolpica": 1.0},
        source_freshness_hours={"jolpica": 1.0},
    )
    assert integrity.status is GateStatus.PASS, integrity.failures
    empty_source_evidence = data_integrity_gate(
        race_snapshot=race,
        identity_coverage=1.0,
        source_coverage={},
        source_freshness_hours={},
    )
    assert empty_source_evidence.status is GateStatus.FAIL
    assert {"source coverage report is empty", "source freshness report is empty"} <= set(
        empty_source_evidence.failures
    )

    manifest = ModelManifest(
        model_name="test",
        model_version="1",
        schema_version=SCHEMA_VERSION,
        trained_at=NOW.isoformat(),
        training_start_event="2024-R01-R",
        training_end_event="2025-R24-R",
        feature_names=("grid",),
        target="win",
        estimator="test-estimator",
        hyperparameters={"depth": 2},
        data_sha256="abc123",
        code_revision="0123456789abcdef",
        metrics={"brier": 0.1},
        calibration_method="isotonic",
        dependency_versions={"numpy": np.__version__},
        calibration_start_event="2025-R01-R",
        calibration_end_event="2025-R24-R",
    )
    reproducibility = reproducibility_gate(
        manifest=manifest,
        repeated_probabilities=([0.2, 0.8], [0.2, 0.8]),
        offline_tests_passed=True,
    )
    assert reproducibility.status is GateStatus.PASS, reproducibility.failures


def test_temporal_and_probability_gates_accept_complete_future_evidence() -> None:
    temporal_frame = pd.DataFrame(
        [
            {"event_id": f"2025-R{round_number:02d}-R", "grandPrixYear": 2025, "round": round_number}
            for round_number in range(1, 9)
        ]
    )
    folds = list(
        expanding_window_splits(
            temporal_frame,
            min_train_events=3,
            test_events=1,
            step_events=1,
            embargo_events=1,
        )
    )
    coverage = AblationCoverage(True, (), (), 75)
    complete_slices = SliceEvaluation(pd.DataFrame(), ())
    temporal = temporal_validation_gate(
        temporal_frame,
        folds,
        final_season_untouched=True,
        pipeline_fit_within_fold=True,
        ablation_coverage=coverage,
        required_slices=complete_slices,
    )
    assert temporal.status is GateStatus.PASS, temporal.failures

    rows = []
    for index in range(40):
        outcome = index % 2
        rows.append(
            {
                "event_id": f"{2024 + index // 20}-R{index % 10 + 1:02d}-R",
                "market": "win",
                "stage": "PRE_RACE",
                "outcome": outcome,
                "probability": 0.9 if outcome else 0.1,
                "champion_probability": 0.7 if outcome else 0.3,
                "fair_market_probability": 0.6 if outcome else 0.4,
                "historical_baseline_probability": 0.55 if outcome else 0.45,
                "season": 2024 + index // 20,
                "circuit_archetype": "street" if index % 3 else "high_speed",
                "is_wet": "true" if index % 5 == 0 else "false",
                "grid_position": index % 20 + 1,
                "is_rookie": index % 7 == 0,
                "constructor_id": f"team-{index % 4}",
                "data_coverage": 0.98 if index % 6 else 0.75,
            }
        )
    probability_records = pd.DataFrame(rows)
    slices = probability_slice_report(probability_records, n_bins=4, n_bootstrap=100)
    wet_values = set(slices.report.loc[slices.report["dimension"].eq("wet_dry"), "value"])
    assert wet_values == {"dry", "wet"}
    quality = probability_quality_gate(
        probability_records,
        champion_probability_col="champion_probability",
        n_bins=4,
        slice_evaluation=slices,
        coherence_issues=(),
        calibration_fitted_on_earlier_data=True,
    )
    assert quality.status is GateStatus.PASS, quality.failures


def test_market_and_software_release_gates_accept_complete_evidence() -> None:
    rows = []
    for event_number in range(20):
        season = 2024 if event_number < 10 else 2025
        for selection_number in range(15):
            rows.append(
                {
                    "event_id": f"{season}-R{event_number % 10 + 1:02d}-R",
                    "selection_id": f"driver-{selection_number}",
                    "market": "win",
                    "forecast_at": f"{season}-03-01T10:05:00Z",
                    "quote_at": f"{season}-03-01T10:00:00Z",
                    "event_start_at": f"{season}-03-02T12:00:00Z",
                    "probability": 0.75,
                    "uncertainty": 0.0,
                    "fair_market_probability": 0.48,
                    "decimal_odds": 2.10,
                    "opening_odds": 2.20,
                    "closing_odds": 1.80,
                    "outcome": 1,
                    "bookmaker": "book-a" if selection_number % 2 else "book-b",
                    "devig_method": "multiplicative",
                    "market_snapshot_complete": True,
                    "rule_version": "book-rules-v1",
                }
            )
    replay = run_backtest(
        pd.DataFrame(rows),
        policy=RiskPolicy(
            staking_mode="flat",
            flat_stake_fraction=0.01,
            max_bet_fraction=0.01,
            max_event_fraction=1.0,
            max_selection_fraction=1.0,
            minimum_edge=0.0,
            minimum_ev=0.0,
            min_stake=0.01,
        ),
    )
    market = market_replay_gate(replay)
    assert market.status is GateStatus.PASS, market.failures
    software = software_release_gate(
        offline_tests_passed=True,
        compile_passed=True,
        browser_smoke_passed=True,
        workflow_permissions_reviewed=True,
        action_versions_pinned=True,
        artifacts_have_manifests=True,
        predictions_loadable=True,
    )
    assert software.status is GateStatus.PASS, software.failures


def test_raw_snapshots_are_content_addressed_immutable_and_secret_safe(tmp_path) -> None:
    path = persist_raw_snapshot(
        {"ok": True},
        tmp_path,
        source="provider",
        request_metadata={"endpoint": "odds", "apiKey": "never-write-me"},
        captured_at=NOW,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.stem == payload["content_sha256"]
    assert payload["request"]["apiKey"] == "[REDACTED]"
    assert "never-write-me" not in path.read_text(encoding="utf-8")
    assert persist_raw_snapshot(
        {"ok": True},
        tmp_path,
        source="provider",
        request_metadata={"endpoint": "odds", "apiKey": "different"},
        captured_at=NOW + timedelta(minutes=1),
    ) == path


def test_temporal_identity_resolution_and_event_snapshot_masking() -> None:
    resolver = IdentityResolver()
    resolver.add("old-team", "Racing", provider="book", valid_to=date(2025, 12, 31))
    resolver.add("new-team", "Racing", provider="book", valid_from=date(2026, 1, 1))
    assert resolver.resolve("Racing", provider="book", at=date(2025, 6, 1)) == "old-team"
    assert resolver.resolve("Racing", provider="book", at=date(2026, 6, 1)) == "new-team"

    legacy = pd.DataFrame(
        [
            {
                "event_id": "2025-R01-R",
                "grandPrixYear": 2025,
                "round": 1,
                "resultsDriverId": "a",
                "constructorName": "team",
                "resultsStartingGridPositionNumber": 2,
                "resultsFinalPositionNumber": 1,
                "DNF": 0,
                "unregistered_post_race_speed": 999,
            },
            {
                "event_id": "2026-R01-R",
                "grandPrixYear": 2026,
                "round": 1,
                "resultsDriverId": "a",
                "constructorName": "team",
                "resultsStartingGridPositionNumber": 3,
                "resultsFinalPositionNumber": 2,
                "DNF": 0,
                "unregistered_post_race_speed": 999,
            },
        ]
    )
    built = build_v2_event_snapshot(
        legacy,
        event_id="2026-R01-R",
        as_of=NOW,
        stage=SessionStage.PRE_RACE,
        source_manifest_id="source-v1",
    )
    assert built.contract_valid
    assert built.frame["resultsFinalPositionNumber"].isna().all()
    assert built.frame["snapshot_id"].notna().all()
    assert built.frame["snapshot_version"].eq(SCHEMA_VERSION).all()
    assert FEATURE_SNAPSHOT_CONTRACT.validate(built.frame).valid
    assert "unregistered_post_race_speed" not in built.frame
    assert built.warnings


def test_correlated_simulation_always_classifies_finishers_before_dnfs() -> None:
    result = simulate_race(
        [
            RaceEntry("finisher", "a", 1_000.0, 0.0, 0.0),
            RaceEntry("dnf", "b", -1_000.0, 1.0, 0.0),
        ],
        SimulationConfig(n_simulations=100, random_seed=1),
    )
    assert (result.positions[:, 0] == 1).all()
    assert (result.positions[:, 1] == 2).all()


def test_release_gates_fail_closed_when_evidence_is_incomplete(tmp_path) -> None:
    decision = evaluate_abstention(weather_age_hours=None, odds_snapshot_complete=False)
    assert drift_abstention_gate(decision).status is GateStatus.FAIL
    assert data_integrity_gate().status is GateStatus.NOT_EVALUATED

    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,0\n", encoding="utf-8")
    manifest = ModelManifest.create(
        model_name="test",
        model_version="v1",
        schema_version=SCHEMA_VERSION,
        training_start_event="2025-R01-R",
        training_end_event="2025-R02-R",
        feature_names=["x"],
        target="y",
        estimator="test",
        hyperparameters={},
        data_path=data,
        code_revision="abc",
        metrics={"brier": 0.2},
    )
    assert manifest.dependency_versions
