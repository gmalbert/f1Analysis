from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from f1bet.artifacts import ModelManifest, champion_challenger_gate
from f1bet.contracts import FORECAST_LEDGER_CONTRACT
from f1bet.identity import IdentityResolver, normalize_label
from f1bet.migrations import migrate_prediction_wide_to_forecasts
from f1bet.monitoring import drift_report, missingness_report, population_stability_index


NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)


def test_model_manifest_round_trip_and_fingerprint(tmp_path) -> None:
    data = tmp_path / "data.csv"
    data.write_text("x,y\n1,2\n", encoding="utf-8")
    manifest = ModelManifest.create(
        model_name="winner",
        model_version="v1",
        schema_version="2",
        training_start_event="2020-R01-R",
        training_end_event="2025-R24-R",
        feature_names=["x"],
        target="y",
        estimator="test",
        hyperparameters={"seed": 42},
        data_path=data,
        code_revision="abc123",
        metrics={"brier": 0.2},
    )
    path = tmp_path / "manifest.json"
    manifest.save(path)
    loaded = ModelManifest.load(path)
    assert loaded == manifest
    assert loaded.verify_data(data)
    data.write_text("changed", encoding="utf-8")
    assert not loaded.verify_data(data)


def test_champion_challenger_gate_requires_probability_and_market_quality() -> None:
    champion = {"brier": 0.20, "log_loss": 0.60, "ece": 0.05, "mean_clv": 0.0}
    good = {"brier": 0.18, "log_loss": 0.55, "ece": 0.04, "mean_clv": 0.01}
    bad = {"brier": 0.18, "log_loss": 0.55, "ece": 0.04, "mean_clv": -0.01}
    assert champion_challenger_gate(champion, good).promote
    assert not champion_challenger_gate(champion, bad).promote


def test_wide_predictions_migrate_to_normalized_forecasts() -> None:
    wide = pd.DataFrame(
        {
            "driver_id": ["a", "b"],
            "win_probability": [0.6, 0.4],
            "podium_probability": [0.8, 0.7],
        }
    )
    migrated = migrate_prediction_wide_to_forecasts(
        wide,
        event_id="2026-R01-R",
        model_version="v1",
        generated_at=NOW,
    )
    assert len(migrated) == 4
    assert FORECAST_LEDGER_CONTRACT.validate(migrated).valid


def test_identity_resolver_detects_ambiguity() -> None:
    assert normalize_label("Sérgio Pérez") == "sergio-perez"
    resolver = IdentityResolver()
    resolver.add("perez", "Sergio Perez")
    assert resolver.resolve("Sérgio Pérez") == "perez"
    resolver.add("other", "Sergio Perez")
    assert resolver.resolve("Sergio Perez", strict=False) is None


def test_drift_and_missingness_reports() -> None:
    reference = pd.DataFrame({"x": range(100), "category": ["a"] * 100})
    current = pd.DataFrame({"x": range(100, 200), "category": ["a"] * 90 + [None] * 10})
    assert population_stability_index(reference.x, current.x) > 0.25
    drift = drift_report(reference, current)
    assert drift.loc[drift.column == "x", "drift_level"].iloc[0] == "action"
    missing = missingness_report(current)
    assert missing.loc[missing.column == "category", "missing_rate"].iloc[0] == 0.1
