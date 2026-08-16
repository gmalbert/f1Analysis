"""Command-line entrypoints for validation, migration, simulation, and replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .artifacts import ModelManifest
from .backtest import run_backtest, run_risk_sensitivity
from .contracts import (
    DECISION_LEDGER_CONTRACT,
    EVENT_CONTRACT,
    FEATURE_SNAPSHOT_CONTRACT,
    FORECAST_LEDGER_CONTRACT,
    ODDS_LEDGER_CONTRACT,
    RACE_MODEL_CONTRACT,
    SETTLEMENT_LEDGER_CONTRACT,
    add_event_identity,
    stamp_feature_snapshot,
)
from .domain import SessionStage
from .domain import MarketType
from .migrations import migrate_legacy_odds, migrate_prediction_wide_to_forecasts
from .pipeline import build_v2_event_snapshot
from .evaluation import probability_slice_report, validate_ablation_coverage
from .monitoring import evaluate_abstention
from .validation import expanding_window_splits, with_event_identity
from .simulation import RaceEntry, SimulationConfig, simulate_race
from .release import (
    build_release_report,
    data_integrity_gate,
    drift_abstention_gate,
    market_replay_gate,
    probability_quality_gate,
    reproducibility_gate,
    risk_gate,
    software_release_gate,
    temporal_validation_gate,
)


def _read(path: str, separator: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=separator, low_memory=False)


def command_validate(args: argparse.Namespace) -> int:
    frame = _read(args.input, args.separator)
    if args.contract == "race":
        if "event_id" not in frame:
            frame = add_event_identity(frame)
        if "feature_as_of" not in frame or "feature_stage" not in frame:
            frame = stamp_feature_snapshot(
                frame,
                as_of=datetime.now(timezone.utc),
                stage=SessionStage[args.stage],
            )
        contract = RACE_MODEL_CONTRACT
    else:
        contract = {
            "event": EVENT_CONTRACT,
            "snapshot": FEATURE_SNAPSHOT_CONTRACT,
            "odds": ODDS_LEDGER_CONTRACT,
            "forecast": FORECAST_LEDGER_CONTRACT,
            "decision": DECISION_LEDGER_CONTRACT,
            "settlement": SETTLEMENT_LEDGER_CONTRACT,
        }[args.contract]
    report = contract.validate(frame)
    print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.valid else 2


def command_simulate(args: argparse.Namespace) -> int:
    frame = _read(args.input, args.separator)
    required = {"driver_id", "constructor_id", "pace_score"}
    missing = required - set(frame)
    if missing:
        raise KeyError(f"simulation input missing columns: {sorted(missing)}")
    entries = [
        RaceEntry(
            driver_id=str(row.driver_id),
            constructor_id=str(row.constructor_id),
            pace_score=float(row.pace_score),
            dnf_probability=float(getattr(row, "dnf_probability", 0.05)),
            uncertainty=float(getattr(row, "uncertainty", 1.0)),
            race_sensitivity=float(getattr(row, "race_sensitivity", 1.0)),
        )
        for row in frame.itertuples(index=False)
    ]
    result = simulate_race(
        entries,
        SimulationConfig(n_simulations=args.simulations, random_seed=args.seed),
    )
    output = result.market_table()
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(args.output, index=False)
    else:
        print(output.to_string(index=False))
    return 0


def command_backtest(args: argparse.Namespace) -> int:
    frame = _read(args.input, args.separator)
    result = run_backtest(
        frame,
        starting_bankroll=args.bankroll,
        commission=args.commission,
    )
    print(json.dumps(result.summary.__dict__ if hasattr(result.summary, "__dict__") else {
        field: getattr(result.summary, field) for field in result.summary.__dataclass_fields__
    }, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        result.ledger.to_csv(args.output, index=False)
    if args.audit_output:
        Path(args.audit_output).parent.mkdir(parents=True, exist_ok=True)
        DECISION_LEDGER_CONTRACT.validate(result.decisions).raise_for_errors()
        result.decisions.to_csv(args.audit_output, index=False)
    if args.sensitivity_output:
        Path(args.sensitivity_output).parent.mkdir(parents=True, exist_ok=True)
        run_risk_sensitivity(
            frame,
            starting_bankroll=args.bankroll,
            commission=args.commission,
        ).to_csv(args.sensitivity_output, index=False)
    return 0


def command_build_snapshot(args: argparse.Namespace) -> int:
    frame = _read(args.input, args.separator)
    result = build_v2_event_snapshot(
        frame,
        event_id=args.event_id,
        as_of=datetime.fromisoformat(args.as_of.replace("Z", "+00:00")),
        stage=SessionStage[args.stage],
        source_manifest_id=args.source_manifest_id,
        collapse_duplicates=args.collapse_duplicates,
    )
    if not result.contract_valid:
        raise ValueError("; ".join(result.errors))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result.frame.to_csv(args.output, index=False)
    print(f"wrote {len(result.frame)} point-in-time feature rows to {args.output}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    return 0


def command_migrate_forecasts(args: argparse.Namespace) -> int:
    frame = _read(args.input, args.separator)
    migrated = migrate_prediction_wide_to_forecasts(
        frame,
        event_id=args.event_id,
        model_version=args.model_version,
        generated_at=datetime.fromisoformat(args.generated_at.replace("Z", "+00:00")),
        selection_col=args.selection_column,
        stage=SessionStage[args.stage],
    )
    report = FORECAST_LEDGER_CONTRACT.validate(migrated)
    report.raise_for_errors()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    migrated.to_csv(args.output, index=False)
    print(f"wrote {len(migrated)} normalized forecasts to {args.output}")
    return 0


def command_migrate_odds(args: argparse.Namespace) -> int:
    frame = _read(args.input, args.separator)
    mapping = {
        column: MarketType(market)
        for column, market in (item.split("=", 1) for item in args.odds_column)
    }
    migrated = migrate_legacy_odds(
        frame,
        event_id=args.event_id,
        event_start_at=datetime.fromisoformat(args.event_start_at.replace("Z", "+00:00")),
        captured_at=datetime.fromisoformat(args.captured_at.replace("Z", "+00:00")),
        bookmaker=args.bookmaker,
        selection_col=args.selection_column,
        odds_columns=mapping,
    )
    report = ODDS_LEDGER_CONTRACT.validate(migrated)
    report.raise_for_errors()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    migrated.to_csv(args.output, index=False)
    print(f"wrote {len(migrated)} normalized quotes to {args.output}")
    return 0


def command_release_audit(args: argparse.Namespace) -> int:
    race = _read(args.race, args.race_separator) if args.race else None
    odds = _read(args.odds, ",") if args.odds else None
    forecasts = _read(args.forecasts, ",") if args.forecasts else None
    decisions = _read(args.decisions, ",") if args.decisions else None
    replay_result = run_backtest(_read(args.replay, ",")) if args.replay else None
    probability_records = _read(args.probability_records, ",") if args.probability_records else None

    source_coverage = source_freshness = None
    if args.source_evidence:
        source_payload = json.loads(Path(args.source_evidence).read_text(encoding="utf-8"))
        source_coverage = {
            source: float(evidence["coverage"])
            for source, evidence in source_payload.items()
        }
        source_freshness = {
            source: float(evidence["freshness_hours"])
            for source, evidence in source_payload.items()
        }

    manifest = ModelManifest.load(args.manifest) if args.manifest else None
    repeated = None
    if args.repeat_probabilities:
        repeat_frame = _read(args.repeat_probabilities, ",")
        repeated = (repeat_frame[args.repeat_first_column], repeat_frame[args.repeat_second_column])

    temporal_frame = _read(args.temporal_history, args.temporal_separator) if args.temporal_history else None
    folds = None
    if temporal_frame is not None:
        temporal_frame = with_event_identity(temporal_frame)
        event_count = temporal_frame["event_id"].nunique()
        folds = list(
            expanding_window_splits(
                temporal_frame,
                min_train_events=max(3, int(event_count * 0.5)),
                test_events=max(1, int(event_count * 0.1)),
                step_events=max(1, int(event_count * 0.1)),
                embargo_events=1,
            )
        )
    ablation_coverage = (
        validate_ablation_coverage(_read(args.ablations, ",")) if args.ablations else None
    )
    slice_evaluation = (
        probability_slice_report(probability_records) if probability_records is not None else None
    )

    drift_decision = None
    if args.drift_evidence:
        drift_payload = json.loads(Path(args.drift_evidence).read_text(encoding="utf-8"))
        drift_decision = evaluate_abstention(**drift_payload)

    gates = [
        data_integrity_gate(
            race_snapshot=race,
            odds_ledger=odds,
            forecast_ledger=forecasts,
            decision_ledger=decisions,
            identity_coverage=args.identity_coverage,
            source_coverage=source_coverage,
            source_freshness_hours=source_freshness,
        ),
        reproducibility_gate(
            manifest=manifest,
            data_path=args.manifest_data,
            schema_version=manifest.schema_version if manifest and args.manifest_data else None,
            feature_names=manifest.feature_names if manifest and args.manifest_data else None,
            repeated_probabilities=repeated,
            offline_tests_passed=args.tests_passed,
        ),
        temporal_validation_gate(
            temporal_frame,
            folds,
            final_season_untouched=args.final_season_untouched,
            pipeline_fit_within_fold=args.pipeline_fit_within_fold,
            ablation_coverage=ablation_coverage,
            required_slices=slice_evaluation,
        ),
        probability_quality_gate(
            probability_records,
            champion_probability_col=args.champion_probability_column,
            historical_probability_col=args.historical_probability_column,
            slice_evaluation=slice_evaluation,
            coherence_issues=[] if args.coherence_passed else None,
            calibration_fitted_on_earlier_data=args.calibration_earlier_only,
        ),
        market_replay_gate(
            replay_result,
            acknowledge_incomplete_season_window=args.acknowledge_incomplete_season_window,
            multiple_bookmakers_available=not args.single_bookmaker_only,
        ),
        risk_gate(),
        drift_abstention_gate(drift_decision),
        software_release_gate(
            offline_tests_passed=args.tests_passed,
            compile_passed=args.compile_passed,
            browser_smoke_passed=args.browser_passed,
            workflow_permissions_reviewed=args.workflows_reviewed,
            action_versions_pinned=args.actions_pinned,
            artifacts_have_manifests=args.manifests_present,
            predictions_loadable=args.predictions_loadable,
        ),
    ]
    report = build_release_report(*gates)
    payload = json.dumps(report.as_dict(), indent=2, default=str)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)
    return 0 if report.passed else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m f1bet")
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser("validate", help="validate a versioned dataset contract")
    validate.add_argument("input")
    validate.add_argument(
        "--contract",
        choices=("race", "event", "snapshot", "odds", "forecast", "decision", "settlement"),
        default="race",
    )
    validate.add_argument("--separator", default="\t")
    validate.add_argument("--stage", choices=tuple(stage.name for stage in SessionStage), default="PRE_RACE")
    validate.set_defaults(func=command_validate)

    simulate = subcommands.add_parser("simulate", help="produce coherent race market probabilities")
    simulate.add_argument("input")
    simulate.add_argument("--separator", default=",")
    simulate.add_argument("--simulations", type=int, default=10_000)
    simulate.add_argument("--seed", type=int, default=42)
    simulate.add_argument("--output")
    simulate.set_defaults(func=command_simulate)

    backtest = subcommands.add_parser("backtest", help="replay timestamped forecasts and odds")
    backtest.add_argument("input")
    backtest.add_argument("--separator", default=",")
    backtest.add_argument("--bankroll", type=float, default=10_000.0)
    backtest.add_argument("--commission", type=float, default=0.0)
    backtest.add_argument("--output")
    backtest.add_argument("--audit-output", help="write every placed, skipped, and rejected decision")
    backtest.add_argument("--sensitivity-output", help="write flat/0.1/0.25/0.5 Kelly sensitivity")
    backtest.set_defaults(func=command_backtest)

    snapshot = subcommands.add_parser("build-snapshot", help="build one point-in-time event snapshot")
    snapshot.add_argument("input")
    snapshot.add_argument("output")
    snapshot.add_argument("--event-id", required=True)
    snapshot.add_argument("--as-of", required=True, help="ISO-8601 timestamp with timezone")
    snapshot.add_argument("--source-manifest-id", required=True)
    snapshot.add_argument("--separator", default="\t")
    snapshot.add_argument("--stage", choices=tuple(stage.name for stage in SessionStage), default="PRE_RACE")
    snapshot.add_argument("--collapse-duplicates", action="store_true")
    snapshot.set_defaults(func=command_build_snapshot)

    migrate = subcommands.add_parser("migrate-forecasts", help="normalize a legacy wide prediction export")
    migrate.add_argument("input")
    migrate.add_argument("output")
    migrate.add_argument("--event-id", required=True)
    migrate.add_argument("--model-version", required=True)
    migrate.add_argument("--generated-at", required=True, help="ISO-8601 timestamp with timezone")
    migrate.add_argument("--selection-column", default="driver_id")
    migrate.add_argument("--separator", default=",")
    migrate.add_argument("--stage", choices=tuple(stage.name for stage in SessionStage), default="PRE_RACE")
    migrate.set_defaults(func=command_migrate_forecasts)

    migrate_odds = subcommands.add_parser("migrate-odds", help="normalize a legacy wide odds export")
    migrate_odds.add_argument("input")
    migrate_odds.add_argument("output")
    migrate_odds.add_argument("--event-id", required=True)
    migrate_odds.add_argument("--event-start-at", required=True)
    migrate_odds.add_argument("--captured-at", required=True)
    migrate_odds.add_argument("--bookmaker", required=True)
    migrate_odds.add_argument("--selection-column", default="driver")
    migrate_odds.add_argument(
        "--odds-column",
        action="append",
        required=True,
        metavar="COLUMN=MARKET",
        help="repeat for each decimal-odds source column",
    )
    migrate_odds.add_argument("--separator", default=",")
    migrate_odds.set_defaults(func=command_migrate_odds)

    audit = subcommands.add_parser("audit-release", help="evaluate available release-gate evidence")
    audit.add_argument("--race")
    audit.add_argument("--race-separator", default="\t")
    audit.add_argument("--odds")
    audit.add_argument("--forecasts")
    audit.add_argument("--decisions")
    audit.add_argument("--replay")
    audit.add_argument("--probability-records")
    audit.add_argument("--champion-probability-column")
    audit.add_argument("--historical-probability-column", default="historical_baseline_probability")
    audit.add_argument("--calibration-earlier-only", action="store_true")
    audit.add_argument("--coherence-passed", action="store_true")
    audit.add_argument("--temporal-history")
    audit.add_argument("--temporal-separator", default="\t")
    audit.add_argument("--ablations")
    audit.add_argument("--final-season-untouched", action="store_true")
    audit.add_argument("--pipeline-fit-within-fold", action="store_true")
    audit.add_argument("--manifest")
    audit.add_argument("--manifest-data")
    audit.add_argument("--repeat-probabilities")
    audit.add_argument("--repeat-first-column", default="run_1")
    audit.add_argument("--repeat-second-column", default="run_2")
    audit.add_argument("--identity-coverage", type=float)
    audit.add_argument("--source-evidence", help="JSON mapping sources to coverage/freshness_hours")
    audit.add_argument("--drift-evidence", help="JSON keyword arguments for the abstention gate")
    audit.add_argument("--acknowledge-incomplete-season-window", action="store_true")
    audit.add_argument("--single-bookmaker-only", action="store_true")
    audit.add_argument("--output")
    audit.add_argument("--tests-passed", action="store_true")
    audit.add_argument("--compile-passed", action="store_true")
    audit.add_argument("--browser-passed", action="store_true")
    audit.add_argument("--workflows-reviewed", action="store_true")
    audit.add_argument("--actions-pinned", action="store_true")
    audit.add_argument("--manifests-present", action="store_true")
    audit.add_argument("--predictions-loadable", action="store_true")
    audit.set_defaults(func=command_release_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
