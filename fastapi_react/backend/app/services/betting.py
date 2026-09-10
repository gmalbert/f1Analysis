from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..config import DATA_DIR
from .data import load_main_data, records


def value_and_stake(payload: Any) -> dict[str, Any]:
    from f1bet.odds import devig_decimal_odds, expected_value
    from f1bet.risk import PortfolioState, RiskPolicy, propose_stake
    market_probability = devig_decimal_odds(
        [payload.decimal_odds, payload.opposing_odds], method=payload.devig_method
    )[0]
    proposal = propose_stake(
        event_id="calculator", selection_id="selection",
        probability=payload.model_probability, decimal_odds=payload.decimal_odds,
        uncertainty=payload.uncertainty, market_probability=market_probability,
        state=PortfolioState(payload.bankroll), policy=RiskPolicy(),
    )
    return {
        "market_probability": market_probability,
        "raw_ev": expected_value(payload.model_probability, payload.decimal_odds),
        "adjusted_probability": proposal.adjusted_probability,
        "stake": proposal.stake,
        "reason_code": proposal.reason_code,
    }


def simulate(payload: Any) -> dict[str, Any]:
    from f1bet.simulation import RaceEntry, SimulationConfig, simulate_race
    entries = [
        RaceEntry(
            driver_id=e.driver_id, constructor_id=e.constructor_id,
            pace_score=e.pace_score, dnf_probability=e.dnf_probability,
            uncertainty=e.uncertainty, race_sensitivity=e.race_sensitivity,
        ) for e in payload.entries
    ]
    output = simulate_race(entries, SimulationConfig(payload.simulations, payload.seed)).market_table()
    return {"columns": list(output.columns), "rows": records(output)}


def backtest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from f1bet.backtest import run_backtest, run_risk_sensitivity
    frame = pd.DataFrame(rows)
    result = run_backtest(frame)
    summary = {field: getattr(result.summary, field) for field in result.summary.__dataclass_fields__}
    return {
        "summary": summary,
        "ledger": records(result.ledger),
        "decisions": records(result.decisions),
        "sensitivity": records(run_risk_sensitivity(frame)),
    }


def calibration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from f1bet.calibration import calibration_table, probability_metrics
    frame = pd.DataFrame(rows)
    missing = {"probability", "outcome"} - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    group_columns = [c for c in ("market", "stage") if c in frame]
    groups = frame.groupby(group_columns, dropna=False, observed=True) if group_columns else [("all", frame)]
    metrics = []
    for key, group in groups:
        row = probability_metrics(group.probability, group.outcome)
        if group_columns:
            values = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(group_columns, values, strict=True)))
        metrics.append(row)
    reliability = calibration_table(frame.probability, frame.outcome)
    return {"metrics": metrics, "reliability": records(reliability)}


def governance() -> dict[str, Any]:
    from f1bet.contracts import RACE_MODEL_CONTRACT, add_event_identity, stamp_feature_snapshot
    from f1bet.domain import SessionStage
    from f1bet.features import default_registry
    registry = default_registry()
    try:
        data = load_main_data()
        audit_columns = [
            c for c in (
                "event_id", "grandPrixYear", "round", "raceId_results", "resultsDriverId",
                "constructorName", "resultsStartingGridPositionNumber", "resultsFinalPositionNumber",
            ) if c in data
        ]
        sample = data[audit_columns].copy()
        if "event_id" not in sample:
            sample = add_event_identity(sample)
        sample = stamp_feature_snapshot(sample, as_of=datetime.now(UTC), stage=SessionStage.PRE_RACE)
        report = RACE_MODEL_CONTRACT.validate(sample).as_dict()
    except Exception as exc:
        report = {"valid": False, "error": str(exc)}

    evidence = None
    evidence_path = DATA_DIR / "release_evidence.json"
    if evidence_path.exists():
        try:
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except Exception as exc:
            evidence = {"read_error": str(exc)}
    return {"registry": registry.manifest(), "contract_audit": report, "release_evidence": evidence}
