"""Streamlit presentation layer for offline betting research and governance."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .backtest import run_backtest, run_risk_sensitivity
from .calibration import calibration_table, probability_metrics
from .contracts import RACE_MODEL_CONTRACT, add_event_identity, stamp_feature_snapshot
from .domain import SessionStage
from .features import default_registry
from .odds import devig_decimal_odds, expected_value
from .risk import PortfolioState, RiskPolicy, propose_stake
from .simulation import RaceEntry, SimulationConfig, simulate_race


def _simulation_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"driver_id": "driver-a", "constructor_id": "team-1", "pace_score": 1.0, "dnf_probability": 0.05, "uncertainty": 0.8, "race_sensitivity": 0.8},
            {"driver_id": "driver-b", "constructor_id": "team-1", "pace_score": 1.4, "dnf_probability": 0.06, "uncertainty": 0.9, "race_sensitivity": 1.0},
            {"driver_id": "driver-c", "constructor_id": "team-2", "pace_score": 2.2, "dnf_probability": 0.08, "uncertainty": 1.0, "race_sensitivity": 1.2},
        ]
    )


def render_betting_research(data: pd.DataFrame | None = None) -> None:
    import streamlit as st

    st.header("Probability & Betting Research")
    st.warning(
        "Paper-research mode only. A finishing-position MAE is not evidence of a betting edge; "
        "release requires frozen real odds, calibration, closing-line value, and walk-forward replay."
    )
    calculator, simulation, replay, calibration, governance = st.tabs(
        ["Value & stake", "Field simulation", "Paper replay", "Calibration", "Release gates"]
    )

    with calculator:
        left, middle, right = st.columns(3)
        model_probability = left.number_input("Model probability", 0.001, 0.999, 0.25, 0.005)
        decimal_odds = middle.number_input("Selection decimal odds", 1.01, 1000.0, 2.10, 0.05)
        uncertainty = right.number_input("Probability uncertainty", 0.0, 0.5, 0.02, 0.005)
        opposing_odds = st.number_input(
            "Opposing decimal odds (complete two-way market)", 1.01, 1000.0, 1.80, 0.05
        )
        devig_method = st.selectbox("De-vig method", ["multiplicative", "additive", "power"])
        market_probability = devig_decimal_odds(
            [decimal_odds, opposing_odds], method=devig_method
        )[0]
        proposal = propose_stake(
            event_id="calculator",
            selection_id="selection",
            probability=model_probability,
            decimal_odds=decimal_odds,
            uncertainty=uncertainty,
            market_probability=market_probability,
            state=PortfolioState(10_000),
            policy=RiskPolicy(),
        )
        metrics = st.columns(4)
        metrics[0].metric("De-vigged market probability", f"{market_probability:.2%}")
        metrics[1].metric("Raw EV / unit", f"{expected_value(model_probability, decimal_odds):+.2%}")
        metrics[2].metric("Conservative probability", f"{proposal.adjusted_probability:.2%}")
        metrics[3].metric("Paper stake on $10k", f"${proposal.stake:,.2f}")
        st.caption(f"Decision: {proposal.reason_code}. This calculator is paper-research only.")

    with simulation:
        st.write(
            "Upload one row per driver. Pace is an arbitrary lower-is-faster score; drivers sharing a "
            "constructor receive correlated shocks and all simulations produce unique finishing positions."
        )
        template = _simulation_template()
        st.download_button(
            "Download input template",
            template.to_csv(index=False),
            "f1_field_simulation_template.csv",
            "text/csv",
        )
        upload = st.file_uploader("Field CSV", type="csv", key="f1bet_field_upload")
        source = pd.read_csv(upload) if upload is not None else template
        simulations = st.slider("Simulations", 1_000, 50_000, 10_000, 1_000)
        if st.button("Run coherent field simulation", key="run_f1bet_simulation"):
            try:
                entries = [
                    RaceEntry(
                        driver_id=str(row.driver_id),
                        constructor_id=str(row.constructor_id),
                        pace_score=float(row.pace_score),
                        dnf_probability=float(row.dnf_probability),
                        uncertainty=float(row.uncertainty),
                        race_sensitivity=float(getattr(row, "race_sensitivity", 1.0)),
                    )
                    for row in source.itertuples(index=False)
                ]
                output = simulate_race(entries, SimulationConfig(simulations, 42)).market_table()
                st.dataframe(output, hide_index=True, width="stretch")
                st.download_button(
                    "Download probabilities", output.to_csv(index=False), "f1_market_probabilities.csv", "text/csv"
                )
            except Exception as exc:
                st.error(f"Simulation input is invalid: {exc}")

    with replay:
        st.write(
            "Replay requires timestamps, real pre-event prices, de-vigged market probability, and settled "
            "outcomes. Records using a forecast or quote after event start are rejected."
        )
        replay_upload = st.file_uploader("Backtest ledger CSV", type="csv", key="f1bet_backtest_upload")
        if replay_upload is None:
            st.info("No odds ledger is bundled, so profitability is intentionally not estimated.")
        elif st.button("Run paper backtest", key="run_f1bet_backtest"):
            try:
                result = run_backtest(pd.read_csv(replay_upload))
                st.json({field: getattr(result.summary, field) for field in result.summary.__dataclass_fields__})
                st.subheader("Placed paper bets")
                st.dataframe(result.ledger, hide_index=True, width="stretch")
                st.subheader("All decisions and abstentions")
                st.dataframe(result.decisions, hide_index=True, width="stretch")
                st.subheader("Required staking sensitivity")
                st.dataframe(
                    run_risk_sensitivity(pd.read_csv(replay_upload)),
                    hide_index=True,
                    width="stretch",
                )
            except Exception as exc:
                st.error(f"Backtest rejected: {exc}")

    with calibration:
        st.write(
            "Upload frozen probabilities and binary outcomes. Diagnostics include Brier score, log loss, "
            "adaptive reliability bins, ECE, calibration slope/intercept, and ROC AUC."
        )
        calibration_upload = st.file_uploader(
            "Calibration CSV", type="csv", key="f1bet_calibration_upload"
        )
        if calibration_upload is None:
            st.info("Required columns: probability and outcome. Optional columns: market and stage.")
        else:
            try:
                calibration_frame = pd.read_csv(calibration_upload)
                missing = {"probability", "outcome"} - set(calibration_frame)
                if missing:
                    raise KeyError(f"missing columns: {sorted(missing)}")
                group_columns = [
                    column for column in ("market", "stage") if column in calibration_frame
                ]
                groups = (
                    calibration_frame.groupby(group_columns, dropna=False, observed=True)
                    if group_columns
                    else [("all", calibration_frame)]
                )
                metric_rows = []
                for key, group in groups:
                    row = probability_metrics(group.probability, group.outcome)
                    if group_columns:
                        values = key if isinstance(key, tuple) else (key,)
                        row.update(dict(zip(group_columns, values)))
                    metric_rows.append(row)
                st.dataframe(pd.DataFrame(metric_rows), hide_index=True, width="stretch")
                reliability = calibration_table(
                    calibration_frame.probability, calibration_frame.outcome
                )
                st.subheader("Adaptive reliability table")
                st.dataframe(reliability, hide_index=True, width="stretch")
                st.line_chart(
                    reliability.set_index("mean_probability")[["observed_rate"]]
                )
            except Exception as exc:
                st.error(f"Calibration input is invalid: {exc}")

    with governance:
        registry = default_registry()
        st.subheader("Feature availability registry")
        st.dataframe(pd.DataFrame(registry.manifest()), hide_index=True, width="stretch")
        if data is not None and not data.empty:
            try:
                audit_columns = [
                    column
                    for column in (
                        "event_id",
                        "grandPrixYear",
                        "round",
                        "raceId_results",
                        "resultsDriverId",
                        "constructorName",
                        "resultsStartingGridPositionNumber",
                        "resultsFinalPositionNumber",
                    )
                    if column in data
                ]
                sample = data[audit_columns].copy()
                if "event_id" not in sample:
                    sample = add_event_identity(sample)
                sample = stamp_feature_snapshot(
                    sample,
                    as_of=datetime.now(timezone.utc),
                    stage=SessionStage.PRE_RACE,
                )
                # Legacy data may contain one row per practice session. Contract validation exposes it.
                report = RACE_MODEL_CONTRACT.validate(sample)
                st.subheader("Current wide-table contract audit")
                if report.valid:
                    st.success("The current table satisfies the v2 core contract.")
                else:
                    st.error("The current table needs migration before it is a valid point-in-time snapshot.")
                st.code(json.dumps(report.as_dict(), indent=2), language="json")
            except Exception as exc:
                st.error(f"Could not audit current data: {exc}")
        st.subheader("Latest automated release evidence")
        evidence_path = Path("data_files/release_evidence.json")
        if evidence_path.exists():
            try:
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                if evidence.get("passed"):
                    st.success("All recorded software release checks passed.")
                else:
                    st.warning("Recorded release evidence is incomplete or contains failures.")
                st.json(evidence)
            except (OSError, json.JSONDecodeError) as exc:
                st.error(f"Release evidence is unreadable: {exc}")
        else:
            st.info(
                "No automated release evidence has been recorded yet. Run the offline suite, compile gate, "
                "and browser smoke check before promotion."
            )
