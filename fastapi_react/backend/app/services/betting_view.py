# Generated offline; no Streamlit dependency.
"""Streamlit presentation layer for offline betting research and governance."""
from __future__ import annotations
import pandas as pd
from f1bet.backtest import run_backtest, run_risk_sensitivity
from f1bet.calibration import calibration_table, probability_metrics
from f1bet.odds import devig_decimal_odds, expected_value
from f1bet.risk import PortfolioState, RiskPolicy, propose_stake
from f1bet.simulation import RaceEntry, SimulationConfig, simulate_race

def _simulation_template() -> pd.DataFrame:
    return pd.DataFrame([{'driver_id': 'driver-a', 'constructor_id': 'team-1', 'pace_score': 1.0, 'dnf_probability': 0.05, 'uncertainty': 0.8, 'race_sensitivity': 0.8}, {'driver_id': 'driver-b', 'constructor_id': 'team-1', 'pace_score': 1.4, 'dnf_probability': 0.06, 'uncertainty': 0.9, 'race_sensitivity': 1.0}, {'driver_id': 'driver-c', 'constructor_id': 'team-2', 'pace_score': 2.2, 'dnf_probability': 0.08, 'uncertainty': 1.0, 'race_sensitivity': 1.2}])

def render_betting_research(ui, data: pd.DataFrame | None=None) -> None:
    ui.header('Probability & Betting Research')
    calculator, simulation, replay, calibration = ui.tabs(['Value & stake', 'Field simulation', 'Paper replay', 'Calibration'])
    with calculator:
        left, middle, right = ui.columns(3)
        model_probability = left.number_input('Model probability', 0.001, 0.999, 0.25, 0.005)
        decimal_odds = middle.number_input('Selection decimal odds', 1.01, 1000.0, 2.1, 0.05)
        uncertainty = right.number_input('Probability uncertainty', 0.0, 0.5, 0.02, 0.005)
        opposing_odds = ui.number_input('Opposing decimal odds (complete two-way market)', 1.01, 1000.0, 1.8, 0.05)
        devig_method = ui.selectbox('De-vig method', ['multiplicative', 'additive', 'power'])
        market_probability = devig_decimal_odds([decimal_odds, opposing_odds], method=devig_method)[0]
        proposal = propose_stake(event_id='calculator', selection_id='selection', probability=model_probability, decimal_odds=decimal_odds, uncertainty=uncertainty, market_probability=market_probability, state=PortfolioState(10000), policy=RiskPolicy())
        metrics = ui.columns(4)
        metrics[0].metric('De-vigged market probability', f'{market_probability:.2%}')
        metrics[1].metric('Raw EV / unit', f'{expected_value(model_probability, decimal_odds):+.2%}')
        metrics[2].metric('Conservative probability', f'{proposal.adjusted_probability:.2%}')
        metrics[3].metric('Paper stake on $10k', f'${proposal.stake:,.2f}')
        ui.caption(f'Decision: {proposal.reason_code}.')
    with simulation:
        ui.write('Upload one row per driver. Pace is an arbitrary lower-is-faster score; drivers sharing a constructor receive correlated shocks and all simulations produce unique finishing positions.')
        template = _simulation_template()
        ui.download_button('Download input template', template.to_csv(index=False), 'f1_field_simulation_template.csv', 'text/csv')
        upload = ui.file_uploader('Field CSV', type='csv', key='f1bet_field_upload')
        source = pd.read_csv(upload) if upload is not None else template
        simulations = ui.slider('Simulations', 1000, 50000, 10000, 1000)
        if ui.button('Run coherent field simulation', key='run_f1bet_simulation'):
            try:
                entries = [RaceEntry(driver_id=str(row.driver_id), constructor_id=str(row.constructor_id), pace_score=float(row.pace_score), dnf_probability=float(row.dnf_probability), uncertainty=float(row.uncertainty), race_sensitivity=float(getattr(row, 'race_sensitivity', 1.0))) for row in source.itertuples(index=False)]
                output = simulate_race(entries, SimulationConfig(simulations, 42)).market_table()
                ui.dataframe(output, hide_index=True, width='stretch')
                ui.download_button('Download probabilities', output.to_csv(index=False), 'f1_market_probabilities.csv', 'text/csv')
            except Exception as exc:
                ui.error(f'Simulation input is invalid: {exc}')
    with replay:
        ui.write('Replay requires timestamps, real pre-event prices, de-vigged market probability, and settled outcomes. Records using a forecast or quote after event start are rejected.')
        replay_upload = ui.file_uploader('Backtest ledger CSV', type='csv', key='f1bet_backtest_upload')
        if replay_upload is None:
            ui.info('No odds ledger is bundled, so profitability is intentionally not estimated.')
        elif ui.button('Run paper backtest', key='run_f1bet_backtest'):
            try:
                result = run_backtest(pd.read_csv(replay_upload))
                ui.json({field: getattr(result.summary, field) for field in result.summary.__dataclass_fields__})
                ui.subheader('Placed paper bets')
                ui.dataframe(result.ledger, hide_index=True, width='stretch')
                ui.subheader('All decisions and abstentions')
                ui.dataframe(result.decisions, hide_index=True, width='stretch')
                ui.subheader('Required staking sensitivity')
                ui.dataframe(run_risk_sensitivity(pd.read_csv(replay_upload)), hide_index=True, width='stretch')
            except Exception as exc:
                ui.error(f'Backtest rejected: {exc}')
    with calibration:
        ui.write('Upload frozen probabilities and binary outcomes. Diagnostics include Brier score, log loss, adaptive reliability bins, ECE, calibration slope/intercept, and ROC AUC.')
        calibration_upload = ui.file_uploader('Calibration CSV', type='csv', key='f1bet_calibration_upload')
        if calibration_upload is None:
            ui.info('Required columns: probability and outcome. Optional columns: market and stage.')
        else:
            try:
                calibration_frame = pd.read_csv(calibration_upload)
                missing = {'probability', 'outcome'} - set(calibration_frame)
                if missing:
                    raise KeyError(f'missing columns: {sorted(missing)}')
                group_columns = [column for column in ('market', 'stage') if column in calibration_frame]
                groups = calibration_frame.groupby(group_columns, dropna=False, observed=True) if group_columns else [('all', calibration_frame)]
                metric_rows = []
                for key, group in groups:
                    row = probability_metrics(group.probability, group.outcome)
                    if group_columns:
                        values = key if isinstance(key, tuple) else (key,)
                        row.update(dict(zip(group_columns, values)))
                    metric_rows.append(row)
                ui.dataframe(pd.DataFrame(metric_rows), hide_index=True, width='stretch')
                reliability = calibration_table(calibration_frame.probability, calibration_frame.outcome)
                ui.subheader('Adaptive reliability table')
                ui.dataframe(reliability, hide_index=True, width='stretch')
                ui.line_chart(reliability.set_index('mean_probability')[['observed_rate']])
            except Exception as exc:
                ui.error(f'Calibration input is invalid: {exc}')
