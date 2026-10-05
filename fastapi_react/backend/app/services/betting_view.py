# Generated offline; no Streamlit dependency.
"""Streamlit presentation layer for offline betting research and governance."""
from __future__ import annotations
import pandas as pd
from f1bet.odds import devig_decimal_odds, expected_value
from f1bet.risk import PortfolioState, RiskPolicy, propose_stake

def render_betting_research(ui, data: pd.DataFrame | None=None) -> None:
    ui.header('Probability & Betting Research')
    ui.subheader('Value & stake')
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
