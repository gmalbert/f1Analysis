"""Timestamp-aware, odds-required paper-betting replay."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

from .risk import PortfolioState, RiskPolicy, propose_stake, record_exposure
from .domain import stable_id
from .odds import expected_value as unit_expected_value


REQUIRED_COLUMNS = {
    "event_id",
    "selection_id",
    "market",
    "forecast_at",
    "quote_at",
    "event_start_at",
    "probability",
    "uncertainty",
    "fair_market_probability",
    "decimal_odds",
    "outcome",
}

REPLAY_EVIDENCE_COLUMNS = (
    "quote_id",
    "forecast_id",
    "opponent_id",
    "bookmaker",
    "opening_odds",
    "closing_odds",
    "devig_method",
    "market_snapshot_complete",
    "rule_version",
    "feature_snapshot_id",
    "source_manifest_id",
    "stage",
)


@dataclass(frozen=True, slots=True)
class BacktestSummary:
    starting_bankroll: float
    ending_bankroll: float
    bets: int
    wins: int
    losses: int
    voids: int
    staked: float
    profit: float
    roi: float
    hit_rate: float
    max_drawdown: float
    mean_clv: float | None
    skipped_no_edge: int
    rejected_lookahead: int
    candidates: int = 0
    abstentions: int = 0
    longest_losing_streak: int = 0
    largest_event_exposure_share: float = 0.0
    largest_selection_exposure_share: float = 0.0
    mean_clv_ci_low: float | None = None
    mean_clv_ci_high: float | None = None


@dataclass(slots=True)
class BacktestResult:
    summary: BacktestSummary
    ledger: pd.DataFrame
    decisions: pd.DataFrame


def _drawdown(bankroll: list[float]) -> float:
    values = np.asarray(bankroll, dtype=float)
    if values.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(values)
    return float(np.max(np.where(peaks > 0, 1.0 - values / peaks, 0.0)))


def _timezone_aware_mask(series: pd.Series) -> pd.Series:
    def aware(value: object) -> bool:
        if pd.isna(value):
            return False
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return False
        return timestamp.tzinfo is not None and timestamp.utcoffset() is not None

    return series.map(aware)


def _coerce_binary_outcome(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and float(value) in {0.0, 1.0}:
        return bool(int(value))
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "won", "win"}:
        return True
    if normalized in {"0", "false", "lost", "loss"}:
        return False
    if normalized in {"void", "push", "cancelled", "canceled", "nan", "none", ""}:
        return None
    raise ValueError(f"outcome must be binary or void, got {value!r}")


def _longest_losing_streak(statuses: Iterable[object]) -> int:
    longest = current = 0
    for status in statuses:
        if status == "lost":
            current += 1
            longest = max(longest, current)
        elif status == "won":
            current = 0
    return longest


def _clustered_mean_interval(
    frame: pd.DataFrame,
    *,
    value_col: str,
    cluster_col: str,
    random_seed: int = 42,
    n_bootstrap: int = 2_000,
) -> tuple[float | None, float | None]:
    clean = frame[[cluster_col, value_col]].dropna()
    clusters = clean[cluster_col].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return None, None
    rng = np.random.default_rng(random_seed)
    means = np.empty(n_bootstrap, dtype=float)
    grouped = {cluster: clean.loc[clean[cluster_col] == cluster, value_col].to_numpy(dtype=float) for cluster in clusters}
    for index in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        values = np.concatenate([grouped[cluster] for cluster in sampled])
        means[index] = values.mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _validate_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ("forecast_at", "quote_at", "event_start_at"):
        aware = _timezone_aware_mask(result[column])
        if not aware.all():
            rows = result.index[~aware].tolist()[:10]
            raise ValueError(f"{column} must contain timezone-aware timestamps; invalid rows: {rows}")
        result[column] = pd.to_datetime(result[column], errors="raise", utc=True)
    for column in (
        "probability",
        "uncertainty",
        "fair_market_probability",
        "decimal_odds",
    ):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"{column} must contain finite values")
    if ((result["probability"] < 0) | (result["probability"] > 1)).any():
        raise ValueError("probability must be in [0, 1]")
    if ((result["uncertainty"] < 0) | (result["uncertainty"] > 1)).any():
        raise ValueError("uncertainty must be in [0, 1]")
    if ((result["fair_market_probability"] < 0) | (result["fair_market_probability"] > 1)).any():
        raise ValueError("fair_market_probability must be in [0, 1]")
    if (result["decimal_odds"] <= 1).any():
        raise ValueError("decimal_odds must be greater than 1")
    if "closing_odds" in result:
        result["closing_odds"] = pd.to_numeric(result["closing_odds"], errors="coerce")
        invalid_closing = result["closing_odds"].notna() & (result["closing_odds"] <= 1)
        if invalid_closing.any():
            raise ValueError("closing_odds must be greater than 1 when provided")
    if "opening_odds" in result:
        result["opening_odds"] = pd.to_numeric(result["opening_odds"], errors="coerce")
        invalid_opening = result["opening_odds"].notna() & (result["opening_odds"] <= 1)
        if invalid_opening.any():
            raise ValueError("opening_odds must be greater than 1 when provided")
    result["settled_outcome"] = result["outcome"].map(_coerce_binary_outcome)
    return result


def run_backtest(
    records: pd.DataFrame,
    *,
    starting_bankroll: float = 10_000.0,
    policy: RiskPolicy | None = None,
    commission: float = 0.0,
) -> BacktestResult:
    """Replay frozen pre-event forecasts; real odds are mandatory for every bet."""
    missing = REQUIRED_COLUMNS - set(records.columns)
    if missing:
        raise KeyError(f"backtest records missing columns: {sorted(missing)}")
    if starting_bankroll <= 0 or not 0 <= commission < 1:
        raise ValueError("invalid starting_bankroll or commission")
    config = policy or RiskPolicy()
    frame = _validate_inputs(records)
    frame = frame.sort_values(["event_start_at", "quote_at", "event_id"], kind="stable")
    state = PortfolioState(starting_bankroll)
    ledger: list[dict[str, object]] = []
    decisions: list[dict[str, object]] = []
    bankroll_path = [starting_bankroll]
    skipped, rejected = 0, 0
    for (_, event_id), event_frame in frame.groupby(["event_start_at", "event_id"], sort=False, observed=True):
        state.event_exposure = {}
        state.selection_exposure = {}
        bankroll_before_event = state.bankroll
        pending: list[dict[str, object]] = []
        h2h_opponents: dict[str, str] = {}
        h2h_rows = event_frame.loc[event_frame["market"].astype("string").eq("head_to_head")]
        h2h_selections = h2h_rows["selection_id"].dropna().astype("string").drop_duplicates().tolist()
        if len(h2h_selections) == 2:
            h2h_opponents = {
                h2h_selections[0]: h2h_selections[1],
                h2h_selections[1]: h2h_selections[0],
            }
        for row in event_frame.itertuples(index=False):
            opponent_id = getattr(row, "opponent_id", None)
            if (
                (opponent_id is None or pd.isna(opponent_id) or not str(opponent_id).strip())
                and str(row.market) == "head_to_head"
            ):
                opponent_id = h2h_opponents.get(str(row.selection_id))
            quote_id = getattr(row, "quote_id", None)
            if quote_id is None or pd.isna(quote_id) or not str(quote_id).strip():
                quote_id = stable_id(
                    row.event_id,
                    row.market,
                    row.selection_id,
                    opponent_id,
                    row.quote_at.isoformat(),
                    row.decimal_odds,
                )
            forecast_id = getattr(row, "forecast_id", None)
            if forecast_id is None or pd.isna(forecast_id) or not str(forecast_id).strip():
                forecast_id = stable_id(
                    row.event_id,
                    row.market,
                    row.selection_id,
                    opponent_id,
                    row.forecast_at.isoformat(),
                    row.probability,
                )
            decided_at = row.forecast_at
            base_decision = {
                "bet_id": stable_id(quote_id, forecast_id, decided_at.isoformat()),
                "quote_id": str(quote_id),
                "forecast_id": str(forecast_id),
                "event_id": row.event_id,
                "selection_id": row.selection_id,
                "market": row.market,
                "opponent_id": opponent_id,
                "decided_at": decided_at,
                "forecast_at": row.forecast_at,
                "quote_at": row.quote_at,
                "event_start_at": row.event_start_at,
                "probability": row.probability,
                "model_probability": row.probability,
                "fair_market_probability": row.fair_market_probability,
                "decimal_odds": row.decimal_odds,
                "edge": float(row.probability) - float(row.fair_market_probability),
                "expected_value": unit_expected_value(float(row.probability), float(row.decimal_odds)),
                "bankroll_before": bankroll_before_event,
            }
            for evidence_column in REPLAY_EVIDENCE_COLUMNS:
                if evidence_column in frame.columns and evidence_column not in base_decision:
                    base_decision[evidence_column] = getattr(row, evidence_column)
            if row.forecast_at > row.event_start_at or row.quote_at > row.event_start_at or row.quote_at > row.forecast_at:
                rejected += 1
                decisions.append({**base_decision, "stake": 0.0, "reason_code": "lookahead_rejected", "status": "rejected"})
                continue
            proposal = propose_stake(
                event_id=str(row.event_id),
                selection_id=str(row.selection_id),
                probability=float(row.probability),
                decimal_odds=float(row.decimal_odds),
                uncertainty=float(row.uncertainty),
                market_probability=float(row.fair_market_probability),
                state=state,
                policy=config,
            )
            decisions.append(
                {
                    **base_decision,
                    "adjusted_probability": proposal.adjusted_probability,
                    "edge": proposal.adjusted_probability - float(row.fair_market_probability),
                    "expected_value": proposal.expected_value,
                    "kelly_fraction": proposal.kelly,
                    "stake": proposal.stake,
                    "reason_code": proposal.reason_code,
                    "status": "placed" if proposal.stake > 0 else "abstained",
                }
            )
            if proposal.stake <= 0:
                skipped += 1
                continue
            record_exposure(state, str(row.event_id), str(row.selection_id), proposal.stake)
            outcome = row.settled_outcome
            if outcome is None:
                profit = 0.0
                status = "void"
            elif outcome:
                profit = proposal.stake * (float(row.decimal_odds) - 1.0) * (1.0 - commission)
                status = "won"
            else:
                profit = -proposal.stake
                status = "lost"
            closing_odds = getattr(row, "closing_odds", np.nan)
            clv = (
                float(row.decimal_odds) / float(closing_odds) - 1.0
                if pd.notna(closing_odds) and float(closing_odds) > 1
                else np.nan
            )
            pending.append(
                {
                    **base_decision,
                    "adjusted_probability": proposal.adjusted_probability,
                    "stake": proposal.stake,
                    "profit": profit,
                    "status": status,
                    "bankroll_before": bankroll_before_event,
                    "expected_value": proposal.expected_value,
                    "kelly_fraction": proposal.kelly,
                    "clv": clv,
                }
            )

        event_profit = float(sum(float(record["profit"]) for record in pending))
        state.settle(event_profit)
        bankroll_path.append(state.bankroll)
        for record in pending:
            record["bankroll_after"] = state.bankroll
            ledger.append(record)

    ledger_frame = pd.DataFrame(ledger)
    staked = float(ledger_frame["stake"].sum()) if not ledger_frame.empty else 0.0
    profit = state.bankroll - starting_bankroll
    settled = ledger_frame[ledger_frame["status"].isin(["won", "lost"])] if not ledger_frame.empty else ledger_frame
    wins = int((ledger_frame.get("status", pd.Series(dtype=str)) == "won").sum())
    losses = int((ledger_frame.get("status", pd.Series(dtype=str)) == "lost").sum())
    voids = int((ledger_frame.get("status", pd.Series(dtype=str)) == "void").sum())
    mean_clv = None
    if not ledger_frame.empty and ledger_frame["clv"].notna().any():
        mean_clv = float(ledger_frame["clv"].mean())
    clv_low, clv_high = _clustered_mean_interval(
        ledger_frame, value_col="clv", cluster_col="event_id"
    ) if not ledger_frame.empty else (None, None)
    event_share = selection_share = 0.0
    if staked:
        event_share = float(ledger_frame.groupby("event_id")["stake"].sum().max() / staked)
        selection_share = float(ledger_frame.groupby("selection_id")["stake"].sum().max() / staked)
    summary = BacktestSummary(
        starting_bankroll=starting_bankroll,
        ending_bankroll=state.bankroll,
        bets=len(ledger_frame),
        wins=wins,
        losses=losses,
        voids=voids,
        staked=staked,
        profit=profit,
        roi=profit / staked if staked else 0.0,
        hit_rate=wins / len(settled) if len(settled) else 0.0,
        max_drawdown=_drawdown(bankroll_path),
        mean_clv=mean_clv,
        skipped_no_edge=skipped,
        rejected_lookahead=rejected,
        candidates=len(frame),
        abstentions=skipped + rejected,
        longest_losing_streak=_longest_losing_streak(ledger_frame.get("status", [])),
        largest_event_exposure_share=event_share,
        largest_selection_exposure_share=selection_share,
        mean_clv_ci_low=clv_low,
        mean_clv_ci_high=clv_high,
    )
    return BacktestResult(summary, ledger_frame, pd.DataFrame(decisions))


def run_risk_sensitivity(
    records: pd.DataFrame,
    *,
    starting_bankroll: float = 10_000.0,
    commission: float = 0.0,
    base_policy: RiskPolicy | None = None,
) -> pd.DataFrame:
    """Replay the required flat/0.1/0.25/0.5 Kelly policy sensitivity."""

    baseline = base_policy or RiskPolicy()
    policies = {
        "flat_1pct": replace(baseline, staking_mode="flat", flat_stake_fraction=0.01),
        "kelly_0.10": replace(baseline, staking_mode="fractional_kelly", kelly_fraction=0.10),
        "kelly_0.25": replace(baseline, staking_mode="fractional_kelly", kelly_fraction=0.25),
        "kelly_0.50": replace(baseline, staking_mode="fractional_kelly", kelly_fraction=0.50),
    }
    rows: list[dict[str, object]] = []
    for label, policy in policies.items():
        summary = run_backtest(
            records,
            starting_bankroll=starting_bankroll,
            policy=policy,
            commission=commission,
        ).summary
        rows.append({"scenario": label, **{field: getattr(summary, field) for field in summary.__dataclass_fields__}})
    return pd.DataFrame(rows)
