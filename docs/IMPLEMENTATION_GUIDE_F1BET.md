# F1 betting research implementation guide

## What is available

`f1bet` is an import-safe companion package for the existing Streamlit application. It does not train or place bets on import. It supports:

- point-in-time schemas and features;
- odds and de-vigging;
- probability calibration;
- race-grouped temporal validation;
- coherent field simulation;
- strategy comparisons;
- conservative paper sizing;
- odds-required replay;
- model/data manifests;
- source adapters and migrations;
- a CLI and Streamlit research page.

## Quick checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -Wall -m compileall -q .
.\.venv\Scripts\python.exe -m f1bet --help
```

The unit suite is offline. Manual FastF1 and odds connectivity scripts are intentionally excluded.

## Start the app

```powershell
.\.venv\Scripts\streamlit.exe run raceAnalysis.py
```

Open **Betting Research** for:

- a probability/price/stake calculator;
- coherent field simulation from an uploaded CSV;
- timestamp-aware paper replay with decision audit and stake sensitivity;
- calibration/reliability diagnostics;
- the feature registry and evidence-backed release-gate audit.

The page shows no ROI when no historical odds ledger exists.

## Validate a dataset

Legacy race TSV:

```powershell
.\.venv\Scripts\python.exe -m f1bet validate data_files\f1ForAnalysis.csv --contract race --separator "`t" --stage PRE_RACE
```

Normalized odds CSV:

```powershell
.\.venv\Scripts\python.exe -m f1bet validate data_files\ledgers\odds.csv --contract odds --separator ,
```

Exit code is `0` for a valid contract and `2` when validation errors exist.

The same command accepts `event`, `snapshot`, `decision`, and `settlement` contracts.

## Simulate a field

Input CSV:

```csv
driver_id,constructor_id,pace_score,dnf_probability,uncertainty
driver-a,team-1,1.0,0.04,0.7
driver-b,team-1,1.3,0.05,0.8
driver-c,team-2,2.1,0.10,1.0
```

Run:

```powershell
.\.venv\Scripts\python.exe -m f1bet simulate field.csv --simulations 20000 --seed 42 --output market_probabilities.csv
```

Lower `pace_score` is faster. The scale is arbitrary but must be comparable within the field. Constructor and race shocks are shared, DNF is simulated separately, and every draw has a unique ordering.

## Migrate a wide prediction export

Supported legacy columns are:

- `win_probability`
- `podium_probability`
- `top_6_probability`
- `top_10_probability`
- `dnf_probability`

Example:

```powershell
.\.venv\Scripts\python.exe -m f1bet migrate-forecasts predictions.csv data_files\ledgers\forecasts.csv `
  --event-id 2026-belgium-R `
  --model-version position-sim-v1 `
  --generated-at 2026-07-25T15:00:00Z `
  --selection-column driver_id `
  --stage PRE_RACE
```

The output is validated before it is written.

## Build a point-in-time event snapshot

CLI example:

```powershell
.\.venv\Scripts\python.exe -m f1bet build-snapshot data_files\f1ForAnalysis.csv data_files\snapshots\2026-R01-R.csv `
  --event-id 2026-R01-R `
  --as-of 2026-03-08T13:00:00Z `
  --source-manifest-id raw-source-manifest-sha `
  --stage PRE_RACE `
  --collapse-duplicates
```

The build fails on conflicting driver-event facts, masks fields unavailable at the selected stage, stamps source/snapshot lineage, and validates both race and feature-snapshot contracts before writing.

Python example:

```python
from datetime import datetime, timezone
import pandas as pd

from f1bet.domain import SessionStage
from f1bet.pipeline import build_v2_event_snapshot

legacy = pd.read_csv("data_files/f1ForAnalysis.csv", sep="\t", low_memory=False)
result = build_v2_event_snapshot(
    legacy,
    event_id="2026-R01-R",
    as_of=datetime.now(timezone.utc),
    stage=SessionStage.PRE_RACE,
    source_manifest_id="raw-source-manifest-sha",
    collapse_duplicates=True,
)

if not result.contract_valid:
    print(result.errors)
```

The existing table can contain multiple session-grain rows per driver/race. Resolve that grain before treating contract duplication as safe to ignore.

## Migrate legacy odds

```powershell
.\.venv\Scripts\python.exe -m f1bet migrate-odds legacy_prices.csv data_files\ledgers\odds.csv `
  --event-id 2026-belgium-R `
  --event-start-at 2026-07-26T13:00:00Z `
  --captured-at 2026-07-26T11:45:00Z `
  --bookmaker example-book `
  --selection-column driver `
  --odds-column win_odds=win
```

Timestamps must be timezone-aware and prices captured after event start are rejected.

## Register a feature

```python
from f1bet.domain import SessionStage
from f1bet.features import FeatureDefinition, default_registry

registry = default_registry()
registry.register(
    FeatureDefinition(
        name="fp2_fuel_corrected_long_run_delta",
        sources=("fastf1_laps", "traffic_estimates"),
        available_at=SessionStage.POST_FP2,
        lookback_events=None,
        description="Median comparable-stint pace relative to field after fuel/traffic adjustment",
        leakage_risk="medium",
    )
)
registry.assert_available(
    ["driver_finish_mean_5r", "fp2_fuel_corrected_long_run_delta"],
    SessionStage.POST_FP2,
)
```

Do not register a feature as pre-race merely because its final CSV is available after the race. Availability means when the underlying fact existed.

## Create and append ledgers

```python
from f1bet.contracts import FORECAST_LEDGER_CONTRACT
from f1bet.ledger import LedgerStore

store = LedgerStore(
    "data_files/ledgers/forecasts.csv",
    contract=FORECAST_LEDGER_CONTRACT,
    id_column="forecast_id",
)
store.append(forecasts)
store.write_metadata(source="position-simulation", owner="research")
```

Appending is idempotent by ID, validates the complete ledger, and uses an atomic sibling-file replacement.

For production volume, preserve the same contract while replacing CSV with SQLite/DuckDB/Parquet.

## Capture sources safely

Copy `.env.example` to `.env` and set values locally. Never paste credentials into code or metadata.

```python
from pathlib import Path

from f1bet.sources import OddsApiClient, persist_raw_snapshot

client = OddsApiClient()
sports = client.sports()
persist_raw_snapshot(
    sports,
    Path("data_files/raw/odds/sports.json"),
    source="the-odds-api-v4",
    request_metadata={"endpoint": "sports"},
)
```

The source adapter applies timeouts, bounded retry/backoff, and 429 handling. It does not log the credential.

## De-vig a complete market

```python
from f1bet.odds import devig_decimal_odds, overround

prices = [1.91, 1.91]
print(overround(prices))
print(devig_decimal_odds(prices, method="multiplicative"))
print(devig_decimal_odds(prices, method="power"))
```

Do not de-vig an incomplete outright board and label the result fair. Store whether the market snapshot is complete.

## Walk-forward training skeleton

```python
from f1bet.validation import assert_strictly_future, expanding_window_splits

for fold in expanding_window_splits(
    snapshot,
    min_train_events=40,
    test_events=1,
    embargo_events=1,
):
    assert_strictly_future(snapshot, fold)
    train = snapshot.iloc[fold.train_index]
    test = snapshot.iloc[fold.test_index]

    # Fit preprocessing and model on train only.
    # Fit calibration on earlier out-of-fold predictions only.
    # Write frozen test forecasts before moving to the next race.
```

Never fit target encoding, imputation, scaling, feature selection, or calibration on the combined frame before the split.

## Probability calibration

```python
from f1bet.calibration import IsotonicProbabilityCalibrator, probability_metrics

metrics = probability_metrics(oof_probability, oof_outcome, n_bins=10)
calibrator = IsotonicProbabilityCalibrator(min_samples=100).fit(
    calibration_probability,
    calibration_outcome,
)
test_probability = calibrator.predict(raw_test_probability)
```

Use a simpler sigmoid/temperature calibrator when samples are too sparse for isotonic. Retain reliability tables; ECE alone is sensitive to binning.

## Backtest file contract

Required CSV columns:

```text
event_id
selection_id
market
forecast_at
quote_at
event_start_at
probability
uncertainty
fair_market_probability
decimal_odds
outcome
```

Optional `closing_odds` enables CLV.

Run:

```powershell
.\.venv\Scripts\python.exe -m f1bet backtest replay.csv --bankroll 10000 --commission 0.0 `
  --output data_files\ledgers\settled-paper-bets.csv `
  --audit-output data_files\ledgers\all-decisions.csv `
  --sensitivity-output data_files\ledgers\risk-sensitivity.csv
```

The default policy is quarter Kelly, maximum 1% per bet, 3% per event, 1.5% per selection, and a 20% drawdown pause. These are conservative research defaults, not financial advice.

## Compare tyre strategies

```python
from f1bet.strategy import Stint, StrategyPlan, TyreCurve, compare_strategies

curves = {
    "S": TyreCurve("S", 0.0, 0.08, 0.004, warmup_penalty=0.1),
    "M": TyreCurve("M", 0.2, 0.04, 0.002),
    "H": TyreCurve("H", 0.5, 0.02, 0.001),
}
plans = [
    StrategyPlan("one-stop", (Stint("M", 1, 25), Stint("H", 26, 50))),
    StrategyPlan("two-stop", (Stint("S", 1, 16), Stint("M", 17, 34), Stint("S", 35, 50))),
]
comparison = compare_strategies(
    plans,
    curves,
    race_laps=50,
    base_lap_time=80.0,
    fuel_gain_per_lap=0.05,
    pit_loss=20.0,
    n_simulations=10_000,
    random_seed=42,
)
```

All plans see identical noise and safety-car draws. The result is a scenario comparison, not causal proof that a historical team made the wrong decision.

## Model manifest and promotion

```python
from f1bet.artifacts import ModelManifest, champion_challenger_gate

decision = champion_challenger_gate(
    champion={"brier": 0.20, "log_loss": 0.60, "ece": 0.05, "mean_clv": 0.002},
    challenger={"brier": 0.18, "log_loss": 0.55, "ece": 0.04, "mean_clv": 0.010},
)
assert decision.promote
```

The gate intentionally requires positive out-of-sample CLV by default. Configure stricter minimum improvements after a stable paper sample exists.

Every new training workflow writes `manifest.json` beside its model. Grandfathered pickles remain loadable for continuity but are visibly marked legacy and cannot be treated as promotion evidence until rebuilt.

## Audit release evidence

```powershell
.\.venv\Scripts\python.exe -m f1bet audit-release `
  --race data_files\snapshots\2026-R01-R.csv `
  --odds data_files\ledgers\odds.csv `
  --forecasts data_files\ledgers\forecasts.csv `
  --decisions data_files\ledgers\decisions.csv `
  --replay replay.csv `
  --probability-records probability_records.csv `
  --champion-probability-column champion_probability `
  --temporal-history data_files\f1db-races-race-results-with-upgrades.csv `
  --ablations ablation_results.csv `
  --manifest models\manifest.json `
  --manifest-data data_files\training_snapshot.csv `
  --repeat-probabilities repeated_probabilities.csv `
  --identity-coverage 1.0 `
  --source-evidence source_evidence.json `
  --drift-evidence drift_evidence.json `
  --final-season-untouched `
  --pipeline-fit-within-fold `
  --calibration-earlier-only `
  --coherence-passed `
  --tests-passed --compile-passed --browser-passed `
  --workflows-reviewed --actions-pinned --manifests-present --predictions-loadable `
  --output data_files\release_evidence.json
```

`source_evidence.json` maps each source to `coverage` and `freshness_hours`; `drift_evidence.json` supplies the documented abstention inputs. Ablation rows must cover all required variants on identical fold IDs, and probability records must contain every required context slice. Unavailable evidence is reported as `not_evaluated` or a failed strict check; it is never silently converted to a pass. The command exits with status 2 until all eight gates pass.

## Operational checklist

- Revoke any credential that ever appeared in source history.
- Store timestamps in UTC and display local time only at the UI edge.
- Keep raw snapshots immutable.
- Record source version and request parameters without credentials.
- Treat unknown identities as quarantine, not fuzzy success.
- Freeze future test blocks.
- Retain abstentions and reason codes.
- Do not promote on ROI alone.
- Do not place real bets from this package.
