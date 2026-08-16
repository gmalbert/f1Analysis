# Implemented feature and change catalog

This catalog records 100 additions delivered by the v2 review. “Feature” includes user-facing capabilities, data-model changes, modeling controls, operational safeguards, and testability improvements. Every item is implemented in code; future extensions are kept out of this list so status is unambiguous.

| # | Addition | Why it matters | Implementation |
|---:|---|---|---|
| 1 | Import-safe `f1bet` package | Training, tests, CLI, and UI share logic without executing Streamlit. | `f1bet/__init__.py` |
| 2 | Canonical event key | Prevents collisions across seasons, rounds, races, and sessions. | `domain.EventKey`, `contracts.add_event_identity` |
| 3 | Session-stage enum | Makes the information set machine-checkable. | `domain.SessionStage` |
| 4 | Market enum | Standardizes winner, podium, top-6, top-10, H2H, DNF, fastest lap, and safety car. | `domain.MarketType` |
| 5 | UTC-only timestamps | Rejects ambiguous naive timestamps and normalizes to UTC. | `domain._utc` |
| 6 | Stable deterministic IDs | Deduplicates quotes, forecasts, and decisions without exposing raw payloads. | `domain.stable_id` |
| 7 | Quote domain model | Captures bookmaker, price, market, selection, opponent, line, and time. | `domain.MarketQuote` |
| 8 | Forecast domain model | Captures probability, uncertainty, stage, snapshot, and model version. | `domain.Forecast` |
| 9 | Paper-decision model | Records price/forecast lineage, edge, EV, stake, and reason. | `domain.BetDecision` |
| 10 | Schema version 2.0 | Gives migrations and artifacts an explicit compatibility boundary. | `contracts.SCHEMA_VERSION` |
| 11 | Column contracts | Checks required fields, types, nullability, ranges, and vocabularies. | `contracts.ColumnRule`, `DatasetContract` |
| 12 | Structured validation reports | Provides codes, severities, columns, and example failing rows. | `contracts.ValidationReport` |
| 13 | Composite-key checks | Detects semantic duplicates even when full rows differ. | `DatasetContract.unique_by` |
| 14 | Feature snapshot stamping | Adds `feature_as_of`, stage, and schema version. | `contracts.stamp_feature_snapshot` |
| 15 | Race snapshot contract | Establishes the minimum point-in-time driver-race grain. | `contracts.RACE_MODEL_CONTRACT` |
| 16 | Odds ledger contract | Prevents malformed prices and duplicate quotes. | `contracts.ODDS_LEDGER_CONTRACT` |
| 17 | Forecast ledger contract | Prevents invalid probabilities, stages, and model metadata. | `contracts.FORECAST_LEDGER_CONTRACT` |
| 18 | Feature registry | Records source, availability, lookback, description, and leakage risk. | `features.FeatureRegistry` |
| 19 | Availability enforcement | Blocks live/post-race fields in pre-race feature sets. | `FeatureRegistry.assert_available` |
| 20 | Leakage-safe rolling builder | Collapses to event grain, shifts one complete event, then rolls without teammate/session leakage. | `features.leakage_safe_event_rolling` |
| 21 | Driver 3/5/10-race form | Adds compact historical mean and volatility features. | `features.add_pre_race_form_features` |
| 22 | Constructor 3/5-race form | Captures team trajectory without current-race leakage. | `features.add_pre_race_form_features` |
| 23 | Driver 5/10-race DNF form | Separates reliability history from finishing pace. | `features.add_pre_race_form_features` |
| 24 | Prior career starts | Measures experience using only previous events. | `features.add_pre_race_form_features` |
| 25 | Log experience | Reduces the leverage of very long careers. | `experience_log` |
| 26 | Season progress | Exposes development phase without using standings outcomes. | `season_progress` |
| 27 | Regulation era | Makes concept drift across major rulesets explicit. | `regulation_era` |
| 28 | Canonical alias normalization | Handles punctuation, case, whitespace, and accents. | `identity.normalize_label` |
| 29 | Ambiguity detection | Refuses aliases mapping to multiple entities. | `identity.IdentityResolver` |
| 30 | Identity resolution report | Separates matched, unknown, and ambiguous selections. | `IdentityResolver.resolution_report` |
| 31 | Decimal/American/fractional conversion | Supports provider odds formats consistently. | `odds.py` |
| 32 | Overround measurement | Quantifies the bookmaker margin of a complete market. | `odds.overround` |
| 33 | Multiplicative de-vig | Stable baseline fair probabilities. | `odds.devig_probabilities` |
| 34 | Additive de-vig | Alternative equal-margin allocation for sensitivity analysis. | `odds.devig_probabilities` |
| 35 | Power de-vig | Sensitivity model for favorite–longshot bias. | `odds.devig_probabilities` |
| 36 | Unit expected value | Calculates probability × decimal odds − 1. | `odds.expected_value` |
| 37 | Probability edge | Separates model probability from fair market probability. | `odds.probability_edge` |
| 38 | Closing-line value | Measures taken price against the closing price. | `odds.closing_line_value` |
| 39 | Best-price selection | Keeps the highest available price per event/market/selection. | `markets.best_available_quotes` |
| 40 | Market consensus | Produces a robust cross-book implied-probability baseline. | `markets.market_consensus` |
| 41 | Rule-aware settlement | Settles each supported F1 market from the appropriate facts. | `markets.settle_market` |
| 42 | Void-on-unknown policy | Avoids guessing when settlement inputs are missing or tied. | `markets.settle_market` |
| 43 | Probability coherence audit | Checks nested finish markets and winner total. | `markets.probability_coherence_issues` |
| 44 | Brier score | Measures squared probability error. | `calibration.brier_score` |
| 45 | Log loss | Strongly penalizes confident incorrect forecasts. | `calibration.logarithmic_loss` |
| 46 | Adaptive reliability bins | Reduces sparse fixed-bin instability in small samples. | `calibration.calibration_table` |
| 47 | Expected calibration error | Adds a readable summary while retaining the full table. | `calibration.expected_calibration_error` |
| 48 | Calibration slope/intercept | Diagnoses overconfidence and systematic bias. | `calibration.calibration_slope_intercept` |
| 49 | Isotonic calibration | Provides bounded nonparametric post-processing. | `calibration.IsotonicProbabilityCalibrator` |
| 50 | Race-grouped expanding window | Tests only on later events and keeps drivers from a race together. | `validation.expanding_window_splits` |
| 51 | Event embargo | Creates a gap between the latest training event and test event. | `validation.expanding_window_splits` |
| 52 | Strict future assertion | Fails on temporal reversal even if row indexes look separate. | `validation.assert_strictly_future` |
| 53 | Leave-season-forward folds | Measures complete future-season generalization. | `validation.leave_season_forward_splits` |
| 54 | Correlated field Monte Carlo | Adds race-wide and constructor shocks instead of independent drivers. | `simulation.simulate_race` |
| 55 | Unique rank in every draw | Prevents impossible duplicate positions. | `SimulationResult.positions` |
| 56 | Separate DNF process | Models survival risk apart from latent pace. | `RaceEntry.dnf_probability`, simulation DNF draws |
| 57 | Coherent H2H probabilities | Derives matchup probabilities from the same field draws. | `SimulationResult.probability` |
| 58 | Plackett–Luce sampler | Converts positive strength scores into full ranking distributions. | `simulation.plackett_luce_simulation` |
| 59 | Finish uncertainty bands | Reports expected position and P10/P50/P90. | `SimulationResult.market_table` |
| 60 | Interpretable tyre curves | Models warm-up, linear wear, and quadratic degradation. | `strategy.TyreCurve` |
| 61 | Robust tyre-curve fit | Trims gross lap anomalies and fits loss relative to clean pace. | `strategy.fit_tyre_curve` |
| 62 | Explicit strategy plans | Validates full-lap stint coverage and compound changes. | `strategy.StrategyPlan` |
| 63 | Common-random strategy comparison | Makes strategy deltas lower variance and reproducible. | `strategy.compare_strategies` |
| 64 | Safety-car pit-loss scenario | Reduces, but does not erase, pit loss under SC. | `strategy.compare_strategies` |
| 65 | Full Kelly calculation | Establishes the theoretical uncapped fraction. | `risk.full_kelly_fraction` |
| 66 | Fractional Kelly default | Uses one-quarter Kelly instead of full Kelly. | `risk.RiskPolicy` |
| 67 | Uncertainty haircut | Lowers the probability before sizing. | `risk.conservative_probability` |
| 68 | Market shrinkage | Pulls a small portion of the forecast toward consensus. | `risk.conservative_probability` |
| 69 | Per-bet cap | Limits one paper position to 1% by default. | `RiskPolicy.max_bet_fraction` |
| 70 | Per-event cap | Limits correlated exposure within one Grand Prix. | `RiskPolicy.max_event_fraction` |
| 71 | Per-selection cap | Limits repeated exposure to the same driver. | `RiskPolicy.max_selection_fraction` |
| 72 | Edge and EV gates | Refuses nominal edges below configured thresholds. | `risk.propose_stake` |
| 73 | Drawdown pause | Stops new paper bets after a configured peak-to-current decline. | `PortfolioState.drawdown`, `propose_stake` |
| 74 | Timestamp-aware replay | Rejects future quotes and post-start information. | `backtest.run_backtest` |
| 75 | Real-odds requirement | Fails if `decimal_odds` are absent. | `backtest.REQUIRED_COLUMNS` |
| 76 | Commission-aware payouts | Supports exchange/operator commission scenarios. | `backtest.run_backtest` |
| 77 | Dynamic bankroll ledger | Records stake, profit, bankroll, EV, Kelly, status, and CLV. | `BacktestResult.ledger` |
| 78 | ROI/hit-rate/drawdown/CLV summary | Reports risk and market quality, not accuracy alone. | `BacktestSummary` |
| 79 | PSI drift | Detects material numeric distribution change. | `monitoring.population_stability_index` |
| 80 | Missingness profile | Measures coverage and cardinality per field. | `monitoring.missingness_report` |
| 81 | Reference/current drift report | Adds PSI, mean shift, missing shift, and severity. | `monitoring.drift_report` |
| 82 | Source coverage/freshness | Represents observed/expected rows and record age with validated bounds. | `monitoring.SourceCoverage` |
| 83 | Model manifest | Records data hash, code, schema, windows, ordered features, dependencies, search count, seed, and metrics. | `artifacts.ModelManifest` |
| 84 | Champion/challenger gate | Requires probability improvement, calibration control, and positive CLV. | `artifacts.champion_challenger_gate` |
| 85 | Append-only ledger store | Deduplicates, validates, and atomically replaces CSV ledgers. | `ledger.LedgerStore` |
| 86 | Wide forecast migration | Converts win/podium/top-6/top-10/DNF columns to normalized facts. | `migrations.migrate_prediction_wide_to_forecasts` |
| 87 | Legacy odds migration | Converts historical wide prices into timestamped quote facts. | `migrations.migrate_legacy_odds` |
| 88 | Retrying JSON client | Adds timeouts, retries, bounded backoff, and 429 handling. | `sources.JsonClient` |
| 89 | Secret-safe odds client | Reads a named environment variable and does not print the key. | `sources.OddsApiClient` |
| 90 | OpenF1 adapter | Provides an isolated laps endpoint. | `sources.OpenF1Client` |
| 91 | Jolpica adapter | Provides an isolated historical results endpoint. | `sources.JolpicaClient` |
| 92 | Raw snapshot envelope | Stores source, capture time, request metadata, and payload. | `sources.persist_raw_snapshot` |
| 93 | V2 snapshot pipeline helper | Builds event-grain identity, prior form, stage masking, lineage, and contract result. | `pipeline.build_v2_event_snapshot` |
| 94 | Validation CLI | Checks event, race, snapshot, odds, forecast, decision, and settlement contracts outside the app. | `python -m f1bet validate` |
| 95 | Simulation CLI | Produces coherent market probabilities from a driver field CSV. | `python -m f1bet simulate` |
| 96 | Backtest CLI | Replays frozen forecasts/prices and exports ledgers, all decisions, and stake sensitivity. | `python -m f1bet backtest` |
| 97 | Migration CLI | Normalizes legacy wide prediction exports. | `python -m f1bet migrate-forecasts` |
| 98 | Streamlit betting-research tab | Adds value/stake, simulation, replay, calibration, and release-gate tools. | `f1bet/streamlit_page.py`, `raceAnalysis.py` |
| 99 | Unit-test isolation | Stops exploratory network scripts from breaking pytest collection. | `pytest.ini` |
| 100 | Credential removal and environment template | Removes the embedded odds key and documents safe configuration. | `betting_test.py`, `.env.example` |

## Count

The review delivers **100 implemented additions**, including more than 30 data-model, validation, market, and risk changes. The count intentionally excludes prose-only future ideas.
