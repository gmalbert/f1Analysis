# Research review: optimizing data for predictive F1 betting

## Bottom line

The highest-value improvement is to optimize for calibrated, market-specific probabilities under a strict information clock—not for a lower aggregate finishing-position MAE. The second is to collect historical odds correctly. The third is to generate coherent race outcomes so winner, podium, points, DNF, and H2H forecasts come from the same simulated world.

F1 is a small-sample, regime-changing, multi-competitor problem. The unit of independence is closer to a Grand Prix than a driver row. Weather, safety cars, red flags, constructor pace, and regulation changes create shared shocks. Models that treat 20 driver rows as 20 independent games will be overconfident.

## GitHub repository review

| Repository | Useful pattern observed | Incorporated here | Caution |
|---|---|---|---|
| [FastF1](https://github.com/theOehrly/Fast-F1) | Session/timing/telemetry access, Pandas-native data, and API caching. | Isolated `OpenF1Client`/`JolpicaClient`, raw snapshots, source metadata, bounded request behavior. | FastF1 cache availability is not the same as a frozen reproducible raw snapshot. |
| [jolpica-f1](https://github.com/jolpica/jolpica-f1) | A normalized schema intended to reduce duplication and adapt to rule changes; enums and database dumps. | Stable IDs, controlled market/stage vocabularies, normalized ledgers, explicit schema version. | Upstream schemas evolve; pin source version and keep mappings. |
| [F1DB](https://github.com/f1db/f1db) | Relational SQLite/SQL artifacts and race-aware CalVer releases. | Relational v2 design, event-version thinking, model/data manifests. | Historical identity and rule corrections can revise prior facts. |
| [F1-PREDICT](https://github.com/XVX-016/F1-PREDICT) | Deterministic race components plus stochastic Monte Carlo and ML residuals; separate backend and frontend. | Interpretable tyre/fuel/pit components, correlated uncertainty, common seeded comparisons, thin UI. | A simulator is only as valid as its calibration and traffic/incident assumptions. |
| [sports-betting](https://github.com/georgedouzas/sports-betting) | Clear extraction, fit, backtest, and value-bet boundaries mirrored in a CLI. | `f1bet` CLI, separate source/forecast/backtest layers, real-odds-required replay. | Generic sports cross-validation must still be made race-grouped and temporal for F1. |
| [NBA_Betting](https://github.com/NBA-Betting/NBA_Betting) | Separates game, market, and feature data; distinguishes daily updates from backfills; environment-based odds key. | Separate ledgers, raw source clients, `.env.example`, bounded backoff. | A maintained historical odds archive is the hard part, not the dashboard. |
| [horse-racing-predictions](https://github.com/gmalbert/horse-racing-predictions) | Walk-forward scripts, calibration diagnostics, drawdown controls, and explicit refusal to claim value without real odds. | Same “no odds, no value backtest” rule, calibration and CLV gates, paper risk policy. | Model-implied odds cannot validate value in the absence of market prices. |
| [OpenF1 Streamlit tutorial](https://github.com/bordanattila/OpenF1_tutorial) | Loader → processor → visualizer separation. | New Streamlit tab delegates all calculations to import-safe modules. | Tutorial code is not a production ingestion contract. |

## Technical literature review

### Calibration before staking

[Walsh and Joshi, “Machine learning for sports betting: should model selection be based on accuracy or calibration?”](https://arxiv.org/abs/2303.06021) directly tests accuracy-selected and calibration-selected betting models and finds calibration selection materially more useful in its study. The portable lesson is not its reported ROI; it is that a probability decision system must be evaluated as probabilities.

Implementation consequences:

- use Brier score and log loss per market;
- render reliability tables and calibration slope/intercept;
- fit calibration only on past/out-of-fold predictions;
- never fit an isotonic calibrator on the final test block;
- apply Kelly only to calibrated, uncertainty-haircut probabilities.

[Nixon et al., “Measuring Calibration in Deep Learning”](https://arxiv.org/abs/1904.01685) shows that calibration conclusions depend on binning and definition, and recommends adaptive binning for more stable comparisons. That is why `calibration_table()` uses equal-frequency bins and why ECE is never the sole gate.

### Ranking rather than 20 independent regressions

[Turner et al., “Modelling rankings in R: the PlackettLuce package”](https://arxiv.org/abs/1810.12068) describes a model for complete and partial rankings with regularization. F1 produces a ranking, not 20 unrelated scalar outcomes.

Implementation consequences:

- provide a Plackett–Luce Gumbel sampler;
- generate exactly one winner per simulation;
- derive H2H and finish-cutoff markets from the same ranks;
- regularize driver/team strengths toward parity where observations are sparse;
- consider dynamic or hierarchical worth parameters by constructor, circuit class, and regulation era.

The current Plackett–Luce helper is a coherent probability layer, not yet a fitted production ranking model. A future fitted version should estimate strengths inside every walk-forward fold.

### Tyre degradation as a latent process

[Cappello and Hoegh, “A state-space approach to modeling tire degradation in Formula 1 racing”](https://journals.sagepub.com/doi/full/10.1177/22150218261446170) separates fuel and latent tyre pace, treats pit stops as state resets, quantifies uncertainty, and notes traffic gap as a needed covariate. Its interpretability is especially valuable with limited public telemetry.

Implementation consequences:

- keep base pace, fuel, tyre age/compound, traffic, track evolution, and noise separate;
- treat pit stops as structural resets;
- estimate distributions, not one degradation coefficient;
- use robust observation error because mistakes and traffic make lap residuals asymmetric;
- pool information hierarchically across drivers/constructors/circuits while allowing event adaptation.

The implemented `TyreCurve` is the deterministic first layer. It intentionally has a path to state-space residuals rather than pretending its quadratic curve is final.

### Strategy is a multi-agent decision problem

[Thomas et al., “Race Strategy Reinforcement Learning”](https://link.springer.com/article/10.1007/s10994-026-07081-3) models tyre, safety-car state, race progress, position, gaps, and last-lap pace, and emphasizes explanations for strategy trust. A recent [pit-stop optimization framework](https://optimization-online.org/wp-content/uploads/2026/02/Pit_Stop_Strategy_Optimization_Model.pdf) similarly centralizes circuit, tyre, fuel, ERS, and pit-loss parameters.

Implementation consequences:

- strategy state should include gaps and traffic, not lap time alone;
- safety-car pit loss is state-dependent;
- compare candidate policies under identical random scenarios;
- do not infer causal pit advantage directly from observed stops;
- surface uncertainty and regret, not only a recommended action.

### Backtest overfitting and temporal leakage

[Bailey et al., “The Probability of Backtest Overfitting”](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253) formalizes how repeated strategy selection can overfit historical simulations. This is directly relevant because the repository contains Monte Carlo feature selection, grid search, Bayesian optimization, ensembles, and many roadmap experiments.

Implementation consequences:

- final test seasons are frozen and inspected once;
- hyperparameters are selected only inside earlier time blocks;
- report the number of tried configurations;
- compare against simple market and grid baselines;
- require consistent results across multiple future blocks, not one maximum ROI.

`expanding_window_splits()` adds an event embargo even though F1 labels do not overlap in exactly the same way as financial returns. The embargo is a conservative guard against source revisions, backfilled features, and closely coupled weekend data.

### Uncertainty under changing regimes

[Adaptive Conformal Predictions for Time Series](https://arxiv.org/abs/2202.07282) addresses interval coverage when exchangeability is unrealistic. F1 regulation changes, upgrades, rookies, calendar changes, and sparse wet races make static residual intervals fragile.

Implementation consequences:

- report rolling coverage by season and regime;
- widen or abstain when source coverage/drift is poor;
- maintain adaptive residual quantiles separately by market and forecast stage;
- do not assume a 2019 residual distribution applies unchanged to 2026.

The current simulation emits empirical P10/P50/P90 intervals and drift metrics. Formal adaptive conformal updating is an evidence-backed next model extension, not falsely labeled complete.

### Kelly is downstream of probability quality

[Kelly, “A New Interpretation of Information Rate”](https://onlinelibrary.wiley.com/doi/abs/10.1002/j.1538-7305.1956.tb03809.x) establishes the long-run growth framework. Kelly is extremely sensitive to an overstated edge, so operational use requires stronger assumptions than simply applying the formula.

Implementation consequences:

- use fractional rather than full Kelly;
- subtract uncertainty before sizing;
- shrink a small amount toward the market;
- cap bet, selection, and event exposure;
- pause on drawdown;
- start with paper decisions only.

## Recommended prediction targets

### First production candidate: head-to-head

Why:

- shared race shocks partially cancel;
- settlement is binary and easier to calibrate;
- model differences can focus on teammate or comparable-car matchups;
- books often expose clearer two-way markets for de-vigging.

Required features:

- pairwise grid/qualifying delta;
- practice long-run and sector delta;
- driver and constructor prior-form delta;
- DNF risk difference;
- circuit/traffic/overtaking interaction;
- grid-penalty and confirmed-start status;
- weather sensitivity;
- market consensus as a benchmark or carefully isolated meta-feature.

### Second: top-10/top-6/podium

These should come from a coherent finishing distribution. Train/calibrate each threshold if needed, then reconcile monotonicity so `P(win) ≤ P(podium) ≤ P(top 6) ≤ P(top 10)`.

### Separate hazards: DNF and safety car

Mechanical DNF, collision DNF, and non-classification are not identical. Circuit, weather, start position density, reliability, and race phase matter. Safety car should eventually be a lap-level hazard, not a constant race probability.

### Avoid initially: correlated parlays and season outrights

Naively multiplying marginal probabilities is invalid. These require joint simulations, book-specific rules, and much more historical price data.

## Point-in-time feature blueprint

### Pre-weekend

- shifted driver and constructor form/volatility;
- shifted mechanical and incident DNF rates;
- teammate-relative rolling pace;
- track archetype and overtaking difficulty;
- driver/team history at similar circuits;
- regulation era and round progress;
- announced upgrade/specification package with effective date;
- historical weather regime, not the future observed weather;
- travel/time-zone/turnaround features if sourced reliably;
- tyre allocation and circuit pit-loss distribution.

### After practice

- fuel-corrected long-run pace distribution;
- sector deltas and theoretical-lap gap;
- stint-specific degradation posterior;
- lap-time residual skew/heavy tails;
- track evolution;
- traffic/clean-air split;
- speed-trap/aero archetype match;
- run-plan coverage and lap count;
- weather and track-temperature changes;
- teammate deltas under comparable compound/tyre age.

### Post-qualifying/pre-race

- full qualifying distribution, not best lap only;
- confirmed grid and penalties;
- start-position incident exposure;
- forecast rain/wind/track temperature distribution;
- estimated starting tyre/strategy scenarios;
- timestamped, de-vigged market consensus;
- source freshness and missingness flags.

### Live only

- first-lap position;
- current gaps and traffic train;
- observed compound/tyre age;
- track status and incident state;
- realized degradation residuals;
- pit window and undercut/overcut state.

## Backtest protocol

1. Freeze raw source and odds snapshots as they arrive.
2. Build feature snapshots using only `captured_at <= forecast_at`.
3. Train on earlier complete races.
4. Calibrate on earlier out-of-fold predictions.
5. Forecast the next race once per declared stage.
6. Join only quotes captured no later than the decision timestamp.
7. De-vig a complete mutually exclusive market from the same book/time.
8. Record every candidate, including abstentions and reason codes.
9. Settle using versioned book rules.
10. Report Brier, log loss, calibration, CLV, ROI, turnover, drawdown, and slices.
11. Keep the final season untouched until the model and thresholds are frozen.

## Required ablations

Run each addition against the same walk-forward folds:

- grid/qualifying only;
- model without market features;
- market consensus only;
- model plus market consensus;
- no practice telemetry;
- no weather;
- no DNF model;
- independent driver simulation versus correlated field simulation;
- fixed tyre curve versus adaptive/state-space residual;
- raw versus calibrated probabilities;
- flat stake versus capped fractional Kelly.

An addition is useful only if it improves future-block probability quality or market-relative performance consistently. A feature-importance chart is not sufficient evidence.

## Data collection priority

1. Historical timestamped prices and closing prices.
2. Stable cross-provider driver/constructor/market IDs.
3. Feature availability/capture timestamps.
4. Confirmed grid and penalty effective times.
5. Practice stint fuel/traffic proxies and source coverage.
6. Book-specific settlement rules.
7. Upgrade/specification chronology.
8. Live odds only after pre-race capture and validation are reliable.

Without priorities 1–3, further model complexity cannot establish a betting edge.
