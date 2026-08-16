# Validation and release gates

## Status definitions

- **Analytics:** exploratory charts and retrospective model diagnostics.
- **Forecast research:** frozen future-event predictions with no wagering claims.
- **Paper betting:** forecasts joined to real timestamped prices and settled virtually.
- **Live execution:** out of scope for this review; requires separate legal, compliance, operator-rule, security, and operational approval.

The repository is currently at **forecast research**, with infrastructure for paper betting once a real odds ledger is populated.

## Gate 1 — Data integrity

All must pass:

- normalized event, driver, constructor, bookmaker, and market IDs;
- 100% timezone-aware capture timestamps;
- no quote or feature later than the decision timestamp;
- explicit feature stage;
- no duplicate quote/forecast IDs;
- source snapshot and schema version retained;
- target absent from pre-race features;
- current-race rolling features proven shifted;
- source coverage and freshness reported;
- unresolved identity joins quarantined.

Failure response: stop the affected event/market pipeline. Do not impute an identity or timestamp.

## Gate 2 — Reproducibility

All must pass:

- deterministic random seed;
- exact feature order in a model manifest;
- dataset SHA-256;
- source-control revision;
- training start/end event;
- hyperparameters and dependency versions;
- calibration method and calibration window;
- a clean offline test run;
- a repeat run produces the same frozen probabilities within numerical tolerance.

## Gate 3 — Temporal validation

Required protocol:

- complete race events remain grouped;
- training events strictly precede test events;
- at least one-event embargo for generated/backfilled weekend sources;
- preprocessing, encoding, imputation, feature selection, and calibration fit inside each fold;
- final full-season block untouched during feature/model selection;
- model searches and ablations counted and reported.

Minimum reports:

- expanding-race walk-forward results;
- leave-season-forward results;
- season, circuit archetype, wet/dry, grid band, rookie, constructor, and data-coverage slices;
- bootstrap confidence intervals grouped by event, not row.

## Gate 4 — Probability quality

Per market and forecast stage, report:

- sample count and base rate;
- Brier score;
- log loss;
- adaptive reliability table;
- ECE with bin count disclosed;
- calibration slope/intercept;
- discrimination metric as secondary evidence;
- comparison with de-vigged market consensus and a simple historical baseline.

Promotion rules:

- challenger improves Brier and log loss on future blocks;
- ECE does not materially regress;
- no single season or circuit drives all improvement;
- finish probabilities are coherent and nested;
- calibration is fitted only on earlier data.

Do not use a universal threshold before market samples exist. Winner and DNF base rates differ too much for one absolute Brier gate.

## Gate 5 — Market replay

Minimum evidence before calling anything a paper edge:

- at least 300 frozen, eligible paper decisions and at least 100 settled bets after filters;
- more than one season or an explicitly acknowledged incomplete evidence window;
- multiple bookmakers where available;
- opening/taken and closing price;
- de-vig method and complete-market status;
- book-specific settlement version;
- positive mean CLV with an event-clustered confidence interval that is not dominated by one race;
- positive ROI after commission as secondary evidence;
- turnover, drawdown, longest losing streak, and exposure concentration;
- all abstentions retained.

CLV is a diagnostic, not proof of future profit. ROI without CLV is especially vulnerable to small-sample outcome luck.

## Gate 6 — Risk

Paper defaults:

- quarter Kelly;
- subtract one uncertainty unit before sizing;
- 10% shrinkage toward market consensus;
- maximum 1% of bankroll per bet;
- maximum 3% per Grand Prix;
- maximum 1.5% per selection;
- minimum 2 percentage-point edge;
- minimum 2% expected value;
- stop new positions at 20% drawdown;
- no parlays;
- no automatic execution.

Before changing any cap, rerun the full event-grouped replay and report sensitivity at flat stakes, 0.1 Kelly, 0.25 Kelly, and 0.5 Kelly.

## Gate 7 — Drift and abstention

Automatic abstention or manual review when:

- material numeric PSI exceeds 0.25;
- critical-source missingness rises by more than 10 percentage points;
- identity coverage is below 100% for offered selections;
- confirmed grid is unavailable for a pre-race model that requires it;
- forecast weather is stale;
- model artifact does not match data hash/schema;
- a regulation-era model has no applicable training history;
- probability coherence checks fail;
- odds snapshot is incomplete for de-vigging.

## Gate 8 — Software release

Required:

- offline unit suite passes;
- core modules compile;
- no live requests during test discovery;
- no embedded secrets;
- least-privilege workflow permissions reviewed;
- third-party action versions reviewed/pinned;
- generated model/data artifacts carry manifests;
- Streamlit starts without training models on request;
- existing predictions remain loadable or migration is documented.

## Current gate assessment

| Gate | Status | Reason |
|---|---|---|
| Data integrity | Implemented in code; evidence pending | Event snapshot migration, stage masking, immutable source snapshots, normalized ledgers, and fail-closed identity/contract checks are implemented. A real forward snapshot corpus is still required. |
| Reproducibility | Implemented for new artifacts | Every training entrypoint writes a complete manifest and loaders reject incompatible manifests. Grandfathered artifacts remain unpromotable until rebuilt. |
| Temporal validation | Implemented in code | Production training, HPO, historical validation, benchmarks, and ablations now use race-grouped chronological holdouts or expanding windows with an event embargo. Historical reports must be regenerated. |
| Probability quality | Not demonstrated | Calibration tools exist, but market-specific future probabilities have not been replayed. |
| Market replay | Not demonstrated | No historical normalized odds ledger is present. |
| Risk | Implemented for paper research | Conservative policy and replay exist; no real-money authority is provided. |
| Drift/abstention | Implemented in code; live evidence pending | PSI, missingness, identity, grid, weather, artifact, era, coherence, and odds-completeness reasons feed a single fail-closed decision. Live source values must still be supplied. |
| Software release | Implemented; legacy artifact exception documented | Offline tests/compilation, secret scanning, least-privilege workflows, immutable action pins, startup smoke checks, and legacy-artifact warnings are in place. Existing pickles need workflow rebuilds for manifests. |

## Required language in reports

Allowed:

- “The model achieved X Brier score on frozen future races.”
- “The paper ledger had positive/negative CLV over N decisions.”
- “This result is uncertain and may not persist.”

Not allowed without additional evidence:

- “The model will beat the market.”
- “MAE of 1.3 means profitable bets.”
- “The strategy is optimal.”
- “Kelly makes the staking safe.”
