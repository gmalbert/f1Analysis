# Comprehensive repository review — August 2026

## Executive verdict

This repository has unusually broad Formula 1 coverage and a functioning precomputation architecture, but it was built as an analytics and finishing-position application rather than a defensible betting research system. The distinction matters. A low finishing-position MAE does not establish calibrated winner, podium, points, DNF, or head-to-head probabilities; without timestamped bookmaker prices there is no way to establish value, closing-line performance, or profit.

The most important change in this review is therefore not another regressor. It is a point-in-time research layer that records exactly what was known, when it was known, what probability was issued, what real price was available, why a paper bet was or was not selected, and how it settled. That layer is implemented in `f1bet/` and exposed in a seventh Streamlit tab.

No part of this review claims that the project has a profitable edge. The correct present status is **forecasting system with a paper-betting research harness**.

## Review scope

The review covered:

- all tracked Python entrypoints, model helpers, feature lists, scripts, tests, data artifacts, documentation, and GitHub Actions workflows;
- the current `data_files/f1ForAnalysis.csv` table and its target coverage;
- runtime model loading, preprocessing, training, validation, cache behavior, and Streamlit execution;
- secret handling, test isolation, network side effects, artifact fingerprints, and CI permissions;
- comparable F1 data, simulation, and sports-betting repositories;
- research on calibration, ranking models, tyre degradation, strategy simulation, backtest overfitting, uncertainty, and bankroll sizing.

## Measured baseline

| Area | Observed state before this change | Consequence |
|---|---:|---|
| Main analysis table | 4,563 rows × 616 columns | Feature count is very high relative to independent race events. |
| Numeric columns | 584 | Strong regularization and ablation are more valuable than indiscriminate feature growth. |
| Object columns | 21 | High-cardinality identity fields require stable IDs and train-only encoding. |
| Target missingness | 13.41% | Future/unsettled rows must be separated from labeled training rows. |
| Columns over 50% missing | 13 | Optional sources need explicit coverage gates. |
| Duplicate complete rows | 0 | The wide output is not duplicated byte-for-byte, but its semantic grain remains implicit. |
| Data range | 2016–2026 | Regulation-era drift is material; the effective independent sample is races, not driver rows. |
| `raceAnalysis.py` | 362,013 bytes | UI, modeling, IO, diagnostics, and domain logic are too tightly coupled. |
| `f1-generate-analysis.py` | 228,356 bytes | A failed merge or late feature can affect the entire wide-table build. |
| Broad exception handlers in three core files | 68 | Failures can become silent missing data or misleading fallbacks. |
| GitHub workflows | 16 | Automation is extensive, but only three workflows declare explicit permissions. |
| Baseline pytest collection | 5 collection errors | Live network calls and stale exploratory scripts prevented a clean offline suite. |
| New offline suite | 62 tests + 3 subtests passing | Core v2 behavior is deterministic and network-free. |

## What the repository already does well

1. It combines FastF1 timing/telemetry, historical result data, weather, qualifying, practice, pit stops, race control, first-lap information, tyre strategy, and circuit context.
2. Training artifacts are precomputed rather than trained on a Streamlit request path.
3. Model data fingerprints already reject artifacts built from different file contents.
4. Runtime model resources and large data loads use bounded Streamlit caches.
5. Multiple model families and track-aware ensembles are available.
6. A temporal leakage audit exists and many rolling features already use `shift(1)`.
7. The full pipeline has explicit stages, critical/non-critical status, and smoke scripts.
8. Existing documentation is extensive and candid about model limitations.

These strengths were preserved. The new package imports independently and does not force a rewrite of the existing pipeline.

## Critical findings

### P0 — A position prediction is not a betting probability

The production target is primarily `resultsFinalPositionNumber`. MAE can be excellent while winner or podium probabilities are poorly calibrated. A point prediction also does not guarantee a valid field ordering: several drivers can be predicted at the same position and independently estimated top-10 probabilities can be mutually inconsistent.

Implemented response:

- coherent Plackett–Luce and correlated field simulations;
- unique finishing positions in every draw;
- nested winner/podium/top-6/top-10 probabilities;
- DNF hazards separated from pace;
- Brier score, log loss, adaptive-bin ECE, calibration slope/intercept, and reliability tables.

### P0 — Current validation is not a clean betting replay

The original repository contained random `train_test_split` use and season-grouped `GroupKFold`. Those paths have now been migrated to race-grouped chronological holdouts or expanding-window folds with a one-event embargo, including UI diagnostics, HPO, benchmark, and ablation scripts.

Implemented response:

- expanding-window folds grouped by complete race event;
- optional event embargo;
- leave-season-forward evaluation;
- hard assertions that the latest train event precedes the earliest test event;
- timestamp guards that reject forecasts, quotes, or decisions unavailable at decision time.

### P0 — There was no historical market ledger

No normalized record connects a model probability to a bookmaker, captured price, capture timestamp, closing price, event start, settlement rule, and outcome. Model-implied odds are not a substitute for offered odds.

Implemented response:

- normalized quote, forecast, paper-decision, and settlement-ready models;
- decimal/American/fractional conversion;
- multiplicative, additive, and power de-vigging;
- price-based closing-line value;
- a backtester that fails if real `decimal_odds` are absent;
- no bundled profitability estimate when an odds ledger is missing.

### P0 — Network side effects and a committed credential broke test collection

`betting_test.py` executed paid external requests at import time and contained an embedded API credential. Several other root `test_*.py` files are exploratory live-network programs rather than unit tests. Baseline collection also failed on stale qualifying columns.

Implemented response:

- the credential was removed;
- the manual connectivity check now reads `THE_ODDS_API_KEY` only from the environment;
- network work runs only under `main()`;
- `.env.example` documents credential names without values;
- `pytest.ini` limits unit collection to `tests/`;
- source adapters never include a secret in raised error messages.

The previously exposed key should be revoked at its provider even though it has been removed from the working tree.

### P1 — The data grain and availability time were implicit

The wide CSV mixes race-level, driver-race, practice-session, qualifying, live-race, and post-race fields. Duplicate-looking columns such as `.1` suffixes reveal historical merge collisions. A column name alone cannot establish whether a value was available pre-weekend, after FP2, after qualifying, live, or only after the race.

Implemented response:

- stable `event_id` and entity IDs;
- `feature_as_of`, `feature_stage`, and `schema_version`;
- a feature registry with source, stage, lookback, description, and leakage risk;
- a contract that makes post-race target fields explicit;
- normalized ledgers rather than adding more market fields to the 616-column CSV.

### P1 — Feature-to-sample ratio is too high

The table has 584 numeric columns but only a few hundred independent races. Driver rows within a race share weather, safety-car, circuit, market, and field shocks. Feature-selection output based on row-wise metrics can overstate independence. Many columns are alternative encodings or near-duplicates.

Recommended operating rule:

- maintain a compact production feature set per forecast stage;
- register every production feature;
- require race-grouped walk-forward ablation before promotion;
- treat feature selection as multiple testing;
- report performance by season, circuit class, wet/dry, rookie/veteran, grid band, and source coverage;
- do not accept claimed MAE reductions written in roadmap documents without frozen out-of-sample artifacts.

### P1 — Identity resolution needs to be a first-class dimension

Driver names, abbreviations, provider IDs, accents, and constructor chronology can differ across FastF1, Jolpica, F1DB, odds books, and local mappings. A failed name join is especially dangerous in betting because it can attach a price to the wrong selection.

Implemented response:

- normalized alias keys;
- explicit canonical IDs;
- ambiguity detection that fails closed;
- matched/unknown/ambiguous resolution reports.

### P1 — Risk and settlement rules were absent

Winner, podium, top-10, DNF, fastest-lap, safety-car, and head-to-head bets do not settle from the same fields. Dead heats, DNS, disqualification, classification, and book-specific rules require source rule metadata.

Implemented response:

- market enum and rule-aware basic settlement;
- void rather than guess when required data are absent;
- fractional Kelly only after probability uncertainty haircuts;
- per-bet, per-event, and per-selection caps;
- a drawdown pause;
- commission support;
- paper-only default.

Book-specific dead-heat and classification rules remain data that must be captured from each operator before any live use.

### P1 — Strategy features need causal humility

Observed pit stops are chosen in response to traffic, gaps, weather, safety cars, damage, and opponent actions. A model trained on observed stops can learn selection bias and should not claim that another strategy would have been better without a simulator or off-policy assumptions.

Implemented response:

- interpretable fuel, tyre-age, warm-up, quadratic degradation, pit-loss, and safety-car components;
- common random numbers for fair side-by-side strategy comparisons;
- uncertainty distributions rather than a single “optimal” lap;
- a path to replace fixed tyre curves with a state-space posterior.

### P2 — Monolithic execution and broad fallbacks hide defects

The app still contains large top-level blocks, duplicate imports, runtime subprocess calls, deprecated compatibility branches, and broad exceptions. This makes unit testing difficult and can convert a true fault into an empty chart.

Implemented first step:

- all new domain, validation, simulation, risk, and market logic lives in import-safe modules;
- the Streamlit page is a thin presentation layer;
- the CLI exercises the same code without Streamlit.

Recommended next refactor: move one existing tab at a time into `pages/` or presentation modules, then extract training functions behind a model service. Avoid a single high-risk rewrite.

### P2 — CI supply-chain and permission hardening is incomplete

Most workflows do not declare top-level least-privilege permissions. Actions use major-version tags rather than immutable commit SHAs. Several workflows can write generated artifacts back to the repository, increasing conflict and provenance risk.

Recommended changes:

- default `permissions: contents: read`;
- grant `contents: write` only to the exact job that publishes artifacts;
- pin third-party actions to full commit SHAs;
- attach model/data manifests to build artifacts instead of committing every generated binary;
- add dependency review, secret scanning, and a minimal CodeQL job;
- serialize workflows that write the same output via `concurrency` groups.

These are documented recommendations, not silently applied policy changes, because workflow permission changes can alter existing automation authority.

## Target architecture

```text
raw immutable snapshots
        ↓
source-normalized tables + identity dimensions
        ↓
point-in-time feature snapshots (stage + as_of + provenance)
        ↓
walk-forward training + calibration + model manifest
        ↓
coherent field simulation / market probabilities
        ↓
timestamped quote join + de-vig + paper decision
        ↓
settlement + CLV + calibration + risk + drift monitoring
        ↓
Streamlit presentation (read-only over precomputed artifacts)
```

## What was implemented in this review

The complete numbered catalog is in [FEATURE_CATALOG_V2.md](FEATURE_CATALOG_V2.md). Major deliverables are:

- `f1bet/domain.py`: canonical event, quote, forecast, and decision models;
- `f1bet/contracts.py`: versioned race/odds/forecast contracts;
- `f1bet/features.py`: point-in-time registry and shifted temporal features;
- `f1bet/identity.py`: fail-closed alias resolution;
- `f1bet/odds.py` and `f1bet/markets.py`: odds, margin, consensus, coherence, settlement;
- `f1bet/calibration.py`: probability scoring and calibration;
- `f1bet/evaluation.py`: comparable ablations and required context-slice reports;
- `f1bet/validation.py`: expanding and season-forward evaluation;
- `f1bet/simulation.py`: coherent field and Plackett–Luce simulations;
- `f1bet/strategy.py`: interpretable strategy comparisons;
- `f1bet/risk.py` and `f1bet/backtest.py`: conservative sizing and replay;
- `f1bet/monitoring.py`: missingness, coverage, and drift;
- `f1bet/artifacts.py`: reproducible model manifests and promotion gates;
- `f1bet/release.py`: fail-closed implementations of release gates 1–8;
- `f1bet/sources.py`: retrying, credential-safe source clients;
- `f1bet/pipeline.py`, `f1bet/migrations.py`, `f1bet/ledger.py`, and `f1bet/cli.py`: point-in-time snapshots, migration, and operations;
- `f1bet/streamlit_page.py`: value, simulation, replay, and governance UI;
- 62 deterministic offline tests plus 3 parametrized subtests, workflow/security validation, full-repository compilation, and a real-entrypoint Streamlit smoke run.

## Recommended delivery order

1. Revoke the previously embedded odds key.
2. Start capturing raw timestamped odds without placing bets.
3. Generate point-in-time feature snapshots for at least three forecast stages: pre-weekend, post-qualifying, and pre-race.
4. Freeze a complete 2026-forward paper ledger.
5. Train market-specific probability models and calibrators inside walk-forward folds.
6. Compare them with market consensus and simple baselines.
7. Promote only after the gates in [VALIDATION_AND_RELEASE_GATES.md](VALIDATION_AND_RELEASE_GATES.md) pass.
8. Keep live bet execution out of scope until book rules, compliance, and operational controls have a separate review.
