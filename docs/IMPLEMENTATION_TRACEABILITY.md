# Predictive F1 betting v2 implementation traceability

This ledger maps the seven v2 specification documents to executable repository evidence. “Implemented” means the capability and its fail-closed checks exist in code. It does not mean that a market edge has been demonstrated: historical timestamped odds, frozen forward forecasts, and settled paper outcomes must still pass the release gates before that claim is allowed.

| Specification area | Primary implementation | Verification evidence | Status |
|---|---|---|---|
| Companion package and thin UI (ADR-001) | `f1bet/`, `f1bet/streamlit_page.py`, `raceAnalysis.py` | package imports in the offline suite; `scripts/smoke_streamlit_app.py` executes the real app entrypoint and value/stake interaction | Implemented |
| Normalized event, quote, forecast, decision, and settlement grains (ADR-002; data model v2) | `f1bet/domain.py`, `f1bet/contracts.py`, `f1bet/ledger.py`, `f1bet/migrations.py` | domain/contract, migration, append-only ledger, and lineage tests | Implemented |
| Information stages and point-in-time snapshots (ADR-003) | `SessionStage`, `FeatureRegistry`, `build_v2_event_snapshot`, stage masking and snapshot contracts | snapshot tests reject duplicates, drop unavailable/unregistered fields, preserve lineage, and validate the v2 contracts | Implemented |
| Probability-first scoring and calibration (ADR-004) | `f1bet/calibration.py`, `f1bet/evaluation.py`, `probability_quality_gate` | Brier/log-loss/reliability/ECE/slope/AUC, baseline comparisons, required slices, and positive/failure release-gate tests | Implemented; future market evidence pending |
| Future-only validation (ADR-005) | `f1bet/validation.py`; temporal CV used by production training, HPO, benchmarks, ablations, and UI diagnostics | tests prove race grouping, strict ordering, embargo, final-season isolation, derived round identity, fold-local evidence requirements, and event-clustered intervals | Implemented; reports must be regenerated |
| Fail closed on odds, identity, and timestamps (ADR-006) | `IdentityResolver`, quote/forecast/decision contracts, `run_backtest`, `data_integrity_gate` | ambiguous identities, naive/post-start timestamps, missing real odds, missing lineage, and incomplete evidence fail in offline tests | Implemented |
| Coherent correlated race simulation (ADR-007) | `f1bet/simulation.py`, `f1bet/markets.py` | unique ranks, common field/constructor shocks, separate DNF process, nested market coherence, H2H, and finite-input tests | Implemented |
| Conservative paper risk only (ADR-008) | `f1bet/risk.py`, `run_backtest`, `run_risk_sensitivity` | quarter Kelly, uncertainty haircut, market shrinkage, caps, edge/EV gates, drawdown pause, full decision retention, and flat/0.1/0.25/0.5 sensitivity tests | Implemented; no live execution |
| Interpretable strategy plus residual uncertainty (ADR-009) | `f1bet/strategy.py` | tyre-curve, complete-stint plan, common-random comparison, safety-car scenario, and adaptive residual tests | Implemented |
| Immutable source and model lineage (ADR-010) | `f1bet/sources.py`, `f1bet/artifacts.py`, training-script manifest writers and manifest-aware loaders | content-addressed snapshot tests, secret-safe clients, SHA/schema/feature checks, deterministic repeat evidence, and legacy-artifact warnings | Implemented for new artifacts; legacy rebuild pending |
| Research target hierarchy | `MarketType`, market settlement, simulation-derived win/podium/top-6/top-10/H2H/DNF/safety-car probabilities | domain, coherent probability, and rule-aware settlement tests | Implemented; parlays and automatic execution intentionally excluded |
| Point-in-time feature blueprint | expanded `default_registry`, shifted event rolling, head-to-head deltas, DNF hazard features, practice/qualifying/weather/strategy metadata | registry availability and leakage tests; strict snapshot output contains only core plus stage-available registered fields | Implemented |
| Required ablations | `REQUIRED_ABLATION_VARIANTS`, `evaluate_probability_variants`, `validate_ablation_coverage` | tests require all 15 documented variants on identical future fold IDs and produce event-clustered intervals | Implemented; experiment results pending |
| Data collection priority and adapters | normalized raw envelopes; Jolpica, OpenF1, and odds clients; coverage/freshness monitoring | retry/timeout/credential/snapshot tests and strict source-evidence release checks | Implemented; continuous collection remains operational work |
| Market replay protocol | opening/taken/closing evidence, latest complete per-book snapshots, de-vig methods, rule versions, CLV/ROI/drawdown/streak/exposure summaries | a 300-decision, 100+-settlement, two-season, multi-book positive-path test plus incomplete-evidence failure tests | Implemented; real replay not demonstrated |
| Release gates 1–8 | `f1bet/release.py`, `python -m f1bet audit-release` | positive and fail-closed tests cover data, reproducibility, temporal, probability, replay, risk, drift, and software gates | Implemented; CLI returns nonzero until supplied evidence passes every gate |
| Security and CI | least-privilege workflow permissions, immutable action SHAs, secret scanner, workflow validator, offline collection | `.github/workflows/f1bet-release.yml`, `scripts/validate_workflow_security.py`, `scripts/scan_embedded_secrets.py`, workflow tests | Implemented |

## Specification coverage

- `PREDICTIVE_F1_BETTING_RESEARCH.md`: probability targets, simulation, feature clock, replay protocol, ablations, and collection priorities are mapped above.
- `FEATURE_CATALOG_V2.md`: all 100 catalog entries remain implemented; later hardening extends those entries without changing the catalog’s fixed count.
- `DATA_MODEL_V2.md`: every documented grain, key, contract, stage, registry field, identity behavior, raw envelope, manifest field, and migration path has a code representation.
- `COMPREHENSIVE_REPOSITORY_REVIEW_2026.md`: P0/P1/P2 findings are addressed through the companion package, temporal migration, normalized ledgers, import-safe tests, risk/settlement, and workflow hardening.
- `ARCHITECTURE_DECISIONS_V2.md`: ADR-001 through ADR-010 are represented in the first ten rows of this ledger.
- `VALIDATION_AND_RELEASE_GATES.md`: all eight gates are executable and fail closed when evidence is absent.
- `IMPLEMENTATION_GUIDE_F1BET.md`: documented validation, simulation, migration, snapshot, ledger, source, de-vig, walk-forward, calibration, replay, strategy, manifest, and audit commands are backed by the CLI or public package APIs.

## Completion boundary

Repository implementation is complete when the offline suite, warning-enabled full compilation, workflow/security checks, CLI checks, and real-entrypoint Streamlit smoke test all pass. Empirical promotion is deliberately separate: Gates 3–5 remain evidence-dependent until regenerated future-block experiments and a real timestamped market ledger are provided.
