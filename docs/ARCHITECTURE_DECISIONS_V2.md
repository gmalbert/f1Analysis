# Architecture decisions for predictive betting v2

## ADR-001 — Add a companion package instead of rewriting the app

**Decision:** put new domain logic in `f1bet/`; keep `raceAnalysis.py` as the entrypoint and call a thin page renderer.

**Reason:** the existing app and pickled artifacts rely on module paths and top-level behavior. A full rewrite would combine model migration, UI migration, and data migration into one high-risk change.

**Consequence:** the legacy monolith remains, but all new logic is independently testable. Existing tabs can migrate incrementally.

## ADR-002 — Normalize markets instead of extending the wide CSV

**Decision:** quote, forecast, decision, and settlement are separate ledgers.

**Reason:** a driver-race can have many books, markets, lines, and timestamps. Appending these to one modeling row creates duplication and lookahead risk.

**Consequence:** market joins must be explicitly timestamped, but their lineage becomes auditable.

## ADR-003 — Treat availability stage as part of feature identity

**Decision:** every production feature has an earliest `SessionStage` and every snapshot has `feature_as_of`.

**Reason:** the same mathematical feature can be valid live and invalid pre-race. Names cannot encode the information clock reliably.

**Consequence:** feature selection can fail fast when a post-race field enters a pre-race model.

## ADR-004 — Optimize probability quality, not finishing-position MAE alone

**Decision:** add market-specific Brier/log-loss/calibration gates and coherent simulations.

**Reason:** bet value and stake depend on probability accuracy. MAE neither calibrates events nor produces valid joint outcomes.

**Consequence:** the position model becomes one input to a probability layer rather than the final betting product.

## ADR-005 — Validate by future race event

**Decision:** expanding-window splits group all drivers in a Grand Prix and train only on earlier events.

**Reason:** driver rows share event shocks and random row splits leak event context. Season GroupKFold can still reverse time.

**Consequence:** fewer effective samples and wider uncertainty, but more honest evidence.

## ADR-006 — Fail closed on missing odds and identity

**Decision:** no value backtest without real decimal odds; ambiguous identity resolution returns no match; unknown settlement becomes void.

**Reason:** silently inventing a price, selection mapping, or result produces precise but invalid profitability.

**Consequence:** more abstentions and incomplete reports, which accurately reflect data quality.

## ADR-007 — Use correlated coherent simulation

**Decision:** simulate race-wide, constructor, individual, and DNF shocks, then rank the complete field.

**Reason:** independent driver distributions can create multiple winners and understate common-event uncertainty.

**Consequence:** all derived markets share the same sample paths and can support H2H and joint extensions.

## ADR-008 — Use conservative paper risk defaults

**Decision:** uncertainty haircut, market shrinkage, quarter Kelly, exposure caps, and drawdown pause.

**Reason:** Kelly assumes the input probability is correct; estimation error makes full Kelly fragile.

**Consequence:** theoretical growth is sacrificed for robustness. The package does not execute bets.

## ADR-009 — Separate deterministic strategy components from residual uncertainty

**Decision:** represent base pace, fuel, tyre loss, pit loss, and safety-car effects explicitly; compare plans with common random numbers.

**Reason:** interpretable components are easier to validate and update with public data than an opaque end-to-end strategy model.

**Consequence:** the deterministic layer is deliberately simplified and must not be labeled a full physics model.

## ADR-010 — Preserve immutable source and artifact lineage

**Decision:** raw snapshot envelopes, SHA-256 data fingerprints, schema versions, code revisions, and model manifests.

**Reason:** historical APIs revise records and feature pipelines change. A result without its exact input version cannot be reproduced.

**Consequence:** storage requirements grow, but audits and rollbacks become possible.
