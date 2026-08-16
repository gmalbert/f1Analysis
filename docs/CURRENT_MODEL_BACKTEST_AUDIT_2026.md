# Current-model backtest audit (2026)

## Verdict

The stored residual audit covers 791 driver-race observations with overall finishing-position MAE 1.354. For 2025 it reports 88 observations, MAE 1.269 and median absolute error 1.004; the partial 2026 slice deteriorates to MAE 1.767 over 48. Error is asymmetric by field position: podium MAE is 0.781, midfield 1.433, and backmarkers 2.120. The artifact pools seasons in its test set and does not document a pure last-season training cutoff, so it is a diagnostic holdout rather than a clean walk-forward betting test. No odds ledger exists.

## Changes justified by the result

1. Train race-by-race expanding windows and report 2025/2026 as truly future blocks. Group by race weekend to prevent row leakage among drivers.
2. Use ranking/listwise objectives or constrained simulations so predicted finishing positions form a valid field ordering.
3. Model DNF separately and combine conditional finish distributions; backmarker error suggests survival/incident uncertainty is being compressed.
4. Add regulation-era, constructor trajectory, track archetype, qualifying/long-run pace, weather, and upgrade effects with strict session cutoffs.

## Betting strategy decision

- **Winner/podium/top-6/top-10:** derive calibrated event probabilities, not point-position MAE alone.
- **Head-to-heads:** best initial market because relative error cancels field-wide shocks; still needs odds/CLV replay.
- **Fastest lap/DNF/safety car:** separate classifiers and rule-aware settlement.
- **Outrights/parlays:** simulate correlated race/season outcomes; no naive multiplication.
- **Staking:** paper-only; no Kelly without event probability calibration.

## Release gate

Leave-future-races-out evaluation, market-relative Brier/log loss for each binary market, 300+ frozen selections, positive CLV, and performance reported by track, weather, grid band, constructor, and regulation era.
