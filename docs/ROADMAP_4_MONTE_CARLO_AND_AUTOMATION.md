# Roadmap Part 4: Monte Carlo & Automation Improvements for MAE ≤ 1.5

**Baseline after ROADMAP-1: 1.69 (80/20) / 1.80 (GroupKFold) → Target: ≤ 1.5 | Estimated impact of this section: 0.05–0.10**

---

## ⏳ 4A. Strengthen Monte Carlo Model (Quick Win) — pending

The Monte Carlo feature selection script uses a much weaker model than production, which means the features it selects aren't necessarily the best ones for the real model. Closing this gap ensures feature selection results generalize to production.

**File:** `scripts/precompute/monte_carlo_features.py`

### Current (weak) config:
```python
model = XGBRegressor(n_estimators=100, max_depth=4, random_state=42)
```

### Improved config matching production:
```python
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Use production-equivalent model settings for feature selection
MONTE_CARLO_MODEL = XGBRegressor(
    n_estimators=300,       # Up from 100 (was too weak for 140+ feature space)
    max_depth=6,            # Up from 4 (matches production)
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
)

# Also add a secondary LightGBM run for agreement checking
MONTE_CARLO_MODEL_LGBM = LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
```

**Estimated MAE impact:** 0.02–0.04 (from better-selected features propagating to production)  
**Effort:** 30 min

---

## ⏳ 4B. Tiered Feature Count Search — pending

The current Monte Carlo search sweeps 8–15 features. With 140+ features in the model, this undershoots the optimal count. A two-stage tiered search finds the right count more efficiently.

**File:** `scripts/precompute/monte_carlo_features.py`

### Current config:
```python
MIN_FEATURES = 8
MAX_FEATURES = 15
N_TRIALS = 1000
```

### Improved tiered config:
```python
# Stage 1: Wide search — identify the best 30–50 feature set
STAGE_1_CONFIG = {
    'min_features': 20,
    'max_features': 60,
    'n_trials': 500,
    'description': 'Wide search — eliminate weak features',
}

# Stage 2: Narrow refinement — within top-50 candidates, find optimal 15–30
STAGE_2_CONFIG = {
    'min_features': 15,
    'max_features': 35,
    'n_trials': 500,
    'candidates': 'from_stage_1_top_50',
    'description': 'Narrow refinement — precision optimization',
}

# Implementation hint: run Stage 1, collect union of top-50 features by frequency,
# then run Stage 2 restricted to that candidate pool.
def run_tiered_monte_carlo(df, n_trials_per_stage=500):
    """Two-stage tiered Monte Carlo feature selection."""
    import random as rnd

    # Stage 1
    stage1_results = []
    for _ in range(n_trials_per_stage):
        k = rnd.randint(STAGE_1_CONFIG['min_features'], STAGE_1_CONFIG['max_features'])
        selected = rnd.sample(all_features, k)
        mae = evaluate_features(df, selected)
        stage1_results.append((mae, selected))

    # Extract top-50 candidate features by frequency in top-20% of Stage 1 runs
    stage1_results.sort(key=lambda x: x[0])
    top_20pct = stage1_results[:len(stage1_results)//5]
    candidate_pool = list({f for _, sel in top_20pct for f in sel})

    # Stage 2 — refined search within candidates
    stage2_results = []
    for _ in range(n_trials_per_stage):
        k = rnd.randint(STAGE_2_CONFIG['min_features'], min(STAGE_2_CONFIG['max_features'], len(candidate_pool)))
        selected = rnd.sample(candidate_pool, k)
        mae = evaluate_features(df, selected)
        stage2_results.append((mae, selected))

    stage2_results.sort(key=lambda x: x[0])
    return stage2_results[0]  # Best (MAE, feature_list) from refined search
```

**Estimated MAE impact:** 0.01–0.02 (from better feature set selection)  
**Effort:** 2–3 hrs

---

## ⏳ 4C. More Frequent Monte Carlo Runs — pending

The current schedule (weekly Sundays) means the feature set is potentially 7 days stale when new race data lands. Increasing to 3×/week ensures features are optimized immediately after each race weekend.

**File:** `.github/workflows/feature-selection-monte-carlo.yml`

### Current schedule:
```yaml
schedule:
  - cron: '0 2 * * 0'   # Weekly Sundays at 2 AM UTC
```

### Improved schedule (Mon/Wed/Fri to cover race weekends):
```yaml
on:
  schedule:
    - cron: '0 3 * * 1'   # Monday 3 AM UTC (day after race Sunday)
    - cron: '0 3 * * 3'   # Wednesday 3 AM UTC (mid-week refresh)
    - cron: '0 3 * * 5'   # Friday 3 AM UTC (before FP1 weekend begins)
  workflow_dispatch:       # Keep manual trigger
```

**Also add a race-triggered run** (when new race data is committed):
```yaml
on:
  push:
    paths:
      - 'data_files/f1ForAnalysis.csv'  # Re-run feature selection when main data updates
```

**Estimated MAE impact:** Indirect — prevents feature staleness  
**Effort:** 15 min

---

## ⏳ 4D. MAE Regression Check in CI — pending

Currently there's no automated check to detect if a code change worsens MAE. Adding a MAE regression guard to the CI pipeline prevents accidental regressions from being deployed.

### New workflow: `.github/workflows/mae-regression-check.yml`

```yaml
name: MAE Regression Check

on:
  pull_request:
    paths:
      - 'f1-generate-analysis.py'
      - 'raceAnalysis.py'
      - 'scripts/precompute/**'

jobs:
  mae-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run MAE regression check
        run: python scripts/check_mae_regression.py --threshold 0.05
        env:
          MAE_BASELINE: ${{ vars.MAE_BASELINE }}   # Set in repo vars, e.g., "1.85"
```

### New script: `scripts/check_mae_regression.py`

```python
#!/usr/bin/env python3
"""CI script: Check that MAE has not regressed beyond threshold vs baseline.
Fails with exit code 1 if new MAE > baseline + threshold.
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBRegressor

DATA_DIR = Path('data_files')

def compute_quick_mae(threshold: float, baseline: float) -> float:
    df = pd.read_csv(DATA_DIR / 'f1ForAnalysis.csv', sep='\t', low_memory=False)

    # Load best features from precomputed list
    feat_file = DATA_DIR / 'precomputed_features' / 'best_features.txt'
    if not feat_file.exists():
        print("WARNING: No feature list found — skipping MAE check")
        sys.exit(0)

    features = feat_file.read_text().strip().split('\n')
    features = [f for f in features if f in df.columns]
    target = 'resultsFinalPositionNumber'

    valid = df[features + [target, 'grandPrixYear']].dropna(subset=[target])
    X = valid[features].select_dtypes(include='number').fillna(valid.median(numeric_only=True))
    y = valid[target]
    groups = valid['grandPrixYear']

    model = XGBRegressor(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(model, X, y, groups=groups, cv=cv, scoring='neg_mean_absolute_error')
    mae = -scores.mean()

    print(f"MAE (5-fold CV): {mae:.4f}")
    print(f"Baseline:        {baseline:.4f}")
    print(f"Threshold:       +{threshold:.4f}")

    if mae > baseline + threshold:
        print(f"FAIL: MAE {mae:.4f} exceeds baseline {baseline:.4f} + threshold {threshold:.4f}")
        sys.exit(1)
    else:
        print(f"PASS: MAE within acceptable range.")
        sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Maximum allowed MAE increase over baseline (default: 0.05)')
    args = parser.parse_args()
    baseline = float(os.environ.get('MAE_BASELINE', '1.69'))
    compute_quick_mae(args.threshold, baseline)
```

**Effort:** 2–3 hrs  
**Value:** Prevents code regressions; critical for team development

---

## ⏳ 4E. Automated Data-Pull Schedule — pending

Currently data pulls are manual. Automating them means the model is always trained on the latest data after each race weekend without manual intervention.

### New workflow: `.github/workflows/weekly-data-refresh.yml`

```yaml
name: Weekly Data Refresh

on:
  schedule:
    - cron: '0 4 * * 1'   # Monday 4 AM UTC (after each race Sunday)
  workflow_dispatch:

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install -r requirements.txt

      - name: Pull tire strategy data (incremental)
        run: python f1-tire-strategy.py
        # Only pulls new races since last run (incremental by design)

      - name: Pull race pace data (incremental)
        run: python f1-race-pace-laps.py

      - name: Pull race control messages (incremental)
        run: python f1-raceMessages.py

      - name: Pull first-lap positions (incremental)
        run: python f1-first-lap-positions.py

      - name: Pull weather for new races
        run: python f1-analysis-weather.py

      - name: Commit updated data files
        uses: stefanzweifel/git-auto-commit-action@v5
        with:
          commit_message: 'data: weekly auto-refresh [skip ci]'
          file_pattern: |
            data_files/tire_strategy_data.csv
            data_files/race_pace_lap_data.csv
            data_files/first_lap_positions.csv
            data_files/f1WeatherData_AllData.csv
            data_files/all_race_control_messages.csv
```

**Note:** Does not run `f1-generate-analysis.py` in CI (too heavy, ~30 min). Trigger that manually or in a separate heavier workflow.

**Effort:** 2–3 hrs  
**Value:** Keeps data fresh automatically; reduces human error in data pull sequencing

---

## ⏳ 4F. Monte Carlo Budget Analysis Dashboard — pending

Add a lightweight diagnostic section to the Streamlit app showing Monte Carlo cost/benefit: how many trials it took to converge, what the within-session MAE standard deviation is, and a plot of MAE vs. feature count.

**File:** `raceAnalysis.py` — add to the Feature Selection tab (Tab5, around line 3782):

```python
# In the "Feature Selection" tab section:
if st.checkbox("Show Monte Carlo convergence analysis"):
    mc_log_file = Path('data_files/precomputed_features/monte_carlo_run_log.json')
    if mc_log_file.exists():
        import json
        mc_log = json.load(mc_log_file.open())
        
        # Plot trial MAE over time
        mc_df = pd.DataFrame(mc_log['trials'])
        fig = alt.Chart(mc_df).mark_line().encode(
            x=alt.X('trial:Q', title='Trial Number'),
            y=alt.Y('mae:Q', title='MAE', scale=alt.Scale(zero=False)),
            color=alt.Color('stage:N', title='Stage')
        ).properties(title='Monte Carlo Convergence', height=250)
        st.altair_chart(fig, width='stretch')
        
        # Feature count vs MAE scatter
        fig2 = alt.Chart(mc_df).mark_circle(size=30, opacity=0.5).encode(
            x=alt.X('n_features:Q', title='Feature Count'),
            y=alt.Y('mae:Q', title='MAE', scale=alt.Scale(zero=False)),
            color=alt.Color('mae:Q', scale=alt.Scale(scheme='redyellowgreen', reverse=True)),
            tooltip=['trial', 'n_features', 'mae'],
        ).properties(title='Feature Count vs MAE', height=250)
        st.altair_chart(fig2, width='stretch')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Best Trial", mc_log.get('best_trial', '–'))
        col2.metric("Best MAE", f"{mc_log.get('best_mae', 0):.4f}")
        col3.metric("Total Trials", mc_log.get('total_trials', 0))
    else:
        st.info("No Monte Carlo run log found. Run the feature selection workflow first.")
```

**Effort:** 2–3 hrs  
**Value:** Visibility into feature selection quality without manual log-reading

---

## Summary

| Improvement | Est. MAE Impact | Effort | Priority | Status |
|-------------|----------------|--------|----------|--------|
| Strengthen MC model (n_estimators/depth) | 0.02–0.04 | 30 min | **P0** | ⏳ pending |
| Tiered feature count search | 0.01–0.02 | 2–3 hrs | **P1** | ⏳ pending |
| 3×/week MC schedule | Indirect | 15 min | **P1** | ⏳ pending |
| MAE regression CI check | 0 (safety) | 2–3 hrs | **P1** | ⏳ pending |
| Automated data-pull schedule | 0 (reliability) | 2–3 hrs | **P2** | ⏳ pending |
| MC convergence dashboard | 0 (insight) | 2–3 hrs | **P2** | ⏳ pending |
| **Total MAE** | **0.03–0.06** | | |

---

## Combined MAE Projection (All 4 Roadmaps)

| Roadmap | Est. MAE Reduction | Cumulative MAE |
|---------|-------------------|----------------|
| Baseline | — | 2.08 |
| ROADMAP_1 (Feature Engineering) | 0.22–0.39 | ~1.70–1.86 |
| ROADMAP_2 (New Data Sources) | 0.10–0.18 | ~1.52–1.76 |
| ROADMAP_3 (Model Architecture) | 0.15–0.25 | ~1.27–1.61 |
| ROADMAP_4 (MC & Automation) | 0.03–0.06 | ~1.21–1.58 |

**Best-case:** MAE ~1.21 | **Expected:** MAE ~1.40–1.60 | **Conservative:** MAE ~1.55–1.75

**Probability of hitting ≤1.5 by implementing all four roadmaps: ~65–75%**

### Implementation Order for Fastest MAE Gains

1. **Week 1 (P0 items, ~4 hrs total):**
   - Strengthen MC model config (30 min)
   - Fix `championship_fight_performance` ordering bug (30 min)
   - Fix `wet_race_vs_quali_delta` groupby bug (1 hr)
   - Add unused features to `get_features_and_target()` (30 min)
   - Expected MAE gain: 0.08–0.15 → projected ~1.93–2.00

2. **Week 2 (Tire strategy data pull, ~1 day):**
   - Run `f1-tire-strategy.py` historical backfill (2–3 hrs background)
   - Add tire features to generator + model (1–2 hrs)
   - Expected MAE gain: 0.08–0.12 → projected ~1.81–1.92

3. **Week 3 (Race pace + First-lap data):**
   - Run `f1-race-pace-laps.py` historical backfill
   - Run `f1-first-lap-positions.py` historical backfill
   - Add features to generator + model
   - Expected MAE gain: 0.11–0.17 → projected ~1.64–1.81

4. **Week 4 (Model architecture):**
   - Position-specific sub-models
   - IterativeImputer
   - Inferred MAE gain: 0.08–0.14 → projected ~1.50–1.73

5. **Ongoing (Automation):**
   - Weather expansion, altitude CSV, grid penalties
   - Monthly HPO, CI MAE regression check
   - Continued data pull automation
