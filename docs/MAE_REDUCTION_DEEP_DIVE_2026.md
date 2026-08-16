# F1 Analysis — MAE Reduction Deep Dive

> Generated: 2026-07-31

---

## Current State: MAE ~1.94 → Target MAE ≤1.5

This document details the highest-impact strategies for reducing MAE in the
F1 final position prediction model.

---

## Feature Engineering (Highest Impact)

### 1. Sector-Specific Practice Performance

Sector times from practice sessions are stronger predictors than overall
practice time. Each sector measures different car attributes:

```python
# Expected MAE reduction: ~0.15 positions
SECTOR_FEATURES = [
    "best_s1_s_practice",     # S1 = speed/aero (straights)
    "best_s2_s_practice",     # S2 = downforce (corners)
    "best_s3_s_practice",     # S3 = combination
    "s1_delta_to_fastest_pct",# How much off fastest in S1
    "s2_delta_to_fastest_pct",
    "s3_delta_to_fastest_pct",
    "qualifying_s1_vs_practice_s1",  # Qualifying improvement from practice
]
```

### 2. Tyre Strategy Prediction

Starting compound choice significantly affects race finishing position.
Add compound type and historical compound performance at circuit:

```python
# Expected MAE reduction: ~0.10 positions
TYRE_FEATURES = [
    "starting_compound_encoded",     # 0=Soft, 1=Medium, 2=Hard, 3=Inter, 4=Wet
    "compound_circuit_avg_stint",    # Typical stint length for this compound here
    "compound_consistency_score",    # How well driver manages this compound
    "expected_stops",                # Model-predicted number of pitstops
    "tyre_advantage_vs_median",      # How starting compound ranks vs field
]
```

### 3. Undercut/Overcut Strategy Feature

```python
# Expected MAE reduction: ~0.08 positions
def predict_strategy_type(driver: str, position: int, 
                            gap_ahead: float, laps_remaining: int) -> int:
    """Encode predicted strategy: 0=standard, 1=undercut, 2=overcut, 3=extend"""
    if gap_ahead < 1.5 and laps_remaining > 20:
        return 1  # Likely undercut
    elif gap_ahead > 5.0 and laps_remaining < 15:
        return 2  # Likely overcut
    elif laps_remaining < 8:
        return 3  # Likely extending
    return 0  # Standard
```

### 4. Race Pace vs Qualifying Pace Differential

Some drivers/cars are better in race trim than qualifying:

```python
# Expected MAE reduction: ~0.12 positions
RACE_QUAL_DIFF_FEATURES = [
    "race_pace_vs_qual_pace_l5",  # Positive = better in race
    "tire_deg_sensitivity",        # How much pace drops per lap
    "fuel_corrected_gap_to_leader", # Gap adjusted for fuel loads
    "long_run_pace_index",         # Practice long run pace rank
]
```

---

## Model Architecture Improvements

### 1. Two-Stage Model (Recommended)

Stage 1: Predict qualifying position (strong signal)
Stage 2: Predict final position using qualifying position as input

```python
from xgboost import XGBRegressor
import numpy as np

class TwoStageF1Model:
    def __init__(self):
        self.qualifying_model = XGBRegressor(n_estimators=600, max_depth=5)
        self.race_model = XGBRegressor(n_estimators=800, max_depth=6)

    def fit(self, X_qual: pd.DataFrame, y_qual: pd.Series,
             X_race: pd.DataFrame, y_race: pd.Series) -> None:
        self.qualifying_model.fit(X_qual, y_qual)
        qual_pred = self.qualifying_model.predict(X_qual)
        X_race_aug = X_race.copy()
        X_race_aug["predicted_qual_pos"] = qual_pred
        X_race_aug["qual_vs_predicted_diff"] = y_qual.values - qual_pred
        self.race_model.fit(X_race_aug, y_race)

    def predict(self, X_qual: pd.DataFrame, X_race: pd.DataFrame) -> np.ndarray:
        qual_pred = self.qualifying_model.predict(X_qual)
        X_race_aug = X_race.copy()
        X_race_aug["predicted_qual_pos"] = qual_pred
        X_race_aug["qual_vs_predicted_diff"] = 0  # Unknown at prediction time
        return self.race_model.predict(X_race_aug)
```

### 2. Position Group Stratification

Train separate models for:
- **Top-10 finishers** (positions 1-10): Need precision for podium/points
- **Mid-field** (positions 11-15): Different dynamics  
- **Backmarkers** (positions 16-20): Safety cars and DNFs dominate

```python
from sklearn.model_selection import GroupKFold

def train_stratified_models(df: pd.DataFrame, features: list) -> dict:
    """Train separate models for different finishing position ranges."""
    df["position_group"] = pd.cut(df["resultsFinalPositionNumber"],
                                    bins=[0, 10, 15, 20], labels=["top10", "mid", "back"])
    models = {}
    for group in ["top10", "mid", "back"]:
        subset = df[df["position_group"] == group]
        X = subset[features].fillna(subset[features].median())
        y = subset["resultsFinalPositionNumber"]
        model = XGBRegressor(n_estimators=500, max_depth=5)
        model.fit(X, y)
        models[group] = model
        print(f"Trained {group} model on {len(subset)} samples")
    return models
```

### 3. Gradient Boosting with Custom Loss (Rank-Aware)

Replace MSE with a rank-aware loss function that penalizes position inversions more:

```python
import numpy as np

def rank_aware_loss(y_true: np.ndarray, y_pred: np.ndarray,
                     inversion_penalty: float = 2.0) -> np.ndarray:
    """
    Custom loss that penalizes rank inversions more than positional error.
    E.g., predicting P1 when actual is P5 is worse than P1 vs P3.
    """
    residuals = y_pred - y_true
    # Double the penalty for large position swaps (>3 positions wrong)
    large_error_mask = np.abs(residuals) > 3
    weights = np.where(large_error_mask, inversion_penalty, 1.0)
    return weights * residuals**2
```

---

## Data Quality Improvements

### 1. First-Lap Position Data

First-lap positions are strong predictors of final result (especially
after safety cars or incidents):

```python
def fetch_first_lap_positions(year: int, round_num: int) -> pd.DataFrame:
    """Fetch position data after first lap from FastF1."""
    session = fastf1.get_session(year, round_num, "R")
    session.load(laps=True, weather=False)
    lap1 = session.laps[session.laps["LapNumber"] == 1]
    return lap1.groupby("Driver")["Position"].first().reset_index()
```

### 2. Safety Car Feature Improvement

Current SC features don't capture when in the race the SC is deployed:

```python
ENHANCED_SC_FEATURES = [
    "sc_deployed_before_lap10",    # Early SC benefits drivers outside top-10
    "sc_deployed_after_lap30",     # Late SC can destroy race leads
    "num_sc_periods",              # Multiple SCs = high chaos
    "sc_duration_pct",             # % of race under SC
    "vsc_periods",                 # Virtual SC count
    "had_red_flag",                # Red flag = complete reset
]
```

---

## Expected MAE Reduction Roadmap

| Enhancement | MAE Reduction | Implementation Time |
|-------------|---------------|---------------------|
| Sector time features | -0.15 | 1 week |
| Two-stage model | -0.12 | 2 weeks |
| Tyre strategy features | -0.10 | 1 week |
| Race/qual pace differential | -0.12 | 1 week |
| First-lap position | -0.08 | 3 days |
| Enhanced SC features | -0.05 | 3 days |
| Hyperparameter optimization | -0.05 | 2 days |
| Stratified position models | -0.08 | 1 week |
| **Total Expected Improvement** | **-0.75** | **~6 weeks** |
| **Projected Final MAE** | **~1.19** | — |

---

## Current Feature Audit

Remove these features (likely leaky or low-importance):

```python
# Features to remove (potential leakage or redundant)
REMOVE_FEATURES = [
    "resultsFinalPositionNumber",  # Target variable — should never be a feature!
    "points_season_total",          # Season total includes current race points
    "championship_position",        # Rank depends on race result
]

# Verify these have shift(1) applied
AUDIT_ROLLING_FEATURES = [
    "constructor_recent_form_3_races",
    "driver_dnf_rate",
    "points_leader_gap",
    "track_experience",
]
```
