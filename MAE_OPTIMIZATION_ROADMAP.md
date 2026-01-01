# MAE Optimization Roadmap
**Goal: Reduce MAE from ~1.94 to ≤1.5**

[← Back to README](README.md)

---

## Current State
- Current MAE: **~1.94**
- Target MAE: **≤1.5**
- Gap to close: **0.44 positions**
- Models: XGBoost, LightGBM, CatBoost, Ensemble
- Features: 140+ engineered features

---

## 🎯 Phase 1: Quick Wins (Est. MAE reduction: 0.15-0.25)

### 1.1 Tire Strategy Features ⚡ HIGH IMPACT
**Complexity:** Medium | **Location:** `f1-generate-analysis.py` (after line 1903)

```python
# Tire compound strategy effectiveness
df['tire_compound_at_start'] = df.groupby(['raceId', 'resultsDriverName'])['compound'].first()
df['tire_stint_count'] = df.groupby(['raceId', 'resultsDriverName'])['stint'].transform('nunique')
df['avg_stint_length'] = df.groupby(['raceId', 'resultsDriverName'])['laps_on_compound'].transform('mean')

# Tire advantage vs field
df['tire_advantage'] = (
    df.groupby('raceId')['compound'].transform(lambda x: (x == 'SOFT').mean()) -
    df.groupby(['raceId', 'resultsDriverName'])['compound'].transform(lambda x: (x == 'SOFT').mean())
)

# Historical tire performance at track
df['driver_tire_performance_at_track'] = df.groupby(
    ['circuitId', 'resultsDriverName', 'tire_compound_at_start']
)['resultsFinalPositionNumber'].transform(lambda x: x.shift(1).mean())
```

### 1.2 Race Pace Features ⚡ HIGH IMPACT
**Complexity:** Medium | **Location:** `f1-generate-analysis.py` (line 1903+)

```python
# Sector time consistency (lower = better racecraft)
df['sector_consistency'] = df.groupby(['raceId', 'resultsDriverName']).apply(
    lambda x: x[['sector1Time', 'sector2Time', 'sector3Time']].std(axis=1).mean()
).reset_index(drop=True)

# Race pace vs qualifying gap
df['race_pace_delta'] = (
    df['avg_lap_time_race'] - df['best_qual_time']
) / df['best_qual_time'] * 100

# Fuel-adjusted pace (early laps vs late laps)
df['fuel_corrected_pace'] = df.groupby(['raceId', 'resultsDriverName']).apply(
    lambda x: (x['LapTime_sec'].iloc[-10:].mean() - x['LapTime_sec'].iloc[:10].mean()) / x['LapTime_sec'].mean()
).reset_index(drop=True)
```

### 1.3 Clean Air Dominance 🏎️ MEDIUM IMPACT
**Complexity:** Easy | **Location:** `f1-generate-analysis.py` (line 1903+)

```python
# Time spent in clean air (top 3 positions)
df['laps_in_clean_air'] = df.groupby(['raceId', 'resultsDriverName']).apply(
    lambda x: (x['position'] <= 3).sum()
).reset_index(drop=True)

# Clean air pace advantage
df['clean_air_advantage'] = df['CleanAirAvg_FP1'] - df['DirtyAirAvg_FP2']
df['clean_air_advantage_bin'] = pd.qcut(df['clean_air_advantage'], q=10, labels=False, duplicates='drop')
```

---

## 🚀 Phase 2: Model Architecture (Est. MAE reduction: 0.10-0.20)

### 2.1 Position-Specific Models 🎯 HIGH IMPACT
**Complexity:** Hard | **Location:** `raceAnalysis.py` (new function after line 1950)

```python
@st.cache_data
def train_position_specific_models(data, CACHE_VERSION):
    """Train separate models for different position groups"""
    models = {}
    position_groups = {
        'podium': (1, 3),      # Top 3
        'points': (4, 10),     # P4-P10
        'midfield': (11, 15),  # P11-P15
        'backfield': (16, 20)  # P16-P20
    }
    
    for group_name, (min_pos, max_pos) in position_groups.items():
        # Filter data for position group
        group_data = data[
            (data['resultsQualificationPositionNumber'] >= min_pos) & 
            (data['resultsQualificationPositionNumber'] <= max_pos)
        ]
        
        X, y = get_features_and_target(group_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost for this position group
        preprocessor = get_preprocessor_position(X_train)
        X_train_prep = preprocessor.fit_transform(X_train)
        
        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,  # Lower for finer granularity
            max_depth=6,
            objective='reg:squarederror'
        )
        model.fit(X_train_prep, y_train)
        models[group_name] = {'model': model, 'preprocessor': preprocessor}
    
    return models
```

### 2.2 Neural Network Ensemble 🧠 HIGH IMPACT
**Complexity:** Hard | **Location:** New file `scripts/neural_ensemble.py`

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_position_predictor_nn(input_dim):
    """Deep learning model for position prediction"""
    model = keras.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Final position
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_absolute_error',
        metrics=['mae', 'mse']
    )
    return model

# Train with early stopping
early_stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=10, restore_best_weights=True)
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=0
)
```

### 2.3 Weighted Ensemble by Track Type 🏁 MEDIUM IMPACT
**Complexity:** Medium | **Location:** `raceAnalysis.py` (line 1950+)

```python
def train_track_specific_ensemble(data, CACHE_VERSION):
    """Weight models differently based on track characteristics"""
    track_types = {
        'street': data['streetRace'] == 1,
        'high_speed': data['avg_speed_kmh'] > 220,
        'technical': (data['turns'] > 15) & (data['streetRace'] == 0)
    }
    
    ensemble_weights = {}
    for track_type, mask in track_types.items():
        track_data = data[mask]
        
        # Train XGB, LGB, CB on this track type
        xgb_mae = cross_val_score(xgb_model, X[mask], y[mask], cv=5, scoring='neg_mean_absolute_error').mean()
        lgb_mae = cross_val_score(lgb_model, X[mask], y[mask], cv=5, scoring='neg_mean_absolute_error').mean()
        cb_mae = cross_val_score(cb_model, X[mask], y[mask], cv=5, scoring='neg_mean_absolute_error').mean()
        
        # Inverse MAE weighting (lower MAE = higher weight)
        total = (1/xgb_mae) + (1/lgb_mae) + (1/cb_mae)
        ensemble_weights[track_type] = {
            'xgb': (1/xgb_mae) / total,
            'lgb': (1/lgb_mae) / total,
            'cb': (1/cb_mae) / total
        }
    
    return ensemble_weights
```

---

## 🔬 Phase 3: Advanced Features (Est. MAE reduction: 0.08-0.15)

### 3.1 Driver vs Team Dynamics 👥 HIGH IMPACT
**Complexity:** Medium | **Location:** `f1-generate-analysis.py` (line 1903+)

```python
# Teammate battle history
df['teammate_head_to_head'] = df.groupby(['grandPrixYear', 'constructorName']).apply(
    lambda x: (x['resultsFinalPositionNumber'].rank() == 1).astype(int)
).reset_index(drop=True)

# Constructor performance trajectory (improving vs declining)
df['constructor_momentum'] = df.groupby('constructorName')['resultsFinalPositionNumber'].transform(
    lambda x: x.rolling(3, min_periods=1).mean().diff()
)

# Driver adaptation rate (new team performance)
df['races_since_team_change'] = df.groupby('resultsDriverName').cumcount() - \
    df.groupby('resultsDriverName')['constructorName'].transform(lambda x: (x != x.shift()).cumsum())

df['new_team_adaptation_curve'] = np.exp(-df['races_since_team_change'] / 5)  # Exponential learning curve
```

### 3.2 Race Strategy Indicators 📊 MEDIUM IMPACT
**Complexity:** Medium | **Location:** `f1-generate-analysis.py` (line 1903+)

```python
# Optimal pit window hit rate
df['optimal_pit_window'] = df.groupby('circuitId')['lap_of_first_stop'].transform(
    lambda x: (x >= x.quantile(0.4)) & (x <= x.quantile(0.6))
)

# Undercut/overcut success rate
df['undercut_success'] = (
    (df['lap_of_first_stop'] < df['lap_of_first_stop'].shift(1)) &
    (df['resultsFinalPositionNumber'] < df['resultsStartingGridPositionNumber'])
).astype(int)

# Strategy aggressiveness score
df['strategy_risk'] = (
    df['numberOfStops'] - df.groupby('raceId')['numberOfStops'].transform('median')
) / df.groupby('raceId')['numberOfStops'].transform('std')
```

### 3.3 Weather Impact Modeling ☁️ MEDIUM IMPACT
**Complexity:** Medium | **Location:** `f1-generate-analysis.py` (line 1903+)

```python
# Rain tire advantage
df['wet_performance_index'] = df.groupby('resultsDriverName').apply(
    lambda x: x[x['total_precipitation'] > 0.5]['resultsFinalPositionNumber'].mean() - 
              x[x['total_precipitation'] <= 0.5]['resultsFinalPositionNumber'].mean()
).reset_index(drop=True)

# Temperature performance band (driver-specific optimal temps)
df['driver_temp_optimal'] = df.groupby('resultsDriverName')['average_temp'].transform(
    lambda x: np.abs(x - x[df['resultsFinalPositionNumber'] <= 3].mean())
)

# Wind sensitivity (impacts downforce-dependent cars differently)
df['wind_penalty'] = df['average_wind_speed'] * (1 - df['downforce_level'] / 100)
```

---

## 🛠️ Phase 4: Data Quality (Est. MAE reduction: 0.05-0.10)

### 4.1 Smart Imputation 🎲 MEDIUM IMPACT
**Complexity:** Easy | **Location:** `f1-generate-analysis.py` (before feature engineering)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Use iterative imputation instead of simple mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
imputer = IterativeImputer(max_iter=10, random_state=42)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

### 4.2 Outlier Capping 📏 LOW IMPACT
**Complexity:** Easy | **Location:** `f1-generate-analysis.py` (line 1903+)

```python
# Cap extreme values at 99th percentile
for col in ['LapTime_sec', 'SpeedST_mph', 'SpeedI2_mph']:
    if col in df.columns:
        upper = df[col].quantile(0.99)
        lower = df[col].quantile(0.01)
        df[col] = df[col].clip(lower, upper)
```

---

## 📈 Phase 5: Hyperparameter Tuning (Est. MAE reduction: 0.03-0.08)

### 5.1 Optuna Bayesian Optimization ⚙️ HIGH IMPACT
**Complexity:** Medium | **Location:** `raceAnalysis.py` (new function)

```python
import optuna

def optimize_xgboost_hyperparameters(X_train, y_train, n_trials=100):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
        }
        
        model = xgb.XGBRegressor(**params, random_state=42)
        mae_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        return -mae_scores.mean()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

---

## 🎓 Implementation Priority Matrix

| Feature/Model | MAE Impact | Complexity | Priority | Est. Dev Time |
|---------------|------------|------------|----------|---------------|
| Tire Strategy Features | High (0.08-0.12) | Medium | 🔴 Critical | 2-3 hours |
| Position-Specific Models | High (0.10-0.15) | Hard | 🔴 Critical | 4-6 hours |
| Race Pace Features | High (0.07-0.10) | Medium | 🟡 High | 2-3 hours |
| Driver vs Team Dynamics | High (0.05-0.08) | Medium | 🟡 High | 3-4 hours |
| Neural Network Ensemble | Med (0.05-0.10) | Hard | 🟡 High | 6-8 hours |
| Weather Impact | Medium (0.04-0.06) | Medium | 🟢 Medium | 2-3 hours |
| Track-Specific Ensemble | Medium (0.03-0.06) | Medium | 🟢 Medium | 2-3 hours |
| Optuna Tuning | Low (0.03-0.05) | Medium | 🟢 Medium | 1-2 hours |
| Smart Imputation | Low (0.02-0.04) | Easy | ⚪ Low | 1 hour |

---

## 🧪 Testing & Validation Strategy

### Cross-Validation Framework
```python
from sklearn.model_selection import GroupKFold, cross_val_score

# Prevent season leakage
gkf = GroupKFold(n_splits=5)
mae_scores = cross_val_score(
    model, X, y, 
    cv=gkf.split(X, y, groups=data['grandPrixYear']),
    scoring='neg_mean_absolute_error'
)
print(f"CV MAE: {-mae_scores.mean():.3f} (+/- {mae_scores.std():.3f})")
```

### A/B Testing Framework
```python
def compare_models(baseline_mae, new_mae, n_races):
    """Statistical significance test"""
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(baseline_mae, new_mae)
    
    improvement = (baseline_mae.mean() - new_mae.mean()) / baseline_mae.mean() * 100
    significant = p_value < 0.05
    
    return {
        'improvement_pct': improvement,
        'significant': significant,
        'p_value': p_value
    }
```

---

## 📊 Expected Cumulative MAE Reduction

| Phase | Features/Models | Est. MAE After | Cumulative Reduction |
|-------|----------------|----------------|----------------------|
| **Current** | Baseline | 1.94 | - |
| **Phase 1** | Quick Wins | 1.69-1.79 | 0.15-0.25 ↓ |
| **Phase 2** | Model Architecture | 1.49-1.69 | 0.25-0.45 ↓ |
| **Phase 3** | Advanced Features | 1.41-1.54 | 0.40-0.53 ↓ |
| **Phase 4** | Data Quality | 1.36-1.49 | 0.45-0.58 ↓ |
| **Phase 5** | Hypertuning | 1.33-1.44 | 0.50-0.61 ↓ |
| **🎯 Target** | **All Phases** | **≤1.5** | **≥0.44 ↓** |

---

## 🚦 Getting Started (Next Steps)

1. **Immediate (This Week):**
   - Implement tire strategy features (Section 1.1)
   - Add race pace features (Section 1.2)
   - Run baseline MAE comparison

2. **Short-term (This Month):**
   - Train position-specific models (Section 2.1)
   - Add driver vs team dynamics (Section 3.1)
   - Implement A/B testing framework

3. **Medium-term (Next Quarter):**
   - Experiment with neural network ensemble (Section 2.2)
   - Complete all Phase 3 advanced features
   - Run Optuna hyperparameter optimization

---

**Remember:** Always measure MAE before and after each change using season-stratified cross-validation to ensure improvements are real and not overfitting!

[← Back to README](README.md)
