"""
Measure MAE true picture - matches the exact split and model params used in raceAnalysis.py.

Methodology:
  - Same model: XGBRegressor(n_estimators=200, lr=0.1, max_depth=4, random_state=42)
  - Same split:  train_test_split(test_size=0.2, random_state=42)            [matches the 2.08 baseline]
  - Same sample weights: 2x winners, 1.5x podium, 1.2x points, 1.0 rest
  - Same early stopping: early_stopping_rounds=20, eval_set on test split
  - Also shows 5-fold GroupKFold by season for a leakage-free reference

The script simulates all ROADMAP-1 bug-fixed and new features *inline* on the existing
CSV so we don't need to wait for a full generator re-run.

Usage:
    python scripts/measure_mae_true_picture.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ── project root ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data_files"

print("=" * 70)
print("F1 MAE TRUE PICTURE — ROADMAP-1 FEATURE ENGINEERING IMPACT")
print("=" * 70)

# ── load CSV ─────────────────────────────────────────────────────────────────
csv_path = DATA_DIR / "f1ForAnalysis.csv"
print(f"\nLoading {csv_path.name} …", end=" ", flush=True)
df = pd.read_csv(csv_path, sep='\t', low_memory=False)
print(f"{len(df):,} rows × {df.shape[1]} columns")

target_col = "resultsFinalPositionNumber"
df = df.dropna(subset=[target_col])
print(f"Valid rows (non-NaN target): {len(df):,}")

# ── sort chronologically so shift/rolling makes sense ────────────────────────
sort_cols = [c for c in ["grandPrixYear", "round", "raceId"] if c in df.columns]
if sort_cols:
    df = df.sort_values(sort_cols).reset_index(drop=True)

# ════════════════════════════════════════════════════════════════════════════
# 1.  INLINE SIMULATION OF BUG-FIXED FEATURES
# ════════════════════════════════════════════════════════════════════════════
print("\n── Simulating ROADMAP-1 fixes inline on loaded CSV ──")

# ── 1a. wet_race_vs_quali_delta (bug was: global mask inside apply → empty) ─
if 'total_precipitation' in df.columns and 'resultsFinalPositionNumber' in df.columns and 'resultsQualificationPositionNumber' in df.columns:
    wet_rows = df['total_precipitation'].fillna(0) > 0
    n_wet = wet_rows.sum()
    print(f"   wet_race_vs_quali_delta   : {n_wet:,} wet-race rows to work with")
    if n_wet >= 20:
        wet_df = df.loc[wet_rows].copy()
        wet_delta = (
            wet_df.groupby('resultsDriverName', group_keys=False)
            .apply(lambda g: (
                g['resultsFinalPositionNumber'] - g['resultsQualificationPositionNumber']
            ).shift(1).expanding().mean())
        )
        df['wet_race_vs_quali_delta_fixed'] = np.nan
        df.loc[wet_rows, 'wet_race_vs_quali_delta_fixed'] = wet_delta.values
        df['wet_race_vs_quali_delta_fixed'] = (
            df.groupby('resultsDriverName')['wet_race_vs_quali_delta_fixed']
            .transform(lambda x: x.ffill())
        )
        coverage = df['wet_race_vs_quali_delta_fixed'].notna().mean()
        print(f"   wet_race_vs_quali_delta_fixed coverage: {coverage:.1%}")
    else:
        df['wet_race_vs_quali_delta_fixed'] = df.get('wet_race_vs_quali_delta', np.nan)
else:
    df['wet_race_vs_quali_delta_fixed'] = np.nan

# ── 1b. championship_fight_performance (bug was: column didn't exist yet when computed) ─
if 'championship_position' in df.columns:
    in_fight = df['championship_position'].fillna(99) <= 3
    n_fight = in_fight.sum()
    print(f"   championship_fight_performance: {n_fight:,} in-fight rows")
    df['championship_fight_performance_fixed'] = np.nan
    fight_vals = (
        df.loc[in_fight]
        .groupby('resultsDriverName', group_keys=False)['resultsFinalPositionNumber']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df.loc[in_fight, 'championship_fight_performance_fixed'] = fight_vals.values
    coverage = df['championship_fight_performance_fixed'].notna().mean()
    print(f"   championship_fight_performance_fixed coverage: {coverage:.1%}")
else:
    df['championship_fight_performance_fixed'] = np.nan

# ── 1c. Interaction features (new, computed from existing CSV columns) ──────

def safe(col, fill=0.0):
    """Return column values with NaN filled."""
    if col in df.columns:
        return df[col].fillna(fill)
    return pd.Series(fill, index=df.index, dtype=float)

# tire_mgmt_x_turns: tire management × number of turns at track
turns_col = next((c for c in ['turns', 'circuitTurns', 'numberOfTurns'] if c in df.columns), None)
if turns_col:
    df['tire_mgmt_x_turns_computed'] = safe('tire_management_score') * safe(turns_col)
else:
    df['tire_mgmt_x_turns_computed'] = safe('tire_management_score') * 15.0  # median turns fallback

# practice_conversion_x_grid: practice-to-race conversion × grid position
df['practice_conversion_x_grid_computed'] = (
    safe('practice_race_conversion') * safe('resultsStartingGridPositionNumber', 10.0)
)

# pressure_x_recent_form: championship pressure × recent form
df['pressure_x_recent_form_computed'] = (
    safe('championship_position_pressure_factor', 0.5) * safe('recent_form_3_races', 10.0)
)

# constructor_reliability_x_form: (1 - dnf_rate) × recent form
dnf_rate = safe('constructor_dnf_rate_3_races', 0.0)
df['constructor_reliability_x_form_computed'] = (
    (1.0 - dnf_rate) * safe('recent_form_3_races', 10.0)
)

# wet_skill_x_precip: wet skill (fixed) × precipitation
df['wet_skill_x_precip_computed'] = (
    df['wet_race_vs_quali_delta_fixed'].fillna(0.0) * safe('total_precipitation', 0.0)
)

# track_exp_x_qual: track experience × qualifying position
df['track_exp_x_qual_computed'] = (
    safe('track_experience', 0.0) * safe('resultsQualificationPositionNumber', 10.0)
)

interaction_cols = [
    'tire_mgmt_x_turns_computed', 'practice_conversion_x_grid_computed',
    'pressure_x_recent_form_computed', 'constructor_reliability_x_form_computed',
    'wet_skill_x_precip_computed', 'track_exp_x_qual_computed',
]
for col in interaction_cols:
    cov = df[col].notna().mean()
    print(f"   {col:45s}: {cov:.1%} coverage")

# ════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE SETS
# ════════════════════════════════════════════════════════════════════════════

OLD_FEATURES = [
    'resultsDriverName', 'constructorName', 'resultsStartingGridPositionNumber',
    'lastFPPositionNumber', 'resultsQualificationPositionNumber', 'grandPrixLaps',
    'constructorTotalRaceStarts', 'activeDriver', 'recent_form_5_races_bin',
    'yearsActive', 'driverDNFAvg', 'best_s1_sec_bin', 'LapTime_sec_bin',
    'SpeedI2_mph_bin', 'SpeedST_mph_bin', 'constructor_recent_form_3_races_bin',
    'constructor_recent_form_5_races_bin', 'CleanAirAvg_FP1_bin', 'Delta_FP1_bin',
    'DirtyAirAvg_FP2_bin', 'Delta_FP2_bin', 'Delta_FP3_bin', 'engineManufacturerId',
    'delta_from_race_avg_bin', 'driverAge', 'finishing_position_std_driver',
    'finishing_position_std_constructor', 'delta_lap_2_historical',
    'delta_lap_10_historical', 'delta_lap_15_historical', 'delta_lap_20_historical',
    'driver_dnf_rate_5_races', 'avg_final_position_per_track_bin',
    'last_final_position_per_track', 'avg_final_position_per_track_constructor_bin',
    'practice_position_improvement_1P_2P', 'practice_position_improvement_2P_3P',
    'practice_position_improvement_1P_3P', 'practice_time_improvement_1T_2T_bin',
    'practice_time_improvement_time_time_bin', 'teammate_practice_delta_bin',
    'last_final_position_per_track_constructor', 'driver_starting_position_3_races_bin',
    'qualPos_x_last_practicePos_bin', 'qualPos_x_avg_practicePos_bin',
    'recent_form_median_3_races', 'recent_form_median_5_races',
    'recent_form_worst_3_races', 'recent_positions_gained_3_races_bin',
    'driver_positionsGained_3_races_bin', 'qual_vs_track_avg_bin',
    'constructor_avg_practice_position_bin', 'practice_position_std_bin',
    'recent_vs_season_bin', 'practice_improvement', 'qual_x_constructor_wins_bin',
    'grid_penalty', 'grid_penalty_x_constructor_bin', 'recent_form_x_qual_bin',
    'driver_rank_x_constructor_rank', 'practice_gap_to_teammate_bin',
    'street_experience', 'fp1_lap_delta_vs_best_bin', 'last_race_vs_track_avg_bin',
    'top_speed_rank_bin', 'historical_avgLapPace_bin', 'pit_delta_x_driver_age_bin',
    'constructor_points_x_grid_bin', 'dnf_rate_x_practice_std_bin',
    'grid_penalty_x_constructor_rank', 'constructor_win_rate_3y', 'driver_podium_rate_3y',
    'track_familiarity', 'recent_podium_streak', 'grid_position_percentile_bin',
    'qual_to_final_delta_5yr_bin', 'qual_to_final_delta_3yr_bin',
    'overtake_potential_3yr_bin', 'overtake_potential_5yr_bin',
    'constructor_avg_qual_pos_at_track_bin', 'driver_avg_grid_pos_at_track_bin',
    'driver_avg_practice_pos_at_track_bin', 'constructor_avg_practice_pos_at_track_bin',
    'constructor_qual_improvement_3r_bin', 'constructor_practice_improvement_3r_bin',
    'driver_teammate_qual_gap_3r_bin', 'driver_teammate_practice_gap_3r_bin',
    'driver_street_qual_avg', 'driver_track_qual_avg', 'driver_high_wind_qual_avg',
    'driver_high_humidity_qual_avg', 'driver_wet_qual_avg', 'driver_safetycar_qual_avg',
    'driver_safetycar_practice_avg', 'races_with_constructor_bin',
    'driver_constructor_avg_final_position_bin', 'constructor_dnf_rate_3_races',
    'constructor_dnf_rate_5_races', 'historical_race_pace_vs_median_bin',
    'practice_consistency_vs_teammate_bin', 'fp3_position_percentile_bin',
    'constructor_practice_improvement_rate_bin', 'track_fp1_fp3_improvement_bin',
    'teammate_practice_delta_at_track_bin', 'qual_vs_track_median',
    'qual_improvement_vs_field_avg_bin', 'driver_podium_rate_at_track',
    'fp3_vs_constructor_avg_bin', 'qual_vs_constructor_avg_bin',
    'practice_lap_time_consistency_bin', 'qual_lap_time_consistency_bin',
    'practice_improvement_vs_teammate_bin', 'qual_improvement_vs_teammate_bin',
    'practice_vs_best_at_track_bin', 'qual_vs_best_at_track', 'qual_vs_worst_at_track',
    'practice_position_percentile_vs_constructor_bin',
    'qualifying_position_percentile_vs_constructor_bin',
    'practice_lap_time_delta_to_constructor_best_bin',
    'qualifying_lap_time_delta_to_constructor_best_bin',
    'qualifying_position_vs_field_best_at_track',
    'practice_position_vs_field_worst_at_track_bin',
    'qualifying_position_vs_field_worst_at_track',
    'qualifying_position_vs_field_median_at_track',
    'practice_position_vs_constructor_best_at_track_bin',
    'qualifying_position_vs_constructor_best_at_track',
    'qualifying_position_vs_constructor_worst_at_track',
    'practice_position_vs_constructor_median_at_track_bin',
    'practice_lap_time_consistency_vs_field_bin',
    'qualifying_lap_time_consistency_vs_field_bin',
    'practice_position_vs_field_recent_form_bin',
    'qualifying_position_vs_field_recent_form_bin',
    'podium_form_3_races', 'wins_last_5_races', 'championship_position',
    'points_leader_gap', 'pole_to_win_rate', 'front_row_conversion',
    'recent_wins_3_races', 'rolling_3_race_win_percentage',
    'recent_qualifying_improvement_trend', 'head_to_head_teammate_performance_delta',
    'championship_position_pressure_factor', 'constructor_recent_mechanical_dnf_rate',
    'driver_performance_at_circuit_type', 'weather_pattern_analysis_by_location',
    'overtaking_difficulty_index', 'q1_q2_q3_sector_consistency',
    'qualifying_position_vs_race_pace_delta_by_track',
    'practice_race_conversion', 'avg_positions_gained_5r', 'race_pace_consistency',
    'overtaking_success_top10', 'tire_management_score',
]

# NEW features = OLD + inline-computed fixes + interaction columns
ROADMAP1_ADDITIONS = [
    # bug-fixed versions
    'wet_race_vs_quali_delta_fixed',
    'championship_fight_performance_fixed',
    # existing columns now being actively used
    'practice_to_qual_improvement_rate',
    'q3_lap_time_delta_to_pole',
    'constructor_podium_rate_at_track',
    'is_first_season_with_constructor',
    'driver_constructor_avg_qual_position',
    'driver_constructor_synergy',
    # interaction features (inline-computed)
    'tire_mgmt_x_turns_computed',
    'practice_conversion_x_grid_computed',
    'pressure_x_recent_form_computed',
    'constructor_reliability_x_form_computed',
    'wet_skill_x_precip_computed',
    'track_exp_x_qual_computed',
]

NEW_FEATURES = OLD_FEATURES + ROADMAP1_ADDITIONS

# filter to only what's in the dataframe
available_old = [f for f in OLD_FEATURES if f in df.columns]
available_new = [f for f in NEW_FEATURES if f in df.columns]
print(f"\nOLD feature set : {len(OLD_FEATURES)} defined → {len(available_old)} present in CSV")
print(f"NEW feature set : {len(NEW_FEATURES)} defined → {len(available_new)} present in CSV")
missing_new = [f for f in ROADMAP1_ADDITIONS if f not in df.columns]
if missing_new:
    print(f"  Still missing (need generator re-run): {missing_new}")

# ════════════════════════════════════════════════════════════════════════════
# 3.  SKLEARN PREPROCESSOR (mirrors raceAnalysis.py)
# ════════════════════════════════════════════════════════════════════════════
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

def make_preprocessor(feature_cols, df):
    num_cols = [c for c in feature_cols if is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not is_numeric_dtype(df[c])]
    transformers = [
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
        ]), num_cols)
    ]
    if cat_cols:
        transformers.append(('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
        ]), cat_cols))
    return ColumnTransformer(transformers)

# ════════════════════════════════════════════════════════════════════════════
# 4.  MODEL EVALUATION — exact params from raceAnalysis.py
# ════════════════════════════════════════════════════════════════════════════
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GroupKFold

def sample_weights(y):
    return np.where(y == 1, 2.0,
           np.where(y <= 3, 1.5,
           np.where(y <= 10, 1.2, 1.0)))

def evaluate_80_20(df, feature_cols, label=""):
    """Evaluate using exact app split: 80/20 random, random_state=42."""
    X = df[feature_cols]
    y = df[target_col]
    preprocessor = make_preprocessor(feature_cols, df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    sw = sample_weights(y_train.values)

    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=20,
        eval_metric='mae',
    )
    model.fit(Xtr, y_train, sample_weight=sw,
              eval_set=[(Xte, y_test)], verbose=False)

    y_pred = model.predict(Xte)
    mae = mean_absolute_error(y_test, y_pred)
    rounds = getattr(model, 'best_iteration', model.n_estimators) + 1
    print(f"  [{label}]  MAE={mae:.4f}  rounds={rounds}  features={len(feature_cols)}")
    return mae

def evaluate_groupkfold(df, feature_cols, label="", n_splits=5):
    """Evaluate using 5-fold GroupKFold by season (leakage-free reference)."""
    year_col = next((c for c in ['grandPrixYear', 'year'] if c in df.columns), None)
    if year_col is None:
        return None

    X = df[feature_cols].values
    y = df[target_col].values
    groups = df[year_col].values

    preprocessor = make_preprocessor(feature_cols, df)
    gkf = GroupKFold(n_splits=n_splits)
    maes = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr_raw = df[feature_cols].iloc[train_idx]
        X_te_raw = df[feature_cols].iloc[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]

        pre = make_preprocessor(feature_cols, df.iloc[train_idx])
        Xtr = pre.fit_transform(X_tr_raw)
        Xte = pre.transform(X_te_raw)
        sw = sample_weights(y_tr)

        model = XGBRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=4,
            random_state=42, n_jobs=-1,
            early_stopping_rounds=20, eval_metric='mae',
        )
        model.fit(Xtr, y_tr, sample_weight=sw,
                  eval_set=[(Xte, y_te)], verbose=False)
        maes.append(mean_absolute_error(y_te, model.predict(Xte)))

    mean_mae = np.mean(maes)
    std_mae = np.std(maes)
    print(f"  [{label}]  MAE={mean_mae:.4f} ± {std_mae:.4f}  (GroupKFold {n_splits}-fold)")
    return mean_mae

# ════════════════════════════════════════════════════════════════════════════
# 5.  RUN EVALUATIONS
# ════════════════════════════════════════════════════════════════════════════
print("\n── Method A: 80/20 random split (matches the app's 2.08 baseline) ──")
mae_old_80 = evaluate_80_20(df, available_old, label="BASELINE (old features)")
mae_new_80 = evaluate_80_20(df, available_new, label="ROADMAP-1  (new features)")

print("\n── Method B: 5-fold GroupKFold by season (leakage-free reference) ──")
mae_old_gkf = evaluate_groupkfold(df, available_old, label="BASELINE (old features)")
mae_new_gkf = evaluate_groupkfold(df, available_new, label="ROADMAP-1  (new features)")

# ════════════════════════════════════════════════════════════════════════════
# 6.  SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Metric':<40} {'BASELINE':>10} {'ROADMAP-1':>10} {'DELTA':>10}")
print("-" * 70)
if mae_old_80 and mae_new_80:
    delta_80 = mae_new_80 - mae_old_80
    sign = "▼ BETTER" if delta_80 < 0 else "▲ WORSE"
    print(f"{'MAE (80/20 random split)':<40} {mae_old_80:>10.4f} {mae_new_80:>10.4f} {delta_80:>+10.4f}  {sign}")
if mae_old_gkf and mae_new_gkf:
    delta_gkf = mae_new_gkf - mae_old_gkf
    sign = "▼ BETTER" if delta_gkf < 0 else "▲ WORSE"
    print(f"{'MAE (GroupKFold by season)':<40} {mae_old_gkf:>10.4f} {mae_new_gkf:>10.4f} {delta_gkf:>+10.4f}  {sign}")
print("-" * 70)
print(f"Feature count old → new: {len(available_old)} → {len(available_new)}")
print()
if mae_new_80:
    pct_to_target = (1.5 - mae_new_80) / (1.5 - mae_old_80) * 100 if mae_old_80 > 1.5 else 100.0
    remaining = max(0, mae_new_80 - 1.5)
    print(f"Target MAE:    ≤ 1.50")
    print(f"Current best:  {mae_new_80:.4f}")
    print(f"Gap to target: {remaining:.4f}")
    if remaining > 0:
        print(f"Still need further improvement of {remaining:.4f} to reach target.")
    else:
        print("TARGET REACHED! ✓")

print()
print("Note: 80/20 random split can include future races in training set (data leakage).")
print("      GroupKFold by season is a stricter, more realistic evaluation.")
print("      The 2.08 figure reported in the app was measured with the 80/20 split.")
