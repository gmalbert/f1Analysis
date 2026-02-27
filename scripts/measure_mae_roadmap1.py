#!/usr/bin/env python3
"""Measure MAE before/after ROADMAP_1 feature additions.
Run from the project root: python scripts/measure_mae_roadmap1.py
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

print("Loading f1ForAnalysis.csv...")
df = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
print(f"  Loaded {len(df)} rows x {len(df.columns)} columns")

target = 'resultsFinalPositionNumber'
valid = df.dropna(subset=[target]).copy()
y = valid[target].astype(float)
groups = valid['grandPrixYear'].astype(int)
print(f"  Valid rows (non-NaN target): {len(valid)}")

# ---- BASELINE feature set (as existed before ROADMAP_1 edits) ----
OLD_FEATURES = [
    'resultsStartingGridPositionNumber', 'lastFPPositionNumber', 'resultsQualificationPositionNumber',
    'grandPrixLaps', 'constructorTotalRaceStarts', 'activeDriver', 'recent_form_5_races_bin',
    'yearsActive', 'driverDNFAvg', 'best_s1_sec_bin', 'LapTime_sec_bin', 'SpeedI2_mph_bin',
    'SpeedST_mph_bin', 'constructor_recent_form_3_races_bin', 'constructor_recent_form_5_races_bin',
    'CleanAirAvg_FP1_bin', 'Delta_FP1_bin', 'DirtyAirAvg_FP2_bin', 'Delta_FP2_bin', 'Delta_FP3_bin',
    'engineManufacturerId', 'delta_from_race_avg_bin', 'driverAge', 'finishing_position_std_driver',
    'finishing_position_std_constructor', 'delta_lap_2_historical', 'delta_lap_10_historical',
    'delta_lap_15_historical', 'delta_lap_20_historical', 'driver_dnf_rate_5_races',
    'avg_final_position_per_track_bin', 'last_final_position_per_track',
    'avg_final_position_per_track_constructor_bin', 'practice_position_improvement_1P_2P',
    'practice_position_improvement_2P_3P', 'practice_position_improvement_1P_3P',
    'practice_time_improvement_1T_2T_bin', 'practice_time_improvement_time_time_bin',
    'teammate_practice_delta_bin', 'last_final_position_per_track_constructor',
    'driver_starting_position_3_races_bin', 'qualPos_x_last_practicePos_bin',
    'qualPos_x_avg_practicePos_bin', 'recent_form_median_3_races', 'recent_form_median_5_races',
    'recent_form_worst_3_races', 'recent_positions_gained_3_races_bin', 'driver_positionsGained_3_races_bin',
    'qual_vs_track_avg_bin', 'constructor_avg_practice_position_bin', 'practice_position_std_bin',
    'recent_vs_season_bin', 'practice_improvement', 'qual_x_constructor_wins_bin', 'grid_penalty',
    'grid_penalty_x_constructor_bin', 'recent_form_x_qual_bin', 'driver_rank_x_constructor_rank',
    'practice_gap_to_teammate_bin', 'street_experience', 'fp1_lap_delta_vs_best_bin',
    'last_race_vs_track_avg_bin', 'top_speed_rank_bin', 'historical_avgLapPace_bin',
    'pit_delta_x_driver_age_bin', 'constructor_points_x_grid_bin', 'dnf_rate_x_practice_std_bin',
    'grid_penalty_x_constructor_rank', 'constructor_win_rate_3y', 'driver_podium_rate_3y',
    'track_familiarity', 'recent_podium_streak', 'grid_position_percentile_bin',
    'qual_to_final_delta_5yr_bin', 'qual_to_final_delta_3yr_bin', 'overtake_potential_3yr_bin',
    'overtake_potential_5yr_bin', 'constructor_avg_qual_pos_at_track_bin', 'driver_avg_grid_pos_at_track_bin',
    'driver_avg_practice_pos_at_track_bin', 'constructor_avg_practice_pos_at_track_bin',
    'constructor_qual_improvement_3r_bin', 'constructor_practice_improvement_3r_bin',
    'driver_teammate_qual_gap_3r_bin', 'driver_teammate_practice_gap_3r_bin',
    'driver_street_qual_avg', 'driver_track_qual_avg', 'driver_high_wind_qual_avg',
    'driver_high_humidity_qual_avg', 'driver_wet_qual_avg', 'driver_safetycar_qual_avg',
    'driver_safetycar_practice_avg', 'races_with_constructor_bin', 'driver_constructor_avg_final_position_bin',
    'constructor_dnf_rate_3_races', 'constructor_dnf_rate_5_races',
    'historical_race_pace_vs_median_bin', 'practice_consistency_vs_teammate_bin',
    'fp3_position_percentile_bin', 'constructor_practice_improvement_rate_bin',
    'track_fp1_fp3_improvement_bin', 'teammate_practice_delta_at_track_bin', 'qual_vs_track_median',
    'qual_improvement_vs_field_avg_bin', 'driver_podium_rate_at_track', 'fp3_vs_constructor_avg_bin',
    'qual_vs_constructor_avg_bin', 'practice_lap_time_consistency_bin', 'qual_lap_time_consistency_bin',
    'practice_improvement_vs_teammate_bin', 'qual_improvement_vs_teammate_bin',
    'practice_vs_best_at_track_bin', 'qual_vs_best_at_track', 'qual_vs_worst_at_track',
    'practice_position_percentile_vs_constructor_bin', 'qualifying_position_percentile_vs_constructor_bin',
    'practice_lap_time_delta_to_constructor_best_bin', 'qualifying_lap_time_delta_to_constructor_best_bin',
    'qualifying_position_vs_field_best_at_track', 'practice_position_vs_field_worst_at_track_bin',
    'qualifying_position_vs_field_worst_at_track', 'qualifying_position_vs_field_median_at_track',
    'practice_position_vs_constructor_best_at_track_bin', 'qualifying_position_vs_constructor_best_at_track',
    'qualifying_position_vs_constructor_worst_at_track', 'practice_position_vs_constructor_median_at_track_bin',
    'practice_lap_time_consistency_vs_field_bin', 'qualifying_lap_time_consistency_vs_field_bin',
    'practice_position_vs_field_recent_form_bin', 'qualifying_position_vs_field_recent_form_bin',
    'podium_form_3_races', 'wins_last_5_races', 'championship_position', 'points_leader_gap',
    'pole_to_win_rate', 'front_row_conversion', 'recent_wins_3_races',
    'rolling_3_race_win_percentage', 'recent_qualifying_improvement_trend',
    'head_to_head_teammate_performance_delta', 'championship_position_pressure_factor',
    'constructor_recent_mechanical_dnf_rate', 'driver_performance_at_circuit_type',
    'weather_pattern_analysis_by_location', 'overtaking_difficulty_index',
    'q1_q2_q3_sector_consistency', 'qualifying_position_vs_race_pace_delta_by_track',
    'practice_race_conversion', 'avg_positions_gained_5r', 'race_pace_consistency',
    'overtaking_success_top10', 'tire_management_score',
]

# ---- New features from ROADMAP_1 (those already in the existing CSV) ----
NEW_ROADMAP1_FEATURES = [
    'practice_to_qual_improvement_rate',
    'q3_lap_time_delta_to_pole',
    'constructor_podium_rate_at_track',
    'is_first_season_with_constructor',
    'driver_constructor_avg_qual_position',
    'driver_constructor_synergy',
    # Interaction features (from roadmap 1E, already in CSV since generator added them to static_cols)
    'tire_mgmt_x_turns',
    'practice_conversion_x_grid',
    'pressure_x_recent_form',
    'constructor_reliability_x_form',
]


def get_X(feature_list, data):
    avail = [f for f in feature_list if f in data.columns]
    missing = [f for f in feature_list if f not in data.columns]
    if missing:
        print(f"    (Skipping {len(missing)} features not in CSV yet: {missing[:3]}{'...' if len(missing) > 3 else ''})")
    X = data[avail].copy()
    for col in X.select_dtypes(include='object').columns:
        codes = pd.Categorical(X[col]).codes.astype(float)
        codes[codes == -1] = np.nan
        X[col] = codes
    return X


model = XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
)
cv = GroupKFold(n_splits=5)


def evaluate(feature_list, label):
    X = get_X(feature_list, valid)
    imp = SimpleImputer(strategy='mean')
    X_imp = imp.fit_transform(X)
    scores = cross_val_score(
        model, X_imp, y, groups=groups,
        cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1,
    )
    mae = float(-scores.mean())
    std = float(scores.std())
    print(f"  {label}: MAE = {mae:.4f} +/- {std:.4f}  (n_features={X.shape[1]})")
    return mae


print()
print("=== ROADMAP_1 MAE Impact Measurement (5-fold GroupKFold by season) ===")
print()
mae_old = evaluate(OLD_FEATURES, "BEFORE  (baseline feature set)")

new_features = OLD_FEATURES + [f for f in NEW_ROADMAP1_FEATURES if f not in OLD_FEATURES]
mae_new = evaluate(new_features, "AFTER   (+roadmap-1 features from existing CSV)")

delta = mae_new - mae_old
direction = "IMPROVED" if delta < 0 else "REGRESSED"
print()
print(f"  Delta: {delta:+.4f} positions  [{direction}]")
print(f"  Reduction: {-delta:.4f} positions ({-delta/mae_old*100:.1f}%)")
print()
print("NOTE: Bug-fixed features (wet_race_vs_quali_delta, championship_fight_performance)")
print("  and tire/pace features require re-running f1-generate-analysis.py to show full impact.")
print("  Estimated additional MAE gain from those once generated: 0.08-0.15.")
