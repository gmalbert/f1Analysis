"""
Debug: Check if lap-level qualifying features are in the data loaded for training
"""
import os
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from raceAnalysis import load_data, CACHE_VERSION

print("Loading data...")
data, _ = load_data(10000, CACHE_VERSION)

print(f"Data shape: {data.shape}")
print(f"Columns: {len(data.columns)}\n")

# Check lap-level qualifying features
lap_level_features = [
    'best_sector1_sec', 'best_sector2_sec', 'best_sector3_sec',
    'theoretical_best_lap', 'actual_best_lap',
    'lap_time_std', 'sector1_std', 'sector2_std', 'sector3_std',
    'avg_sector1_sec', 'avg_sector2_sec', 'avg_sector3_sec',
    'primary_compound', 'theoretical_gap',
    'total_qualifying_laps', 'valid_laps', 'deleted_laps'
]

print("=== LAP-LEVEL QUALIFYING FEATURES IN DATA ===")
for feat in lap_level_features:
    if feat in data.columns:
        non_null = data[feat].notna().sum()
        pct = (non_null / len(data)) * 100
        print(f"✓ {feat:30s} - {non_null:5d} rows ({pct:5.1f}%)")
    else:
        print(f"✗ {feat:30s} - NOT IN DATA")

# Check if they're in the feature list
print("\n=== CHECKING FEATURES LOADED FROM FILE ===")
from raceAnalysis import load_f1_position_model_features
numerical_features, categorical_features = load_f1_position_model_features()
print(f"Loaded {len(numerical_features)} numerical features from file")
print(f"Loaded {len(categorical_features)} categorical features from file")

lap_in_numerical = [f for f in lap_level_features if f in numerical_features]
print(f"\nLap-level features in numerical_features: {len(lap_in_numerical)}")
if lap_in_numerical:
    for f in lap_in_numerical:
        print(f"  - {f}")

lap_in_categorical = [f for f in lap_level_features if f in categorical_features]
print(f"\nLap-level features in categorical_features: {len(lap_in_categorical)}")
if lap_in_categorical:
    for f in lap_in_categorical:
        print(f"  - {f}")
