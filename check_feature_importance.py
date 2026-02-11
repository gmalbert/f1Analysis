"""
Check feature importance for newly added lap-level qualifying features
"""
import pickle
import pandas as pd
from pathlib import Path

# Load the trained model
model_path = Path("data_files/models/position_model.pkl")
if not model_path.exists():
    print(f"Model not found at {model_path}")
    exit(1)

with open(model_path, 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    preprocessor = model_data['preprocessor']

# Get feature names after preprocessing
feature_names = []
for name, _, cols in preprocessor.transformers:
    if name == 'num':
        feature_names.extend(cols)
    elif name == 'cat':
        # For categorical features, we'd need to get the encoded names
        # but for now just track the original column names
        feature_names.extend(cols)

print(f"Total features: {len(feature_names)}\n")

# Get feature importance from XGBoost
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Check lap-level qualifying features
    lap_level_features = [
        'best_sector1_sec', 'best_sector2_sec', 'best_sector3_sec',
        'theoretical_best_lap', 'actual_best_lap',
        'lap_time_std', 'sector1_std', 'sector2_std', 'sector3_std',
        'avg_sector1_sec', 'avg_sector2_sec', 'avg_sector3_sec',
        'primary_compound', 'theoretical_gap',
        'total_qualifying_laps', 'valid_laps', 'deleted_laps'
    ]
    
    print("=== LAP-LEVEL QUALIFYING FEATURE IMPORTANCE ===")
    lap_level_importance = feature_importance_df[feature_importance_df['feature'].isin(lap_level_features)]
    
    if len(lap_level_importance) > 0:
        print(f"\n{len(lap_level_importance)} lap-level features found in model:\n")
        print(lap_level_importance.to_string(index=False))
        print(f"\nMean importance: {lap_level_importance['importance'].mean():.6f}")
        print(f"Max importance: {lap_level_importance['importance'].max():.6f}")
        print(f"\nTop-ranked lap-level feature: {lap_level_importance.iloc[0]['feature']} (rank #{feature_importance_df[feature_importance_df['feature'] == lap_level_importance.iloc[0]['feature']].index[0] + 1})")
    else:
        print("\n[WARNING] No lap-level qualifying features found in trained model!")
        print("This means they were filtered out during preprocessing or not in the data.")
    
    print("\n=== TOP 20 MOST IMPORTANT FEATURES OVERALL ===")
    print(feature_importance_df.head(20).to_string(index=False))
    
else:
    print("Model does not have feature_importances_ attribute")
