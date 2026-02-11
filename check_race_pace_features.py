import pickle
import pandas as pd
import xgboost as xgb

model_path = 'data_files/models/position_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
preprocessor = model_data.get('preprocessor')

# Get feature names
num_features = preprocessor.transformers_[0][2] if preprocessor else []

print(f'Total features: {len(num_features)}')

# Get importance
if isinstance(model, xgb.Booster):
    importances_dict = model.get_score(importance_type='weight')
else:
    importances_dict = dict(zip(num_features, model.feature_importances_))

imp_df = pd.DataFrame(list(importances_dict.items()), columns=['feature', 'importance'])
imp_df = imp_df.sort_values('importance', ascending=False)

# Race pace features
race_pace_features = [
    'practice_race_conversion',
    'avg_positions_gained_5r', 
    'race_pace_consistency',
    'overtaking_success_top10',
    'tire_management_score',
    # Binned versions
    'practice_race_conversion_bin',
    'avg_positions_gained_5r_bin',
    'race_pace_consistency_bin',
    'overtaking_success_top10_bin',
    'tire_management_score_bin'
]

print('\n=== RACE PACE/STRATEGY FEATURES ===')
race_feats = imp_df[imp_df['feature'].isin(race_pace_features)]
if len(race_feats) > 0:
    print(race_feats.to_string(index=False))
    print(f'\nMean importance: {race_feats["importance"].mean():.6f}')
    print(f'Max importance: {race_feats["importance"].max():.6f}')
    
    # Find rank
    top = race_feats.iloc[0]
    rank = imp_df[imp_df['feature'] == top['feature']].index[0] + 1
    print(f'Top-ranked race pace feature: {top["feature"]} (rank #{rank}, importance: {top["importance"]:.6f})')
else:
    print('No race pace features found in model!')

print('\n=== TOP 15 FEATURES OVERALL ===')
print(imp_df.head(15).to_string(index=False))

# Compare with lap-level qualifying
print('\n=== LAP-LEVEL QUALIFYING FEATURES (for comparison) ===')
lap_level = imp_df[imp_df['feature'].str.contains('best_sector|theoretical_best_lap|lap_time_std|sector._std', regex=True, na=False)]
if len(lap_level) > 0:
    print(lap_level.head(10).to_string(index=False))
    print(f'\nMean importance: {lap_level["importance"].mean():.6f}')
else:
    print('No lap-level features found')
