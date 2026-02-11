import pickle
import pandas as pd

model_path = 'data_files/models/position_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
preprocessor = model_data.get('preprocessor')

# Get feature names from preprocessor
num_features = preprocessor.transformers_[0][2] if preprocessor else []

print(f'Total numerical features: {len(num_features)}')

# Get importance scores
import xgboost as xgb
import numpy as np

if isinstance(model, xgb.Booster):
    # XGBoost Booster format
    importances_dict = model.get_score(importance_type='weight')
elif hasattr(model, 'feature_importances_'):
    # sklearn-style model
    importances_dict = dict(zip(num_features, model.feature_importances_))
else:
    print("Unable to extract feature importances")
    importances_dict = {}

if importances_dict:
    
    # Convert to DataFrame
    importances_df = pd.DataFrame(list(importances_dict.items()), columns=['feature', 'importance'])
    importances_df = importances_df.sort_values('importance', ascending=False)
    
    # Filter for engineered features  
    engineered = importances_df[importances_df['feature'].str.contains('sector_consistency|sector_balance|theoretical_gap_x|consistency_x_mistakes', regex=True)]
    
    print(f'\n=== ENGINEERED QUALIFYING FEATURES ===')
    print(engineered.to_string(index=False))
    
    if len(engineered) > 0:
        print(f'\nMean importance: {engineered["importance"].mean():.6f}')
        print(f'Max importance: {engineered["importance"].max():.6f}')
        
        # Find rank of top engineered feature
        top_eng_feature = engineered.iloc[0]['feature']
        rank = importances_df[importances_df['feature'] == top_eng_feature].index[0] + 1
        print(f'Top-ranked engineered feature: {top_eng_feature} (rank #{rank})')
    
    print(f'\n=== COMPARISON: LAP-LEVEL BASE FEATURES ===')
    lap_level = importances_df[importances_df['feature'].str.contains('best_sector|theoretical_best_lap|lap_time_std|sector._std|deleted_laps', regex=True)]
    print(lap_level.head(10).to_string(index=False))
    
