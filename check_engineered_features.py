import pickle

model_path = 'data_files/models/position_model.pkl'
with open(model_path, 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_names = model_data.get('feature_names', [])

print(f'Total features: {len(feature_names)}')

# Check for engineered features
eng_features = [f for f in feature_names if any(x in f for x in [
    'sector_consistency_score',
    'sector_balance', 
    'theoretical_gap_x_deleted_laps',
    'consistency_x_mistakes'
])]

print(f'\nEngineered qualifying features in model: {len(eng_features)}')
for ef in eng_features:
    print(f'  - {ef}')

if eng_features and hasattr(model, 'get_score'):
    importances = model.get_score(importance_type='weight')
    print(f'\nImportance scores:')
    for ef in eng_features:
        imp = importances.get(ef, 0)
        print(f'  {ef}: {imp}')
