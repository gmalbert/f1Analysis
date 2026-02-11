import pandas as pd

df = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t')

features = [
    'wet_race_vs_quali_delta',
    'championship_fight_performance',
    'practice_race_conversion',
    'race_pace_consistency',
    'tire_management_score'
]

print('Feature Coverage Analysis:')
print('=' * 60)
for feat in features:
    if feat in df.columns:
        count = df[feat].notna().sum()
        pct = df[feat].notna().mean() * 100
        print(f'{feat:40s} {count:5d}/{len(df)} ({pct:5.1f}%)')
    else:
        print(f'{feat:40s} NOT FOUND')
