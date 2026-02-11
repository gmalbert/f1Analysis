import pandas as pd
import numpy as np

df = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t')

features = [
    'practice_race_conversion',
    'avg_positions_gained_5r',
    'race_pace_consistency',
    'overtaking_success_top10',
    'tire_management_score'
]

print('Race Pace Feature Distributions')
print('=' * 80)

for feat in features:
    if feat not in df.columns:
        print(f'\n{feat}: NOT FOUND')
        continue
    
    data = df[feat].dropna()
    if len(data) == 0:
        print(f'\n{feat}: NO DATA')
        continue
    
    print(f'\n{feat}:')
    print(f'  Count: {len(data):,} / {len(df):,} ({len(data)/len(df)*100:.1f}%)')
    print(f'  Min: {data.min():.3f}')
    print(f'  25%: {data.quantile(0.25):.3f}')
    print(f'  50%: {data.quantile(0.50):.3f}')
    print(f'  75%: {data.quantile(0.75):.3f}')
    print(f'  Max: {data.max():.3f}')
    print(f'  Mean: {data.mean():.3f}')
    print(f'  Std: {data.std():.3f}')
    
    # Suggest bins
    print(f'  Suggested bins:')
    if data.min() < 0:
        # Symmetric around 0
        q25, q75 = data.quantile(0.25), data.quantile(0.75)
        print(f'    [{data.min():.1f}, {q25:.1f}, 0, {abs(q25):.1f}, {data.max():.1f}]')
    else:
        # Positive only
        q33, q67 = data.quantile(0.33), data.quantile(0.67)
        print(f'    [0, {q33:.1f}, {q67:.1f}, {data.max():.1f}]')
