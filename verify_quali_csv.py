"""Verify enriched qualifying CSV has correct structure"""
import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')

print(f"Total rows: {len(df)}")
print(f"\nKey columns present:")

key_cols = ['driverId', 'constructorId', 'constructorName', 'Q1_sec', 'Q2_sec', 'Q3_sec', 
            'best_sector1_sec', 'theoretical_best_lap', 'raceId']

for col in key_cols:
    exists = col in df.columns
    if exists:
        non_null = df[col].notna().sum()
        print(f"  ✓ {col}: {non_null:,} non-null values")
    else:
        print(f"  ✗ {col}: MISSING")

print(f"\nSample rows:")
sample_cols = ['Year', 'Round', 'driverId', 'constructorName', 'Q1_sec', 'best_sector1_sec']
print(df[sample_cols].head(3).to_string())

print(f"\n2018-2025 coverage:")
coverage = df.groupby('Year').size()
for year, count in coverage.items():
    print(f"  {year}: {count} records")
