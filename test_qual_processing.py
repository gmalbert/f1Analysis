"""
Quick test to verify qualifying data processing works correctly
"""
import pandas as pd
import numpy as np
from os import path

DATA_DIR = 'data_files/'

# Test the time conversion function
def time_str_to_seconds(time_str):
    """Convert qualifying time string (e.g., '1:21.164') to seconds float."""
    if pd.isna(time_str) or time_str is None:
        return np.nan
    try:
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                minutes = float(parts[0])
                seconds = float(parts[1])
                return minutes * 60 + seconds
        return float(time_str)
    except Exception:
        return np.nan

# Load f1db qualifying JSON
qualifying_json = pd.read_json(path.join(DATA_DIR, 'f1db-races-qualifying-results.json'))

print("Testing qualifying data processing...")
print(f"Total records: {len(qualifying_json)}")

# Process times
qualifying = qualifying_json.copy()
qualifying['q1_sec'] = qualifying['q1'].apply(time_str_to_seconds)
qualifying['q2_sec'] = qualifying['q2'].apply(time_str_to_seconds)
qualifying['q3_sec'] = qualifying['q3'].apply(time_str_to_seconds)
qualifying['best_qual_time'] = qualifying[['q1_sec', 'q2_sec', 'q3_sec']].min(axis=1)

print(f"\nPopulated fields:")
print(f"  q1_sec: {qualifying['q1_sec'].notna().sum()} non-null ({qualifying['q1_sec'].notna().sum()/len(qualifying)*100:.1f}%)")
print(f"  q2_sec: {qualifying['q2_sec'].notna().sum()} non-null ({qualifying['q2_sec'].notna().sum()/len(qualifying)*100:.1f}%)")
print(f"  q3_sec: {qualifying['q3_sec'].notna().sum()} non-null ({qualifying['q3_sec'].notna().sum()/len(qualifying)*100:.1f}%)")
print(f"  best_qual_time: {qualifying['best_qual_time'].notna().sum()} non-null ({qualifying['best_qual_time'].notna().sum()/len(qualifying)*100:.1f}%)")

print(f"\nSample data (2024):")
sample = qualifying[qualifying['year'] == 2024].head(5)
print(sample[['year', 'round', 'driverId', 'q1', 'q1_sec', 'q2_sec', 'q3_sec', 'best_qual_time']])

print("\n✅ Qualifying data processing successful!")
