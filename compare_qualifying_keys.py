"""Compare raceId/driverId between f1db qualifying JSON and all_qualifying_races.csv"""
import pandas as pd
from os import path

DATA_DIR = 'data_files'

# Load qualifying from f1db JSON (what the generator uses)
qualifying_json = pd.read_json(path.join(DATA_DIR, 'f1db-races-qualifying-results.json'))
print(f"f1db-races-qualifying-results.json: {len(qualifying_json)} rows")
print(f"Sample JSON data:")
print(qualifying_json[['raceId', 'driverId', 'positionNumber']].head())
print(f"\nJSON raceId dtype: {qualifying_json['raceId'].dtype}")
print(f"JSON driverId dtype: {qualifying_json['driverId'].dtype}")
print(f"Sample driverId values: {qualifying_json['driverId'].head().tolist()}")

# Load qualifying from CSV (FastF1 enhanced)
qualifying_csv = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')
print(f"\n\nall_qualifying_races.csv: {len(qualifying_csv)} rows")
print(f"Sample CSV data:")
print(qualifying_csv[['Year', 'Round', 'raceId', 'driverId', 'best_sector1_sec']].head())
print(f"\nCSV raceId dtype: {qualifying_csv['raceId'].dtype}")
print(f"CSV driverId dtype: {qualifying_csv['driverId'].dtype}")
print(f"Sample driverId values: {qualifying_csv['driverId'].head().tolist()}")

# Check for overlap
json_race_driver_pairs = set(zip(qualifying_json['raceId'], qualifying_json['driverId']))
csv_race_driver_pairs = set(zip(qualifying_csv['raceId'].astype(int), qualifying_csv['driverId']))

overlapping_pairs = json_race_driver_pairs & csv_race_driver_pairs
print(f"\n\nOverlapping (raceId, driverId) pairs: {len(overlapping_pairs)}")
print(f"JSON unique pairs: {len(json_race_driver_pairs)}")
print(f"CSV unique pairs: {len(csv_race_driver_pairs)}")
print(f"Overlap percentage: {(len(overlapping_pairs) / len(csv_race_driver_pairs) * 100):.1f}%")

if len(overlapping_pairs) < 10:
    print(f"\nSample overlapping pairs: {list(overlapping_pairs)[:5]}")
    
    # Check if any CSV pairs match JSON pairs
    sample_csv_pairs = list(csv_race_driver_pairs)[:5]
    print(f"\nSample CSV pairs: {sample_csv_pairs}")
    print(f"Are they in JSON? {[p in json_race_driver_pairs for p in sample_csv_pairs]}")
