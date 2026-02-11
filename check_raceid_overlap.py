"""Check if CSV raceIds exist in JSON at all"""
import pandas as pd
from os import path

DATA_DIR = 'data_files'

qualifying_json = pd.read_json(path.join(DATA_DIR, 'f1db-races-qualifying-results.json'))
qualifying_csv = pd.read_csv(path.join(DATA_DIR, 'all_qualifying_races.csv'), sep='\t')

csv_race_ids = set(qualifying_csv['raceId'].dropna().astype(int))
json_race_ids = set(qualifying_json['raceId'])

print(f"CSV total raceIds: {len(csv_race_ids)}")
print(f"JSON total raceIds: {len(json_race_ids)}")
print(f"CSV sample raceIds: {list(csv_race_ids)[:10]}")
print(f"JSON max raceId: {max(json_race_ids)}")
print(f"JSON raceIds in CSV range: {sorted([r for r in json_race_ids if r >= min(csv_race_ids)])[:10]}")

matching_race_ids = csv_race_ids & json_race_ids
print(f"\nMatching raceIds: {len(matching_race_ids)}")

if len(matching_race_ids) > 0:
    print(f"Sample matching: {list(matching_race_ids)[:5]}")
    
    # For a matching raceId, check drivers
    sample_race = list(matching_race_ids)[0]
    json_drivers = qualifying_json[qualifying_json['raceId'] == sample_race]['driverId'].tolist()
    csv_drivers = qualifying_csv[qualifying_csv['raceId'] == sample_race]['driverId'].tolist()
    
    print(f"\nFor raceId {sample_race}:")
    print(f"JSON drivers: {json_drivers[:5]}")
    print(f"CSV drivers: {csv_drivers[:5]}")
    print(f"Overlapping drivers: {set(json_drivers) & set(csv_drivers)}")
else:
    print("\nNo matching raceIds found!")
    print(f"CSV raceIds NOT in JSON: {list(csv_race_ids - json_race_ids)[:5]}")
