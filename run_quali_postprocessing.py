"""
Run only the post-processing section of fastF1-qualifying.py
This bypasses the data collection phase and any cached bytecode issues.
"""
import pandas as pd
from os import path

DATA_DIR = 'data_files'
csv_path = path.join(DATA_DIR, 'all_qualifying_races.csv')

print(f"\n{'='*60}")
print("POST-PROCESSING QUALIFYING DATA")
print(f"{'='*60}\n")

# Check CSV exists
if not path.exists(csv_path):
    print("[ERROR] CSV file not found:", csv_path)
    exit(1)

# Load CSV
try:
    qualifying = pd.read_csv(csv_path, sep='\t')
    print(f"[SUCCESS] Loaded {len(qualifying)} qualifying records")
except Exception as e:
    print(f"[ERROR] Failed to load CSV:", e)
    exit(1)

# Validate CSV structure
required_cols = ['DriverId', 'TeamId', 'raceId']
if not all(col in qualifying.columns for col in required_cols):
    print(f"[ERROR] CSV missing required columns. Found: {qualifying.columns.tolist()[:10]}")
    exit(1)

print(f"[SUCCESS] CSV has required columns: {required_cols}")

# Rename columns to match data pipeline expectations
column_mapping = {
    'DriverId': 'driverId',
    'TeamId': 'constructorId',
    'TeamName': 'constructorName'
}

qualifying = qualifying.rename(columns=column_mapping)
print(f"\n[SUCCESS] Renamed columns: {list(column_mapping.keys())} -> {list(column_mapping.values())}")

# Verify driverId exists after rename
if 'driverId' not in qualifying.columns:
    print("[ERROR] driverId column not found after renaming")
    exit(1)

# Load active drivers and merge
active_drivers_path = path.join(DATA_DIR, 'active_drivers.csv')
if path.exists(active_drivers_path):
    try:
        active_drivers = pd.read_csv(active_drivers_path, sep='\t')
        print(f"\n[SUCCESS] Loaded {len(active_drivers)} active drivers")
        
        # Merge to add driver metadata
        qualifying = qualifying.merge(
            active_drivers[['driverId', 'abbreviation', 'name']],
            on='driverId',
            how='left',
            suffixes=('', '_active')
        )
        
        matched_count = qualifying['driverId'].notna().sum()
        unique_drivers = qualifying['driverId'].nunique()
        
        print(f"[SUCCESS] Merged with active_drivers:")
        print(f"  - Total rows: {len(qualifying)}")
        print(f"  - Rows with driver match: {matched_count}")
        print(f"  - Unique drivers: {unique_drivers}")
        
    except Exception as e:
        print(f"[WARNING] Failed to merge with active_drivers: {e}")
        print("  Continuing without merge.")

# Load constructors and merge (if file exists)
constructors_path = path.join(DATA_DIR, 'f1db-constructors.json')
if path.exists(constructors_path):
    try:
        import json
        with open(constructors_path, 'r', encoding='utf-8') as f:
            constructors_data = json.load(f)
        
        constructors_df = pd.DataFrame(constructors_data)
        if 'id' in constructors_df.columns:
            constructors_df = constructors_df.rename(columns={'id': 'constructorId'})
        
        # Merge to add constructor metadata  
        qualifying = qualifying.merge(
            constructors_df[['constructorId', 'name']].rename(columns={'name': 'constructor_full_name'}),
            on='constructorId',
            how='left'
        )
        
        print(f"[SUCCESS] Merged with {len(constructors_df)} constructors")
        
    except Exception as e:
        print(f"[WARNING] Failed to merge with constructors: {e}")

# Save enriched CSV
try:
    qualifying.to_csv(csv_path, sep='\t', index=False)
    print(f"\n[SUCCESS] Saved enriched qualifying data to:")
    print(f"  {csv_path}")
    print(f"\n  Total rows: {len(qualifying)}")
    print(f"  Total columns: {len(qualifying.columns)}")
    print(f"\n  Key columns: {[c for c in qualifying.columns if c in ['driverId', 'constructorId', 'Q1_sec', 'Q2_sec', 'Q3_sec', 'best_sector1_sec', 'theoretical_best_lap']]}")
    
except Exception as e:
    print(f"[ERROR] Failed to save CSV:", e)
    exit(1)

print(f"\n{'='*60}")
print("POST-PROCESSING COMPLETE")
print(f"{'='*60}\n")
