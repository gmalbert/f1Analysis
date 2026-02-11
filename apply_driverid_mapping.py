"""
Apply driverId mapping to all_qualifying_races.csv
Converts short driverIds (hamilton, vettel) to full f1db format (lewis-hamilton, sebastian-vettel)
"""
import pandas as pd
from os import path

DATA_DIR = 'data_files'

# Load the mapping
mapping_path = path.join(DATA_DIR, 'driverId_mapping.csv')
mapping_df = pd.read_csv(mapping_path, sep='\t')
mapping_dict = dict(zip(mapping_df['short_id'], mapping_df['full_id']))

print(f"Loaded {len(mapping_dict)} driverId mappings")
print(f"Sample: {list(mapping_dict.items())[:3]}")

# Load the qualifying CSV
quali_path = path.join(DATA_DIR, 'all_qualifying_races.csv')
quali_df = pd.read_csv(quali_path, sep='\t')

print(f"\nOriginal CSV: {len(quali_df)} rows")
print(f"Unique driverIds before mapping: {quali_df['driverId'].nunique()}")
print(f"Sample driverIds before: {quali_df['driverId'].unique()[:5].tolist()}")

# Count NaN values before mapping
nan_before = quali_df['driverId'].isna().sum()
print(f"\nNaN driverIds before mapping: {nan_before}")

# Apply the mapping (NaN values will remain NaN)
quali_df['driverId'] = quali_df['driverId'].map(mapping_dict).fillna(quali_df['driverId'])

# Check for any unmapped values
nan_after = quali_df['driverId'].isna().sum()
print(f"NaN driverIds after mapping: {nan_after}")

if nan_after > nan_before:
    print(f"WARNING: Mapping failed! {nan_after - nan_before} additional NaN values created")
    print("This suggests some short driverIds were not in the mapping dictionary")
else:
    print("[SUCCESS] All non-NaN driverIds mapped successfully")
    print(f"Unique driverIds after mapping: {quali_df['driverId'].nunique()}")
    print(f"Sample driverIds after: {quali_df['driverId'].unique()[:5].tolist()}")
    
    if nan_after > 0:
        print(f"\n[INFO] {nan_after} rows still have NaN driverId (were NaN before mapping)")
        print("These drivers were not in active_drivers.csv:")
        print(quali_df[quali_df['driverId'].isna()][['Year', 'Round', 'FullName']].head())
    
    # Save the updated CSV
    quali_df.to_csv(quali_path, sep='\t', index=False)
    print(f"\nSaved updated CSV to {quali_path}")
