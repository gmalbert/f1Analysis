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
print(f"Sample mappings:")
for i, (short, full) in enumerate(list(mapping_dict.items())[:3]):
    print(f"  {short:15s} → {full}")

# Load the qualifying CSV
quali_path = path.join(DATA_DIR, 'all_qualifying_races.csv')
quali_df = pd.read_csv(quali_path, sep='\t')

print(f"\n=== BEFORE MAPPING ===")
print(f"Total rows: {len(quali_df)}")
nan_before = quali_df['driverId'].isna().sum()
print(f"NaN driverIds: {nan_before}")
print(f"Valid driverIds: {quali_df['driverId'].notna().sum()}")
print(f"Sample driverIds: {quali_df[quali_df['driverId'].notna()]['driverId'].unique()[:5].tolist()}")

# Keep copy of original for comparison
original_driverid = quali_df['driverId'].copy()

# Apply the mapping
quali_df['driverId'] = quali_df['driverId'].map(mapping_dict)

print(f"\n=== AFTER MAPPING ===")
nan_after = quali_df['driverId'].isna().sum()
newly_unmapped = nan_after - nan_before

if newly_unmapped > 0:
    print(f"[ERROR] Mapping created {newly_unmapped} NEW NaN values!")
    print("These short IDs were not in the mapping dictionary:")
    failed_mask = quali_df['driverId'].isna() & original_driverid.notna()
    print(f"Failed IDs: {original_driverid[failed_mask].unique().tolist()}")
    print("\nNOT saving - mapping incomplete!")
else:
    print(f"[SUCCESS] All {quali_df['driverId'].notna().sum()} driverIds mapped successfully")
    print(f"NaN driverIds: {nan_after} (same as before - these were already NaN)")
    print(f"Sample mapped driverIds: {quali_df[quali_df['driverId'].notna()]['driverId'].unique()[:5].tolist()}")
    
    if nan_after > 0:
        print(f"\n[INFO] {nan_after} rows have NaN driverId (were missing before mapping)")
        print("Example drivers with NaN:")
        print(quali_df[quali_df['driverId'].isna()][['Year', 'Round', 'FullName']].head())
    
    # Save the updated CSV
    quali_df.to_csv(quali_path, sep='\t', index=False)
    print(f"\n✓ Saved updated CSV to {quali_path}")
