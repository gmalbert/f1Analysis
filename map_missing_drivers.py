"""
Map the 4 missing drivers manually to their full f1db format
Albon, Giovinazzi, Schumacher, Bortoleto not in active_drivers.csv
"""
import pandas as pd
from os import path

DATA_DIR = 'data_files'

# Manual mappings for the 4 missing drivers based on f1db format
missing_mappings = {
    'Alexander Albon': 'alexander-albon',
    'Antonio Giovinazzi': 'antonio-giovinazzi',
    'Mick Schumacher': 'mick-schumacher',
    'Gabriel Bortoleto': 'gabriel-bortoleto'
}

print("Loading qualifying CSV...")
quali_path = path.join(DATA_DIR, 'all_qualifying_races.csv')
quali_df = pd.read_csv(quali_path, sep='\t')

print(f"Rows with NaN driverId: {quali_df['driverId'].isna().sum()}")

# Apply mappings based on FullName
for full_name, driver_id in missing_mappings.items():
    mask = quali_df['driverId'].isna() & (quali_df['FullName'] == full_name)
    count = mask.sum()
    if count > 0:
        quali_df.loc[mask, 'driverId'] = driver_id
        print(f"  {full_name:20s} → {driver_id:25s} ({count} rows)")

print(f"\nAfter mapping:")
print(f"Rows with NaN driverId: {quali_df['driverId'].isna().sum()}")

if quali_df['driverId'].isna().sum() == 0:
    print("\n[SUCCESS] All driverIds now populated!")
    quali_df.to_csv(quali_path, sep='\t', index=False)
    print(f"Saved updated CSV to {quali_path}")
else:
    print("\n[WARNING] Some rows still have NaN driverId")
    print(quali_df[quali_df['driverId'].isna()][['Year', 'Round', 'FullName', 'BroadcastName']])
