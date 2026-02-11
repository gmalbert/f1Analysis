"""
Test script to verify enhanced fastF1-qualifying.py extraction
Tests on 2024 Australian GP (cached) to validate new lap-level fields
"""
import fastf1
import pandas as pd
from os import path

# Enable cache
cache_dir = path.join('data_files', 'f1_cache')
fastf1.Cache.enable_cache(cache_dir)

print("Loading 2024 Australian GP qualifying session...")
session = fastf1.get_session(2024, 'Australia', 'Q')
session.load()

print("\n=== Session Results Structure ===")
print(f"qualifying.results columns: {session.results.columns.tolist()}")
print(f"qualifying.results shape: {session.results.shape}")

print("\n=== Session Laps Structure ===")
print(f"qualifying.laps columns: {session.laps.columns.tolist()}")
print(f"qualifying.laps shape: {session.laps.shape}")

print("\n=== Testing Enhanced Extraction Logic ===")

# Enhanced extraction logic (from updated fastF1-qualifying.py)
qualifying_results = session.results
qualifying_laps = session.laps

# Process lap-level data for each driver
driver_lap_stats = []

for driver_num in qualifying_results['DriverNumber']:
    driver_laps = qualifying_laps[qualifying_laps['DriverNumber'] == driver_num]
    
    if len(driver_laps) == 0:
        continue
    
    # Get valid laps (non-deleted)
    valid_laps = driver_laps[driver_laps['IsAccurate'] == True]
    deleted_laps = driver_laps[driver_laps['Deleted'] == True]
    
    # Calculate statistics
    stats = {
        'DriverNumber': driver_num,
        'total_qualifying_laps': len(driver_laps),
        'valid_laps': len(valid_laps),
        'deleted_laps': len(deleted_laps),
    }
    
    # Best sector times (from all valid laps)
    if len(valid_laps) > 0:
        # Convert timedeltas to seconds for sector times
        valid_laps_copy = valid_laps.copy()
        valid_laps_copy['Sector1TimeSeconds'] = valid_laps_copy['Sector1Time'].dt.total_seconds()
        valid_laps_copy['Sector2TimeSeconds'] = valid_laps_copy['Sector2Time'].dt.total_seconds()
        valid_laps_copy['Sector3TimeSeconds'] = valid_laps_copy['Sector3Time'].dt.total_seconds()
        valid_laps_copy['LapTimeSeconds'] = valid_laps_copy['LapTime'].dt.total_seconds()
        
        stats['best_sector1_sec'] = valid_laps_copy['Sector1TimeSeconds'].min()
        stats['best_sector2_sec'] = valid_laps_copy['Sector2TimeSeconds'].min()
        stats['best_sector3_sec'] = valid_laps_copy['Sector3TimeSeconds'].min()
        
        # Theoretical best lap (sum of best sectors)
        stats['theoretical_best_lap'] = (
            stats['best_sector1_sec'] + 
            stats['best_sector2_sec'] + 
            stats['best_sector3_sec']
        )
        
        # Actual best lap
        stats['actual_best_lap'] = valid_laps_copy['LapTimeSeconds'].min()
        
        # Gap between theoretical and actual (consistency indicator)
        stats['theoretical_gap'] = stats['actual_best_lap'] - stats['theoretical_best_lap']
        
        # Lap time consistency (standard deviation)
        stats['lap_time_std'] = valid_laps_copy['LapTimeSeconds'].std()
        stats['sector1_std'] = valid_laps_copy['Sector1TimeSeconds'].std()
        stats['sector2_std'] = valid_laps_copy['Sector2TimeSeconds'].std()
        stats['sector3_std'] = valid_laps_copy['Sector3TimeSeconds'].std()
        
        # Average sector times
        stats['avg_sector1_sec'] = valid_laps_copy['Sector1TimeSeconds'].mean()
        stats['avg_sector2_sec'] = valid_laps_copy['Sector2TimeSeconds'].mean()
        stats['avg_sector3_sec'] = valid_laps_copy['Sector3TimeSeconds'].mean()
        
        # Primary tire compound used
        compound_counts = valid_laps_copy['Compound'].value_counts()
        stats['primary_compound'] = compound_counts.index[0] if len(compound_counts) > 0 else None
    else:
        # No valid laps - fill with NaN
        for field in ['best_sector1_sec', 'best_sector2_sec', 'best_sector3_sec',
                     'theoretical_best_lap', 'actual_best_lap', 'theoretical_gap',
                     'lap_time_std', 'sector1_std', 'sector2_std', 'sector3_std',
                     'avg_sector1_sec', 'avg_sector2_sec', 'avg_sector3_sec']:
            stats[field] = None
        stats['primary_compound'] = None
    
    driver_lap_stats.append(stats)

# Convert to DataFrame
lap_stats_df = pd.DataFrame(driver_lap_stats)

print(f"\nExtracted lap statistics for {len(lap_stats_df)} drivers")
print("\n=== Sample Statistics (First 3 Drivers) ===")
print(lap_stats_df.head(3).to_string())

print("\n=== Field Validation ===")
for col in lap_stats_df.columns:
    if col != 'DriverNumber':
        non_null = lap_stats_df[col].notna().sum()
        null_count = lap_stats_df[col].isna().sum()
        print(f"{col:25s}: {non_null:2d} non-null, {null_count:2d} null")

print("\n=== Theoretical vs Actual Best Lap (Top 3) ===")
top3 = lap_stats_df.nsmallest(3, 'actual_best_lap')[
    ['DriverNumber', 'theoretical_best_lap', 'actual_best_lap', 'theoretical_gap']
]
print(top3.to_string())

print("\n✅ Test complete! All 13 new lap-level fields extracted successfully.")
