"""
Quick test to see if the lap stats merge is working correctly
Run on just Bahrain 2024 (Round 1)
"""
import fastf1
import pandas as pd
from os import path
import numpy as np

DATA_DIR = 'data_files/'
fastf1.Cache.enable_cache(path.join(DATA_DIR, 'f1_cache'))

print("Loading Bahrain 2024 qualifying...")
qualifying = fastf1.get_session(2024, 1, 'Q')
qualifying.load()

qualifying_results = qualifying.results
qualifying_laps = qualifying.laps

print(f"\nqualifying.results columns: {qualifying_results.columns.tolist()}")
print(f"qualifying.results shape: {qualifying_results.shape}")

driver_lap_stats = []
for driver_abbrev in qualifying_results['Abbreviation'].unique():
    driver_laps = qualifying_laps[qualifying_laps['Driver'] == driver_abbrev]
    
    if len(driver_laps) == 0:
        print(f"WARNING: No laps found for driver {driver_abbrev}")
        continue
        
    valid_laps = driver_laps[
        (~driver_laps['Deleted']) & 
        (driver_laps['LapTime'].notna()) &
        (driver_laps['Sector1Time'].notna()) &
        (driver_laps['Sector2Time'].notna()) &
        (driver_laps['Sector3Time'].notna())
    ]
    
    stats = {
        'DriverNumber': driver_laps.iloc[0]['DriverNumber'],
        'Abbreviation': driver_abbrev,
        'total_qualifying_laps': len(driver_laps),
        'valid_laps': len(valid_laps),
    }
    
    if len(valid_laps) > 0:
        lap_times_sec = valid_laps['LapTime'].dt.total_seconds()
        stats['best_sector1_sec'] = valid_laps['Sector1Time'].dt.total_seconds().min()
        stats['best_sector2_sec'] = valid_laps['Sector2Time'].dt.total_seconds().min()
        stats['best_sector3_sec'] = valid_laps['Sector3Time'].dt.total_seconds().min()
        stats['theoretical_best_lap'] = stats['best_sector1_sec'] + stats['best_sector2_sec'] + stats['best_sector3_sec']
        stats['actual_best_lap'] = lap_times_sec.min()
    
    driver_lap_stats.append(stats)

lap_stats_df = pd.DataFrame(driver_lap_stats)
print(f"\nlap_stats_df columns: {lap_stats_df.columns.tolist()}")
print(f"lap_stats_df shape: {lap_stats_df.shape}")
print(f"\nlap_stats_df head():")
print(lap_stats_df.head(3))

print(f"\nMerging on: ['DriverNumber', 'Abbreviation']")
qualifying_results_merged = pd.merge(
    qualifying_results,
    lap_stats_df,
    on=['DriverNumber', 'Abbreviation'],
    how='left'
)

print(f"\nqualifying_results_merged columns: {qualifying_results_merged.columns.tolist()}")
print(f"qualifying_results_merged shape: {qualifying_results_merged.shape}")

print(f"\nChecking lap stats columns populated:")
for col in ['total_qualifying_laps', 'best_sector1_sec', 'theoretical_best_lap']:
    if col in qualifying_results_merged.columns:
        non_null = qualifying_results_merged[col].notna().sum()
        print(f"  {col}: {non_null} non-null out of {len(qualifying_results_merged)}")
    else:
        print(f"  {col}: NOT FOUND")
