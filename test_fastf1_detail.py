"""
Test to see what FastF1 actually provides for qualifying data
This will show the difference between qualifying.results and qualifying.laps
"""
import fastf1
import pandas as pd

# Enable cache
fastf1.Cache.enable_cache('data_files/f1_cache')

print("Loading 2024 Australian GP Qualifying...")
session = fastf1.get_session(2024, 1, 'Q')
session.load()

print("\n" + "="*80)
print("1. session.results (What the current script uses)")
print("="*80)
print("\nColumns available:")
print(session.results.columns.tolist())
print("\nSample data for one driver:")
verstappen_result = session.results[session.results['Abbreviation'] == 'VER']
print(verstappen_result[['DriverNumber', 'Abbreviation', 'Q1', 'Q2', 'Q3', 'Position']].to_string())

print("\n" + "="*80)
print("2. session.laps (What FastF1 CAN provide - not used by current script)")
print("="*80)
print("\nColumns available:")
print(session.laps.columns.tolist())

verstappen_laps = session.laps[session.laps['Driver'] == 'VER']
print(f"\nVerstappen's qualifying laps: {len(verstappen_laps)} total laps")
print("\nAll of Verstappen's lap times:")
print(verstappen_laps[['LapNumber', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 
                       'Compound', 'IsPersonalBest', 'Deleted', 'DeletedReason']].to_string())

print("\n" + "="*80)
print("3. KEY DIFFERENCE")
print("="*80)
print("\nsession.results provides:")
print("  - Final Q1, Q2, Q3 best times (one per session)")
print("  - Overall position")
print("  - Basic driver metadata")

print("\nsession.laps provides:")
print("  - EVERY individual lap during qualifying")
print("  - Sector times for each lap")
print("  - Tire compound for each lap")
print("  - Which lap was deleted (track limits)")
print("  - Lap-by-lap progression")
print("  - Can calculate theoretical best lap (best S1 + best S2 + best S3)")
