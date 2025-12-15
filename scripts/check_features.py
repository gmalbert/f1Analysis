import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to path to import functions
sys.path.append(os.getcwd())

# Import the functions we created
from raceAnalysis import (
    load_data, 
    create_constructor_adjusted_driver_features,
    create_recent_performance_features, 
    create_constructor_compatibility_features
)

print("Loading data...")
data, pitStops = load_data(1000)  # Load smaller dataset for testing

print(f"Original data shape: {data.shape}")
print(f"Original columns with team info: {[col for col in data.columns if 'constructor' in col.lower() or 'driver' in col.lower()][:10]}")

# Apply column renaming (same as main app)
if 'constructorName_results_with_qualifying' in data.columns:
    data.rename(columns={'constructorName_results_with_qualifying': 'constructorName'}, inplace=True)
elif 'constructorName_qualifying' in data.columns:
    data.rename(columns={'constructorName_qualifying': 'constructorName'}, inplace=True)

# Apply data type conversions (simplified version)
data['resultsStartingGridPositionNumber'] = data['resultsStartingGridPositionNumber'].astype('Float64')
data['resultsFinalPositionNumber'] = data['resultsFinalPositionNumber'].astype('Float64')

print(f"After renaming - constructorName column exists: {'constructorName' in data.columns}")
print("\nCreating team-aware features...")

try:
    print("1. Constructor adjusted features...")
    original_cols = set(data.columns)
    data = create_constructor_adjusted_driver_features(data)
    new_cols = set(data.columns) - original_cols
    print(f"   Added: {new_cols}")
    
    print("2. Recent performance features...")
    original_cols = set(data.columns)
    data = create_recent_performance_features(data, recent_races=5)
    new_cols = set(data.columns) - original_cols
    print(f"   Added: {new_cols}")
    
    print("3. Constructor compatibility features...")
    original_cols = set(data.columns)
    data = create_constructor_compatibility_features(data)
    new_cols = set(data.columns) - original_cols
    print(f"   Added: {new_cols}")
    
    print(f"\nFinal data shape: {data.shape}")
    
    # Test with Carlos Sainz example
    if 'resultsDriverName' in data.columns:
        sainz_data = data[data['resultsDriverName'].str.contains('Sainz', na=False)]
        if not sainz_data.empty:
            print(f"\nCarlos Sainz sample data (showing team-aware features):")
            team_cols = [col for col in data.columns if any(x in col for x in ['constructor', 'recent', 'compatibility', 'Career', 'Experience'])]
            print(sainz_data[['grandPrixYear', 'constructorName', 'resultsFinalPositionNumber'] + team_cols[:5]].head(3))
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()