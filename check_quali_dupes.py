"""Check for duplicate rows in qualifying CSV"""
import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')

print(f"Total rows: {len(df)}")

# Check for exact duplicates
exact_dupes = df.duplicated().sum()
print(f"Exact duplicate rows: {exact_dupes}")

# Check for duplicates based on year/round/driver
key_cols = ['Year', 'Round', 'driverId']
if all(col in df.columns for col in key_cols):
    logical_dupes = df.duplicated(subset=key_cols).sum()
    print(f"Logical duplicates (same Year/Round/driverId): {logical_dupes}")
    
    # Show sample duplicates
    if logical_dupes > 0:
        dupes_df = df[df.duplicated(subset=key_cols, keep=False)]
        print(f"\nSample duplicate records:")
        print(dupes_df[key_cols + ['constructorName', 'Q1_sec']].sort_values(key_cols).head(6).to_string())

# Check unique qualifying sessions
unique_sessions = df[['Year', 'Round']].drop_duplicates()
print(f"\nUnique qualifying sessions: {len(unique_sessions)}")

# Expected count (assuming ~20 drivers per session)
expected_rows = len(unique_sessions) * 20
print(f"Expected rows (~20 drivers/session): {expected_rows}")
print(f"Actual rows: {len(df)}")
print(f"Difference: {len(df) - expected_rows}")
