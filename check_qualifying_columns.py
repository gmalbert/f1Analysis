"""Check if previously-NaN qualifying columns are now populated"""
import pandas as pd

print("Loading f1ForAnalysis.csv...")
df = pd.read_csv('data_files/f1ForAnalysis.csv', sep='\t', low_memory=False)
print(f"Total rows: {len(df):,}")
print(f"Total columns: {len(df.columns):,}\n")

# Previously NaN qualifying columns from all_nan_fields_investigation.txt
qualifying_cols = [
    # Basic times (8 fields)
    'q1_sec', 'q2_sec', 'q3_sec', 'best_qual_time',
    'q1_sec_bin', 'q2_sec_bin', 'q3_sec_bin', 'best_qual_time_bin',
    
    # Deltas (3 fields)
    'teammate_qual_delta', 'qualifying_gap_to_pole',
    'teammate_qual_delta_bin',
    
    # Improvements (3 fields)
    'qual_improvement_vs_teammate', 'qual_improvement_q1_to_q3',
    'qual_improvement_vs_teammate_bin',
    
    # Consistency (3 fields)
    'qual_lap_time_consistency', 'qualifying_consistency_vs_constructor_avg',
    'qual_lap_time_consistency_bin',
    
    # Constructor (1 field)
    'qualifying_lap_time_delta_to_constructor_best',
    
    # Additional binned (17 fields)
    'qualifying_gap_to_pole_bin',
    'qual_improvement_q1_to_q3_bin',
    'qualifying_consistency_vs_constructor_avg_bin',
    'qualifying_lap_time_delta_to_constructor_best_bin',
]

# Also check new lap-level fields
lap_level_cols = [
    'best_sector1_sec', 'best_sector2_sec', 'best_sector3_sec',
    'theoretical_best_lap', 'actual_best_lap', 'lap_time_std',
    'sector1_std', 'sector2_std', 'sector3_std',
    'avg_sector1_sec', 'avg_sector2_sec', 'avg_sector3_sec',
    'primary_compound', 'theoretical_gap',
    'total_qualifying_laps', 'valid_laps', 'deleted_laps'
]

print("=" * 80)
print("PREVIOUSLY-NaN QUALIFYING COLUMNS STATUS")
print("=" * 80)

populated = []
still_nan = []
missing = []

for col in qualifying_cols:
    if col not in df.columns:
        missing.append(col)
        print(f"[MISSING] {col:50s}")
    else:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        if non_null > 0:
            populated.append((col, non_null, pct))
            print(f"[OK] {col:50s} {non_null:6,} rows ({pct:5.1f}%)")
        else:
            still_nan.append(col)
            print(f"[NaN] {col:50s} Still 100% NaN")

print("\n" + "=" * 80)
print("NEW LAP-LEVEL QUALIFYING COLUMNS")
print("=" * 80)

for col in lap_level_cols:
    if col not in df.columns:
        print(f"[MISSING] {col:50s}")
    else:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        if non_null > 0:
            print(f"[OK] {col:50s} {non_null:6,} rows ({pct:5.1f}%)")
        else:
            print(f"[NaN] {col:50s} Still 100% NaN")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Populated columns: {len(populated)}")
print(f"Still NaN: {len(still_nan)}")
print(f"Missing from dataset: {len(missing)}")

if populated:
    print(f"\nAverage coverage: {sum(p[2] for p in populated) / len(populated):.1f}%")
    
if missing:
    print(f"\nMissing columns: {missing}")
