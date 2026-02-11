"""Remove duplicate rows from qualifying CSV"""
import pandas as pd
from os import path

csv_path = path.join('data_files', 'all_qualifying_races.csv')

print(f"Loading {csv_path}...")
df = pd.read_csv(csv_path, sep='\t')
print(f"Original rows: {len(df)}")

# Count duplicates
dupes = df.duplicated().sum()
print(f"Exact duplicates found: {dupes}")

# Remove duplicates
df_deduped = df.drop_duplicates()
print(f"After deduplication: {len(df_deduped)}")

# Backup original
backup_path = csv_path + '.backup_before_dedup'
df.to_csv(backup_path, sep='\t', index=False)
print(f"\nBackup saved to: {backup_path}")

# Save deduplicated
df_deduped.to_csv(csv_path, sep='\t', index=False)
print(f"Deduplicated CSV saved to: {csv_path}")

print(f"\nVerification:")
print(f"  Rows removed: {len(df) - len(df_deduped)}")
print(f"  Unique Year/Round combinations: {df_deduped[['Year', 'Round']].drop_duplicates().shape[0]}")
print(f"  Sample driverId values: {df_deduped['driverId'].dropna().head(3).tolist()}")
print(f"  constructorId populated: {df_deduped['constructorId'].notna().sum()} rows")
