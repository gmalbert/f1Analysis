import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')

print("Current CSV Coverage by Year:")
print("=" * 50)
for year in range(2018, 2026):
    total = len(df[df['Year'] == year])
    with_lap_stats = df[(df['Year'] == year) & (df['total_qualifying_laps'].notna())].shape[0]
    print(f"{year}: {total:3d} rows total, {with_lap_stats:3d} with lap stats")

print("\n" + "=" * 50)
print(f"Total rows in CSV: {len(df)}")
lap_stats_count = df['total_qualifying_laps'].notna().sum()
pct = round(lap_stats_count / len(df) * 100, 1) if len(df) > 0 else 0
print(f"Rows with lap-level stats: {lap_stats_count} ({pct}%)")
