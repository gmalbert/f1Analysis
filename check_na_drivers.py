import pandas as pd

df = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')

print(f'Total rows: {len(df)}')
print(f'Rows with NaN driverId: {df["driverId"].isna().sum()}')
print(f'Rows with valid driverId: {df["driverId"].notna().sum()}')
print(f'\nSample valid driverIds (should be full format after mapping):')
print(df[df["driverId"].notna()]["driverId"].unique()[:10])
print(f'\nSample rows with NaN driverId:')
print(df[df["driverId"].isna()][['Year', 'Round', 'DriverNumber', 'BroadcastName', 'FullName']].head())
