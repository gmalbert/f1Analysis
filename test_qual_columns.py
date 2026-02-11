import pandas as pd

qual = pd.read_csv('data_files/all_qualifying_races.csv', sep='\t')
print(f'Total rows: {len(qual)}')

id_cols = [c for c in qual.columns if 'Id' in c]
print(f'\nColumns with "Id": {id_cols}')

name_cols = [c for c in qual.columns if 'Name' in c]
print(f'Columns with "Name": {name_cols}')

print(f'\nDriverId exists: {"DriverId" in qual.columns}')
print(f'TeamId exists: {"TeamId" in qual.columns}')

if 'DriverId' in qual.columns:
    print(f'\nSample DriverId values: {qual["DriverId"].head(3).tolist()}')
