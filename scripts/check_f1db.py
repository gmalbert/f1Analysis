import pandas as pd, os, datetime
files = [
    'data_files/f1db-races.json',
    'data_files/f1db-races-race-results.json'
]
for p in files:
    print('\nChecking', p)
    if not os.path.exists(p):
        print('  MISSING')
        continue
    try:
        df = pd.read_json(p)
        print('  rows:', len(df))
        if 'date' in df.columns:
            df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
            print('  parsed nulls:', df['date_parsed'].isna().sum())
            print('  max date:', df['date_parsed'].max())
            print('  last 5 dates:', sorted(df['date_parsed'].dropna().unique())[-5:])
        else:
            # try finding any date-like column
            date_cols = [c for c in df.columns if 'date' in c.lower()]
            print('  date-like cols:', date_cols)
            for c in date_cols:
                try:
                    s = pd.to_datetime(df[c], errors='coerce')
                    print(f'   {c} max:', s.max())
                except Exception as e:
                    print('   parse failed for', c, e)
    except Exception as e:
        print('  read failed:', e)
print('\nDone')
