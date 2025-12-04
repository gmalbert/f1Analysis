import pandas as pd, os, datetime
p='data_files/f1ForAnalysis.csv'
print('path:', p)
print('exists:', os.path.exists(p))
if os.path.exists(p):
    stat=os.path.getmtime(p)
    print('modified:', datetime.datetime.fromtimestamp(stat).isoformat())
    try:
        df=pd.read_csv(p, sep='\t', usecols=['short_date'])
        print('read rows:', len(df))
        df['short_date_parsed']=pd.to_datetime(df['short_date'], errors='coerce')
        nulls=df['short_date_parsed'].isna().sum()
        print('parse nulls:', nulls)
        print('max short_date:', df['short_date_parsed'].max())
        print('min short_date:', df['short_date_parsed'].min())
        s=sorted(df['short_date_parsed'].dropna().unique())
        print('last 5 unique dates:', s[-5:])
    except Exception as e:
        print('read failed:', e)
else:
    print('file not found')
