import os
import pandas as pd
from datetime import datetime
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_path = os.path.join(ROOT, 'data_files', 'f1ForAnalysis.csv')
print('Reading:', csv_path)
if not os.path.exists(csv_path):
    print('MISSING CSV')
    raise SystemExit(1)
df = pd.read_csv(csv_path, sep='\t', low_memory=False)
print('\nColumns (total {}):'.format(len(df.columns)))
print(', '.join(df.columns.tolist()))

for col in ['PredictedFinalPosition', 'PredictedPositionMAE']:
    exists = col in df.columns
    nonnull = int(df[col].notna().sum()) if exists else 'N/A'
    print(f"\nColumn {col}: exists={exists}, non-null count={nonnull}")

# parse short_date and find next upcoming
if 'short_date' in df.columns:
    df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
    now = pd.to_datetime(datetime.now())
    upcoming = df[df['short_date'] >= now]
    print('\nUpcoming rows count:', len(upcoming))
    if len(upcoming):
        next_date = upcoming['short_date'].min()
        print('Next race date:', next_date)
        send_df = upcoming[upcoming['short_date'] == next_date].copy()
        print('Rows for next race:', len(send_df))
        # counts in send_df
        for col in ['PredictedFinalPosition', 'PredictedPositionMAE']:
            exists = col in send_df.columns
            nonnull = int(send_df[col].notna().sum()) if exists else 'N/A'
            print(f"  next race: {col}: exists={exists}, non-null={nonnull}")
        print('\nSample rows (first 10) with key cols:')
        keys = ['short_date','grandPrixName','resultsDriverName','PredictedFinalPosition','PredictedPositionMAE']
        keys_present = [k for k in keys if k in send_df.columns]
        print(send_df[keys_present].head(10).to_string(index=False))
    else:
        print('No upcoming races found in data (short_date >= now)')
else:
    print('No short_date column present')

print('\nDone')
