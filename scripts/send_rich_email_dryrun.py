#!/usr/bin/env python3
"""Dry-run helper: determine which predictions_*.csv would be chosen by the sender
and print the 'Predictive Results for Active Drivers' table preview.

Selection logic:
- Compute next race from `data_files/f1db-races.json` (preferred).
- Prefer a predictions file whose `grandPrixName` matches next race name.
- Else prefer predictions file with `short_date >= today` and earliest date.
- Else fallback to the most recently modified predictions file.

No email is sent; this is for diagnostics only.
"""
import os
import sys
import glob
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    import pandas as pd
    import json
except Exception as e:
    print('Missing dependency:', e)
    sys.exit(2)


def compute_next_race_from_calendar(root):
    races_json = os.path.join(root, 'data_files', 'f1db-races.json')
    if not os.path.exists(races_json):
        return None, None
    try:
        with open(races_json, 'r', encoding='utf-8') as fh:
            races = json.load(fh)
    except Exception:
        return None, None

    now = pd.to_datetime(datetime.now())
    candidates = []
    for r in races:
        sd = r.get('short_date') or r.get('date') or r.get('race_date') or r.get('raceDate')
        try:
            sd_dt = pd.to_datetime(sd, errors='coerce')
        except Exception:
            sd_dt = None
        if sd_dt is not None and not pd.isna(sd_dt) and sd_dt >= now:
            name = r.get('grandPrixName') or r.get('name') or r.get('raceName')
            candidates.append((sd_dt, name))
    if not candidates:
        return None, None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1], candidates[0][0]


def choose_predictions_file(root, next_race_name):
    pred_glob = os.path.join(root, 'data_files', 'predictions_*.csv')
    pred_files = sorted(glob.glob(pred_glob))
    if not pred_files:
        return None

    # 1) try to find file where grandPrixName in file matches next_race_name
    if next_race_name:
        for p in pred_files:
            try:
                tmp = pd.read_csv(p, nrows=50)
            except Exception:
                continue
            if 'grandPrixName' in tmp.columns:
                names = tmp['grandPrixName'].dropna().astype(str).str.lower().unique()
                if any(next_race_name.lower() in n for n in names):
                    return p

    # 2) prefer predictions files with embedded short_date >= today, pick earliest
    now = pd.to_datetime(datetime.now())
    candidates = []
    for p in pred_files:
        try:
            tmp = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        if 'short_date' in tmp.columns:
            sd = pd.to_datetime(tmp['short_date'], errors='coerce')
            if not sd.dropna().empty:
                candidates.append((p, sd.dropna().iloc[0]))
    upcoming = [(p, sd) for (p, sd) in candidates if sd >= now]
    if upcoming:
        upcoming.sort(key=lambda x: x[1])
        return upcoming[0][0]

    # 3) final fallback: newest by mtime
    pred_files.sort(key=os.path.getmtime, reverse=True)
    return pred_files[0]


def build_display_table_from_predictions(pred_path):
    try:
        pred = pd.read_csv(pred_path)
    except Exception as e:
        print('Failed to read predictions file:', e)
        return None

    if 'resultsDriverName' not in pred.columns and 'resultsDriver' in pred.columns:
        pred = pred.rename(columns={'resultsDriver': 'resultsDriverName'})

    # compute low/high bounds
    if 'PredictedFinalPositionStd' in pred.columns:
        pred['PredictedFinalPosition_Low'] = pred['PredictedFinalPosition'] - pred['PredictedFinalPositionStd']
        pred['PredictedFinalPosition_High'] = pred['PredictedFinalPosition'] + pred['PredictedFinalPositionStd']
    elif 'PredictedPositionMAE' in pred.columns:
        pred['PredictedFinalPosition_Low'] = pred['PredictedFinalPosition'] - pred['PredictedPositionMAE']
        pred['PredictedFinalPosition_High'] = pred['PredictedFinalPosition'] + pred['PredictedPositionMAE']

    if 'PredictedPositionMAE' in pred.columns:
        pred['PredictedPositionMAE_Low'] = pred['PredictedFinalPosition'] - pred['PredictedPositionMAE']
        pred['PredictedPositionMAE_High'] = pred['PredictedFinalPosition'] + pred['PredictedPositionMAE']

    cols_to_show = ['constructorName', 'resultsDriverName', 'PredictedFinalPosition', 'PredictedFinalPositionStd',
                    'PredictedFinalPosition_Low', 'PredictedFinalPosition_High', 'PredictedPositionMAE',
                    'PredictedPositionMAE_Low', 'PredictedPositionMAE_High']
    present = [c for c in cols_to_show if c in pred.columns]
    if not present:
        return pred.head(20)
    out = pred[present].sort_values(by='PredictedFinalPosition', na_position='last')
    return out


def main():
    next_race_name, next_race_date = compute_next_race_from_calendar(ROOT)
    print('Derived next race from calendar:', next_race_name, next_race_date)

    chosen = choose_predictions_file(ROOT, next_race_name)
    print('\nChosen predictions file:')
    print(chosen)

    if chosen:
        print('\nPreview of chosen predictions file (first rows):')
        try:
            preview = pd.read_csv(chosen).head(10)
            print(preview.to_string(index=False))
        except Exception as e:
            print('Failed to preview chosen file:', e)

        display = build_display_table_from_predictions(chosen)
        if display is None:
            print('\nNo displayable prediction table.')
        else:
            print('\n-- Predictive Results for Active Drivers (from chosen predictions file) --')
            print(display.to_string(index=False))
    else:
        print('No predictions file found.')


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Dry-run helper: print which predictions source would be used and preview the
"Predictive Results for Active Drivers" table.

This replicates selection logic used by `send_rich_email_now.py` but only prints
the chosen file and a small preview (no email send).
"""
import os
import sys
from datetime import datetime
import glob
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def find_next_race(df):
    if 'short_date' not in df.columns:
        return None, None
    df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
    now = pd.to_datetime(datetime.now())
    upcoming = df[df['short_date'] >= now].copy()
    if upcoming.empty:
        return None, None
    nd = upcoming['short_date'].min()
    candidates = upcoming[upcoming['short_date'] == nd]
    gp = None
    if 'grandPrixName' in candidates.columns:
        vals = candidates['grandPrixName'].dropna().unique()
        if len(vals) > 0:
            gp = str(vals[0])
    return gp, nd

def choose_predictions_file(root, next_race_name):
    pred_glob = os.path.join(root, 'data_files', 'predictions_*.csv')
    pred_files = sorted(glob.glob(pred_glob))
    candidate = None
    # prefer a file that mentions the next race name
    if next_race_name:
        for p in pred_files:
            try:
                tmp = pd.read_csv(p, nrows=20)
                if 'grandPrixName' in tmp.columns:
                    names = tmp['grandPrixName'].dropna().astype(str).str.lower().unique()
                    if any(next_race_name.lower() in n for n in names):
                        candidate = p
                        break
            except Exception:
                continue
    # If analysis didn't provide a next_race_name, prefer an obvious filename match
    # (helps when predictions files lack embedded short_date/grandPrixName metadata).
    if candidate is None and not next_race_name and pred_files:
        # Prefer Abu Dhabi when the next race is in early December 2025
        for slug in ('abu-dhabi', 'abu_dhabi', 'abu-dhabi_2025'):
            for p in pred_files:
                if slug in os.path.basename(p).lower():
                    candidate = p
                    break
            if candidate:
                break
    # fallback to earliest short_date >= today
    if candidate is None and pred_files:
        now = pd.to_datetime(datetime.now())
        candidates = []
        for p in pred_files:
            try:
                tmp = pd.read_csv(p, nrows=5)
                if 'short_date' in tmp.columns:
                    sd = pd.to_datetime(tmp['short_date'], errors='coerce')
                    if not sd.dropna().empty:
                        candidates.append((p, sd.dropna().iloc[0]))
            except Exception:
                continue
        upcoming = [(p, sd) for (p, sd) in candidates if sd >= now]
        if upcoming:
            upcoming.sort(key=lambda x: x[1])
            candidate = upcoming[0][0]
    # final fallback: newest file by mtime
    if candidate is None and pred_files:
        pred_files.sort(key=os.path.getmtime, reverse=True)
        candidate = pred_files[0]
    return candidate

def build_table_from_analysis(df, next_date):
    # select rows for the next_date
    sel = df.copy()
    sel['short_date'] = pd.to_datetime(sel['short_date'], errors='coerce')
    rows = sel[sel['short_date'] == next_date].copy()
    return rows

def main():
    csv_path = os.path.join(ROOT, 'data_files', 'f1ForAnalysis.csv')
    if not os.path.exists(csv_path):
        print('Analysis CSV not found:', csv_path)
        sys.exit(1)
    df = pd.read_csv(csv_path, sep='\t')
    next_race_name, next_race_date = find_next_race(df)
    print('Next race from analysis CSV:', next_race_name, next_race_date)

    chosen_pred = choose_predictions_file(ROOT, next_race_name)
    if chosen_pred:
        print('Found predictions file candidate:', chosen_pred)
        try:
            pred_df = pd.read_csv(chosen_pred)
            preview = pred_df.head(10)
            print('\nPreview of predictions file (first 10 rows):')
            print(preview.to_string(index=False))
        except Exception as e:
            print('Failed to read predictions file:', e)
    else:
        print('No predictions_*.csv file found.')

    if next_race_date is not None:
        analysis_table = build_table_from_analysis(df, next_race_date)
        if analysis_table.empty:
            print('\nNo rows in analysis CSV for next race date; showing latest 10 rows instead:')
            print(df.tail(10).to_string(index=False))
        else:
            print(f"\nPredictive Results for Active Drivers (analysis rows for {next_race_date.date()}):")
            # display the key columns if present, otherwise all columns
            cols = ['constructorName', 'resultsDriverName', 'PredictedFinalPosition', 'PredictedFinalPositionStd',
                    'PredictedFinalPosition_Low', 'PredictedFinalPosition_High', 'PredictedPositionMAE',
                    'PredictedPositionMAE_Low', 'PredictedPositionMAE_High']
            present = [c for c in cols if c in analysis_table.columns]
            if not present:
                print('\nNo predicted columns found in analysis slice; showing head of the slice:')
                print(analysis_table.head(10).to_string(index=False))
            else:
                sort_col = present[2] if len(present) > 2 else present[0]
                out = analysis_table[present].sort_values(by=sort_col, na_position='last')
                print(out.to_string(index=False))
    else:
        print('\nCould not determine next race date from analysis CSV.')

    # Also, if we found a predictions file, build the 'Predictive Results for Active Drivers'
    # table as Streamlit would show it (compute low/high bounds and MAE columns if needed).
    if chosen_pred:
        try:
            pred = pd.read_csv(chosen_pred)
            # Ensure key columns exist
            if 'resultsDriverName' not in pred.columns and 'resultsDriver' in pred.columns:
                pred = pred.rename(columns={'resultsDriver': 'resultsDriverName'})
            display_cols = ['constructorName', 'resultsDriverName', 'PredictedFinalPosition',
                            'PredictedFinalPositionStd', 'PredictedPositionMAE']
            # Compute bounds
            if 'PredictedFinalPositionStd' in pred.columns:
                pred['PredictedFinalPosition_Low'] = pred['PredictedFinalPosition'] - pred['PredictedFinalPositionStd']
                pred['PredictedFinalPosition_High'] = pred['PredictedFinalPosition'] + pred['PredictedFinalPositionStd']
            elif 'PredictedPositionMAE' in pred.columns:
                pred['PredictedFinalPosition_Low'] = pred['PredictedFinalPosition'] - pred['PredictedPositionMAE']
                pred['PredictedFinalPosition_High'] = pred['PredictedFinalPosition'] + pred['PredictedPositionMAE']
            # PredictedPositionMAE low/high are the same around predicted final position
            if 'PredictedPositionMAE' in pred.columns:
                pred['PredictedPositionMAE_Low'] = pred['PredictedFinalPosition'] - pred['PredictedPositionMAE']
                pred['PredictedPositionMAE_High'] = pred['PredictedFinalPosition'] + pred['PredictedPositionMAE']

            cols_to_show = ['constructorName', 'resultsDriverName', 'PredictedFinalPosition', 'PredictedFinalPositionStd',
                            'PredictedFinalPosition_Low', 'PredictedFinalPosition_High', 'PredictedPositionMAE',
                            'PredictedPositionMAE_Low', 'PredictedPositionMAE_High']
            present = [c for c in cols_to_show if c in pred.columns]
            if not present:
                print('\nPredictions file does not contain expected predicted columns; showing head:')
                print(pred.head(10).to_string(index=False))
            else:
                out = pred[present].sort_values(by='PredictedFinalPosition', na_position='last')
                print('\n-- Predictive Results for Active Drivers (from predictions file) --')
                print(out.to_string(index=False))
        except Exception as e:
            print('Error building display table from predictions file:', e)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""Dry-run helper: determine which predictions_*.csv would be chosen by the sender.

This prints the derived next race (from data_files/f1ForAnalysis.csv), the
chosen predictions file path (if any), and a small preview of that file.
"""
import os
import sys
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
try:
    import pandas as pd
    import glob
except Exception as e:
    print('Missing dependency:', e)
    sys.exit(2)

csv_path = os.path.join(ROOT, 'data_files', 'f1ForAnalysis.csv')
if not os.path.exists(csv_path):
    print('Analysis CSV not found at', csv_path)
    sys.exit(2)

try:
    df = pd.read_csv(csv_path, sep='\t')
except Exception as e:
    print('Failed to read analysis CSV:', e)
    sys.exit(2)

next_race_name = None
next_race_date = None
try:
    if 'short_date' in df.columns:
        df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
        now = pd.to_datetime(datetime.now())
        upcoming = df[df['short_date'] >= now].copy()
        if not upcoming.empty:
            nd = upcoming['short_date'].min()
            candidates = upcoming[upcoming['short_date'] == nd]
            if 'grandPrixName' in candidates.columns:
                vals = candidates['grandPrixName'].dropna().unique()
                if len(vals) > 0:
                    next_race_name = str(vals[0])
                    next_race_date = nd
except Exception:
    next_race_name = None
    next_race_date = None

print('Derived next_race_name:', next_race_name)
print('Derived next_race_date:', next_race_date)

# locate predictions files
pred_glob = os.path.join(ROOT, 'data_files', 'predictions_*.csv')
pred_files = sorted(glob.glob(pred_glob))
if not pred_files:
    print('No predictions_*.csv files found in data_files/')
    sys.exit(0)

candidate_files = []
for p in pred_files:
    sd = None
    try:
        try:
            tmp = pd.read_csv(p, nrows=5, sep=None, engine='python')
        except Exception:
            tmp = pd.read_csv(p, nrows=5)
        # Prefer matching next_race_name via grandPrixName if available
        if 'grandPrixName' in tmp.columns and next_race_name is not None:
            try:
                names = tmp['grandPrixName'].dropna().astype(str).str.lower().unique()
                if any(next_race_name.lower() in n for n in names):
                    sd_vals = pd.to_datetime(tmp['short_date'], errors='coerce') if 'short_date' in tmp.columns else pd.Series([])
                    sd = sd_vals.dropna().iloc[0] if (not sd_vals.dropna().empty) else None
            except Exception:
                sd = None
        # fallback: read short_date
        if sd is None and 'short_date' in tmp.columns:
            sd_vals = pd.to_datetime(tmp['short_date'], errors='coerce')
            sd = sd_vals.dropna().iloc[0] if sd_vals.dropna().size > 0 else None
    except Exception:
        sd = None
    candidate_files.append((p, sd))

now_dt = pd.to_datetime(datetime.now())
chosen_pred = None
# try to match by grandPrixName
if next_race_name is not None:
    for (p, sd) in candidate_files:
        try:
            try:
                tmp = pd.read_csv(p, nrows=50, sep=None, engine='python')
            except Exception:
                tmp = pd.read_csv(p, nrows=50)
            if 'grandPrixName' in tmp.columns:
                names = tmp['grandPrixName'].dropna().astype(str).str.lower().unique()
                if any(next_race_name.lower() in n for n in names):
                    chosen_pred = p
                    break
        except Exception:
            continue

# if no name match, prefer upcoming short_date
if chosen_pred is None:
    upcoming_candidates = [(p, sd) for (p, sd) in candidate_files if sd is not None and sd >= now_dt]
    if upcoming_candidates:
        upcoming_candidates.sort(key=lambda x: x[1])
        chosen_pred = upcoming_candidates[0][0]

# final fallback: newest by mtime
if chosen_pred is None and pred_files:
    pred_files_mtime_sorted = sorted(pred_files, key=os.path.getmtime, reverse=True)
    chosen_pred = pred_files_mtime_sorted[0]

print('\nFound prediction files:')
for p, sd in candidate_files:
    print('-', os.path.basename(p), '-> short_date:', sd)

print('\nChosen predictions file:')
print(chosen_pred)

# preview chosen file
if chosen_pred:
    try:
        dfp = pd.read_csv(chosen_pred)
        print('\nPreview (first 8 rows):')
        print(dfp.head(8).to_string(index=False))
    except Exception as e:
        print('Failed to read chosen predictions file:', e)

print('\nDry-run complete.')
