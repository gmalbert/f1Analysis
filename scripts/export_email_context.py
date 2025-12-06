import pandas as pd
import os
from pathlib import Path

DATA_DIR = Path("data_files")


def find_predictions_file(preferred_name=None):
    # Look for candidate prediction CSVs in DATA_DIR; prefer an explicit name match,
    # then try to match next-race from calendar JSON (grandPrixName), else newest.
    candidates = sorted(DATA_DIR.glob('predictions_*.csv'))
    if not candidates:
        return None
    if preferred_name:
        p = DATA_DIR / preferred_name
        if p.exists():
            return p
    # try calendar lookup for next race name
    races_json = DATA_DIR / 'f1db-races.json'
    next_name = None
    if races_json.exists():
        try:
            import json
            from datetime import datetime
            with open(races_json, 'r', encoding='utf-8') as fh:
                races = json.load(fh)
            now = pd.to_datetime(datetime.now())
            future = []
            for r in races:
                sd = r.get('short_date') or r.get('date') or r.get('race_date') or r.get('raceDate')
                sd_dt = pd.to_datetime(sd, errors='coerce')
                if sd_dt is not None and not pd.isna(sd_dt) and sd_dt >= now:
                    name = r.get('grandPrixName') or r.get('name') or r.get('raceName')
                    future.append((sd_dt, name))
            if future:
                future.sort(key=lambda x: x[0])
                next_name = future[0][1]
        except Exception:
            next_name = None
    if next_name:
        # try to find a candidate that contains the slugified race name
        slug = ''.join(ch.lower() if ch.isalnum() else '-' for ch in str(next_name))
        for c in candidates:
            if slug in c.name.lower():
                return c
    # fallback: choose newest file
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


INPUT = None
TSV_OUT = DATA_DIR / "email_context_abu-dhabi_2025.tsv"
HTML_OUT = DATA_DIR / "email_context_abu-dhabi_2025.html"


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export email context (TSV + HTML) from a predictions CSV')
    parser.add_argument('--input', '-i', help='Predictions CSV to use (path relative to repo root or absolute)')
    args = parser.parse_args()

    global INPUT
    if args.input:
        INPUT = Path(args.input)
        if not INPUT.exists():
            # allow relative to data_files
            cand = DATA_DIR / Path(args.input).name
            if cand.exists():
                INPUT = cand
    else:
        INPUT = find_predictions_file()

    if INPUT is None or not INPUT.exists():
        print(f"No predictions input found in {DATA_DIR}. Looked for predictions_*.csv")
        return

    # predictions CSV is comma-delimited in this repo
    df = pd.read_csv(INPUT, sep=',')

    # Ensure expected columns exist (fallback to common alternatives)
    col_map = {
        'constructorName': None,
        'resultsDriverName': None,
        'PredictedFinalPosition': None,
        # PredictedPositionMAE is optional; detect when present
        'PredictedPositionMAE': None,
    }

    for c in df.columns:
        lc = c.lower()
        if lc == 'constructorname':
            col_map['constructorName'] = c
        if lc in ('resultsdrivername', 'driver'):
            col_map['resultsDriverName'] = c
        if lc == 'predictedfinalposition' or lc == 'predictedposition':
            col_map['PredictedFinalPosition'] = c
        if lc in ('predictedpositionmae','predictedposition_mae','predictedpositionmae_l','predictedpositionmae_h','predictedposition_mae_l'):
            # primary MAE column
            if col_map.get('PredictedPositionMAE') is None:
                col_map['PredictedPositionMAE'] = c

    # PredictedPositionMAE is optional; require only constructorName, resultsDriverName, PredictedFinalPosition
    required = [k for k in ('constructorName', 'resultsDriverName', 'PredictedFinalPosition')]
    missing = [k for k in required if col_map.get(k) is None]
    if missing:
        print(f"Missing expected columns: {missing} — available columns: {list(df.columns)}")
        return

    # Include Rank if present and order columns: Rank, Constructor, Driver, PredictedFinalPosition
    cols = []
    rank_col = None
    for c in df.columns:
        if c.lower() == 'rank':
            rank_col = c
            break
    if rank_col:
        cols.append(rank_col)
    cols.extend([col_map['constructorName'], col_map['resultsDriverName'], col_map['PredictedFinalPosition']])
    # include MAE if present
    if col_map.get('PredictedPositionMAE'):
        cols.append(col_map['PredictedPositionMAE'])
    out = df[cols].copy()
    # normalize column names
    rename_map = {}
    if rank_col:
        rename_map[rank_col] = 'Rank'
    rename_map[col_map['constructorName']] = 'Constructor'
    rename_map[col_map['resultsDriverName']] = 'Driver'
    rename_map[col_map['PredictedFinalPosition']] = 'PredictedFinalPosition'
    if col_map.get('PredictedPositionMAE'):
        rename_map[col_map['PredictedPositionMAE']] = 'MAE'
    out = out.rename(columns=rename_map)
    # If there's a Rank column, use it to order top->bottom
    rank_col = None
    for c in df.columns:
        if c.lower() == 'rank':
            rank_col = c
            break
    if 'Rank' in out.columns:
        try:
            out['Rank'] = out['Rank'].astype(int)
            out = out.sort_values('Rank', ascending=True)
        except Exception:
            pass

    # Round predicted position to 3 decimals for email readability
    out['PredictedFinalPosition'] = out['PredictedFinalPosition'].astype(float).round(3)
    if 'MAE' in out.columns:
        try:
            out['MAE'] = out['MAE'].astype(float).round(3)
        except Exception:
            pass

    # Write TSV for email attachment
    out.to_csv(TSV_OUT, sep='\t', index=False)
    print(f"Wrote TSV: {TSV_OUT}")

    # Create a prettier HTML table for inline email context
    style = '''
<style>
  table.f1 { border-collapse: collapse; width: 100%; font-family: Arial, Helvetica, sans-serif; }
  table.f1 th, table.f1 td { padding: 6px 8px; }
  table.f1 th { background: #f2f2f2; text-align: left; font-weight: 600; }
  table.f1 tr:nth-child(even) { background: #fafafa; }
  table.f1 td.num { text-align: right; font-family: Menlo, Monaco, monospace; }
</style>
'''

    # extract race name & date from the predictions CSV when available
    race_name = None
    race_date = None
    for c in df.columns:
        if c.lower() == 'grandprixname' and race_name is None:
            race_name = df[c].dropna().astype(str).mode().iloc[0] if not df[c].dropna().empty else None
        if c.lower() == 'short_date' and race_date is None:
            try:
                sd = pd.to_datetime(df[c], errors='coerce')
                if not sd.dropna().empty:
                    race_date = sd.dropna().iloc[0].strftime('%Y-%m-%d')
            except Exception:
                race_date = None

    html = [style]
    # include race name/date above the table so it's always present in emails
    # If we don't have race_date but have race_name, try to look it up in the calendar JSON
    if not race_date and race_name:
        try:
            races_json = DATA_DIR / 'f1db-races.json'
            if races_json.exists():
                import json
                with open(races_json, 'r', encoding='utf-8') as fh:
                    races = json.load(fh)
                for r in races:
                    name = r.get('grandPrixName') or r.get('name') or r.get('raceName')
                    if name and race_name.lower() in str(name).lower():
                        sd = r.get('short_date') or r.get('date') or r.get('race_date') or r.get('raceDate')
                        try:
                            sd_dt = pd.to_datetime(sd, errors='coerce')
                            if not pd.isna(sd_dt):
                                race_date = sd_dt.strftime('%Y-%m-%d')
                                break
                        except Exception:
                            continue
        except Exception:
            pass

    if race_name:
        html.append(f"<div><strong>Race:</strong> {race_name}</div>")
    if race_date:
        html.append(f"<div><strong>Date:</strong> {race_date}</div>")
    html.append('<table class="f1">')
    # include Rank header if present
    if 'Rank' in out.columns:
        # Predicted header left-aligned per request; numeric values remain right-aligned
        if 'MAE' in out.columns:
            html.append('<thead><tr><th style="width:48px">Rank</th><th>Constructor</th><th>Driver</th><th>Predicted</th><th>MAE</th></tr></thead>')
        else:
            html.append('<thead><tr><th style="width:48px">Rank</th><th>Constructor</th><th>Driver</th><th>Predicted</th></tr></thead>')
    else:
        if 'MAE' in out.columns:
            html.append('<thead><tr><th>Constructor</th><th>Driver</th><th>Predicted</th><th>MAE</th></tr></thead>')
        else:
            html.append('<thead><tr><th>Constructor</th><th>Driver</th><th>Predicted</th></tr></thead>')
    html.append('<tbody>')
    for _, r in out.iterrows():
        pred = float(r['PredictedFinalPosition'])
        pred_s = f"{pred:.3f}"
        if 'Rank' in out.columns:
            rank_v = '' if pd.isna(r['Rank']) else int(r['Rank'])
            if 'MAE' in out.columns:
                mae_v = '' if pd.isna(r.get('MAE')) else f"{float(r.get('MAE')):.3f}"
                html.append(f"<tr><td>{rank_v}</td><td>{r['Constructor']}</td><td>{r['Driver']}</td><td class=\"num\">{pred_s}</td><td class=\"num\">{mae_v}</td></tr>")
            else:
                html.append(f"<tr><td>{rank_v}</td><td>{r['Constructor']}</td><td>{r['Driver']}</td><td class=\"num\">{pred_s}</td></tr>")
        else:
            if 'MAE' in out.columns:
                mae_v = '' if pd.isna(r.get('MAE')) else f"{float(r.get('MAE')):.3f}"
                html.append(f"<tr><td>{r['Constructor']}</td><td>{r['Driver']}</td><td class=\"num\">{pred_s}</td><td class=\"num\">{mae_v}</td></tr>")
            else:
                html.append(f"<tr><td>{r['Constructor']}</td><td>{r['Driver']}</td><td class=\"num\">{pred_s}</td></tr>")
    html.append('</tbody></table>')

    HTML_OUT.write_text('\n'.join(html), encoding='utf-8')
    print(f"Wrote HTML snippet: {HTML_OUT}")

    # Print a concise plain-text summary to stdout (ordered by Rank if available)
    print('\nConstructor — Driver — PredictedFinalPosition (top→bottom)')
    for _, r in out.iterrows():
        print(f"- {r['Constructor']} — {r['Driver']} — {r['PredictedFinalPosition']}")


if __name__ == '__main__':
    main()
