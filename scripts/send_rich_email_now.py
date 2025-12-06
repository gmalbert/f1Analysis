#!/usr/bin/env python3
"""Send a rich HTML race predictions email now using project files and env vars.

This script reads `.env`, loads a small slice of `data_files/f1ForAnalysis.csv`,
renders a Jinja2 template in `notifications/templates`, and sends via SMTP.

It's intended for local/manual use; it uses environment variables for secrets.
"""
import os
import sys
from datetime import datetime

# load dotenv if available (best-effort)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import pandas as pd
    from jinja2 import Environment, FileSystemLoader
    from notifications.emailer import send_email
except Exception as e:
    print('Missing dependency or import error:', e)
    sys.exit(2)

# Read env vars
smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
smtp_port = int(os.getenv('SMTP_PORT', '587'))
email_from = os.getenv('EMAIL_FROM')
email_to = os.getenv('EMAIL_TO')
email_pass = os.getenv('EMAIL_PASSWORD')

missing = [k for k in ('EMAIL_FROM', 'EMAIL_TO', 'EMAIL_PASSWORD') if not os.getenv(k)]
if missing:
    print('Missing required environment variables:', ', '.join(missing))
    sys.exit(2)

# Load a small subset of the analysis CSV
csv_path = os.path.join(ROOT, 'data_files', 'f1ForAnalysis.csv')
if not os.path.exists(csv_path):
    print('Analysis CSV not found at', csv_path)
    sys.exit(2)

try:
    df = pd.read_csv(csv_path, sep='\t')
except Exception as e:
    print('Failed to read analysis CSV:', e)
    sys.exit(2)

# Determine next race name/date from the canonical calendar JSON (preferred).
next_race_name = None
next_race_date = None
try:
    now = pd.to_datetime(datetime.now())
    races_json = os.path.join(ROOT, 'data_files', 'f1db-races.json')
    if os.path.exists(races_json):
        import json
        with open(races_json, 'r', encoding='utf-8') as fh:
            races = json.load(fh)
        candidates = []
        for r in races:
            sd = r.get('short_date') or r.get('date') or r.get('race_date') or r.get('raceDate')
            try:
                sd_dt = pd.to_datetime(sd, errors='coerce')
            except Exception:
                sd_dt = None
            if sd_dt is not None and not pd.isna(sd_dt) and sd_dt >= now:
                # prefer explicit 'grandPrixName' but fall back to name fields
                name = r.get('grandPrixName') or r.get('name') or r.get('raceName')
                candidates.append((sd_dt, name))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            next_race_date, next_race_name = candidates[0]
    else:
        # fallback: derive next race from analysis CSV (older behavior)
        if 'short_date' in df.columns:
            df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
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

# Prefer an existing predictions_*.csv when available (most recent). This allows
# scheduled emails to use the produced predictions file instead of attempting
# to re-run model logic against the full analysis CSV.
try:
    import glob
    pred_glob = os.path.join(ROOT, 'data_files', 'predictions_*.csv')
    pred_files = sorted(glob.glob(pred_glob))
    # Prefer the predictions file that matches the next upcoming race by short_date.
    chosen_pred = None
    candidate_files = []
    for p in pred_files:
        try:
            # lightweight read to find short_date
            try:
                tmp = pd.read_csv(p, nrows=5, sep=None, engine='python')
            except Exception:
                tmp = pd.read_csv(p, nrows=5)
            sd = None
            # Prefer matching next_race_name via grandPrixName if available
            if 'grandPrixName' in tmp.columns and next_race_name is not None:
                try:
                    names = tmp['grandPrixName'].dropna().astype(str).str.lower().unique()
                    if any(next_race_name.lower() in n for n in names):
                        sd_vals = pd.to_datetime(tmp['short_date'], errors='coerce') if 'short_date' in tmp.columns else pd.Series([])
                        sd = sd_vals.dropna().iloc[0] if (not sd_vals.dropna().empty) else None
                except Exception:
                    sd = None
            # If no match by grandPrixName, fall back to reading short_date if present
            if sd is None and 'short_date' in tmp.columns:
                sd_vals = pd.to_datetime(tmp['short_date'], errors='coerce')
                sd = sd_vals.dropna().iloc[0] if sd_vals.dropna().size > 0 else None
        except Exception:
            sd = None
        candidate_files.append((p, sd))

    # choose the file that matches the next_race_name if found, otherwise pick the file whose short_date is >= today and minimal; otherwise fallback to newest by mtime
    now_dt = pd.to_datetime(datetime.now())
    chosen_pred = None
    # if we computed next_race_name, try to find a file whose grandPrixName matches it
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

    # If no file matched by name, prefer upcoming short_date files
    if chosen_pred is None:
        upcoming_candidates = [(p, sd) for (p, sd) in candidate_files if sd is not None and sd >= now_dt]
        if upcoming_candidates:
            # pick the file with the smallest short_date
            upcoming_candidates.sort(key=lambda x: x[1])
            chosen_pred = upcoming_candidates[0][0]

    # final fallback: most recently modified predictions file
    if chosen_pred is None and pred_files:
        pred_files_mtime_sorted = sorted(pred_files, key=os.path.getmtime, reverse=True)
        chosen_pred = pred_files_mtime_sorted[0]

    if chosen_pred:
        try:
            try:
                pred_df = pd.read_csv(chosen_pred, sep=None, engine='python')
            except Exception:
                pred_df = pd.read_csv(chosen_pred)
            if not pred_df.empty:
                send_df = pred_df.copy()
                if 'short_date' in send_df.columns:
                    send_df['short_date'] = pd.to_datetime(send_df['short_date'], errors='coerce')
                used_predictions_file = True
                # make pred_files list available for attachment logic
                pred_files = [chosen_pred]
            else:
                used_predictions_file = False
        except Exception:
            used_predictions_file = False
    else:
        used_predictions_file = False
    # If no predictions file was selected, attempt a headless generation as fallback
    if not (('used_predictions_file' in locals() and used_predictions_file) or (chosen_pred is not None)):
        try:
            headless_script = os.path.join(ROOT, 'scripts', 'headless_predict_and_write.py')
            if os.path.exists(headless_script):
                # run headless generator to produce a predictions file for the next race
                import subprocess
                subprocess.run([sys.executable, headless_script], check=False)
                # prefer headless predictions file if produced
                import glob
                headless_glob = os.path.join(ROOT, 'data_files', 'predictions_headless_*.csv')
                headless_files = sorted(glob.glob(headless_glob), key=os.path.getmtime, reverse=True)
                if headless_files:
                    chosen_pred = headless_files[0]
                    try:
                        pred_df = pd.read_csv(chosen_pred)
                        if not pred_df.empty:
                            send_df = pred_df.copy()
                            if 'short_date' in send_df.columns:
                                send_df['short_date'] = pd.to_datetime(send_df['short_date'], errors='coerce')
                            used_predictions_file = True
                            pred_files = [chosen_pred]
                    except Exception:
                        used_predictions_file = False
        except Exception:
            pass
except Exception:
    used_predictions_file = False

# For the email, pick rows for the next upcoming race (earliest future short_date)
# If a predictions file was already selected above, skip re-selecting from the
# large analysis CSV to avoid overwriting the predictions content.
if not ('used_predictions_file' in locals() and used_predictions_file):
    try:
        df['short_date'] = pd.to_datetime(df['short_date'], errors='coerce')
        now = pd.to_datetime(datetime.now())
        upcoming = df[df['short_date'] >= now].copy()
        if not upcoming.empty:
            # find the earliest upcoming race date and select all rows for that date
            next_date = upcoming['short_date'].min()
            send_df = upcoming[upcoming['short_date'] == next_date].copy()
            # Only send if the next race is within 3 days
            try:
                delta = (next_date - now).total_seconds() / 86400.0
                if delta > 3:
                    print(f"Next race at {next_date.date()} is more than 3 days away ({delta:.1f} days). Skipping send.")
                    sys.exit(0)
            except Exception:
                # if date arithmetic fails, continue and send (conservative)
                pass
            # Prefer sorting by predicted final position (ascending). Fall back to driver name.
            sort_col = 'PredictedFinalPosition' if 'PredictedFinalPosition' in send_df.columns else 'resultsDriverName'
            send_df = send_df.sort_values(by=sort_col, na_position='last')
        else:
            # no future races: fallback to the most recent race in data
            latest_date = df['short_date'].max()
            if pd.isna(latest_date):
                send_df = df.head(50)
            else:
                send_df = df[df['short_date'] == latest_date].copy()
                sort_col = 'PredictedFinalPosition' if 'PredictedFinalPosition' in send_df.columns else 'resultsDriverName'
                send_df = send_df.sort_values(by=sort_col, na_position='last')
    except Exception:
        send_df = df.head(50)
else:
    # We're using a predictions file already loaded into send_df; ensure it's sorted
    try:
        sort_col = 'PredictedFinalPosition' if 'PredictedFinalPosition' in send_df.columns else 'resultsDriverName'
        send_df = send_df.sort_values(by=sort_col, na_position='last')
    except Exception:
        pass

# Prepare rows and render Jinja2 template
rows = send_df.to_dict(orient='records')
# Add formatted display fields for template
for r in rows:
    # short_date formatting
    try:
        sd = r.get('short_date')
        if sd is None:
            r['short_date_fmt'] = ''
        else:
            r['short_date_fmt'] = pd.to_datetime(sd, errors='coerce').strftime('%Y-%m-%d')
    except Exception:
        r['short_date_fmt'] = str(r.get('short_date'))
    # Predicted final position: integer if effectively integer, otherwise one decimal
    p = r.get('PredictedFinalPosition')
    try:
        if p is None or (isinstance(p, float) and pd.isna(p)):
            r['PredictedFinalPositionFmt'] = ''
        else:
            pf = float(p)
            if abs(pf - round(pf)) < 0.01:
                r['PredictedFinalPositionFmt'] = str(int(round(pf)))
            else:
                r['PredictedFinalPositionFmt'] = f"{pf:.1f}"
    except Exception:
        r['PredictedFinalPositionFmt'] = str(p)

    # MAE formatting
    mae = r.get('PredictedPositionMAE')
    try:
        if mae is None or (isinstance(mae, float) and pd.isna(mae)):
            r['PredictedPositionMAEFmt'] = ''
        else:
            r['PredictedPositionMAEFmt'] = f"{float(mae):.2f}"
    except Exception:
        r['PredictedPositionMAEFmt'] = str(mae)

# setup template rendering and determine which columns to show
tmpl_dir = os.path.join(ROOT, 'notifications', 'templates')
env = Environment(loader=FileSystemLoader(tmpl_dir), autoescape=True)
tmpl = env.get_template('race_results.html.j2')

# Determine race date string to display in the email header
race_date = None
try:
    if not send_df.empty and 'short_date' in send_df.columns:
        try:
            first_dt = pd.to_datetime(send_df['short_date'].dropna().iloc[0])
            race_date = first_dt.strftime('%Y-%m-%d')
        except Exception:
            race_date = str(send_df['short_date'].dropna().iloc[0])
except Exception:
    race_date = None

# determine gp_name for template (race display name)
gp_name = None
try:
    if 'send_df' in locals() and send_df is not None and 'grandPrixName' in send_df.columns:
        vals = [str(x) for x in send_df['grandPrixName'].dropna().unique()]
        if vals:
            gp_name = vals[0]
except Exception:
    gp_name = None

# Flags for template to decide which columns to render
include_prediction = 'PredictedFinalPosition' in send_df.columns or any(('PredictedFinalPositionFmt' in r and r['PredictedFinalPositionFmt']) for r in rows)
include_mae = 'PredictedPositionMAE' in send_df.columns or any(('PredictedPositionMAEFmt' in r and r['PredictedPositionMAEFmt']) for r in rows)
include_constructor = 'constructorName' in send_df.columns or any(('constructorName' in r and r['constructorName']) for r in rows)

html_body = tmpl.render(
    title='F1 Predictions',
    generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
    # summary intentionally omitted; race_date is shown separately in the template
    race_date=race_date,
    race_name=gp_name,
    rows=rows,
    include_prediction=include_prediction,
    include_mae=include_mae,
    include_constructor=include_constructor,
    predictions_mode = (used_predictions_file if 'used_predictions_file' in locals() else False),
)

# Determine gp_name (race display name) to help pick matching snippet files
gp_name = None
try:
    if not send_df.empty and 'grandPrixName' in send_df.columns:
        try:
            names = [str(x) for x in send_df['grandPrixName'].dropna().unique()]
            if names:
                gp_name = names[0]
        except Exception:
            gp_name = None
except Exception:
    gp_name = None

# Optionally embed a pre-rendered HTML snippet and attach its TSV if present
inline_html = None
attached_tsv_path = None
try:
    import glob
    html_glob = os.path.join(ROOT, 'data_files', 'email_context_*.html')
    tsv_glob = os.path.join(ROOT, 'data_files', 'email_context_*.tsv')
    html_files = sorted(glob.glob(html_glob))
    tsv_files = sorted(glob.glob(tsv_glob))
    chosen_html = None
    chosen_tsv = None
    # Try to pick a snippet that matches the gp_name if available
    if gp_name:
        slug = gp_name.lower().replace(' ', '-').replace('_', '-')
        for h in html_files:
            if slug in os.path.basename(h).lower():
                chosen_html = h
                break
        for t in tsv_files:
            if slug in os.path.basename(t).lower():
                chosen_tsv = t
                break
    # Always run the exporter to regenerate the inline snippet/TSV for this send.
    try:
        exporter = os.path.join(ROOT, 'scripts', 'export_email_context.py')
        if os.path.exists(exporter):
            import subprocess
            args = [sys.executable, exporter]
            # If we have a specific predictions file selected earlier, pass it to the exporter
            if 'pred_files' in locals() and pred_files:
                args.extend(['--input', pred_files[0]])
            subprocess.run(args, check=False)
            # refresh file lists after exporter runs
            html_files = sorted(glob.glob(html_glob))
            tsv_files = sorted(glob.glob(tsv_glob))
    except Exception:
        pass

    # Try to pick a snippet that matches the gp_name if available
    if gp_name:
        slug = gp_name.lower().replace(' ', '-').replace('_', '-')
        for h in html_files:
            if slug in os.path.basename(h).lower():
                chosen_html = h
                break
        for t in tsv_files:
            if slug in os.path.basename(t).lower():
                chosen_tsv = t
                break

    # If none matched by name, pick the newest HTML/TSV if present
    if chosen_html is None and html_files:
        chosen_html = sorted(html_files, key=os.path.getmtime, reverse=True)[0]
    if chosen_tsv is None and tsv_files:
        chosen_tsv = sorted(tsv_files, key=os.path.getmtime, reverse=True)[0]

    if chosen_html and os.path.exists(chosen_html):
        inline_html = open(chosen_html, 'r', encoding='utf-8').read()
    if chosen_tsv and os.path.exists(chosen_tsv):
        attached_tsv_path = chosen_tsv
except Exception:
    inline_html = None
    attached_tsv_path = None

# If we found an inline snippet, try to extract race name/date from it (fallbacks)
if inline_html:
    try:
        import re
        # look for <div><strong>Date:</strong> YYYY-MM-DD</div>
        m = re.search(r"<div>\s*<strong>\s*Date:\s*</strong>\s*([^<]+)\s*</div>", inline_html, flags=re.I)
        if m:
            race_date = m.group(1).strip()
        # look for <div><strong>Race:</strong> Name</div>
        m2 = re.search(r"<div>\s*<strong>\s*Race:\s*</strong>\s*([^<]+)\s*</div>", inline_html, flags=re.I)
        if m2 and not gp_name:
            gp_name = m2.group(1).strip()
    except Exception:
        pass

    # If we still don't have race_date but have gp_name, try to look it up in the calendar JSON
    if (not race_date) and gp_name:
        try:
            races_json = os.path.join(ROOT, 'data_files', 'f1db-races.json')
            if os.path.exists(races_json):
                import json
                with open(races_json, 'r', encoding='utf-8') as fh:
                    races = json.load(fh)
                for r in races:
                    name = r.get('grandPrixName') or r.get('name') or r.get('raceName')
                    if name and gp_name.lower() in str(name).lower():
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

# If we found an inline snippet, inject it into the template rendering
if inline_html:
    html_body = tmpl.render(
        title='F1 Predictions',
        generated_at=datetime.now().strftime('%Y-%m-%d %H:%M'),
        race_date=race_date,
        race_name=gp_name,
        rows=rows,
        include_prediction=include_prediction,
        include_mae=include_mae,
        include_constructor=include_constructor,
        predictions_mode = (used_predictions_file if 'used_predictions_file' in locals() else False),
        inline_table_html=inline_html,
    )

# Attachment bytes (CSV)
# If we used a real predictions file, attach the original file bytes and
# use its filename. Otherwise, attach a generated TSV of `send_df`.
attachment_name = 'f1_predictions.csv'
csv_bytes = None
try:
    # Prefer a prepared TSV attachment if available (produced by export_email_context)
    if 'attached_tsv_path' in locals() and attached_tsv_path and os.path.exists(attached_tsv_path):
        with open(attached_tsv_path, 'rb') as fh:
            csv_bytes = fh.read()
        attachment_name = os.path.basename(attached_tsv_path)
    elif 'used_predictions_file' in locals() and used_predictions_file and 'pred_files' in locals() and pred_files:
        pred_file_path = pred_files[0]
        with open(pred_file_path, 'rb') as fh:
            csv_bytes = fh.read()
        attachment_name = os.path.basename(pred_file_path)
    else:
        csv_bytes = send_df.to_csv(index=False, sep='\t').encode('utf-8')
        attachment_name = 'f1_predictions.tsv'
except Exception:
    # fallback to in-memory CSV
    csv_bytes = send_df.to_csv(index=False, sep='\t').encode('utf-8')
    attachment_name = 'f1_predictions.tsv'

recipients = [a.strip() for a in email_to.split(',') if a.strip()]
# Build a descriptive subject using the upcoming race name when available
gp_name = None
if not send_df.empty and 'grandPrixName' in send_df.columns:
    try:
        names = [str(x) for x in send_df['grandPrixName'].dropna().unique()]
        if names:
            gp_name = names[0]
    except Exception:
        gp_name = None

if gp_name:
    subject = f"Predictions for the upcoming {gp_name} race"
else:
    subject = f"F1 Predictions — {datetime.now().strftime('%Y-%m-%d')}"

try:
    send_email(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        username=email_from,
        password=email_pass,
        from_addr=email_from,
        to_addrs=recipients,
        subject=subject,
        html_body=html_body,
        attachment_bytes=csv_bytes,
        attachment_name='f1_predictions.csv',
    )
    print('Rich email sent successfully to:', ', '.join(recipients))
    sys.exit(0)
except Exception as e:
    print('Error sending rich email:', e)
    sys.exit(3)
