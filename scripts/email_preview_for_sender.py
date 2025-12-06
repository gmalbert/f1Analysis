"""Preview what `send_rich_email_now.py` would send without performing SMTP.

Finds the chosen predictions file, then looks for `data_files/email_context_*.html` and
`*.tsv` (preferring a file matching the race name) and prints:
 - Subject
 - Recipients (from env)
 - Attachment path chosen
 - First 800 chars of the HTML body that would be sent
"""
import os
import sys
import glob
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    import pandas as pd
except Exception as e:
    print('Missing dependency:', e)
    sys.exit(2)


def main():
    # reuse the existing dry-run selection logic to find chosen_pred by loading the dryrun module
    import runpy
    dry_globals = runpy.run_path(os.path.join(ROOT, 'scripts', 'send_rich_email_dryrun.py'))
    compute_next = dry_globals.get('compute_next_race_from_calendar')
    choose_pred = dry_globals.get('choose_predictions_file')
    if compute_next:
        next_race_name, next_race_date = compute_next(ROOT)
    else:
        next_race_name, next_race_date = (None, None)
    if choose_pred:
        chosen = choose_pred(ROOT, next_race_name)
    else:
        chosen = None

    # build minimal send_df similar to send_rich_email_now
    send_df = None
    used_predictions_file = False
    if chosen:
        try:
            send_df = pd.read_csv(chosen)
            used_predictions_file = True
        except Exception:
            send_df = None

    # determine gp_name for matching
    gp_name = None
    try:
        if send_df is not None and 'grandPrixName' in send_df.columns:
            vals = send_df['grandPrixName'].dropna().unique()
            if len(vals) > 0:
                gp_name = str(vals[0])
    except Exception:
        gp_name = None

    # find email_context files
    html_files = sorted(glob.glob(os.path.join(ROOT, 'data_files', 'email_context_*.html')))
    tsv_files = sorted(glob.glob(os.path.join(ROOT, 'data_files', 'email_context_*.tsv')))
    chosen_html = None
    chosen_tsv = None
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

    if chosen_html is None and html_files:
        chosen_html = sorted(html_files, key=os.path.getmtime, reverse=True)[0]
    if chosen_tsv is None and tsv_files:
        chosen_tsv = sorted(tsv_files, key=os.path.getmtime, reverse=True)[0]

    print('Preview of email send')
    print('---------------------')
    print('Next race name:', next_race_name)
    print('Chosen predictions file:', chosen)
    print('Found inline HTML snippet:', chosen_html)
    print('Found TSV attachment:', chosen_tsv)
    print('')
    if chosen_html:
        print('--- HTML snippet head (first 800 chars) ---')
        with open(chosen_html, 'r', encoding='utf-8') as fh:
            s = fh.read()
            print(s[:800])
    if chosen_tsv:
        print('\n--- TSV attachment sample (first 10 lines) ---')
        with open(chosen_tsv, 'r', encoding='utf-8') as fh:
            for i, ln in enumerate(fh):
                if i >= 10:
                    break
                print(ln.rstrip('\n'))


if __name__ == '__main__':
    main()
