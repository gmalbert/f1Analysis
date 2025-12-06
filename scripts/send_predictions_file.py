#!/usr/bin/env python3
"""Send the latest predictions_*.csv (or a specified file) as an email attachment.

Usage:
  python scripts\send_predictions_file.py              # send latest predictions_*.csv
  python scripts\send_predictions_file.py path/to/file.csv  # send specified file

Relies on `.env` for SMTP creds (same variables used elsewhere).
"""
import os
import sys
from pathlib import Path
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data_files'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    from notifications.emailer import send_email
except Exception as e:
    print('Missing imports:', e)
    sys.exit(2)

smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
smtp_port = int(os.getenv('SMTP_PORT', '587'))
email_from = os.getenv('EMAIL_FROM')
email_to = os.getenv('EMAIL_TO')
email_pass = os.getenv('EMAIL_PASSWORD')

missing = [k for k in ('EMAIL_FROM','EMAIL_TO','EMAIL_PASSWORD') if not os.getenv(k)]
if missing:
    print('Missing required env vars:', ', '.join(missing))
    sys.exit(2)

# Determine file to send
arg = sys.argv[1] if len(sys.argv) > 1 else None
if arg:
    send_path = Path(arg)
    if not send_path.exists():
        print('Specified file not found:', send_path)
        sys.exit(1)
else:
    # pick latest predictions_*.csv in data_files
    preds = sorted(DATA_DIR.glob('predictions_*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not preds:
        print('No predictions_*.csv files found in', DATA_DIR)
        sys.exit(1)
    send_path = preds[0]

print('Sending file:', send_path)
with open(send_path, 'rb') as fh:
    csv_bytes = fh.read()

recipients = [a.strip() for a in email_to.split(',') if a.strip()]
subject = f"Predictions file: {send_path.name}"
html_body = f"<p>Attached is the predictions file <strong>{send_path.name}</strong> generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}.</p>"

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
        attachment_name=send_path.name,
    )
    print('Sent:', ', '.join(recipients))
except Exception as e:
    print('Error sending file:', e)
    sys.exit(1)
