#!/usr/bin/env python3
"""Send a single test email using the project's `notifications.emailer` module.

This script loads `.env` (via python-dotenv when available), reads required
SMTP variables, and sends a minimal HTML test message to verify SMTP connectivity.
It prints only a success/failure message and never echoes secrets.
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # fallback: try to read a .env in the repo root if present
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    try:
        with open(env_path, 'r', encoding='utf-8') as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                k, v = line.split('=', 1)
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from notifications.emailer import send_email

required = ['SMTP_SERVER', 'SMTP_PORT', 'EMAIL_FROM', 'EMAIL_TO', 'EMAIL_PASSWORD']
missing = [k for k in required if not os.getenv(k)]
if missing:
    print('Missing environment variables:', ', '.join(missing))
    sys.exit(2)

smtp_server = os.getenv('SMTP_SERVER')
try:
    smtp_port = int(os.getenv('SMTP_PORT', '587'))
except Exception:
    smtp_port = 587
username = os.getenv('EMAIL_FROM')
password = os.getenv('EMAIL_PASSWORD')
from_addr = os.getenv('EMAIL_FROM')
to_addrs = [addr.strip() for addr in os.getenv('EMAIL_TO').split(',') if addr.strip()]

subject = 'Test — Gridlocked F1 App (automated)'
html_body = '<p>This is a test message from the <strong>Gridlocked F1</strong> app. If you received this, SMTP is working.</p>'

try:
    send_email(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        username=username,
        password=password,
        from_addr=from_addr,
        to_addrs=to_addrs,
        subject=subject,
        html_body=html_body,
    )
    print('Email sent successfully to:', ', '.join(to_addrs))
    sys.exit(0)
except Exception as e:
    print('Error sending email:', str(e))
    sys.exit(3)
