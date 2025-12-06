import smtplib
from email.message import EmailMessage
from typing import List, Optional


def _html_to_text(h: str) -> str:
    """Simple HTML->text fallback for email clients that prefer plain text."""
    try:
        import re
        text = h
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</tr>|</div>|</p>|</table>|</h[1-6]>", "\n", text, flags=re.I)
        text = re.sub(r"</t[dh]>", "\t", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()
    except Exception:
        return "See HTML part for details."


def send_email(
    smtp_server: str,
    smtp_port: int,
    username: str,
    password: str,
    from_addr: str,
    to_addrs: List[str],
    subject: str,
    html_body: str,
    attachment_bytes: Optional[bytes] = None,
    attachment_name: str = "attachment.csv",
):
    """Send a simple multipart email with HTML and a plain-text fallback.

    Uses STARTTLS on the given `smtp_port` (commonly 587).
    """
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg["Subject"] = subject
    plain = _html_to_text(html_body)
    msg.set_content(plain)
    msg.add_alternative(html_body, subtype="html")

    if attachment_bytes is not None:
        msg.add_attachment(attachment_bytes, maintype="text", subtype="csv", filename=attachment_name)

    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls()
        server.ehlo()
        if username and password:
            server.login(username, password)
        server.send_message(msg)
