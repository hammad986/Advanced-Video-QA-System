"""Production email delivery via SMTP with STARTTLS.

Required secrets (set in Replit Secrets tab):
    SMTP_EMAIL     – sender address  (e.g. you@gmail.com)
    SMTP_PASSWORD  – SMTP password / Gmail app-password

Optional env vars:
    SMTP_HOST      – SMTP server hostname  (default: smtp.gmail.com)
    SMTP_PORT      – SMTP port             (default: 587, STARTTLS)
    SMTP_FROM_NAME – Display name in From  (default: Video QA)

If SMTP_EMAIL / SMTP_PASSWORD are absent, send_email() is a no-op that logs
a warning — the API never crashes due to missing mail configuration.

Emails are sent in a daemon thread so the API endpoint returns immediately.
"""

from __future__ import annotations

import logging
import os
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

_SMTP_HOST  = os.environ.get("SMTP_HOST",      "smtp.gmail.com")
_SMTP_PORT  = int(os.environ.get("SMTP_PORT",  "587"))
_FROM_NAME  = os.environ.get("SMTP_FROM_NAME", "Video QA")

# Resolved at import time — also re-read inside _send_sync so a restart
# after secret injection picks up the new values.
_SMTP_EMAIL    = os.environ.get("SMTP_EMAIL",    "")
_SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")


def configured() -> bool:
    """Return True when both SMTP credentials are present."""
    email = os.environ.get("SMTP_EMAIL",    _SMTP_EMAIL)
    pw    = os.environ.get("SMTP_PASSWORD", _SMTP_PASSWORD)
    return bool(email and pw)


def _send_sync(to: str, subject: str, body: str) -> bool:
    """Blocking SMTP send — runs in a daemon thread; never raises."""
    email = os.environ.get("SMTP_EMAIL",    _SMTP_EMAIL)
    pw    = os.environ.get("SMTP_PASSWORD", _SMTP_PASSWORD)

    if not email or not pw:
        logger.warning(
            "[email] SMTP not configured. Skipping send to=%s subject=%r",
            to, subject,
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = f"{_FROM_NAME} <{email}>"
        msg["To"]      = to
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(email, pw)
            server.sendmail(email, [to], msg.as_string())

        logger.info("[email] ✓ Sent to=%s subject=%r", to, subject)
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "[email] Authentication failed. Check SMTP_EMAIL / SMTP_PASSWORD."
        )
    except smtplib.SMTPException as exc:
        logger.error("[email] SMTP error sending to=%s: %s", to, exc)
    except OSError as exc:
        logger.error("[email] Network error sending to=%s: %s", to, exc)
    except Exception as exc:  # noqa: BLE001
        logger.error("[email] Unexpected error sending to=%s: %s", to, exc)

    return False


def send_email(to: str, subject: str, body: str) -> None:
    """Fire-and-forget email — returns immediately, sends in background thread.

    Never raises; failures are logged.
    """
    if not configured():
        logger.warning(
            "[email] SMTP not configured (SMTP_EMAIL/SMTP_PASSWORD missing). "
            "Email to=%s will not be sent. subject=%r",
            to, subject,
        )
        return

    t = threading.Thread(
        target=_send_sync,
        args=(to, subject, body),
        daemon=True,
        name=f"email-{to[:20]}",
    )
    t.start()
