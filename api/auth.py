"""Password hashing + JWT-based bearer authentication + OTP utilities.

The signing key comes from `JWT_SECRET` env var; if absent we generate one at
process start and persist it under `data/.jwt_secret` so tokens survive
restarts on the same instance. For multi-replica deployments, set `JWT_SECRET`
explicitly to a stable value.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jwt
from passlib.context import CryptContext

from . import db

_pwd = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto",
                    pbkdf2_sha256__rounds=29000)

_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days
_ALGORITHM = "HS256"

OTP_TTL_SECONDS = 600          # 10 minutes
OTP_MAX_ATTEMPTS = 5


def _resolve_secret() -> str:
    """Pick up JWT_SECRET from env, else create/load a persistent local one."""
    s = os.environ.get("JWT_SECRET")
    if s:
        return s
    p = Path("data/.jwt_secret")
    if p.exists():
        return p.read_text().strip()
    p.parent.mkdir(parents=True, exist_ok=True)
    new = secrets.token_urlsafe(48)
    p.write_text(new)
    try:
        os.chmod(p, 0o600)
    except Exception:
        pass
    return new


_SECRET = _resolve_secret()


# ── Password policy ────────────────────────────────────────────────────

_SPECIAL_CHARS = r"""!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?`~"""

_PASSWORD_RULES = [
    (r".{8,}",                  "at least 8 characters"),
    (r"[A-Z]",                  "at least one uppercase letter"),
    (r"[a-z]",                  "at least one lowercase letter"),
    (r"[0-9]",                  "at least one digit"),
    (rf"[{_SPECIAL_CHARS}]",    "at least one special character"),
]


def validate_password_strength(password: str) -> Tuple[bool, str]:
    """Return (ok, error_message). error_message is '' when ok=True."""
    for pattern, desc in _PASSWORD_RULES:
        if not re.search(pattern, password):
            return False, f"Password must contain {desc}."
    return True, ""


# ── Password hashing ───────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd.verify(plain, hashed)
    except Exception:
        return False


# ── OTP ────────────────────────────────────────────────────────────────

def generate_otp() -> str:
    """Return a 6-digit numeric OTP (zero-padded)."""
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()


def verify_otp_hash(otp: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_otp(otp), stored_hash)


# ── Tokens ─────────────────────────────────────────────────────────────

def issue_token(user_id: str) -> Dict[str, Any]:
    now = int(time.time())
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + _TOKEN_TTL_SECONDS,
    }
    token = jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "expires_in": _TOKEN_TTL_SECONDS}


def decode_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        return jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def user_from_token(token: str) -> Optional[Dict[str, Any]]:
    payload = decode_token(token)
    if not payload:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    user = db.get_user_by_id(user_id)
    if not user:
        return None
    # Reject tokens issued before the last password reset
    iat = payload.get("iat", 0)
    invalidated_before = user.get("tokens_invalidated_before") or 0
    if iat < invalidated_before:
        return None
    return user
