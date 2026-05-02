"""Password hashing + JWT bearer auth + OTP utilities + Google OAuth helpers.

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

_pwd = CryptContext(
    schemes=["pbkdf2_sha256"],
    deprecated="auto",
    pbkdf2_sha256__rounds=29000,
)

_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days
_ALGORITHM = "HS256"

OTP_TTL_SECONDS = 600       # 10 minutes
OTP_MAX_ATTEMPTS = 5


def _resolve_secret() -> str:
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


# ── Google OAuth config ────────────────────────────────────────────────

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
GOOGLE_AUTH_URL      = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL     = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL  = "https://www.googleapis.com/oauth2/v2/userinfo"
GOOGLE_SCOPES        = "openid email profile"

# In-memory CSRF state store {state_token: created_at}
# Single-worker safe; for multi-replica deployments use Redis or a DB table.
_oauth_states: Dict[str, float] = {}
_OAUTH_STATE_TTL = 300  # 5 minutes


def google_oauth_configured() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)


def generate_oauth_state() -> str:
    """Generate a cryptographically random state token and store it."""
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = time.time()
    # Prune expired states
    cutoff = time.time() - _OAUTH_STATE_TTL
    expired = [k for k, v in _oauth_states.items() if v < cutoff]
    for k in expired:
        _oauth_states.pop(k, None)
    return state


def consume_oauth_state(state: str) -> bool:
    """Return True and remove state if valid and not expired."""
    created = _oauth_states.pop(state, None)
    if created is None:
        return False
    return (time.time() - created) < _OAUTH_STATE_TTL


def build_google_auth_url(redirect_uri: str) -> str:
    from urllib.parse import urlencode
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         GOOGLE_SCOPES,
        "state":         generate_oauth_state(),
        "access_type":   "online",
        "prompt":        "select_account",
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def exchange_google_code(code: str, redirect_uri: str) -> Dict[str, Any]:
    """Exchange auth code for tokens and return Google user info dict."""
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code":          code,
                "client_id":     GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri":  redirect_uri,
                "grant_type":    "authorization_code",
            },
        )
        token_resp.raise_for_status()
        token_data = token_resp.json()

        user_resp = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        user_resp.raise_for_status()
        return user_resp.json()


# ── Password policy ────────────────────────────────────────────────────

_SPECIAL_CHARS = r"""!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?`~"""

_PASSWORD_RULES = [
    (r".{8,}",               "at least 8 characters"),
    (r"[A-Z]",               "at least one uppercase letter"),
    (r"[a-z]",               "at least one lowercase letter"),
    (r"[0-9]",               "at least one digit"),
    (rf"[{_SPECIAL_CHARS}]", "at least one special character"),
]


def validate_password_strength(password: str) -> Tuple[bool, str]:
    for pattern, desc in _PASSWORD_RULES:
        if not re.search(pattern, password):
            return False, f"Password must contain {desc}."
    return True, ""


# ── Password hashing ───────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)


def verify_password(plain: str, hashed: Optional[str]) -> bool:
    if not hashed or hashed.startswith("oauth:"):
        return False
    try:
        return _pwd.verify(plain, hashed)
    except Exception:
        return False


# ── OTP ────────────────────────────────────────────────────────────────

def generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode()).hexdigest()


def verify_otp_hash(otp: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_otp(otp), stored_hash)


# ── Tokens ─────────────────────────────────────────────────────────────

def issue_token(user_id: str) -> Dict[str, Any]:
    now = int(time.time())
    payload = {"sub": user_id, "iat": now, "exp": now + _TOKEN_TTL_SECONDS}
    token = jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": _TOKEN_TTL_SECONDS,
    }


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
    iat = payload.get("iat", 0)
    invalidated_before = user.get("tokens_invalidated_before") or 0
    if iat < invalidated_before:
        return None
    return user
