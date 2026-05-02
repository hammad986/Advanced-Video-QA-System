"""FastAPI application — SaaS surface over the Video-QA RAG pipeline.

Endpoints
---------
Auth
  POST /auth/register              create account (password policy enforced)
  POST /auth/login                 obtain bearer token (blocked if unverified)
  GET  /auth/me                    current user (auth required)
  POST /auth/verify_email          verify email address with 6-digit code
  POST /auth/resend_verification   resend verification code (rate-limited)
  POST /auth/change_password       change password (auth required)
  POST /auth/request_reset         OTP password reset, step 1
  POST /auth/verify_otp            OTP password reset, step 2 (consumes code)
  POST /auth/reset_password        OTP password reset, step 3
  GET  /auth/google                redirect to Google OAuth consent screen
  GET  /auth/google/callback       handle Google OAuth callback

Videos
  POST /upload_video               multipart upload (auth)
  POST /process_video              transcribe + index an uploaded video (auth)
  POST /process_url                download + index a YouTube video (auth, rate-limited)
  GET  /videos                     list current user's videos (auth)

Q&A
  POST /ask_question               ask a question (auth)

System
  GET  /health                     public health probe
  GET  /docs                       Swagger UI
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import (
    Depends,
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from . import auth, db
from .compare import router as compare_router
from .schemas import (
    AskIn,
    AskOut,
    ChangePasswordIn,
    ChangePasswordOut,
    HealthOut,
    LoginIn,
    ProcessIn,
    ProcessOut,
    ProcessURLIn,
    ProcessURLOut,
    RegisterIn,
    RequestResetIn,
    RequestResetOut,
    ResendVerificationIn,
    ResendVerificationOut,
    ResetPasswordIn,
    ResetPasswordOut,
    TokenOut,
    UploadOut,
    UserOut,
    VerifyEmailIn,
    VerifyEmailOut,
    VerifyOTPIn,
    VerifyOTPOut,
    VideoOut,
)

logger = logging.getLogger("video_qa.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# ── Constants ──────────────────────────────────────────────────────────
MAX_UPLOAD_MB            = int(os.environ.get("VIDEO_QA_MAX_UPLOAD_MB", "300"))
MAX_URL_MB               = 200
MAX_URL_DURATION_MINUTES = 30
ALLOWED_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".mpeg4"}
USER_UPLOAD_ROOT = Path("data/users")

_ALLOWED_URL_DOMAINS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}

# Rate limit windows
_RATE_LOGIN_MAX         = 10   # per IP per 15 min
_RATE_LOGIN_WIN         = 900
_RATE_REGISTER_MAX      = 5    # per IP per hour
_RATE_REGISTER_WIN      = 3600
_RATE_OTP_MAX           = 3    # per user per hour
_RATE_OTP_WIN           = 3600
_RATE_URL_MAX           = 3    # per user per hour
_RATE_URL_WIN           = 3600
_RATE_EMAIL_VER_MAX     = 5    # resend verification per IP per hour
_RATE_EMAIL_VER_WIN     = 3600

_EMAIL_VER_TTL = 60 * 60 * 24  # 24 hours

# Compute the Google OAuth redirect URI
_REPLIT_DEV_DOMAIN = os.environ.get("REPLIT_DEV_DOMAIN", "")
_GOOGLE_REDIRECT_URI = os.environ.get(
    "GOOGLE_REDIRECT_URI",
    f"https://{_REPLIT_DEV_DOMAIN}/auth/google/callback" if _REPLIT_DEV_DOMAIN else "",
)

# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Video-QA SaaS API",
    version="1.0.0",
    description=(
        "Production API for the Video-QA RAG system. Upload videos, index them, "
        "and ask grounded questions with timestamps, confidence, and "
        "hallucination-aware status."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)

# ── Lazy pipeline singleton ────────────────────────────────────────────
_pipeline = None
_pipeline_lock = threading.Lock()
_index_write_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from video_qa.pipeline import VideoQAPipeline
                logger.info("[boot] initializing VideoQAPipeline…")
                _pipeline = VideoQAPipeline()
                logger.info("[boot] pipeline ready")
    return _pipeline


# ── Auth dependency ────────────────────────────────────────────────────
def current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    user = auth.user_from_token(creds.credentials)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return user


# ── Rate limiting helpers ──────────────────────────────────────────────
def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(
    key: str, max_count: int, window_seconds: float, label: str
) -> None:
    if not db.check_rate_limit(key, max_count, window_seconds):
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Too many {label} attempts. Please try again later.",
        )
    db.record_rate_event(key)


# ── Lifecycle ──────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    USER_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    backend = "postgresql" if db._USE_PG else "sqlite"
    logger.info("[boot] DB backend=%s path=%s", backend, db.DB_PATH if not db._USE_PG else "DATABASE_URL")
    if auth.google_oauth_configured():
        logger.info("[boot] Google OAuth configured — redirect URI: %s", _GOOGLE_REDIRECT_URI)
    else:
        logger.warning("[boot] Google OAuth NOT configured (GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET missing)")


app.include_router(compare_router)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_static_dir), html=True), name="ui")


# ── Health ─────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthOut, tags=["system"])
def health() -> HealthOut:
    chunks = 0
    try:
        from video_qa.embeddings import VectorStore
        vs = VectorStore()
        if vs._load():
            chunks = len(vs.metadata) if vs.metadata is not None else 0
    except Exception as e:
        logger.warning("[health] index introspection failed: %s", e)
    n_users = n_videos = 0
    try:
        with db._conn() as c:
            n_users  = c.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()["cnt"]
            n_videos = c.execute("SELECT COUNT(*) AS cnt FROM videos").fetchone()["cnt"]
    except Exception as e:
        logger.warning("[health] db introspection failed: %s", e)
    return HealthOut(
        status="ok",
        indexed_chunks=chunks,
        db_users=n_users,
        db_videos=n_videos,
        db_backend="postgresql" if db._USE_PG else "sqlite",
    )


# ── Auth routes ────────────────────────────────────────────────────────
@app.post("/auth/register", response_model=TokenOut, tags=["auth"])
def register(payload: RegisterIn, request: Request) -> TokenOut:
    ip = _client_ip(request)
    _enforce_rate_limit(f"register:{ip}", _RATE_REGISTER_MAX, _RATE_REGISTER_WIN, "registration")

    ok, err = auth.validate_password_strength(payload.password)
    if not ok:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, err)

    if db.get_user_by_email(payload.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    user = db.create_user(payload.email, auth.hash_password(payload.password))

    # Generate email verification code (24h TTL), log it server-side
    ver_code   = auth.generate_otp()
    ver_hash   = auth.hash_otp(ver_code)
    ver_expiry = time.time() + _EMAIL_VER_TTL
    db.set_email_verification(user["id"], ver_hash, ver_expiry)

    logger.info(
        "[email_verify] Code generated for user=%s email=%s (dev mode — not emailed)",
        user["id"], payload.email,
    )
    logger.info("[email_verify] CODE=%s  ← POST /auth/verify_email to confirm", ver_code)

    return TokenOut(**auth.issue_token(user["id"]))


@app.post("/auth/login", response_model=TokenOut, tags=["auth"])
def login(payload: LoginIn, request: Request) -> TokenOut:
    ip = _client_ip(request)
    _enforce_rate_limit(f"login:{ip}", _RATE_LOGIN_MAX, _RATE_LOGIN_WIN, "login")

    user = db.get_user_by_email(payload.email)
    if not user or not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")

    if not user.get("email_verified"):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Email not verified. Check server logs for code → POST /auth/verify_email.",
        )

    return TokenOut(**auth.issue_token(user["id"]))


@app.get("/auth/me", response_model=UserOut, tags=["auth"])
def me(user: Dict[str, Any] = Depends(current_user)) -> UserOut:
    return UserOut(
        id=user["id"],
        email=user["email"],
        created_at=user["created_at"],
        email_verified=bool(user.get("email_verified", True)),
        auth_provider=user.get("auth_provider") or "local",
    )


# ── Email Verification ─────────────────────────────────────────────────
@app.post("/auth/verify_email", response_model=VerifyEmailOut, tags=["auth"])
def verify_email(payload: VerifyEmailIn) -> VerifyEmailOut:
    user = db.get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid email or code.")

    if user.get("email_verified"):
        return VerifyEmailOut(message="Email is already verified.", verified=True)

    ver_hash   = user.get("email_ver_hash")
    ver_expiry = user.get("email_ver_expiry") or 0

    if not ver_hash:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "No verification code found. Request a new one via /auth/resend_verification.",
        )
    if time.time() > ver_expiry:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Verification code has expired. Request a new one via /auth/resend_verification.",
        )
    if not auth.verify_otp_hash(payload.code, ver_hash):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid email or code.")

    db.mark_email_verified(user["id"])
    logger.info("[email_verify] Email verified for user=%s", user["id"])
    return VerifyEmailOut(message="Email verified. You can now sign in.", verified=True)


@app.post("/auth/resend_verification", response_model=ResendVerificationOut, tags=["auth"])
def resend_verification(
    payload: ResendVerificationIn, request: Request
) -> ResendVerificationOut:
    ip = _client_ip(request)
    _enforce_rate_limit(
        f"email_ver:{ip}", _RATE_EMAIL_VER_MAX, _RATE_EMAIL_VER_WIN, "verification resend"
    )
    _GENERIC = "If the account exists and is unverified, a new code has been sent."

    user = db.get_user_by_email(payload.email)
    if not user:
        return ResendVerificationOut(message=_GENERIC)
    if user.get("email_verified"):
        return ResendVerificationOut(message="Email is already verified.")

    ver_code   = auth.generate_otp()
    ver_hash   = auth.hash_otp(ver_code)
    ver_expiry = time.time() + _EMAIL_VER_TTL
    db.set_email_verification(user["id"], ver_hash, ver_expiry)

    logger.info("[email_verify] Resent code for user=%s (dev mode)", user["id"])
    logger.info("[email_verify] CODE=%s  ← POST /auth/verify_email", ver_code)
    return ResendVerificationOut(message=_GENERIC)


# ── Change Password ────────────────────────────────────────────────────
@app.post("/auth/change_password", response_model=ChangePasswordOut, tags=["auth"])
def change_password(
    payload: ChangePasswordIn,
    user: Dict[str, Any] = Depends(current_user),
) -> ChangePasswordOut:
    # OAuth-only accounts have no local password
    full_user = db.get_user_by_email(user["email"])
    if not full_user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")

    provider = full_user.get("auth_provider") or "local"
    if provider != "local":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"This account uses {provider} sign-in and has no local password to change.",
        )

    if not auth.verify_password(payload.old_password, full_user.get("password_hash")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Current password is incorrect.")

    if payload.old_password == payload.new_password:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "New password must differ from the current password.",
        )

    ok, err = auth.validate_password_strength(payload.new_password)
    if not ok:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, err)

    db.update_password(full_user["id"], auth.hash_password(payload.new_password))
    db.invalidate_user_tokens(full_user["id"])

    logger.info("[auth] Password changed for user=%s; all sessions invalidated.", full_user["id"])
    return ChangePasswordOut(
        message="Password updated. Please sign in again with your new password."
    )


# ── OTP / Password Reset ────────────────────────────────────────────────
@app.post("/auth/request_reset", response_model=RequestResetOut, tags=["auth"])
def request_reset(payload: RequestResetIn, request: Request) -> RequestResetOut:
    ip = _client_ip(request)
    _enforce_rate_limit(f"otp_req:{ip}", _RATE_OTP_MAX, _RATE_OTP_WIN, "OTP request")
    _GENERIC = "If the account exists, an OTP has been sent."

    user = db.get_user_by_email(payload.email)
    if not user:
        return RequestResetOut(message=_GENERIC)

    _enforce_rate_limit(
        f"otp_req_user:{user['id']}", _RATE_OTP_MAX, _RATE_OTP_WIN, "OTP request"
    )

    otp    = auth.generate_otp()
    expiry = time.time() + auth.OTP_TTL_SECONDS
    db.set_otp(user["id"], auth.hash_otp(otp), expiry)

    logger.info("[otp] OTP generated for user=%s (dev mode)", user["id"])
    logger.info("[otp] OTP=%s  ← POST /auth/verify_otp", otp)
    return RequestResetOut(message=_GENERIC)


@app.post("/auth/verify_otp", response_model=VerifyOTPOut, tags=["auth"])
def verify_otp(payload: VerifyOTPIn) -> VerifyOTPOut:
    user = db.get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid OTP or email.")

    otp_data = db.get_otp_data(user["id"])
    if not otp_data or not otp_data.get("otp_hash"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No active OTP for this account.")

    attempts = otp_data.get("otp_attempts") or 0
    if attempts >= auth.OTP_MAX_ATTEMPTS:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            f"OTP blocked after {auth.OTP_MAX_ATTEMPTS} failed attempts. Request a new OTP.",
        )

    expiry = otp_data.get("otp_expiry") or 0
    if time.time() > expiry:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OTP has expired. Request a new one.")

    if not auth.verify_otp_hash(payload.otp, otp_data["otp_hash"]):
        new_attempts = db.increment_otp_attempts(user["id"])
        remaining = auth.OTP_MAX_ATTEMPTS - new_attempts
        if remaining <= 0:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"Incorrect OTP. Account locked after {auth.OTP_MAX_ATTEMPTS} failed attempts.",
            )
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Incorrect OTP. {remaining} attempt(s) remaining.",
        )

    # Consume the hash on first successful verify (prevents reuse)
    db.mark_otp_verified(user["id"])
    return VerifyOTPOut(message="OTP verified. You may now reset your password.", verified=True)


@app.post("/auth/reset_password", response_model=ResetPasswordOut, tags=["auth"])
def reset_password(payload: ResetPasswordIn) -> ResetPasswordOut:
    user = db.get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid request.")

    otp_data = db.get_otp_data(user["id"])
    if not otp_data or not otp_data.get("otp_verified"):
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "OTP not verified. Complete /auth/verify_otp first.",
        )

    expiry = otp_data.get("otp_expiry") or 0
    if time.time() > expiry:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OTP session expired. Start over.")

    ok, err = auth.validate_password_strength(payload.new_password)
    if not ok:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, err)

    db.update_password(user["id"], auth.hash_password(payload.new_password))
    db.invalidate_user_tokens(user["id"])
    db.clear_otp(user["id"])

    logger.info("[auth] Password reset for user=%s; all sessions invalidated.", user["id"])
    return ResetPasswordOut(message="Password updated. Please log in with your new password.")


# ── Google OAuth ────────────────────────────────────────────────────────
def _google_redirect_uri(request: Request) -> str:
    """Compute the redirect URI from the configured env var or the request's base URL."""
    if _GOOGLE_REDIRECT_URI:
        return _GOOGLE_REDIRECT_URI
    base = str(request.base_url).rstrip("/")
    return f"{base}/auth/google/callback"


@app.get("/auth/google", tags=["auth"], include_in_schema=True)
def google_login(request: Request):
    """Redirect user to Google's OAuth2 consent screen."""
    if not auth.google_oauth_configured():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Google OAuth is not configured. "
            "Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in Secrets.",
        )
    redirect_uri = _google_redirect_uri(request)
    url = auth.build_google_auth_url(redirect_uri)
    return RedirectResponse(url, status_code=302)


@app.get("/auth/google/callback", tags=["auth"], include_in_schema=True)
async def google_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    """Handle the OAuth2 callback from Google."""
    if not auth.google_oauth_configured():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Google OAuth is not configured.",
        )

    if error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Google OAuth error: {error}",
        )

    if not code:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing authorization code.")

    if not auth.consume_oauth_state(state):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Invalid or expired OAuth state. Please try signing in again.",
        )

    redirect_uri = _google_redirect_uri(request)
    try:
        google_user = await auth.exchange_google_code(code, redirect_uri)
    except Exception as e:
        logger.exception("[oauth] Google token exchange failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            f"Failed to complete Google sign-in: {e}",
        )

    google_id = google_user.get("id") or google_user.get("sub")
    email     = (google_user.get("email") or "").lower().strip()

    if not google_id or not email:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Google did not return a valid user ID or email.",
        )

    # 1. Find by google_id (returning user)
    user = db.get_user_by_google_id(google_id)

    # 2. Find by email → link existing account
    if not user:
        user = db.get_user_by_email(email)
        if user:
            db.link_google_account(user["id"], google_id)
            logger.info("[oauth] Linked Google account to existing user=%s", user["id"])

    # 3. New user — create account (email pre-verified by Google)
    if not user:
        user = db.create_user(
            email=email,
            password_hash="oauth:google",
            auth_provider="google",
            google_id=google_id,
        )
        logger.info("[oauth] Created new Google user=%s email=%s", user["id"], email)

    tok = auth.issue_token(user["id"])
    token_str = tok["access_token"]

    # Redirect to /ui with JWT in the fragment (never appears in server logs)
    redirect_url = f"/ui#token={token_str}&type=bearer"
    return RedirectResponse(redirect_url, status_code=302)


# ── Upload helpers ─────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    base = Path(name).stem
    s = _SLUG_RE.sub("-", base).strip("-").lower()
    return s[:48] or "video"


def _namespaced_video_id(user_id: str, original_filename: str) -> str:
    return f"{user_id}__{_slugify(original_filename)}-{uuid.uuid4().hex[:6]}"


def _ensure_owner(
    video: Optional[Dict[str, Any]], user_id: str, video_id: str
) -> Dict[str, Any]:
    if not video:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Video '{video_id}' not found")
    if video["user_id"] != user_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Video '{video_id}' not found")
    return video


# ── Video routes ───────────────────────────────────────────────────────
@app.post("/upload_video", response_model=UploadOut, tags=["videos"])
async def upload_video(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(current_user),
) -> UploadOut:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unsupported extension '{ext}'")

    video_id = _namespaced_video_id(user["id"], file.filename or "video")
    user_dir = USER_UPLOAD_ROOT / user["id"] / "videos"
    user_dir.mkdir(parents=True, exist_ok=True)
    target = user_dir / f"{video_id}{ext}"

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    try:
        with target.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    out.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        f"File exceeds {MAX_UPLOAD_MB} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        target.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Upload failed: {e}")

    db.register_video(
        video_id=video_id, user_id=user["id"],
        filename=file.filename or target.name, status="uploaded",
    )
    logger.info("[upload] user=%s video_id=%s bytes=%d", user["id"], video_id, written)
    return UploadOut(
        video_id=video_id, filename=file.filename or target.name, status="uploaded"
    )


@app.post("/process_video", response_model=ProcessOut, tags=["videos"])
def process_video(
    payload: ProcessIn,
    user: Dict[str, Any] = Depends(current_user),
) -> ProcessOut:
    video = _ensure_owner(db.get_video(payload.video_id), user["id"], payload.video_id)

    user_dir = USER_UPLOAD_ROOT / user["id"] / "videos"
    matches = list(user_dir.glob(f"{payload.video_id}.*"))
    if not matches:
        db.update_video_status(payload.video_id, "failed", "Source file missing on disk")
        raise HTTPException(status.HTTP_410_GONE, "Source file missing on disk")

    db.update_video_status(payload.video_id, "processing")
    pipe = get_pipeline()
    try:
        with _index_write_lock:
            result_id = pipe.process_video(str(matches[0]))
    except Exception as e:
        db.update_video_status(payload.video_id, "failed", str(e))
        logger.exception("[process_video] failed for %s", payload.video_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Processing failed: {e}")

    if not result_id:
        db.update_video_status(payload.video_id, "failed", "Pipeline returned no video_id")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Processing failed")

    db.update_video_status(payload.video_id, "ready")
    return ProcessOut(
        video_id=payload.video_id, status="ready",
        detail=f"Indexed under id '{result_id}'",
    )


# ── URL Video Processing ───────────────────────────────────────────────

def _validate_youtube_url(url: str) -> None:
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid URL format.")
        netloc = parsed.netloc.lower()
        if netloc not in _ALLOWED_URL_DOMAINS:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"Only YouTube URLs are allowed (youtube.com / youtu.be). "
                f"Received domain: '{parsed.netloc}'",
            )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid URL format.")


def _download_youtube_audio(url: str, tmp_dir: str) -> str:
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed.")

    max_bytes   = MAX_URL_MB * 1024 * 1024
    max_seconds = MAX_URL_DURATION_MINUTES * 60

    def _progress_hook(d: Dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            if (d.get("downloaded_bytes") or 0) > max_bytes:
                raise yt_dlp.utils.DownloadError(f"File exceeds {MAX_URL_MB} MB size limit")

    ydl_opts = {
        "outtmpl":    str(Path(tmp_dir) / "%(id)s.%(ext)s"),
        "format":     "bestaudio/best",
        "quiet":      True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [_progress_hook],
        "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {max_seconds}"),
        "http_headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "128",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        duration = (info or {}).get("duration") or 0
        if duration > max_seconds:
            raise RuntimeError(
                f"Video is {duration // 60:.0f} min — exceeds the "
                f"{MAX_URL_DURATION_MINUTES} min limit."
            )
        ydl.download([url])

    candidates = [
        f for f in Path(tmp_dir).glob("*.*")
        if f.suffix.lower() not in {".webp", ".jpg", ".png", ".json"}
    ]
    if not candidates:
        raise RuntimeError("Download completed but audio file not found.")
    return str(candidates[0])


@app.post("/process_url", response_model=ProcessURLOut, tags=["videos"])
async def process_url(
    payload: ProcessURLIn,
    user: Dict[str, Any] = Depends(current_user),
) -> ProcessURLOut:
    """Download a YouTube video and run the full RAG pipeline on it.

    Both the yt-dlp download and the pipeline run are offloaded to
    background threads via asyncio.to_thread() so the FastAPI event
    loop is never blocked.
    """
    _validate_youtube_url(payload.url)
    _enforce_rate_limit(
        f"url_proc:{user['id']}", _RATE_URL_MAX, _RATE_URL_WIN, "URL processing"
    )

    url_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", payload.url.split("?v=")[-1])[:32] or "yt"
    video_id = _namespaced_video_id(user["id"], url_slug)
    db.register_video(
        video_id=video_id, user_id=user["id"],
        filename=payload.url, status="processing",
    )

    tmp_dir = tempfile.mkdtemp(prefix="videoqa_url_")
    try:
        logger.info("[url] user=%s downloading %s", user["id"], payload.url)
        try:
            audio_path = await asyncio.to_thread(
                _download_youtube_audio, payload.url, tmp_dir
            )
        except RuntimeError as e:
            db.update_video_status(video_id, "failed", str(e))
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
        except Exception as e:
            db.update_video_status(video_id, "failed", f"Download failed: {e}")
            logger.exception("[url] download failed for %s", payload.url)
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Failed to download video: {e}")

        user_dir = USER_UPLOAD_ROOT / user["id"] / "videos"
        user_dir.mkdir(parents=True, exist_ok=True)
        dest = user_dir / f"{video_id}{Path(audio_path).suffix}"
        shutil.copy2(audio_path, str(dest))

        pipe = get_pipeline()

        def _run_pipeline() -> Optional[str]:
            with _index_write_lock:
                return pipe.process_video(str(dest))

        try:
            result_id = await asyncio.to_thread(_run_pipeline)
        except Exception as e:
            db.update_video_status(video_id, "failed", str(e))
            logger.exception("[url] pipeline failed for video_id=%s", video_id)
            raise HTTPException(
                status.HTTP_500_INTERNAL_SERVER_ERROR, f"Processing failed: {e}"
            )

        if not result_id:
            db.update_video_status(video_id, "failed", "Pipeline returned no video_id")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Processing failed")

        db.update_video_status(video_id, "ready")
        logger.info("[url] user=%s video_id=%s indexed OK", user["id"], video_id)
        return ProcessURLOut(
            video_id=video_id, status="ready",
            detail=f"YouTube audio indexed under id '{result_id}'",
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/videos", response_model=List[VideoOut], tags=["videos"])
def list_videos(user: Dict[str, Any] = Depends(current_user)) -> List[VideoOut]:
    return [VideoOut(**r) for r in db.list_user_videos(user["id"])]


# ── Q&A ────────────────────────────────────────────────────────────────
@app.post("/ask_question", response_model=AskOut, tags=["qa"])
def ask_question(
    payload: AskIn,
    user: Dict[str, Any] = Depends(current_user),
) -> AskOut:
    if payload.video_id:
        video = _ensure_owner(db.get_video(payload.video_id), user["id"], payload.video_id)
        if video["status"] != "ready":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Video status is '{video['status']}'. Call /process_video first.",
            )
        scope = payload.video_id
    else:
        owned = [v for v in db.list_user_videos(user["id"]) if v["status"] == "ready"]
        if not owned:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                "You have no processed videos yet. Upload and process one first.",
            )
        scope = None

    pipe = get_pipeline()
    t0   = time.time()

    if scope is not None:
        result = pipe.ask(payload.query, active_video_id=scope, use_cache=payload.use_cache)
    else:
        best: Optional[Dict[str, Any]] = None
        for v in owned:
            r = pipe.ask(
                payload.query, active_video_id=v["video_id"], use_cache=payload.use_cache
            )
            if r.get("status") == "UNSUPPORTED":
                continue
            score = (r.get("support") or {}).get("support_score", 0.0)
            if best is None or score > (best.get("support") or {}).get("support_score", 0.0):
                best = r
        result = best or {
            "answer": "Not found in your videos.",
            "confidence": 10,
            "confidence_label": "Low",
            "status": "UNSUPPORTED",
            "timestamp": None,
            "chunk_ids": [],
            "provider": "oos_gate",
            "support": {
                "status": "UNSUPPORTED",
                "semantic_similarity": 0.0,
                "keyword_overlap": 0.0,
                "support_score": 0.0,
            },
            "neighbors": [],
            "cached": False,
        }

    latency_ms = int((time.time() - t0) * 1000)
    return AskOut(
        answer=result.get("answer", ""),
        confidence=int(result.get("confidence", 0) or 0),
        confidence_label=result.get("confidence_label"),
        status=result.get("status", "UNSUPPORTED"),
        timestamp=result.get("timestamp"),
        chunk_ids=result.get("chunk_ids", []) or [],
        provider=result.get("provider"),
        support=result.get("support"),
        neighbors=result.get("neighbors", []) or [],
        cached=bool(result.get("cached", False)),
        latency_ms=latency_ms,
        fallback_level=result.get("fallback_level"),
        llm_latency_ms=result.get("llm_latency_ms"),
        providers_tried=result.get("providers_tried", []) or [],
        bedrock_calls_used=result.get("bedrock_calls_used"),
    )


# ── Root ───────────────────────────────────────────────────────────────
@app.get("/", tags=["system"])
def root() -> Dict[str, Any]:
    return {
        "service": "Video-QA SaaS API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui",
        "google_oauth": auth.google_oauth_configured(),
    }
