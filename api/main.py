"""FastAPI application — SaaS surface over the Video-QA RAG pipeline.

Endpoints
---------
  POST /auth/register           create account (password policy enforced)
  POST /auth/login              obtain bearer token (rate-limited)
  GET  /auth/me                 current user (auth required)
  POST /auth/verify_email       verify email address with 6-digit code
  POST /auth/resend_verification resend email verification code (rate-limited)
  POST /auth/request_reset      request OTP for password reset (rate-limited)
  POST /auth/verify_otp         verify OTP — consumes the code (max 5 attempts)
  POST /auth/reset_password     reset password after OTP verification (no OTP re-entry)
  POST /upload_video            multipart upload (auth required)
  POST /process_video           transcribe + index an uploaded video (auth)
  POST /process_url             download + index a YouTube video (auth, rate-limited)
  POST /ask_question            ask a question (auth)
  GET  /videos                  list current user's videos (auth)
  GET  /health                  public health probe
  GET  /docs                    Swagger UI

Per-user data isolation
-----------------------
Each upload gets a namespaced video_id of the form `{user_id}__{slug}`. The
existing pipeline already filters retrieval by `video_id`. For unscoped
questions we filter retrieval by **all** of the caller's video_ids (server-side)
so a user can never see another user's chunks.
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
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from . import auth, db
from .compare import router as compare_router
from .schemas import (
    AskIn,
    AskOut,
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
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Constants ──────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("VIDEO_QA_MAX_UPLOAD_MB", "300"))
MAX_URL_MB = 200
MAX_URL_DURATION_MINUTES = 30
ALLOWED_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".mpeg4"}
USER_UPLOAD_ROOT = Path("data/users")

# Allowed YouTube domains for URL processing
_ALLOWED_URL_DOMAINS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}

# Rate limit windows
_RATE_LOGIN_MAX        = 10   # per IP per 15 min
_RATE_LOGIN_WIN        = 900
_RATE_REGISTER_MAX     = 5    # per IP per hour
_RATE_REGISTER_WIN     = 3600
_RATE_OTP_MAX          = 3    # per user per hour
_RATE_OTP_WIN          = 3600
_RATE_URL_MAX          = 3    # per user per hour
_RATE_URL_WIN          = 3600
_RATE_EMAIL_VER_MAX    = 5    # resend verification per IP per hour
_RATE_EMAIL_VER_WIN    = 3600

# Email verification code TTL
_EMAIL_VER_TTL = 60 * 60 * 24  # 24 hours

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
                logger.info("[boot] initializing VideoQAPipeline (cold start)…")
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


def _enforce_rate_limit(key: str, max_count: int, window_seconds: float, label: str) -> None:
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
    logger.info("[boot] DB initialized at %s", db.DB_PATH)


# Multi-Video Compare module
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
            n_users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            n_videos = c.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    except Exception as e:
        logger.warning("[health] db introspection failed: %s", e)
    return HealthOut(status="ok", indexed_chunks=chunks, db_users=n_users, db_videos=n_videos)


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

    # Generate a 6-digit email verification code and store it
    ver_code = auth.generate_otp()
    ver_hash = auth.hash_otp(ver_code)
    ver_expiry = time.time() + _EMAIL_VER_TTL
    db.set_email_verification(user["id"], ver_hash, ver_expiry)

    # In production wire this to an email service; for now log it server-side.
    logger.info(
        "[email_verify] Verification code generated for user=%s email=%s (dev mode — not emailed)",
        user["id"], payload.email,
    )
    logger.info("[email_verify] CODE=%s  ← use POST /auth/verify_email to confirm", ver_code)

    tok = auth.issue_token(user["id"])
    return TokenOut(**tok)


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
            "Email address not verified. "
            "Check your server logs for the verification code and POST to /auth/verify_email.",
        )

    tok = auth.issue_token(user["id"])
    return TokenOut(**tok)


@app.get("/auth/me", response_model=UserOut, tags=["auth"])
def me(user: Dict[str, Any] = Depends(current_user)) -> UserOut:
    return UserOut(
        id=user["id"],
        email=user["email"],
        created_at=user["created_at"],
        email_verified=bool(user.get("email_verified", True)),
    )


# ── Email Verification ─────────────────────────────────────────────────
@app.post("/auth/verify_email", response_model=VerifyEmailOut, tags=["auth"])
def verify_email(payload: VerifyEmailIn) -> VerifyEmailOut:
    user = db.get_user_by_email(payload.email)
    if not user:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid email or code.")

    if user.get("email_verified"):
        return VerifyEmailOut(message="Email is already verified.", verified=True)

    ver_hash = user.get("email_ver_hash")
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
def resend_verification(payload: ResendVerificationIn, request: Request) -> ResendVerificationOut:
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

    ver_code = auth.generate_otp()
    ver_hash = auth.hash_otp(ver_code)
    ver_expiry = time.time() + _EMAIL_VER_TTL
    db.set_email_verification(user["id"], ver_hash, ver_expiry)

    logger.info(
        "[email_verify] Resent verification code for user=%s (dev mode — not emailed)",
        user["id"],
    )
    logger.info("[email_verify] CODE=%s  ← use POST /auth/verify_email to confirm", ver_code)

    return ResendVerificationOut(message=_GENERIC)


# ── OTP / Password Reset ────────────────────────────────────────────────
@app.post("/auth/request_reset", response_model=RequestResetOut, tags=["auth"])
def request_reset(payload: RequestResetIn, request: Request) -> RequestResetOut:
    ip = _client_ip(request)
    _enforce_rate_limit(f"otp_req:{ip}", _RATE_OTP_MAX, _RATE_OTP_WIN, "OTP request")

    # Always return generic message — never reveal whether the email exists
    _GENERIC = "If the account exists, an OTP has been sent."

    user = db.get_user_by_email(payload.email)
    if not user:
        return RequestResetOut(message=_GENERIC)

    # Per-user rate limit (in addition to per-IP)
    _enforce_rate_limit(f"otp_req_user:{user['id']}", _RATE_OTP_MAX, _RATE_OTP_WIN, "OTP request")

    otp = auth.generate_otp()
    otp_hash = auth.hash_otp(otp)
    expiry = time.time() + auth.OTP_TTL_SECONDS
    db.set_otp(user["id"], otp_hash, expiry)

    # In production wire this to an email service; for now log it server-side.
    logger.info("[otp] Generated OTP for user=%s (dev mode — not emailed)", user["id"])
    logger.info("[otp] OTP=%s  ← use POST /auth/verify_otp to verify", otp)

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
            f"OTP blocked after {auth.OTP_MAX_ATTEMPTS} failed attempts. "
            "Request a new OTP.",
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

    # Consume the OTP hash so it cannot be reused
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

    # Check the OTP session hasn't expired since verification
    expiry = otp_data.get("otp_expiry") or 0
    if time.time() > expiry:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "OTP session expired. Start over.")

    # Note: no OTP re-verification here. The hash was consumed by /auth/verify_otp
    # to prevent reuse. Checking otp_verified=1 + expiry is the authoritative gate.

    ok, err = auth.validate_password_strength(payload.new_password)
    if not ok:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, err)

    db.update_password(user["id"], auth.hash_password(payload.new_password))
    db.invalidate_user_tokens(user["id"])
    db.clear_otp(user["id"])

    logger.info("[auth] Password reset for user=%s; all sessions invalidated.", user["id"])
    return ResetPasswordOut(message="Password updated. Please log in with your new password.")


# ── Helpers ────────────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    base = Path(name).stem
    s = _SLUG_RE.sub("-", base).strip("-").lower()
    return s[:48] or "video"


def _namespaced_video_id(user_id: str, original_filename: str) -> str:
    return f"{user_id}__{_slugify(original_filename)}-{uuid.uuid4().hex[:6]}"


def _ensure_owner(video: Optional[Dict[str, Any]], user_id: str, video_id: str) -> Dict[str, Any]:
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

    db.register_video(video_id=video_id, user_id=user["id"],
                      filename=file.filename or target.name, status="uploaded")
    logger.info("[upload] user=%s video_id=%s bytes=%d", user["id"], video_id, written)
    return UploadOut(video_id=video_id, filename=file.filename or target.name, status="uploaded")


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
    src = matches[0]

    db.update_video_status(payload.video_id, "processing")
    pipe = get_pipeline()
    try:
        with _index_write_lock:
            result_id = pipe.process_video(str(src))
    except Exception as e:
        db.update_video_status(payload.video_id, "failed", str(e))
        logger.exception("[process_video] failed for %s", payload.video_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Processing failed: {e}")

    if not result_id:
        db.update_video_status(payload.video_id, "failed", "Pipeline returned no video_id")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Processing failed")

    db.update_video_status(payload.video_id, "ready")
    return ProcessOut(video_id=payload.video_id, status="ready",
                      detail=f"Indexed under id '{result_id}'")


# ── URL Video Processing ───────────────────────────────────────────────

def _validate_youtube_url(url: str) -> None:
    """Raise HTTP 400 if the URL is not a supported YouTube URL."""
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
    """
    Download audio-only from a YouTube URL using yt-dlp.
    Runs synchronously — caller must wrap in asyncio.to_thread().
    Returns path to the downloaded audio file.
    Raises RuntimeError on failure.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    max_bytes = MAX_URL_MB * 1024 * 1024
    max_seconds = MAX_URL_DURATION_MINUTES * 60

    out_tmpl = str(Path(tmp_dir) / "%(id)s.%(ext)s")

    def _progress_hook(d: Dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            downloaded = d.get("downloaded_bytes") or 0
            if downloaded > max_bytes:
                raise yt_dlp.utils.DownloadError(
                    f"File exceeds {MAX_URL_MB} MB size limit"
                )

    ydl_opts = {
        "outtmpl": out_tmpl,
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "progress_hooks": [_progress_hook],
        "match_filter": yt_dlp.utils.match_filter_func(
            f"duration <= {max_seconds}"
        ),
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        },
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
                f"Video is {duration // 60:.0f} min — exceeds the {MAX_URL_DURATION_MINUTES} min limit."
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

    Security:
    - Only youtube.com / youtu.be are accepted (HTTP 400 otherwise).
    - Max duration: 30 minutes. Max file size: 200 MB.
    - Rate limited to 3 requests per user per hour.

    The yt-dlp download runs in a background thread so the FastAPI event loop
    is never blocked.
    """
    # 1. Validate URL domain
    _validate_youtube_url(payload.url)

    # 2. Per-user rate limit
    _enforce_rate_limit(
        f"url_proc:{user['id']}", _RATE_URL_MAX, _RATE_URL_WIN, "URL processing"
    )

    # 3. Build a namespaced video_id from the URL slug
    url_slug = re.sub(r"[^a-zA-Z0-9_-]", "-", payload.url.split("?v=")[-1])[:32] or "yt"
    video_id = _namespaced_video_id(user["id"], url_slug)
    db.register_video(video_id=video_id, user_id=user["id"],
                      filename=payload.url, status="processing")

    tmp_dir = tempfile.mkdtemp(prefix="videoqa_url_")
    audio_path: Optional[str] = None
    try:
        # 4. Download audio in a background thread — NEVER block the event loop
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
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY,
                f"Failed to download video: {e}",
            )

        # 5. Copy the audio into the user upload directory so the pipeline
        #    produces a stable video_id (derived from the filename stem).
        user_dir = USER_UPLOAD_ROOT / user["id"] / "videos"
        user_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(audio_path).suffix
        dest = user_dir / f"{video_id}{ext}"
        shutil.copy2(audio_path, str(dest))

        # 6. Run the pipeline synchronously inside a thread to keep FastAPI
        #    responsive. The index write lock prevents concurrent FAISS writes.
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
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                f"Processing failed: {e}",
            )

        if not result_id:
            db.update_video_status(video_id, "failed", "Pipeline returned no video_id")
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Processing failed")

        db.update_video_status(video_id, "ready")
        logger.info("[url] user=%s video_id=%s indexed OK", user["id"], video_id)
        return ProcessURLOut(
            video_id=video_id,
            status="ready",
            detail=f"YouTube audio indexed under id '{result_id}'",
        )

    finally:
        # 7. Always clean up the temp directory regardless of outcome
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/videos", response_model=List[VideoOut], tags=["videos"])
def list_videos(user: Dict[str, Any] = Depends(current_user)) -> List[VideoOut]:
    rows = db.list_user_videos(user["id"])
    return [VideoOut(**r) for r in rows]


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
    t0 = time.time()
    if scope is not None:
        result = pipe.ask(payload.query, active_video_id=scope, use_cache=payload.use_cache)
    else:
        best: Optional[Dict[str, Any]] = None
        for v in owned:
            r = pipe.ask(payload.query, active_video_id=v["video_id"], use_cache=payload.use_cache)
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
            "support": {"status": "UNSUPPORTED", "semantic_similarity": 0.0,
                        "keyword_overlap": 0.0, "support_score": 0.0},
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
    }
