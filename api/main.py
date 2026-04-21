"""FastAPI application — SaaS surface over the Video-QA RAG pipeline.

Endpoints
---------
  POST /auth/register         create account
  POST /auth/login            obtain bearer token
  GET  /auth/me               current user (auth required)
  POST /upload_video          multipart upload (auth required)
  POST /process_video         transcribe + index an uploaded video (auth)
  POST /ask_question          ask a question (auth)
  GET  /videos                list current user's videos (auth)
  GET  /health                public health probe
  GET  /docs                  Swagger UI

Per-user data isolation
-----------------------
Each upload gets a namespaced video_id of the form `{user_id}__{slug}`. The
existing pipeline already filters retrieval by `video_id`. For unscoped
questions we filter retrieval by **all** of the caller's video_ids (server-side)
so a user can never see another user's chunks.

Persistent embeddings
---------------------
The FAISS index at `models/video_index.faiss` is shared across users but every
chunk's metadata records its `video_id` (which embeds the user's id). All
retrieval paths apply that filter, so the index is **multi-tenant by namespace**
rather than per-user files. This gives O(1) cold start vs reloading per-user
indexes per request.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is importable so `video_qa.*` and `config_loader` resolve
# regardless of how uvicorn is launched.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import auth, db
from .schemas import (
    AskIn,
    AskOut,
    HealthOut,
    LoginIn,
    ProcessIn,
    ProcessOut,
    RegisterIn,
    TokenOut,
    UploadOut,
    UserOut,
    VideoOut,
)

logger = logging.getLogger("video_qa.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# ── Constants ──────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.environ.get("VIDEO_QA_MAX_UPLOAD_MB", "300"))
ALLOWED_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mp3", ".wav", ".m4a", ".mpeg4"}
USER_UPLOAD_ROOT = Path("data/users")

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
    allow_origins=["*"],  # Tighten in production via env-driven allowlist
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)

# ── Lazy pipeline singleton ────────────────────────────────────────────
# Loading the embedding model + FAISS index takes ~2s, so we do it on the
# first request that needs it, not at import time. This keeps `/health`,
# `/auth/register`, and `/auth/login` snappy and lets the container start
# before the first heavy call.
_pipeline = None
# Serializes both pipeline init and mutations of the shared FAISS index.
# /process_video rebuilds the global index file, so concurrent calls would
# race on the same on-disk artifacts and produce inconsistent retrieval.
# A process-wide lock is sufficient because we run a single uvicorn worker
# (see workflow command). For multi-replica deployments, replace this with
# a durable job queue + cross-process advisory lock.
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


# ── Lifecycle ──────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    USER_UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("[boot] DB initialized at %s", db.DB_PATH)


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
def register(payload: RegisterIn) -> TokenOut:
    if db.get_user_by_email(payload.email):
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered")
    user = db.create_user(payload.email, auth.hash_password(payload.password))
    tok = auth.issue_token(user["id"])
    return TokenOut(**tok)


@app.post("/auth/login", response_model=TokenOut, tags=["auth"])
def login(payload: LoginIn) -> TokenOut:
    user = db.get_user_by_email(payload.email)
    if not user or not auth.verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
    tok = auth.issue_token(user["id"])
    return TokenOut(**tok)


@app.get("/auth/me", response_model=UserOut, tags=["auth"])
def me(user: Dict[str, Any] = Depends(current_user)) -> UserOut:
    return UserOut(id=user["id"], email=user["email"], created_at=user["created_at"])


# ── Helpers ────────────────────────────────────────────────────────────
_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def _slugify(name: str) -> str:
    base = Path(name).stem
    s = _SLUG_RE.sub("-", base).strip("-").lower()
    return s[:48] or "video"


def _namespaced_video_id(user_id: str, original_filename: str) -> str:
    """Build a per-user video_id. Short uuid suffix avoids collisions on re-upload."""
    return f"{user_id}__{_slugify(original_filename)}-{uuid.uuid4().hex[:6]}"


def _ensure_owner(video: Optional[Dict[str, Any]], user_id: str, video_id: str) -> Dict[str, Any]:
    if not video:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Video '{video_id}' not found")
    if video["user_id"] != user_id:
        # Same as 404 to avoid leaking existence of other users' video_ids.
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Video '{video_id}' not found")
    return video


# ── Video routes ───────────────────────────────────────────────────────
@app.post("/upload_video", response_model=UploadOut, tags=["videos"])
async def upload_video(
    file: UploadFile = File(...),
    user: Dict[str, Any] = Depends(current_user),
) -> UploadOut:
    """Upload a media file. The file is persisted under
    ``data/users/{user_id}/videos/{video_id}{ext}`` and registered in the DB
    with status ``uploaded``. Use ``/process_video`` next to transcribe+index it.
    """
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
    """Transcribe + chunk + embed an uploaded video. This is synchronous and
    can be slow for long media; the response only returns once indexing is
    complete and the video is queryable.
    """
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
        # The pipeline derives video_id from `Path(video_path).stem`, so naming
        # the file `{video_id}{ext}` makes the stored chunks carry our
        # namespaced id automatically. That id includes the user's uuid prefix,
        # which the retrieval filter then uses for tenant isolation.
        # Hold the index lock for the entire processing call: the pipeline
        # rebuilds the shared FAISS index file at the end, and concurrent
        # processing would otherwise produce torn writes.
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
    """Ask a grounded question over the caller's videos.

    - If ``video_id`` is provided, the answer is restricted to that video.
    - If omitted, the answer is restricted to **all** videos the caller owns.
      (We never search other users' content — enforced server-side.)
    """
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
        # Multi-video scope: ask the pipeline once per owned video and pick the
        # answer with the highest support_score that isn't UNSUPPORTED. Cheap
        # because cache + small per-user video counts.
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
    )


# ── Root ───────────────────────────────────────────────────────────────
@app.get("/", tags=["system"])
def root() -> Dict[str, Any]:
    return {
        "service": "Video-QA SaaS API",
        "version": app.version,
        "docs": "/docs",
        "health": "/health",
    }
