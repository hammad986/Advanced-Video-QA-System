"""Storage abstraction — S3 when configured, local disk otherwise.

Environment variables
---------------------
AWS_S3_BUCKET         required for S3 mode
AWS_ACCESS_KEY_ID     required for S3 mode
AWS_SECRET_ACCESS_KEY required for S3 mode
AWS_REGION            optional, default us-east-1
STORAGE_BACKEND       "s3" | "local"  (auto-detected if not set)
S3_SIGNED_URL_EXPIRY  presigned URL lifetime in seconds (default 3600)

Local mode
----------
Files live under  data/media/<video_id>/<filename>
Served by the FastAPI  GET /media/{path:path}?token=<jwt>  endpoint.
The file_url stored in the DB is the path   /media/<video_id>/<filename>.

S3 mode
-------
Files stored as  videos/<video_id>/<filename>  in the configured bucket.
The file_url stored in the DB is the S3 key (no leading slash).
resolve_url() generates a presigned GET URL valid for S3_SIGNED_URL_EXPIRY
seconds on each call — no direct public access to S3.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

_BUCKET       = os.environ.get("AWS_S3_BUCKET", "")
_REGION       = os.environ.get("AWS_REGION", "us-east-1")
_EXPLICIT     = os.environ.get("STORAGE_BACKEND", "")
_SIGNED_EXPIRY = int(os.environ.get("S3_SIGNED_URL_EXPIRY", "3600"))

USE_S3 = (
    _EXPLICIT == "s3"
    or (not _EXPLICIT and bool(_BUCKET and os.environ.get("AWS_ACCESS_KEY_ID")))
)

MEDIA_ROOT = Path("data/media")


# ── S3 client (lazy singleton) ─────────────────────────────────────────

_s3_client = None

def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3", region_name=_REGION)
    return _s3_client


# ── Public API ──────────────────────────────────────────────────────────

def upload(local_path: str, video_id: str, filename: str) -> str:
    """Copy *local_path* to storage; return the stored key / path."""
    if USE_S3:
        key = f"videos/{video_id}/{filename}"
        _s3().upload_file(
            local_path, _BUCKET, key,
            ExtraArgs={"ContentType": _content_type(filename)},
        )
        return key
    else:
        dest_dir = MEDIA_ROOT / video_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / filename
        shutil.copy2(local_path, str(target))
        return f"/media/{video_id}/{filename}"


def resolve_url(stored: str) -> str:
    """Convert a stored key/path to an accessible URL.

    * Local paths (/media/…) → returned as-is; client appends ?token=<jwt>
    * S3 keys → presigned GET URL (expires in S3_SIGNED_URL_EXPIRY seconds)
    """
    if not stored:
        return ""
    if stored.startswith("/"):
        return stored
    if USE_S3:
        return _s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": _BUCKET, "Key": stored},
            ExpiresIn=_SIGNED_EXPIRY,
        )
    return stored


def download(stored: str, local_path: str) -> None:
    """Download from storage to *local_path* (for worker use)."""
    if USE_S3 and not stored.startswith("/"):
        _s3().download_file(_BUCKET, stored, local_path)
    else:
        parts = stored.lstrip("/").split("/", 2)   # ["media", video_id, filename]
        src   = MEDIA_ROOT / parts[1] / parts[2]
        shutil.copy2(str(src), local_path)


def delete(stored: str, video_id: str) -> None:
    if USE_S3 and not stored.startswith("/"):
        try:
            _s3().delete_object(Bucket=_BUCKET, Key=stored)
        except Exception:
            pass
    else:
        shutil.rmtree(str(MEDIA_ROOT / video_id), ignore_errors=True)


def local_path(stored: str) -> Optional[str]:
    """Return the absolute filesystem path if using local storage; else None."""
    if stored.startswith("/media/"):
        parts = stored.lstrip("/").split("/", 2)
        return str(MEDIA_ROOT / parts[1] / parts[2])
    return None


def backend_name() -> str:
    return "s3" if USE_S3 else "local"


# ── Helpers ────────────────────────────────────────────────────────────

def _content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    return {
        ".mp4": "video/mp4",     ".mkv": "video/x-matroska",
        ".avi": "video/x-msvideo", ".mov": "video/quicktime",
        ".webm": "video/webm",   ".mp3": "audio/mpeg",
        ".wav": "audio/wav",     ".m4a": "audio/mp4",
    }.get(ext, "application/octet-stream")
