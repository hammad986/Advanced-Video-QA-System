"""Video processing job — runs in the background worker.

Works in both execution modes:
  threading mode  → called from api/jobs.py worker thread, shares the
                    process with FastAPI; uses api.pipeline_singleton.
  RQ/Redis mode   → called by an external rq worker process; uses
                    workers.model_cache (preloaded at worker startup).

Progress stages
---------------
  0 %   queued
 10 %   downloading   — pull from S3/local storage OR yt-dlp (for URL jobs)
 25 %   validating    — ffprobe: confirm valid audio/video stream
 40 %   normalizing   — ffmpeg → 16 kHz mono WAV
 60 %   transcribing  — Whisper transcription
 80 %   indexing      — chunk + embed + FAISS write
100 %   ready
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("video_qa.worker")

_FFMPEG_TIMEOUT          = int(os.environ.get("FFMPEG_TIMEOUT", "600"))
_WORKER_MODE             = os.environ.get("WORKER_MODE", "threading")   # "rq" | "threading"
_MAX_URL_MB              = int(os.environ.get("MAX_URL_MB", "200"))
_MAX_URL_DURATION_MINS   = int(os.environ.get("MAX_URL_DURATION_MINUTES", "30"))


# ── Pipeline accessor (mode-aware) ──────────────────────────────────────

def _get_pipeline():
    """Return the pipeline singleton appropriate for this execution mode."""
    if _WORKER_MODE == "rq":
        from workers.model_cache import get_pipeline
        return get_pipeline()
    # In-process (threading) mode — share with the API's singleton
    from api.pipeline_singleton import get_pipeline
    return get_pipeline()


def _get_index_write_lock():
    """Return the FAISS write lock for this execution mode.

    In RQ mode the worker is single-process so a module-level lock suffices;
    we still import api.pipeline_singleton's lock so both code-paths use the
    same reference when running in-process.
    """
    from api.pipeline_singleton import index_write_lock
    return index_write_lock


# ── yt-dlp download (URL-sourced jobs) ──────────────────────────────────

def _ytdlp_download(url: str, tmp_dir: str) -> str:
    """Download audio-only from a YouTube URL via yt-dlp.

    Returns the path to the resulting WAV file.
    Raises RuntimeError for duration / size violations.
    """
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp is not installed — run: pip install yt-dlp")

    max_bytes   = _MAX_URL_MB * 1024 * 1024
    max_seconds = _MAX_URL_DURATION_MINS * 60

    def _hook(d: Dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            if (d.get("downloaded_bytes") or 0) > max_bytes:
                raise yt_dlp.utils.DownloadError(
                    f"Download aborted — file exceeds {_MAX_URL_MB} MB size limit"
                )

    ydl_opts: Dict[str, Any] = {
        "outtmpl":        str(Path(tmp_dir) / "%(id)s.%(ext)s"),
        "format":         "bestaudio/best",
        "quiet":          True,
        "no_warnings":    True,
        "noplaylist":     True,
        "progress_hooks": [_hook],
        "match_filter":   yt_dlp.utils.match_filter_func(f"duration <= {max_seconds}"),
        "http_headers":   {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        "postprocessors": [{
            "key":              "FFmpegExtractAudio",
            "preferredcodec":   "wav",
            "preferredquality": "128",
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info     = ydl.extract_info(url, download=False)
        duration = (info or {}).get("duration") or 0
        if duration > max_seconds:
            raise RuntimeError(
                f"Video is {int(duration // 60)} min — "
                f"exceeds the {_MAX_URL_DURATION_MINS} min limit."
            )
        ydl.download([url])

    candidates = [
        f for f in Path(tmp_dir).glob("*.*")
        if f.suffix.lower() not in {".webp", ".jpg", ".png", ".json"}
    ]
    if not candidates:
        raise RuntimeError("yt-dlp finished but no audio file was found in output directory.")
    # Return the largest file (most likely the audio, not a thumbnail)
    return str(max(candidates, key=lambda p: p.stat().st_size))


# ── Helpers ─────────────────────────────────────────────────────────────

def _set_progress(
    job_id: str, progress: int, stage: str, status: str,
    error: Optional[str] = None,
) -> None:
    try:
        from api import db
        db.update_job_progress(job_id, progress, stage, status, error)
    except Exception as exc:
        logger.warning("[worker] DB progress update failed: %s", exc)


def _ffprobe_validate(path: str) -> None:
    """Raise RuntimeError if path contains no audio or video stream."""
    def _has_stream(spec: str) -> bool:
        res = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", spec,
             "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30,
        )
        return res.returncode == 0 and bool(res.stdout.strip())

    if not _has_stream("a:0") and not _has_stream("v:0"):
        raise RuntimeError("ffprobe: no audio or video stream found in file")


def _normalize_audio(input_path: str, output_wav: str) -> None:
    """Convert any media file to 16 kHz mono PCM-WAV (Whisper-optimal)."""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vn",                    # strip video track
        "-acodec", "pcm_s16le",   # 16-bit signed PCM
        "-ar",  "16000",          # 16 kHz
        "-ac",  "1",              # mono
        output_wav,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg normalization failed: {(res.stderr or '')[-600:]}")


# ── Job entry point ─────────────────────────────────────────────────────

def process_video_job(
    job_id: str,
    video_id: str,
    user_id: str,
    file_url: str,
    original_filename: str,
    attempt: int = 1,
    source_url: Optional[str] = None,
) -> None:
    """Process one video end-to-end.

    Called by:
      • api/jobs.py  _run_with_retry()   (threading mode)
      • rq worker                        (RQ/Redis mode — RQ handles retries)

    source_url: if set, download from YouTube via yt-dlp (URL-sourced jobs).
                Otherwise pull from storage using file_url (uploaded jobs).
    """
    logger.info(
        "[worker] START  job_id=%s  video_id=%s  file=%s  attempt=%d  url_job=%s",
        job_id, video_id, original_filename, attempt, bool(source_url),
    )

    # Reset to processing at the top of every attempt so polling stays live
    _set_progress(job_id, 0, "queued", "processing")

    tmp_dir = tempfile.mkdtemp(prefix="vqa_job_")
    try:
        from api import storage, db

        # 1 · Download ──────────────────────────────────────────────────
        _set_progress(job_id, 10, "downloading", "processing")

        if source_url:
            # URL-sourced job: download from YouTube with yt-dlp
            raw_path = _ytdlp_download(source_url, tmp_dir)
            logger.info("[worker] yt-dlp  %s  (%d B)", raw_path, os.path.getsize(raw_path))
            # Store the downloaded file so the player can serve it
            safe_name  = f"{video_id}.wav"
            stored_url = storage.upload(raw_path, video_id, safe_name)
            db.update_video_file_url(video_id, stored_url)
            logger.info("[worker] stored  %s → %s", safe_name, stored_url)
        else:
            # Upload-sourced job: pull from our own storage
            ext      = Path(original_filename).suffix or ".mp4"
            raw_path = os.path.join(tmp_dir, f"raw{ext}")
            storage.download(file_url, raw_path)
            logger.info("[worker] downloaded  %s  (%d B)", raw_path, os.path.getsize(raw_path))

        # 2 · Validate ──────────────────────────────────────────────────
        _set_progress(job_id, 25, "validating", "processing")
        _ffprobe_validate(raw_path)

        # 3 · Normalise audio ────────────────────────────────────────────
        _set_progress(job_id, 40, "normalizing", "processing")
        wav_path = os.path.join(tmp_dir, "audio.wav")
        _normalize_audio(raw_path, wav_path)
        logger.info("[worker] normalised  %s  (%d B)", wav_path, os.path.getsize(wav_path))

        # 4 · Transcribe ─────────────────────────────────────────────────
        _set_progress(job_id, 60, "transcribing", "processing")
        pipe = _get_pipeline()

        # 5 · Chunk + embed + FAISS write ────────────────────────────────
        _set_progress(job_id, 80, "indexing", "processing")
        with _get_index_write_lock():
            result_id = pipe.process_video(wav_path)

        if not result_id:
            raise RuntimeError("Pipeline returned no video_id after processing")

        # Done ────────────────────────────────────────────────────────────
        db.update_job_progress(job_id, 100, "ready", "ready")
        db.update_video_status(video_id, "ready")
        logger.info("[worker] DONE  job_id=%s  result_id=%s", job_id, result_id)

    except Exception as exc:
        logger.exception("[worker] FAILED  job_id=%s  attempt=%d  error=%s",
                         job_id, attempt, exc)
        _set_progress(job_id, 0, "failed", "failed", str(exc))
        try:
            from api import db
            db.update_video_status(video_id, "failed", str(exc))
        except Exception:
            pass
        raise   # let RQ / threading-retry caller handle back-off
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
