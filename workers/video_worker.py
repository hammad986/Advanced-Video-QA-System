"""RQ job function — full video processing pipeline.

Runs inside a background thread (RQ SimpleWorker) in the same process as
the FastAPI application.  All api.* imports are available since we share
the same Python process and sys.path.

Progress stages
---------------
  0 %   queued
 10 %   downloading   — pull from S3 / local storage
 25 %   validating    — ffprobe probe for valid audio/video stream
 40 %   normalizing   — ffmpeg → 16 kHz mono WAV
 60 %   transcribing  — pipeline Whisper transcription
 80 %   indexing      — chunking + embeddings + FAISS write
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
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger("video_qa.worker")

_FFMPEG_TIMEOUT = int(os.environ.get("FFMPEG_TIMEOUT", "600"))  # seconds


# ── Helpers ─────────────────────────────────────────────────────────────

def _set_progress(
    job_id: str, progress: int, stage: str, status: str, error: Optional[str] = None
) -> None:
    try:
        from api import db
        db.update_job_progress(job_id, progress, stage, status, error)
    except Exception as exc:
        logger.warning("[worker] DB progress update failed: %s", exc)


def _ffprobe_validate(path: str) -> None:
    """Raise RuntimeError if path contains no audio or video stream."""
    def _probe(stream_spec: str) -> bool:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", stream_spec,
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0 and bool(result.stdout.strip())

    if not _probe("a:0") and not _probe("v:0"):
        raise RuntimeError("ffprobe: no audio or video stream found in file")


def _normalize_audio(input_path: str, output_wav: str) -> None:
    """Convert any media file to 16 kHz mono PCM-WAV (Whisper-optimal)."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",                   # drop video track
        "-acodec", "pcm_s16le",  # 16-bit signed PCM
        "-ar", "16000",          # 16 kHz sample rate
        "-ac", "1",              # mono
        output_wav,
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=_FFMPEG_TIMEOUT,
    )
    if result.returncode != 0:
        tail = (result.stderr or "")[-600:]
        raise RuntimeError(f"ffmpeg normalization failed: {tail}")


# ── Job entry point ─────────────────────────────────────────────────────

def process_video_job(
    job_id: str,
    video_id: str,
    user_id: str,
    file_url: str,
    original_filename: str,
) -> None:
    """Called by the RQ worker for every video upload."""
    logger.info(
        "[worker] START  job_id=%s  video_id=%s  file=%s",
        job_id, video_id, original_filename,
    )

    tmp_dir = tempfile.mkdtemp(prefix="vqa_job_")
    try:
        from api import storage, db

        # ── 1 · Download ───────────────────────────────────────────────
        _set_progress(job_id, 10, "downloading", "processing")
        ext = Path(original_filename).suffix or ".mp4"
        raw_path = os.path.join(tmp_dir, f"raw{ext}")
        storage.download(file_url, raw_path)
        logger.info("[worker] downloaded  %s  (%d B)", raw_path, os.path.getsize(raw_path))

        # ── 2 · Validate ───────────────────────────────────────────────
        _set_progress(job_id, 25, "validating", "processing")
        _ffprobe_validate(raw_path)

        # ── 3 · Normalise audio ────────────────────────────────────────
        _set_progress(job_id, 40, "normalizing", "processing")
        wav_path = os.path.join(tmp_dir, "audio.wav")
        _normalize_audio(raw_path, wav_path)
        logger.info("[worker] normalised  %s  (%d B)", wav_path, os.path.getsize(wav_path))

        # ── 4 · Transcribe ─────────────────────────────────────────────
        _set_progress(job_id, 60, "transcribing", "processing")
        from api.pipeline_singleton import get_pipeline, index_write_lock
        pipe = get_pipeline()

        # ── 5 · Index ──────────────────────────────────────────────────
        _set_progress(job_id, 80, "indexing", "processing")
        with index_write_lock:
            result_id = pipe.process_video(wav_path)

        if not result_id:
            raise RuntimeError("Pipeline returned no video_id after processing")

        # ── Done ───────────────────────────────────────────────────────
        db.update_job_progress(job_id, 100, "ready", "ready")
        db.update_video_status(video_id, "ready")
        logger.info("[worker] DONE  job_id=%s  result_id=%s", job_id, result_id)

    except Exception as exc:
        logger.exception("[worker] FAILED  job_id=%s  error=%s", job_id, exc)
        _set_progress(job_id, 0, "failed", "failed", str(exc))
        try:
            from api import db
            db.update_video_status(video_id, "failed", str(exc))
        except Exception:
            pass
        raise  # re-raise so RQ marks the job as failed
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
