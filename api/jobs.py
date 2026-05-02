"""Job queue — dual-mode: threading.Queue (default) or RQ + Redis.

Mode selection (automatic at startup)
--------------------------------------
No REDIS_URL  →  Python threading.Queue, in-process worker.
               Retry logic (max 3 attempts, exponential back-off) is built in.
               Safe for single-process Replit deployments.

REDIS_URL set →  RQ queue.  The API process only enqueues jobs.
               The worker runs as a *separate* process:
                 python -m workers.rq_worker_entrypoint
               (see Dockerfile.worker / docker-compose.yml)
               Retry via rq.Retry(max=3, interval=[10, 30, 60]).
               Job timeout: 3600 s (configurable via JOB_TIMEOUT_SECONDS).

In both modes GET /job_status/{job_id} and GET /job_stream/{job_id} read
progress from PostgreSQL, so the polling/SSE contract is identical.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("video_qa.jobs")

_REDIS_URL     = os.environ.get("REDIS_URL", "")
_JOB_TIMEOUT   = int(os.environ.get("JOB_TIMEOUT_SECONDS", "3600"))
_MAX_RETRIES   = int(os.environ.get("JOB_MAX_RETRIES", "3"))
_RETRY_DELAYS  = [10, 30, 60]          # seconds between threading-mode retries

# ── Internal thread-queue (threading mode) ─────────────────────────────
_thread_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None
_start_lock = threading.Lock()


# ── Public API ──────────────────────────────────────────────────────────

def start_worker() -> Optional[threading.Thread]:
    """Start the background worker.

    * threading mode  → starts a daemon thread, returns it.
    * Redis/RQ mode   → no in-process worker (external process handles jobs).
    Returns None in Redis mode.  Safe to call multiple times.
    """
    global _worker_thread
    if _REDIS_URL:
        logger.info(
            "[jobs] RQ mode active — start worker separately: "
            "python -m workers.rq_worker_entrypoint  (REDIS_URL=%s)",
            _REDIS_URL.split("@")[-1],
        )
        return None

    with _start_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return _worker_thread
        _worker_thread = _start_thread_worker()
    return _worker_thread


def enqueue_video_job(
    job_id: str,
    video_id: str,
    user_id: str,
    file_url: str,
    original_filename: str,
) -> None:
    """Submit a video-processing job to the active queue."""
    payload: Dict[str, Any] = {
        "job_id":            job_id,
        "video_id":          video_id,
        "user_id":           user_id,
        "file_url":          file_url,
        "original_filename": original_filename,
    }
    if _REDIS_URL:
        _rq_enqueue(payload)
    else:
        _thread_queue.put(payload)
        logger.info("[jobs] thread-queued  job_id=%s  depth=%d",
                    job_id, _thread_queue.qsize())


def queue_mode() -> str:
    """Return 'redis' or 'threading'."""
    return "redis" if _REDIS_URL else "threading"


# ── Threading-mode worker ──────────────────────────────────────────────

def _start_thread_worker() -> threading.Thread:
    def _loop() -> None:
        logger.info("[jobs] threading.Queue worker ready")
        while True:
            payload = _thread_queue.get()
            try:
                _run_with_retry(payload)
            except Exception as exc:
                logger.exception("[jobs] job permanently failed job_id=%s: %s",
                                 payload.get("job_id", "?"), exc)
            finally:
                _thread_queue.task_done()

    t = threading.Thread(target=_loop, daemon=True, name="video-worker")
    t.start()
    return t


def _run_with_retry(payload: Dict[str, Any]) -> None:
    """Execute the job up to _MAX_RETRIES times with back-off."""
    from workers.video_worker import process_video_job
    job_id = payload["job_id"]
    last_exc: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            if attempt > 1:
                logger.info("[jobs] retry %d/%d  job_id=%s", attempt, _MAX_RETRIES, job_id)
            process_video_job(
                job_id,
                payload["video_id"],
                payload["user_id"],
                payload["file_url"],
                payload["original_filename"],
                attempt=attempt,
            )
            return          # success
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                delay = _RETRY_DELAYS[min(attempt - 1, len(_RETRY_DELAYS) - 1)]
                logger.warning(
                    "[jobs] attempt %d/%d failed for job_id=%s, retrying in %ds: %s",
                    attempt, _MAX_RETRIES, job_id, delay, exc,
                )
                time.sleep(delay)

    raise last_exc  # type: ignore[misc]


# ── Redis / RQ mode ────────────────────────────────────────────────────

def _rq_enqueue(payload: Dict[str, Any]) -> None:
    from redis import Redis
    from rq import Queue, Retry

    conn = Redis.from_url(_REDIS_URL)
    q    = Queue(connection=conn)
    from workers.video_worker import process_video_job

    job = q.enqueue(
        process_video_job,
        payload["job_id"],
        payload["video_id"],
        payload["user_id"],
        payload["file_url"],
        payload["original_filename"],
        job_id=payload["job_id"],
        job_timeout=_JOB_TIMEOUT,
        result_ttl=86400,
        failure_ttl=86400,
        retry=Retry(max=_MAX_RETRIES, interval=_RETRY_DELAYS),
    )
    logger.info("[jobs] RQ-enqueued  job_id=%s  rq_job=%s", payload["job_id"], job.id)
