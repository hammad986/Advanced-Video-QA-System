"""Job queue + in-process worker.

Two modes, selected automatically at startup:

1. Native threading.Queue  (default, no external service needed)
   * Job state is persisted in PostgreSQL by the worker itself.
   * Safe for single-worker / single-process deployments.

2. RQ + real Redis          (set REDIS_URL in Secrets)
   * Enables durable queue, external workers, and the RQ dashboard.
   * Requires an external Redis server.

The GET /job_status/{job_id} endpoint reads progress from PostgreSQL in
both modes, so the polling contract is identical.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("video_qa.jobs")

_REDIS_URL = os.environ.get("REDIS_URL", "")

# ── Internal thread-queue (used when REDIS_URL is absent) ───────────────
_thread_queue: queue.Queue = queue.Queue()
_worker_thread: Optional[threading.Thread] = None


# ── Public API ─────────────────────────────────────────────────────────

def start_worker() -> threading.Thread:
    """Start the background worker thread and return it.

    Safe to call multiple times — only the first call creates the thread.
    """
    global _worker_thread
    if _worker_thread is not None and _worker_thread.is_alive():
        return _worker_thread

    if _REDIS_URL:
        _worker_thread = _start_rq_worker()
    else:
        _worker_thread = _start_thread_worker()

    return _worker_thread


def enqueue_video_job(
    job_id: str,
    video_id: str,
    user_id: str,
    file_url: str,
    original_filename: str,
) -> None:
    """Submit a video processing job to the active queue."""
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
        logger.info("[jobs] queued job_id=%s (thread-queue depth=%d)",
                    job_id, _thread_queue.qsize())


# ── Native threading.Queue worker ──────────────────────────────────────

def _start_thread_worker() -> threading.Thread:
    def _loop() -> None:
        logger.info("[jobs] native threading.Queue worker started")
        while True:
            payload = _thread_queue.get()
            try:
                _run_job(payload)
            except Exception as exc:
                logger.exception("[jobs] unhandled exception in job loop: %s", exc)
            finally:
                _thread_queue.task_done()

    t = threading.Thread(target=_loop, daemon=True, name="video-worker")
    t.start()
    return t


def _run_job(payload: Dict[str, Any]) -> None:
    from workers.video_worker import process_video_job
    process_video_job(
        payload["job_id"],
        payload["video_id"],
        payload["user_id"],
        payload["file_url"],
        payload["original_filename"],
    )


# ── RQ + real Redis worker (optional) ──────────────────────────────────

def _rq_enqueue(payload: Dict[str, Any]) -> None:
    from redis import Redis
    from rq import Queue

    redis_conn = Redis.from_url(_REDIS_URL)
    q = Queue(connection=redis_conn)
    from workers.video_worker import process_video_job
    q.enqueue(
        process_video_job,
        payload["job_id"],
        payload["video_id"],
        payload["user_id"],
        payload["file_url"],
        payload["original_filename"],
        job_id=payload["job_id"],
        job_timeout=3600,
        result_ttl=86400,
        failure_ttl=86400,
    )
    logger.info("[jobs] enqueued to RQ job_id=%s", payload["job_id"])


def _start_rq_worker() -> threading.Thread:
    from redis import Redis
    from rq import Queue, SimpleWorker

    redis_conn = Redis.from_url(_REDIS_URL)
    q = Queue(connection=redis_conn)

    def _loop() -> None:
        logger.info("[jobs] RQ SimpleWorker started (REDIS_URL set)")
        worker = SimpleWorker([q], connection=redis_conn)
        worker.work(with_scheduler=False)

    t = threading.Thread(target=_loop, daemon=True, name="rq-worker")
    t.start()
    return t
