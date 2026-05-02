"""RQ worker entry point — run this as a separate process when REDIS_URL is set.

Usage
-----
  # directly
  REDIS_URL=redis://localhost:6379 python -m workers.rq_worker_entrypoint

  # via docker-compose (see docker-compose.yml)
  docker compose up worker

The process:
  1. Adds the project root to sys.path.
  2. Sets WORKER_MODE=rq so video_worker.py uses workers.model_cache.
  3. Preloads all ML models into memory (pipeline_singleton / model_cache).
  4. Starts an RQ Worker that processes the "default" queue.

Environment variables
---------------------
REDIS_URL             Redis connection string  (required)
RQ_QUEUE              Queue name               (default: default)
RQ_WORKER_NAME        Worker name for RQ dashboard (optional)
DATABASE_URL          PostgreSQL DSN           (required)
WORKER_MODE           Set automatically to "rq"
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── Path ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("WORKER_MODE", "rq")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("video_qa.rq_worker")


def main() -> None:
    redis_url  = os.environ.get("REDIS_URL", "redis://localhost:6379")
    queue_name = os.environ.get("RQ_QUEUE", "default")
    worker_name = os.environ.get("RQ_WORKER_NAME", None)

    logger.info("[rq_worker] Starting  queue=%s  redis=%s",
                queue_name, redis_url.split("@")[-1])

    # ── Preload models before first job arrives ─────────────────────────
    logger.info("[rq_worker] Preloading ML models…")
    try:
        from workers.model_cache import preload
        preload()
    except Exception as exc:
        logger.warning(
            "[rq_worker] Model preload failed — will load on first job: %s", exc
        )

    # ── Start RQ worker ────────────────────────────────────────────────
    from redis import Redis
    from rq import Queue, Worker

    conn   = Redis.from_url(redis_url)
    queues = [Queue(queue_name, connection=conn)]

    kwargs: dict = dict(connection=conn, queues=queues)
    if worker_name:
        kwargs["name"] = worker_name

    worker = Worker(**kwargs)
    logger.info("[rq_worker] Listening on queue '%s' — ready for jobs", queue_name)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
