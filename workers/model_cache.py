"""Process-level pipeline singleton for RQ worker processes.

Importing this module in a worker process lets models be loaded once at
startup (via preload()) and reused across every job in that process,
avoiding the costly re-initialisation on each job call.

In threading-mode (no REDIS_URL), the API and worker share the same
process and therefore the same api.pipeline_singleton instance, so this
module is not used — workers/video_worker.py always delegates to
api.pipeline_singleton.get_pipeline().

Usage in rq_worker_entrypoint.py
---------------------------------
    from workers.model_cache import preload
    preload()   # warm up before first job
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger("video_qa.worker")

_lock:     threading.Lock = threading.Lock()
_pipeline: Optional[object] = None


def get_pipeline():
    """Return the shared VideoQAPipeline, initialising it on first call."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                from video_qa.pipeline import VideoQAPipeline
                logger.info("[model_cache] Initialising VideoQAPipeline…")
                _pipeline = VideoQAPipeline()
                logger.info("[model_cache] Pipeline ready")
    return _pipeline


def preload() -> None:
    """Eagerly warm up all models.  Call once at worker startup."""
    logger.info("[model_cache] Preloading models…")
    get_pipeline()
    logger.info("[model_cache] All models preloaded ✓")
