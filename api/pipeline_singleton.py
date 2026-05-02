"""Shared pipeline singleton and index write-lock.

Both api/main.py and workers/video_worker.py import from here so they share
the same pipeline instance and serialise writes to the FAISS index.
"""

from __future__ import annotations

import threading
import logging

logger = logging.getLogger("video_qa.pipeline_singleton")

_pipeline = None
_pipeline_lock = threading.Lock()
index_write_lock = threading.Lock()


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from video_qa.pipeline import VideoQAPipeline
                logger.info("[pipeline] initialising VideoQAPipeline…")
                _pipeline = VideoQAPipeline()
                logger.info("[pipeline] ready")
    return _pipeline
