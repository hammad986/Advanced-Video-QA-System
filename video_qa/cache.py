"""
Query-level LRU cache for the Video-QA pipeline.

Keyed by (normalized_query, video_id). Stores the full pipeline result dict.
Bounded size (default 256) with simple OrderedDict-based LRU eviction.
Thread-safe via a single lock — Streamlit can issue concurrent requests.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


def _normalize(query: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation noise at edges."""
    return " ".join((query or "").lower().strip().split())


def _make_key(query: str, video_id: Optional[str]) -> str:
    raw = f"{_normalize(query)}||{video_id or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class QueryCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._store: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, query: str, video_id: Optional[str]) -> Optional[Dict[str, Any]]:
        key = _make_key(query, video_id)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self.hits += 1
                # Return a shallow copy so downstream mutations don't corrupt cache
                return dict(self._store[key])
            self.misses += 1
            return None

    def set(self, query: str, video_id: Optional[str], value: Dict[str, Any]) -> None:
        key = _make_key(query, video_id)
        with self._lock:
            self._store[key] = dict(value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"size": len(self._store), "hits": self.hits, "misses": self.misses}


# Shared module-level cache instance
_GLOBAL_CACHE = QueryCache(max_size=256)


def get_cache() -> QueryCache:
    return _GLOBAL_CACHE
