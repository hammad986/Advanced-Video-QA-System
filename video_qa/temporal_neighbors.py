"""
Temporal reasoning helper: resolve previous/next chunks of a given chunk
using the FAISS metadata (video_id, chunk_id, start, end).

This enables questions like "how was this derived?" / "what came before/after?"
by expanding retrieved chunks with their temporal neighbours.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .embeddings import VectorStore
from .config import config
from .logger import get_logger

logger = get_logger(__name__)


def _load_metadata() -> List[Dict[str, Any]]:
    index_path = config.get("retrieval.index_path", "models/video_index.faiss")
    metadata_path = config.get("retrieval.metadata_path", "models/metadata.pkl")
    vs = VectorStore(index_path, metadata_path)
    return vs.metadata or []


def _index_by_video(metadata: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group metadata by video_id and sort each group by (chunk_id, start)."""
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for m in metadata:
        vid = m.get("video_id") or ""
        buckets.setdefault(vid, []).append(m)
    for vid, items in buckets.items():
        items.sort(key=lambda x: (x.get("chunk_id", 0), x.get("start", 0.0)))
    return buckets


def get_neighbors(
    chunk: Dict[str, Any],
    *,
    window: int = 1,
    _cache: Dict[str, Any] = {},
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return previous and next chunks (up to `window` each side) of `chunk`
    within the same video_id.

    Returns {"previous": [...], "next": [...]}. Each neighbour is a copy of
    the metadata dict with an added 'relation' key ('prev'/'next').
    """
    if "index" not in _cache:
        meta = _load_metadata()
        _cache["index"] = _index_by_video(meta)

    vid = chunk.get("video_id")
    cid = chunk.get("chunk_id")
    if vid is None or cid is None:
        return {"previous": [], "next": []}

    bucket = _cache["index"].get(vid, [])
    if not bucket:
        return {"previous": [], "next": []}

    # Find position of this chunk
    pos = None
    for i, m in enumerate(bucket):
        if m.get("chunk_id") == cid:
            pos = i
            break
    if pos is None:
        return {"previous": [], "next": []}

    prev = bucket[max(0, pos - window):pos]
    nxt = bucket[pos + 1:pos + 1 + window]

    prev_out = [{**m, "relation": "prev"} for m in prev]
    next_out = [{**m, "relation": "next"} for m in nxt]
    return {"previous": prev_out, "next": next_out}


def expand_with_neighbors(
    chunks: List[Dict[str, Any]],
    *,
    window: int = 1,
) -> List[Dict[str, Any]]:
    """
    Given a list of retrieved chunks, return a new list that includes each
    chunk's prev/next neighbours (de-duplicated by (video_id, chunk_id)).
    Original chunks keep their similarity score; neighbours get score=None
    and relation tag.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []

    def _key(c: Dict[str, Any]) -> Tuple[Any, Any]:
        return (c.get("video_id"), c.get("chunk_id"))

    for c in chunks:
        k = _key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
        nb = get_neighbors(c, window=window)
        for n in nb["previous"] + nb["next"]:
            nk = _key(n)
            if nk not in seen:
                seen.add(nk)
                out.append(n)
    return out
