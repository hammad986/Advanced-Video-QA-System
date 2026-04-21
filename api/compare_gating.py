"""Intelligent gating layer for /compare_videos.

Three deterministic, explainable checks decide whether a comparison should run:

  1. TOPIC ALIGNMENT — cosine similarity between per-video centroid embeddings.
       >= 0.75 → SAME TOPIC
       0.55 .. 0.75 → PARTIAL OVERLAP
       <  0.55 → DIFFERENT TOPIC
  2. QUERY ALIGNMENT — cosine of the query embedding to each video's centroid.
       Marks LOW RELEVANCE when the query doesn't align with both videos.
  3. SUFFICIENCY — top-1 retrieval score must be >= 0.65 *and* there must be
       at least 2 retrieved chunks, per video.

Decision precedence:
       topic_sim < 0.55                    → NOT_COMPARABLE
       insufficient on any video           → INSUFFICIENT
       0.55 <= topic_sim < 0.75            → PARTIAL
       low query relevance on any video    → PARTIAL  (still informative)
       otherwise                           → COMPARABLE

Centroids are built once per video_id and cached in-process. We use the BGE
chunk text already on disk (data/chunks/{video_id}.json) so this never touches
the FAISS index, never re-transcribes, and never makes an LLM call. Pure,
deterministic, explainable.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("video_qa.api.compare.gating")

# ── Thresholds (centralised so they can be tuned in one place) ─────────
TOPIC_SAME = 0.75
TOPIC_PARTIAL = 0.55
QUERY_RELEVANT = 0.45        # query↔centroid cosine for "relevant"
SUFFICIENCY_TOP_SCORE = 0.65 # min top-1 retrieval score per video
SUFFICIENCY_MIN_CHUNKS = 2   # min retrieved chunks per video
CENTROID_MAX_CHUNKS = 24     # cap to keep centroid build cheap on huge videos
CHUNKS_DIR = Path("data/chunks")

# ── Per-process centroid cache. Cleared by clear_centroid_cache() if needed. ──
_centroid_cache: Dict[str, np.ndarray] = {}
_cache_lock = threading.Lock()


def clear_centroid_cache() -> None:
    with _cache_lock:
        _centroid_cache.clear()


def _load_video_texts(video_id: str) -> List[str]:
    """Read chunk texts for a video from disk. Returns [] if not found."""
    p = CHUNKS_DIR / f"{video_id}.json"
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning("[gating] failed to read %s: %s", p, e)
        return []
    raw = data.get("chunks") or data.get("segments") or []
    title = (data.get("title") or "").strip()
    texts: List[str] = [title] if title else []
    for c in raw:
        t = (c.get("text") or c.get("content") or c.get("title") or "").strip()
        if t:
            texts.append(t)
    return texts[:CENTROID_MAX_CHUNKS]


def _embed_mean(texts: List[str]) -> Optional[np.ndarray]:
    """Embed a list of texts with the existing BGE model and return the
    L2-normalized mean vector. Returns None if embeddings fail or list empty.
    Catches all exceptions: an embedding-model crash must downgrade to a
    safe refusal, never propagate as a 500."""
    if not texts:
        return None
    try:
        # Local import: keeps this module importable without paying the model
        # load cost at API startup.
        from video_qa.embeddings import create_embeddings, normalize_embeddings
        vecs = create_embeddings(texts)
        if vecs is None or len(vecs) == 0:
            return None
        centroid = np.asarray(vecs, dtype=np.float32).mean(axis=0, keepdims=True)
        centroid = normalize_embeddings(centroid)
        return centroid[0]
    except Exception as e:
        logger.warning("[gating] _embed_mean failed (%d texts): %s", len(texts), e)
        return None


def get_video_centroid(video_id: str) -> Optional[np.ndarray]:
    """Return cached centroid for a video, computing on first access.

    The cache is keyed on ``(video_id, mtime)`` of the chunks file so that
    re-processing a video (which rewrites ``data/chunks/{video_id}.json``)
    transparently invalidates the cached centroid — no manual flush needed."""
    p = CHUNKS_DIR / f"{video_id}.json"
    try:
        mtime = p.stat().st_mtime
    except FileNotFoundError:
        logger.warning("[gating] no chunks on disk for video_id=%s — centroid unavailable", video_id)
        return None

    cache_key = f"{video_id}::{mtime}"
    with _cache_lock:
        cached = _centroid_cache.get(cache_key)
    if cached is not None:
        return cached

    texts = _load_video_texts(video_id)
    if not texts:
        return None
    centroid = _embed_mean(texts)
    if centroid is None:
        return None
    with _cache_lock:
        # Drop any previous entry for this video_id (older mtime) before
        # inserting the fresh one — keeps the cache from growing unboundedly
        # across reprocessings.
        prefix = f"{video_id}::"
        for k in [k for k in _centroid_cache if k.startswith(prefix)]:
            _centroid_cache.pop(k, None)
        _centroid_cache[cache_key] = centroid
    return centroid


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Both inputs are L2-normalized, so cosine == dot. Defensive renorm anyway.
    na = float(np.linalg.norm(a)) or 1.0
    nb = float(np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / (na * nb))


def _embed_query(query: str) -> Optional[np.ndarray]:
    """Embed the query. Returns None on any failure (caller handles refusal)."""
    try:
        from video_qa.embeddings import create_embeddings
        vecs = create_embeddings([query])
        if vecs is None or len(vecs) == 0:
            return None
        return np.asarray(vecs, dtype=np.float32)[0]
    except Exception as e:
        logger.warning("[gating] _embed_query failed: %s", e)
        return None


# ── Public API ─────────────────────────────────────────────────────────
def evaluate(
    question: str,
    per_video_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run the 3-check gating layer.

    Args:
        question: the user query.
        per_video_chunks: a list of {"video_id", "filename", "chunks"}; the
            "chunks" entry must be the list returned by RetrievalSystem.retrieve
            (so each chunk has at least a "score" field).

    Returns:
        {
          "decision": "COMPARABLE" | "PARTIAL" | "NOT_COMPARABLE" | "INSUFFICIENT",
          "reason":   str,                              # human-readable
          "topic_similarity": float | None,             # min pairwise centroid cosine
          "pairwise_topic": [                           # all O(n²) pairs
              {"a": vid, "b": vid, "similarity": float}, ...
          ],
          "query_relevance": {video_id: float},         # query↔centroid cosine
          "sufficiency": {video_id: {                   # per-video gate
              "ok": bool,
              "top_score": float,
              "n_chunks": int,
              "issue": str | None,
          }},
        }
    """
    video_ids = [e["video_id"] for e in per_video_chunks]

    # 1. Topic alignment (centroid-centroid).
    centroids: Dict[str, Optional[np.ndarray]] = {
        vid: get_video_centroid(vid) for vid in video_ids
    }
    pairwise: List[Dict[str, Any]] = []
    sims: List[float] = []
    for i in range(len(video_ids)):
        for j in range(i + 1, len(video_ids)):
            a, b = video_ids[i], video_ids[j]
            ca, cb = centroids[a], centroids[b]
            if ca is None or cb is None:
                pairwise.append({"a": a, "b": b, "similarity": None})
                continue
            s = _cosine(ca, cb)
            pairwise.append({"a": a, "b": b, "similarity": round(s, 4)})
            sims.append(s)
    topic_similarity = min(sims) if sims else None  # weakest link governs

    # 2. Query alignment (query↔centroid).
    qvec = _embed_query(question)
    query_relevance: Dict[str, float] = {}
    for vid, cent in centroids.items():
        if cent is None or qvec is None:
            query_relevance[vid] = 0.0
        else:
            query_relevance[vid] = round(_cosine(qvec, cent), 4)

    # 3. Sufficiency (per-video retrieval signal).
    sufficiency: Dict[str, Dict[str, Any]] = {}
    insufficient_video_ids: List[str] = []
    for entry in per_video_chunks:
        vid = entry["video_id"]
        chunks = entry["chunks"] or []
        scores = [float(c.get("score", 0.0)) for c in chunks]
        top = max(scores) if scores else 0.0
        n = len(chunks)
        issue: Optional[str] = None
        ok = True
        if n < SUFFICIENCY_MIN_CHUNKS:
            ok = False
            issue = f"only {n} relevant chunk(s) (need ≥{SUFFICIENCY_MIN_CHUNKS})"
        elif top < SUFFICIENCY_TOP_SCORE:
            ok = False
            issue = (f"top retrieval score {top:.2f} below threshold "
                     f"{SUFFICIENCY_TOP_SCORE:.2f}")
        sufficiency[vid] = {"ok": ok, "top_score": round(top, 4),
                            "n_chunks": n, "issue": issue}
        if not ok:
            insufficient_video_ids.append(vid)

    # ── Decision engine ────────────────────────────────────────────────
    # FAIL-SAFE: if we couldn't build centroids for one or more videos
    # (missing chunks file, embedding model crash, etc.) we cannot
    # responsibly judge topic alignment. Refuse with INSUFFICIENT rather
    # than silently falling through to PARTIAL/COMPARABLE.
    missing_centroids = [vid for vid, c in centroids.items() if c is None]
    if missing_centroids:
        return {
            "decision": "INSUFFICIENT",
            "reason": (f"Cannot evaluate comparability — embeddings unavailable "
                       f"for: {', '.join(missing_centroids)}. The video(s) may "
                       f"not be fully processed yet."),
            "topic_similarity": None,
            "pairwise_topic": pairwise,
            "query_relevance": query_relevance,
            "sufficiency": sufficiency,
        }

    # Hard NO: completely different topics → never compare, even if retrieval
    # happened to surface chunks (those would be misleading).
    if topic_similarity is not None and topic_similarity < TOPIC_PARTIAL:
        return {
            "decision": "NOT_COMPARABLE",
            "reason": (f"Videos cover different topics "
                       f"(centroid similarity {topic_similarity:.2f} < {TOPIC_PARTIAL:.2f}). "
                       f"Refusing to compare to avoid a misleading answer."),
            "topic_similarity": round(topic_similarity, 4),
            "pairwise_topic": pairwise,
            "query_relevance": query_relevance,
            "sufficiency": sufficiency,
        }

    if insufficient_video_ids:
        bullets = "; ".join(
            f"'{vid}': {sufficiency[vid]['issue']}" for vid in insufficient_video_ids
        )
        return {
            "decision": "INSUFFICIENT",
            "reason": f"Insufficient evidence to compare — {bullets}.",
            "topic_similarity": round(topic_similarity, 4) if topic_similarity is not None else None,
            "pairwise_topic": pairwise,
            "query_relevance": query_relevance,
            "sufficiency": sufficiency,
        }

    if topic_similarity is not None and topic_similarity < TOPIC_SAME:
        return {
            "decision": "PARTIAL",
            "reason": (f"Videos partially overlap "
                       f"(centroid similarity {topic_similarity:.2f}, between "
                       f"{TOPIC_PARTIAL:.2f} and {TOPIC_SAME:.2f}). Comparison "
                       f"may be uneven; treat as directional."),
            "topic_similarity": round(topic_similarity, 4),
            "pairwise_topic": pairwise,
            "query_relevance": query_relevance,
            "sufficiency": sufficiency,
        }

    # Topic is fine and evidence is sufficient. Last guard: query relevance.
    low_relevance = [vid for vid, s in query_relevance.items() if s < QUERY_RELEVANT]
    if len(low_relevance) >= 1 and qvec is not None:
        return {
            "decision": "PARTIAL",
            "reason": (f"Query is loosely related to: "
                       f"{', '.join(low_relevance)} "
                       f"(query↔centroid cosine < {QUERY_RELEVANT:.2f}). "
                       f"Returning a comparison but flagging weaker matches."),
            "topic_similarity": round(topic_similarity, 4) if topic_similarity is not None else None,
            "pairwise_topic": pairwise,
            "query_relevance": query_relevance,
            "sufficiency": sufficiency,
        }

    return {
        "decision": "COMPARABLE",
        "reason": (f"Videos are on the same topic "
                   f"(centroid similarity "
                   f"{topic_similarity:.2f} ≥ {TOPIC_SAME:.2f}) and each has "
                   f"sufficient relevant evidence."
                   if topic_similarity is not None
                   else "Sufficient evidence retrieved for each video."),
        "topic_similarity": round(topic_similarity, 4) if topic_similarity is not None else None,
        "pairwise_topic": pairwise,
        "query_relevance": query_relevance,
        "sufficiency": sufficiency,
    }
