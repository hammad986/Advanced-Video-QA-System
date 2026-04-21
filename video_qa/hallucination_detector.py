"""
Hallucination detector — fast, deterministic post-check for generated answers.

Given an answer and the retrieved contexts, compute:
  - semantic similarity (cosine) between the answer embedding and the
    concatenated-context embedding
  - keyword overlap ratio (answer content words ∩ context)
  - a fused support score in [0, 1]

Return a status:
  - SUPPORTED    — high semantic sim AND high keyword overlap
  - PARTIAL      — one signal strong, the other weak
  - UNSUPPORTED  — both signals weak (likely hallucination)

No LLM calls. Uses the existing embedding model already loaded by the pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import numpy as np

from .logger import get_logger

logger = get_logger(__name__)

# Status thresholds — tuned conservatively; evaluation script will report
# how these map to actual hallucination rate on the test set.
SIM_SUPPORTED   = 0.70
SIM_PARTIAL     = 0.45
OVERLAP_SUPPORTED = 0.50
OVERLAP_PARTIAL   = 0.25

# Generic stopwords we don't want to count as "supporting" keywords
_STOPWORDS = {
    "the", "this", "that", "these", "those", "there", "here", "with", "from",
    "your", "into", "about", "what", "when", "where", "which", "while", "will",
    "would", "could", "should", "have", "been", "being", "does", "done", "such",
    "then", "than", "also", "just", "only", "very", "much", "more", "most",
    "some", "many", "like", "over", "under", "between", "both", "each", "other",
    "they", "them", "their", "because", "however", "therefore", "answer", "question",
}


def _content_words(text: str) -> List[str]:
    toks = re.findall(r"\b[a-zA-Z]{4,}\b", (text or "").lower())
    return [t for t in toks if t not in _STOPWORDS]


def _keyword_overlap(answer: str, contexts: List[Dict[str, Any]]) -> float:
    ans_words = set(_content_words(answer))
    if not ans_words:
        return 0.0
    blob = " ".join((c.get("text") or "") for c in contexts).lower()
    matched = sum(1 for w in ans_words if w in blob)
    return matched / len(ans_words)


def _strip_citation(answer: str) -> str:
    """Remove the '(see [mm:ss - mm:ss])' tail before embedding — it's meta, not content."""
    return re.sub(r"\(\s*see\s*\[\d{1,3}:\d{2}\s*-\s*\d{1,3}:\d{2}\]\s*\)", "", answer or "").strip()


def _semantic_similarity(answer: str, contexts: List[Dict[str, Any]]) -> float:
    """Cosine similarity between the answer embedding and concatenated-context embedding."""
    try:
        from .embeddings import create_embeddings
    except Exception as e:
        logger.warning(f"[HALLUCINATION] embedding import failed: {e}")
        return 0.0

    ans_text = _strip_citation(answer)
    if not ans_text.strip():
        return 0.0
    ctx_text = " ".join((c.get("text") or "") for c in contexts).strip()
    if not ctx_text:
        return 0.0

    try:
        embs = create_embeddings([ans_text, ctx_text])
    except Exception as e:
        logger.warning(f"[HALLUCINATION] embedding call failed: {e}")
        return 0.0

    if embs is None or getattr(embs, "shape", (0,))[0] < 2:
        return 0.0

    a = embs[0].astype("float32")
    b = embs[1].astype("float32")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _classify(sim: float, overlap: float) -> str:
    if sim >= SIM_SUPPORTED and overlap >= OVERLAP_SUPPORTED:
        return "SUPPORTED"
    if sim >= SIM_PARTIAL and overlap >= OVERLAP_PARTIAL:
        return "PARTIAL"
    # One strong + one weak → PARTIAL; both weak → UNSUPPORTED
    if (sim >= SIM_SUPPORTED) ^ (overlap >= OVERLAP_SUPPORTED):
        return "PARTIAL"
    return "UNSUPPORTED"


def check_answer(
    answer: str,
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run the full hallucination post-check.

    Returns:
        {
            "status":             "SUPPORTED" | "PARTIAL" | "UNSUPPORTED",
            "semantic_similarity": float in [0, 1],
            "keyword_overlap":     float in [0, 1],
            "support_score":       float in [0, 1]   # fused
        }
    """
    if not answer or not contexts:
        return {
            "status": "UNSUPPORTED",
            "semantic_similarity": 0.0,
            "keyword_overlap": 0.0,
            "support_score": 0.0,
        }

    # "Not found in video" is technically faithful (model declined) — treat as SUPPORTED
    if "not found in video" in answer.lower():
        return {
            "status": "SUPPORTED",
            "semantic_similarity": 1.0,
            "keyword_overlap": 1.0,
            "support_score": 1.0,
        }

    sim = max(0.0, min(1.0, _semantic_similarity(answer, contexts)))
    overlap = max(0.0, min(1.0, _keyword_overlap(answer, contexts)))
    status = _classify(sim, overlap)
    support_score = round(0.6 * sim + 0.4 * overlap, 4)

    return {
        "status": status,
        "semantic_similarity": round(sim, 4),
        "keyword_overlap": round(overlap, 4),
        "support_score": support_score,
    }
