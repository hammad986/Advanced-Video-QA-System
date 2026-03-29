"""
Confidence Scorer — Video-QA System
─────────────────────────────────────────────────────────────────
Computes a weighted confidence score (0-100) for a generated answer
based on retrieval quality signals.

Formula:
    confidence = (avg_similarity * 0.5) + (context_overlap * 0.3)
               + (chunk_agreement  * 0.2)
    → Normalised to 0-100, capped at 100

No LLM call required — fully deterministic and instant.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────────
# Config thresholds
# ─────────────────────────────────────────────────────────────────

_HIGH_SIM_THRESH   = 0.75   # avg cosine similarity considered "high"
_MED_SIM_THRESH    = 0.50
_HIGH_OVERLAP      = 0.60   # fraction of answer words found in context
_MED_OVERLAP       = 0.30
_HIGH_AGREEMENT    = 0.85   # top-2 chunk scores within 0.05 = perfect agreement
_AGREEMENT_GAP_MAX = 0.20   # gap at which agreement score bottoms out to 0.5

USEFUL_CHUNK_THRESHOLD = 0.50   # similarity score above which a chunk counts as "used"


# ─────────────────────────────────────────────────────────────────
# Internal factor calculators
# ─────────────────────────────────────────────────────────────────

def _avg_similarity(contexts: List[Dict[str, Any]]) -> float:
    """Mean cosine-similarity score across all retrieved chunks."""
    scores = [float(c.get("score", 0)) for c in contexts if c.get("score") is not None]
    return (sum(scores) / len(scores)) if scores else 0.0


def _context_overlap(answer: str, contexts: List[Dict[str, Any]]) -> float:
    """
    Fraction of unique answer words (≥4 chars) that appear in the
    concatenated context.  Stop-words are not excluded — simple and fast.
    """
    if not answer or not contexts:
        return 0.0

    context_blob = " ".join(c.get("text", "") for c in contexts).lower()
    answer_words = set(
        w for w in re.findall(r"\b[a-zA-Z]{4,}\b", answer.lower())
    )
    if not answer_words:
        return 0.0

    matched = sum(1 for w in answer_words if w in context_blob)
    return matched / len(answer_words)


def _chunk_agreement(contexts: List[Dict[str, Any]]) -> float:
    """
    Score based on how close the top-2 chunk similarity scores are.
    Perfect agreement (gap < 0.05) → 1.0
    Large gap (≥ _AGREEMENT_GAP_MAX)  → 0.5
    """
    if len(contexts) < 2:
        return 1.0  # single chunk, no disagreement possible

    scores = sorted(
        [float(c.get("score", 0)) for c in contexts if c.get("score") is not None],
        reverse=True,
    )
    gap = scores[0] - scores[1] if len(scores) >= 2 else 0.0
    gap = max(0.0, gap)

    if gap < 0.05:
        return 1.0
    if gap >= _AGREEMENT_GAP_MAX:
        return 0.5
    # Linear interpolation between 1.0 and 0.5
    return 1.0 - 0.5 * (gap - 0.05) / (_AGREEMENT_GAP_MAX - 0.05)


# ─────────────────────────────────────────────────────────────────
# Explanation builder
# ─────────────────────────────────────────────────────────────────

def _build_explanation(
    avg_sim: float,
    overlap: float,
    agreement: float,
) -> List[str]:
    """Convert factor values into human-readable bullet explanations."""
    bullets: List[str] = []

    # Similarity
    if avg_sim >= _HIGH_SIM_THRESH:
        bullets.append("High similarity across multiple retrieved chunks")
    elif avg_sim >= _MED_SIM_THRESH:
        bullets.append("Moderate similarity with retrieved chunks")
    else:
        bullets.append("Low retrieval similarity — context may be only partially relevant")

    # Overlap
    if overlap >= _HIGH_OVERLAP:
        bullets.append("Strong keyword overlap between answer and transcript")
    elif overlap >= _MED_OVERLAP:
        bullets.append("Partial keyword overlap with transcript")
    else:
        bullets.append("Low keyword overlap — answer may extend beyond retrieved context")

    # Agreement
    if agreement >= _HIGH_AGREEMENT:
        bullets.append("Consistent relevance across top retrieved chunks")
    else:
        bullets.append("Top chunks have slightly inconsistent relevance scores")

    return bullets


# ─────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────

def compute_confidence(
    answer: str,
    contexts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute a weighted confidence score for the generated answer.

    Args:
        answer:   LLM-generated answer string.
        contexts: All retrieved chunks (list of dicts with 'score' and 'text').

    Returns:
        {
            "score":          int   (0–100),
            "label":          str   ("High" | "Medium" | "Low"),
            "breakdown":      dict,
            "explanation":    List[str],
            "useful_chunks":  int,    # chunks above USEFUL_CHUNK_THRESHOLD
            "total_chunks":   int,
        }
    """
    if not contexts:
        return {
            "score":         0,
            "label":         "Low",
            "breakdown":     {"avg_similarity": 0.0, "context_overlap": 0.0, "chunk_agreement": 0.0},
            "explanation":   ["No context retrieved — cannot assess confidence"],
            "useful_chunks": 0,
            "total_chunks":  0,
        }

    avg_sim   = _avg_similarity(contexts)
    overlap   = _context_overlap(answer, contexts)
    agreement = _chunk_agreement(contexts)

    raw_score = (avg_sim * 0.5) + (overlap * 0.3) + (agreement * 0.2)
    score = min(100, int(raw_score * 100))

    if score >= 70:
        label = "High"
    elif score >= 40:
        label = "Medium"
    else:
        label = "Low"

    useful_chunks = sum(
        1 for c in contexts if float(c.get("score", 0)) >= USEFUL_CHUNK_THRESHOLD
    )

    return {
        "score":  score,
        "label":  label,
        "breakdown": {
            "avg_similarity":  round(avg_sim,   3),
            "context_overlap": round(overlap,   3),
            "chunk_agreement": round(agreement, 3),
        },
        "explanation":   _build_explanation(avg_sim, overlap, agreement),
        "useful_chunks": useful_chunks,
        "total_chunks":  len(contexts),
    }
