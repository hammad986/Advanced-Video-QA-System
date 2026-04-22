"""Topic-Strength + Recommendation layer for /compare_videos.

Pure, deterministic, retrieval-driven ranking. NO LLM calls. Every score is
derived from numbers that were already produced upstream:

  similarity  →  mean of top-k FAISS retrieval scores (per video)
  coverage    →  fraction of requested top-k actually returned (per video)
  clarity     →  text-statistics heuristic on the retrieved chunk text
                  · shorter sentences and shorter words ⇒ higher clarity
                  · combined into one 0..1 score, deterministic, no LLM

  topic_strength = 0.5·sim + 0.3·cov + 0.2·clarity   (all in 0..1)

Best-video selection: argmax of topic_strength, but ONLY when the gate said
the comparison is valid (status COMPARABLE or PARTIAL). For NOT_COMPARABLE
or INSUFFICIENT we deliberately return ``best_video = None`` and an empty
recommendation — the system must refuse to rank when it cannot compare.

Recommendation:
  · "beginner" → highest *clarity* (simplest language). Tie-broken by score.
  · "revision" → highest *coverage × similarity* (most depth on the topic).

Differences:
  Pairwise text-statistics deltas turned into short, factual one-liners.
  Examples produced by this module (never invented prose):
    "Lecture A uses shorter sentences (avg 11 words) than Lecture B (avg 22)."
    "Lecture B has denser vocabulary (avg word length 5.4 vs 4.6)."

This module is import-safe (no model loads, no network) and side-effect free.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("video_qa.api.compare.ranking")

# ── Tunables (single source of truth) ──────────────────────────────────
WEIGHT_SIM = 0.5
WEIGHT_COV = 0.3
WEIGHT_CLARITY = 0.2

# Heuristic anchors for clarity: a "very clear" sentence averages ≤ 12 words
# and ≤ 4.5 chars/word. These are not magic — they're standard readability
# proxies (close to Flesch reading-ease intuitions) that work without any
# external dependency.
CLARITY_TARGET_SENT_WORDS = 12.0
CLARITY_TARGET_WORD_CHARS = 4.5

# A pairwise difference is only emitted when the underlying delta is large
# enough to be meaningful (avoids "Lecture A: 12.0 words vs Lecture B: 12.1").
MIN_SENT_DELTA = 3.0          # words/sentence
MIN_WORD_DELTA = 0.4          # chars/word
MIN_SCORE_DELTA = 0.05        # topic_strength

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SENT_RE = re.compile(r"[.!?]+\s+|[.!?]+$|\n+")


# ── Text statistics ────────────────────────────────────────────────────
def _text_stats(texts: List[str]) -> Dict[str, float]:
    """Return deterministic statistics over a list of chunk texts."""
    blob = " ".join(t for t in texts if t).strip()
    if not blob:
        return {"n_words": 0, "avg_sent_words": 0.0, "avg_word_chars": 0.0}
    words = _WORD_RE.findall(blob)
    n_words = len(words)
    avg_word_chars = (sum(len(w) for w in words) / n_words) if n_words else 0.0
    sentences = [s for s in _SENT_RE.split(blob) if s.strip()]
    n_sent_obs = max(1, len(sentences))
    # ASR transcripts often have no terminators, which would collapse a
    # 600-word chunk into ONE 600-word "sentence" and crush clarity. Put a
    # principled floor on the sentence count: assume a sentence is at most
    # ~30 words (well above natural English speech). Deterministic, bounded,
    # never reduces sentence count below what the regex actually saw.
    n_sent = max(n_sent_obs, math.ceil(n_words / 30.0))
    avg_sent_words = n_words / n_sent
    return {
        "n_words": float(n_words),
        "avg_sent_words": float(avg_sent_words),
        "avg_word_chars": float(avg_word_chars),
    }


def _clarity_score(stats: Dict[str, float]) -> float:
    """Map sentence/word length stats to a 0..1 clarity score.

    Clarity = mean of two sub-scores:
      sent_clarity = clamp01( target_sent / max(target, avg_sent_words) )
      word_clarity = clamp01( target_word / max(target, avg_word_chars) )

    Either component is 1.0 when the text is *at or below* the target;
    it decays smoothly above the target. If we have no text at all, clarity
    is 0 (we can't claim a clear explanation we never saw).
    """
    avg_sent = stats.get("avg_sent_words", 0.0)
    avg_word = stats.get("avg_word_chars", 0.0)
    if avg_sent <= 0 or avg_word <= 0:
        return 0.0
    sent_clarity = min(1.0, CLARITY_TARGET_SENT_WORDS / max(avg_sent, CLARITY_TARGET_SENT_WORDS))
    word_clarity = min(1.0, CLARITY_TARGET_WORD_CHARS / max(avg_word, CLARITY_TARGET_WORD_CHARS))
    return round((sent_clarity + word_clarity) / 2.0, 4)


def _normalise_similarity(scores: List[float]) -> float:
    """Mean of top retrieval scores, clamped to 0..1.
    FAISS IP on normalised BGE vectors usually sits in 0..1 already; we clamp
    defensively in case future retrievers return cosine in [-1, 1]."""
    if not scores:
        return 0.0
    m = sum(scores) / len(scores)
    return round(max(0.0, min(1.0, float(m))), 4)


def _coverage(n_returned: int, top_k_per_video: int) -> float:
    """How much of the requested top-k actually came back?"""
    if top_k_per_video <= 0:
        return 0.0
    return round(min(1.0, n_returned / float(top_k_per_video)), 4)


# ── Public API ─────────────────────────────────────────────────────────
def compute_topic_strength(
    per_video_chunks: List[Dict[str, Any]],
    top_k_per_video: int,
) -> Dict[str, Dict[str, float]]:
    """Per-video {similarity, coverage, clarity, score, n_words} dict."""
    out: Dict[str, Dict[str, float]] = {}
    for entry in per_video_chunks:
        chunks = entry.get("chunks") or []
        scores = [float(c.get("score", 0.0)) for c in chunks]
        texts = [str(c.get("text", "")) for c in chunks]
        stats = _text_stats(texts)
        sim = _normalise_similarity(scores)
        cov = _coverage(len(chunks), top_k_per_video)
        cla = _clarity_score(stats)
        score = round(WEIGHT_SIM * sim + WEIGHT_COV * cov + WEIGHT_CLARITY * cla, 4)
        out[entry["video_id"]] = {
            "similarity": sim,
            "coverage": cov,
            "clarity": cla,
            "score": score,
            "avg_sent_words": round(stats["avg_sent_words"], 2),
            "avg_word_chars": round(stats["avg_word_chars"], 2),
            "n_words": int(stats["n_words"]),
        }
    return out


def select_best(
    topic_strength: Dict[str, Dict[str, float]],
    status: str,
) -> Optional[str]:
    """Argmax video_id by topic_strength.score. ONLY when ranking is valid."""
    if status not in ("COMPARABLE", "PARTIAL"):
        return None
    if not topic_strength:
        return None
    return max(topic_strength.items(), key=lambda kv: kv[1]["score"])[0]


def make_recommendation(
    topic_strength: Dict[str, Dict[str, float]],
    filenames_by_id: Dict[str, str],
    status: str,
) -> Dict[str, Optional[str]]:
    """Return {beginner, revision, beginner_video_id, revision_video_id, explanation}.

    Both videos can be None when ranking is not valid (e.g. NOT_COMPARABLE).
    The explanation field is always populated — the spec requires it."""
    if status not in ("COMPARABLE", "PARTIAL") or not topic_strength:
        return {
            "beginner": None, "revision": None,
            "beginner_video_id": None, "revision_video_id": None,
            "explanation": ("Recommendations are withheld because the videos "
                            "are not safely comparable — see the gating reason."),
        }
    # Beginner: highest clarity; tie-break with overall score.
    beginner_id = max(
        topic_strength.items(),
        key=lambda kv: (kv[1]["clarity"], kv[1]["score"]),
    )[0]
    # Revision: highest "depth" = coverage × similarity; tie-break with score.
    revision_id = max(
        topic_strength.items(),
        key=lambda kv: (kv[1]["coverage"] * kv[1]["similarity"], kv[1]["score"]),
    )[0]
    b_name = filenames_by_id.get(beginner_id, beginner_id)
    r_name = filenames_by_id.get(revision_id, revision_id)
    b_clarity = topic_strength[beginner_id]["clarity"]
    r_depth = round(topic_strength[revision_id]["coverage"]
                    * topic_strength[revision_id]["similarity"], 3)
    return {
        "beginner": (f"For beginners: '{b_name}' — clearest language "
                     f"(clarity {b_clarity:.2f})."),
        "revision": (f"For revision: '{r_name}' — deepest coverage of the "
                     f"query (depth {r_depth:.2f})."),
        "beginner_video_id": beginner_id,
        "revision_video_id": revision_id,
        "explanation": ("Beginner pick maximises text clarity (shorter "
                        "sentences and simpler words). Revision pick "
                        "maximises retrieval depth (coverage × similarity)."),
    }


def extract_differences(
    topic_strength: Dict[str, Dict[str, float]],
    filenames_by_id: Dict[str, str],
    status: str,
) -> List[str]:
    """Pairwise factual deltas. Empty list when ranking is not valid."""
    if status not in ("COMPARABLE", "PARTIAL"):
        return []
    out: List[str] = []
    items = list(topic_strength.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a_id, a = items[i]
            b_id, b = items[j]
            a_name = filenames_by_id.get(a_id, a_id)
            b_name = filenames_by_id.get(b_id, b_id)

            # Sentence length: shorter ⇒ easier to follow.
            d_sent = a["avg_sent_words"] - b["avg_sent_words"]
            if abs(d_sent) >= MIN_SENT_DELTA:
                shorter, longer = (a_name, b_name) if d_sent < 0 else (b_name, a_name)
                short_v = min(a["avg_sent_words"], b["avg_sent_words"])
                long_v = max(a["avg_sent_words"], b["avg_sent_words"])
                out.append(
                    f"'{shorter}' uses shorter sentences "
                    f"(avg {short_v:.0f} words) than '{longer}' "
                    f"(avg {long_v:.0f}) — likely easier to follow."
                )

            # Word length: longer words ⇒ denser vocabulary.
            d_word = a["avg_word_chars"] - b["avg_word_chars"]
            if abs(d_word) >= MIN_WORD_DELTA:
                denser, plainer = (a_name, b_name) if d_word > 0 else (b_name, a_name)
                d_v = max(a["avg_word_chars"], b["avg_word_chars"])
                p_v = min(a["avg_word_chars"], b["avg_word_chars"])
                out.append(
                    f"'{denser}' has denser vocabulary "
                    f"(avg word length {d_v:.1f}) vs '{plainer}' "
                    f"({p_v:.1f})."
                )

            # Topic-strength gap: who explains the topic better overall.
            d_score = a["score"] - b["score"]
            if abs(d_score) >= MIN_SCORE_DELTA:
                stronger, weaker = (a_name, b_name) if d_score > 0 else (b_name, a_name)
                hi = max(a["score"], b["score"])
                lo = min(a["score"], b["score"])
                out.append(
                    f"'{stronger}' explains the topic more strongly overall "
                    f"(score {hi:.2f} vs {lo:.2f})."
                )
    if not out:
        out.append("Both videos cover the topic at a comparable depth and "
                   "reading level — no significant differences detected.")
    return out
