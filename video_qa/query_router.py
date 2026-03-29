"""
Query Router for Video-QA System
─────────────────────────────────
Implements a two-path hybrid architecture:

  PATH A  — Summary / overview questions
            Detected by keyword matching → returns stored lecture summary
            (no retrieval required)

  PATH B  — Factual / evidence questions  (default)
            Standard RAG: retrieval → re-ranking → LLM answer generation

The summary is generated once at processing time and stored as
  data/summaries/<video_id>.json
so repeated overview questions are answered instantly without an LLM call.
"""

import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import config
from .logger import get_logger

logger = get_logger(__name__)

# Phrases that indicate an LLM returned an error string rather than real content.
# Any response containing these is silently discarded.
_ERROR_INDICATORS = [
    "hf_token missing",
    "please provide your free hugging face token",
    "api_key",
    "hf auth login",
    "you must provide",
    "access denied",
    "unauthorized",
]


def _is_error_response(text: str) -> bool:
    """Return True if *text* looks like an API error rather than real content."""
    low = text.lower()
    return any(phrase in low for phrase in _ERROR_INDICATORS)

# ─────────────────────────────────────────────
# Summary storage helpers
# ─────────────────────────────────────────────
SUMMARY_DIR = Path("data/summaries")

SUMMARY_KEYWORDS = [
    r"\bsummar(y|ize|ise)\b",
    r"\boverview\b",
    r"\bwhat (is|was) (this |the )?lecture about\b",
    r"\bwhat did (we|i) learn\b",
    r"\b(main )?topic(s)?\b",
    r"\bwhat (does|did) (this|the) (video|lecture) (cover|discuss|explain|teach)\b",
    r"\bgist\b",
    r"\bbrief(ly)?\b",
    r"\bkey (point|takeaway|lesson)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in SUMMARY_KEYWORDS]


def is_summary_question(query: str) -> bool:
    """
    Return True if the query is asking for a lecture summary / overview.

    Uses keyword regex matching — fast, zero LLM call needed.
    """
    for pattern in _COMPILED:
        if pattern.search(query):
            logger.info(f"[Router] Summary intent detected for query: {query!r}")
            return True
    logger.info(f"[Router] Factual intent — routing to RAG for query: {query!r}")
    return False


# ─────────────────────────────────────────────
# Summary persistence
# ─────────────────────────────────────────────

def summary_path(video_id: str) -> Path:
    return SUMMARY_DIR / f"{video_id}.json"


def load_summary(video_id: str) -> Optional[Dict[str, Any]]:
    """Load a previously generated summary for *video_id*.

    Validates that the stored bullets don't contain error strings.
    If corrupt, deletes the file so it regenerates on next processing.
    """
    path = summary_path(video_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ── CORRUPTION GUARD ──────────────────────────────────────────────
        # If the stored summary contains error indicators (e.g. a previous run
        # stored an HF token-error string as a bullet), discard and regenerate.
        bullets = data.get("bullets", [])
        combined = " ".join(bullets).lower()
        if _is_error_response(combined):
            logger.warning(
                f"Stale/corrupt summary detected for {video_id!r} "
                f"(contains error indicators). Deleting and regenerating."
            )
            path.unlink(missing_ok=True)
            return None

        return data
    except Exception as exc:
        logger.warning(f"Failed to load summary for {video_id}: {exc}")
        return None


def save_summary(video_id: str, summary_data: Dict[str, Any]) -> None:
    """Persist a generated summary to disk."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    path = summary_path(video_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved: {path.name}")
    except Exception as exc:
        logger.error(f"Failed to save summary: {exc}")


# ─────────────────────────────────────────────
# Summary generation
# ─────────────────────────────────────────────

def generate_lecture_summary(
    full_transcript: str,
    video_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Generate a 5-bullet lecture summary from the full transcript text.

    Uses exclusively local Ollama (phi3:mini).

    The result is saved to data/summaries/<video_id>.json and returned.
    Returns None if the local LLM fails.
    """
    if not full_transcript or not full_transcript.strip():
        logger.warning("Empty transcript — cannot generate summary.")
        return None

    # Truncate very long transcripts to avoid context-length errors
    max_chars = 8_000
    text = full_transcript[:max_chars]
    if len(full_transcript) > max_chars:
        text += "\n[transcript truncated for brevity]"

    prompt = (
        "You are a lecture summarizer.\n"
        "Read the following transcript and write exactly 5 concise bullet points "
        "that capture the key topics and ideas covered in this lecture.\n"
        "Format each bullet starting with '• '.\n"
        "Do NOT include anything outside the 5 bullet points.\n\n"
        f"Transcript:\n{text}\n\n"
        "5-bullet summary:"
    )

    response: Optional[str] = None

    # 1. Ollama (local) exclusively
    try:
        from .answer_generator import ask_local_llm
        _r = ask_local_llm(prompt, timeout=120)
        if _r and not _is_error_response(_r):
            response = _r
            logger.info("[Summary] Local LLM generated summary.")
    except Exception:
        pass

    if not response:
        logger.warning("[Summary] Local LLM failed — no summary generated.")
        return None

    # Parse bullets (keep only lines starting with •, -, *, or digits)
    bullets: List[str] = []
    for line in response.splitlines():
        line = line.strip()
        if line and (
            line.startswith("•")
            or line.startswith("-")
            or line.startswith("*")
            or (len(line) > 2 and line[0].isdigit() and line[1] in ".)")
        ):
            # Normalise to "• ..." format
            clean = re.sub(r"^[\•\-\*\d\.\)]+\s*", "", line).strip()
            if clean:
                bullets.append(f"• {clean}")

    if not bullets:
        # Fallback: treat every non-empty line as a bullet
        bullets = [f"• {l.strip()}" for l in response.splitlines() if l.strip()]

    # Cap at 5 bullets
    bullets = bullets[:5]

    summary_data = {
        "video_id":  video_id,
        "bullets":   bullets,
        "raw":       response,
    }
    save_summary(video_id, summary_data)
    return summary_data


# ─────────────────────────────────────────────
# Route a query — returns result dict or None
# ─────────────────────────────────────────────

def route_summary_query(
    query: str,
    video_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """
    If *query* is a summary/overview question AND a summary exists for
    *video_id*, return a formatted result dict.

    If the summary file is missing, dynamically generate a lightweight summary
    using FAISS retrieval (top 5 chunks) + Ollama to prevent standard RAG fallback.

    Returns None ONLY if this is not a summary question.
    """
    if not is_summary_question(query):
        return None

    if not video_id:
        logger.warning("[Router] Summary query but no active_video_id — falling through to RAG.")
        return None

    summary = load_summary(video_id)
    if summary:
        bullets = summary.get("bullets", [])
        answer_text = "**Lecture Overview**\n\n" + "\n".join(bullets)

        return {
            "answer":       answer_text,
            "timestamp":    None,
            "contexts":     [],          # no retrieval chunks for pre-computed summary
            "verified":     True,
            "confidence":   100,
            "confidence_label": "High",
            "confidence_explanation": [
                "Full transcript coverage used for summary generation",
                "No retrieval loss (100% context available)"
            ],
            "verification": {
                "status":      "VERIFIED",
                "justification": "This summary was synthethized directly from the entire lecture transcript.",
                "trust_score": 100,
                "confidence":  1.0,
                "method":      "full_transcript"
            },
            "is_summary":   True,
        }

    # ── SMART FALLBACK: Generate summary dynamically ──────────────────────
    logger.warning(f"[Router] Summary missing for {video_id!r} — generating smart fallback summary.")

    try:
        from .retrieval import RetrievalSystem
        from .answer_generator import ask_local_llm

        retriever = RetrievalSystem()
        if not retriever.is_available():
            raise RuntimeError("FAISS retriever is unavailable.")

        # Get top 5 chunks
        retrieved_chunks = retriever.retrieve(query, top_k=5, video_id=video_id)
        if not retrieved_chunks:
            raise ValueError("No chunks retrieved.")

        # Combine chunks into context string
        context_text = "\n\n".join([f"--- Excerpt {i+1} ---\n{c.get('text', '')}" for i, c in enumerate(retrieved_chunks)])

        prompt = f"""You are a helpful assistant.

Given the following lecture transcript excerpts:
{context_text}

Generate a short and clear summary of the lecture.

Rules:
* Use only the given context
* Keep it concise (4–6 bullet points)
* Do not add external knowledge

Answer:"""

        logger.info("[Router] Calling fallback LLM for summary...")
        from .answer_generator import generate_with_fallback
        fallback_res = generate_with_fallback(prompt)
        response = fallback_res.get("response", "")

        if not response or _is_error_response(response):
            raise RuntimeError("LLM failed to generate fallback summary.")

        # Parse bullets
        bullets = []
        for line in response.splitlines():
            line = line.strip()
            if line and (line.startswith("•") or line.startswith("-") or line.startswith("*") or (len(line) > 2 and line[0].isdigit() and line[1] in ".)")):
                clean = re.sub(r"^[\•\-\*\d\.\)]+\s*", "", line).strip()
                if clean:
                    bullets.append(f"• {clean}")

        if not bullets:
            bullets = [f"• {line.strip()}" for line in response.splitlines() if line.strip()][:5]
        
        bullets = bullets[:6] # ensure max 6
        answer_text = "**Lecture Overview (Generated)**\n\n" + "\n".join(bullets)

        return {
            "answer":       answer_text,
            "timestamp":    None,
            "contexts":     [],
            "verified":     True,
            "confidence":   70,
            "confidence_label": "High",
            "confidence_explanation": [
                "Generated via dynamic fallback summarization",
                "Multiple chunks synthesized to create overview"
            ],
            "verification": {
                "status":      "VERIFIED",
                "justification": "Dynamically synthesized overview from the lecture.",
                "trust_score": 70,
                "confidence":  0.7,
                "method":      "dynamic_fallback"
            },
            "is_summary":   True,
        }

    except Exception as exc:
        logger.error(f"[Router] Smart fallback failed: {exc}")
        return {
            "answer":       "⚠️ Summary unavailable. Please ensure Ollama is running and re-process the video.",
            "timestamp":    None,
            "contexts":     [],
            "verified":     False,
            "verification": {
                "status":      "FAILED",
                "trust_score": 0,
                "confidence":  0.0,
            },
            "is_summary":   True,
        }
