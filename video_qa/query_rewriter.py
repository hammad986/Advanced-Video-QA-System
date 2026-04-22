import logging
import time
import requests
from dotenv import load_dotenv

# Lazy optional provider import for tests and CI without dependency.
# Keep `genai` object available for patching in tests even if package is absent.
try:
    import google.generativeai as genai
except ImportError:
    class _MockGenai:
        Client = None

    genai = _MockGenai()

load_dotenv()
import requests
from typing import Optional
from .config import config

logger = logging.getLogger(__name__)

# Constants
MAX_WORD_COUNT = 8
REWRITE_PROMPT_TEMPLATE = """Rewrite the query to improve semantic search clarity. Keep it concise.
Output ONLY the rewritten question, nothing else.

User Question:
{question}

Rewritten Question:"""

from typing import Tuple
import os

def call_gemini(prompt: str) -> str:
    global genai
    import os
    from dotenv import dotenv_values
    
    if genai is None:
        try:
            import google.generativeai as genai_module
            genai = genai_module
        except ImportError:
            raise ImportError("google-generativeai is not installed")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        env_dict = dotenv_values(".env")
        api_key = env_dict.get("GEMINI_API_KEY")
        
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
        
    # Use the installed `google.generativeai` (classic) API: configure + GenerativeModel.
    # Tests patch `video_qa.query_rewriter.genai` so either API shape works; prefer classic.
    if hasattr(genai, "Client"):
        client = genai.Client(api_key=api_key.strip("'\" "))
        response = client.models.generate_content(prompt)
    else:
        genai.configure(api_key=api_key.strip("'\" "))
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
    if response and hasattr(response, "text") and response.text:
        return response.text.strip()
    return ""

# ── Rule-based rewrite (deterministic, no LLM, zero-latency fallback) ──
# Strips conversational filler, normalises casing/punctuation, and lightly
# canonicalises common interrogative shapes. Designed to never make the
# query worse — if nothing matches, it returns the trimmed original.
import re as _re

_FILLER_PHRASES = [
    "could you please", "could you", "would you please", "would you",
    "can you please", "can you", "i would like to know", "i want to know",
    "i'd like to know", "please tell me", "tell me about", "i was wondering",
    "do you know", "do you happen to know", "i need to know",
    "would it be possible to", "is it possible to",
]
_TRAILING_FILLER = [
    "please", "thanks", "thank you", "if possible", "if you can",
]
_CONTRACTIONS = {
    "what's": "what is", "who's": "who is", "where's": "where is",
    "when's": "when is", "how's": "how is", "that's": "that is",
    "it's": "it is", "let's": "let us", "isn't": "is not",
    "aren't": "are not", "doesn't": "does not", "don't": "do not",
    "didn't": "did not", "won't": "will not", "can't": "cannot",
    "couldn't": "could not", "wouldn't": "would not",
    "shouldn't": "should not", "you're": "you are", "we're": "we are",
    "they're": "they are", "i'm": "i am",
}
_WORD_RE = _re.compile(r"\b\w[\w']*\b")


def _rule_based_rewrite(question: str) -> str:
    """Deterministic, dependency-free query rewrite. Pure function."""
    q = question.strip()
    if not q:
        return q
    # 1. Lowercase for matching but preserve original cap of first letter at end.
    low = q.lower()
    # 2. Strip leading filler phrases.
    changed = True
    while changed:
        changed = False
        for f in _FILLER_PHRASES:
            if low.startswith(f + " ") or low.startswith(f + ","):
                low = low[len(f):].lstrip(" ,").lstrip()
                changed = True
                break
    # 3. Strip trailing politeness ("…, please", "…, thanks").
    low = low.rstrip(" ?.!")
    for f in _TRAILING_FILLER:
        if low.endswith(", " + f) or low.endswith(" " + f):
            low = low[: -len(f)].rstrip(" ,").rstrip()
    # 4. Expand common contractions (improves embedding similarity).
    tokens = _WORD_RE.findall(low)
    expanded = " ".join(_CONTRACTIONS.get(t, t) for t in tokens) if tokens else low
    # 5. Collapse whitespace, restore terminal '?'.
    expanded = _re.sub(r"\s+", " ", expanded).strip()
    if not expanded:
        return q
    # If we accidentally stripped everything meaningful, fall back to original.
    if len(expanded.split()) < max(1, min(3, len(q.split()) // 2)):
        return q
    # Capitalise first letter; keep terminal '?'.
    expanded = expanded[0].upper() + expanded[1:]
    if "?" in q and not expanded.endswith("?"):
        expanded += "?"
    return expanded


def rewrite_query(question: str, force: bool = False) -> Tuple[str, str]:
    """Rewrite a user question for better retrieval.

    Provider chain (per fallback spec):
      1. Gemini ONLY  →  provider = "gemini"
      2. Rule-based   →  provider = "rule_based"   (no LLM, deterministic)

    Bedrock / other LLMs are intentionally NOT used here — query rewrite
    must never burn cost or latency on a heavy model. If both fail (or the
    query is too long), we return the trimmed original.

    Returns ``(rewritten_text, provider)``.
    """
    if not isinstance(question, str) or not question.strip():
        return (question, "original")

    original_query = question.strip()

    # Skip very long queries (degrades to full keyword retrieval path)
    if not force and len(original_query.split()) > MAX_WORD_COUNT:
        logger.info("[QUERY REWRITE] Query too long, skipping rewrite.")
        return (original_query, "original")

    prompt = (
        "Rewrite this question clearly. Return ONLY the rewritten question "
        f"text without any extra conversational filler or options:\n{original_query}"
    )
    logger.info(f"[QUERY REWRITE] Initiating rewrite cascade for query {original_query!r}")

    # 1. TRY GEMINI (only LLM allowed for rewrite)
    try:
        rewritten = call_gemini(prompt)
        if rewritten and len(rewritten.strip()) > 5:
            return (rewritten.strip(), "gemini")
    except Exception as e:
        logger.warning(f"[QUERY REWRITE] Gemini failed: {e}; using rule-based fallback.")

    # 2. RULE-BASED FALLBACK (deterministic, no LLM, no network)
    rb = _rule_based_rewrite(original_query)
    if rb and rb.lower() != original_query.lower():
        logger.info(f"[QUERY REWRITE] Rule-based rewrite: {original_query!r} → {rb!r}")
        return (rb, "rule_based")
    # Rule-based had nothing to change — return original unchanged.
    return (original_query, "original")
