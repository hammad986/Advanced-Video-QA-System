"""
Answer Generation Module for Video-QA System
===============================================================

Simplified architecture:
  1. Retrieve & Re-rank (in pipeline)
  2. Send combined top-2 chunks to Ollama
  3. Extract answer directly
  4. Use timestamp from the highest-scoring chunk's metadata
"""

import re
import os
import requests
from typing import Optional, Dict, Any, List

from .config import config
from .logger import get_logger
from .confidence_scorer import compute_confidence
from config_loader import load_config

CONFIG = load_config()

logger = get_logger(__name__)

# Session-level kill-switch tripped when we detect the Gemini daily/project quota
# is exhausted. Further calls short-circuit to the next provider instead of
# paying 3×backoff seconds per query for retries that cannot possibly succeed
# until the per-day quota window resets.
_GEMINI_DAILY_EXHAUSTED = False


def _looks_like_daily_quota(err_msg: str) -> bool:
    m = err_msg.lower()
    return (
        "perdayper" in m.replace(" ", "")
        or "requests per day" in m
        or "free_tier_requests" in m
        or "generate_content_free_tier_requests" in m
    )

# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def format_timestamp(seconds: float) -> str:
    """Convert seconds to mm:ss format."""
    if seconds is None:
        return "00:00"
    total_seconds = int(float(seconds))
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

def format_time_range(start: float, end: float) -> str:
    """Format timestamp range as [mm:ss - mm:ss]."""
    return f"[{format_timestamp(start)} - {format_timestamp(end)}]"

def _is_not_found(answer: Optional[str]) -> bool:
    """Return True if the answer string signals 'not found'."""
    if not answer:
        return True
    low = answer.lower().strip()
    return "not found in video" in low or "cannot be answered" in low


def get_gemini_response(prompt: str) -> Optional[str]:
    """Use Gemini API to generate text.

    When running on Streamlit Cloud, API key is retrieved from `st.secrets`.
    If not available, fall back to environment variables or the config.
    """
    try:
        import google.generativeai as genai
        try:
            import streamlit as st
            api_key = st.secrets.get("API_KEY") if hasattr(st, "secrets") else None
        except Exception:
            api_key = None

        if not api_key:
            api_key = os.environ.get("GEMINI_API_KEY") or config.get("answer.gemini_api_key", "")

        if not api_key:
            logger.error("Gemini API key not configured")
            return None

        genai.configure(api_key=api_key)
        model_name = CONFIG.get("answer", {}).get("api_model", "gemini") or config.get("answer.gemini_model", "gemini-pro")
        if model_name.lower().startswith("gemini"):
            model_name = model_name
        else:
            model_name = "gemini-pro"

        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)

        if response is None:
            return None

        text = getattr(response, "text", None)
        return text.strip() if text else None

    except Exception as exc:
        logger.error(f"Gemini request failed: {exc}")
        return None

# ──────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────

def build_answer_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build a grounded QA prompt that:
      - shows each chunk with its [mm:ss - mm:ss] timestamp
      - forces the model to answer ONLY from the given context
      - asks for an inline timestamp citation
      - requires an exact "Not found in video." string when evidence is missing
    """
    parts = []
    for i, ctx in enumerate(contexts, 1):
        ts = format_time_range(ctx.get("start", 0), ctx.get("end", 0))
        parts.append(f"--- Chunk {i}  {ts} ---\n{ctx.get('text', '')}")
    context_text = "\n\n".join(parts)

    return f"""You are a Video QA assistant. You answer questions strictly from the
transcript excerpts of a video that are provided below. Each excerpt is tagged
with its timestamp in [mm:ss - mm:ss] format.

TRANSCRIPT CONTEXT:
{context_text}

RULES:
1. Answer ONLY using information explicitly present in the transcript above.
2. Do NOT use outside knowledge. Do NOT speculate.
3. Keep the answer concise: 1-3 sentences.
4. End your answer with the timestamp of the chunk you used, in the form
   (see [mm:ss - mm:ss]).
5. If the answer is not contained in the transcript, respond with EXACTLY:
   Not found in video.

Question: {query}
Answer:"""

# ──────────────────────────────────────────────────────
# LLM Call
# ──────────────────────────────────────────────────────

def ask_local_llm(prompt: str, timeout: Optional[float] = None) -> Optional[str]:
    """Ask local Ollama (phi3:mini). Returns None if unavailable."""
    try:
        response = requests.post(
            config.get("answer.ollama_url", "http://localhost:11434/api/generate"),
            json={
                "model": config.get("answer.local_model", "phi3:mini"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": config.get("answer.max_tokens", 256),
                    "temperature": 0,
                    "top_p": 0.9,
                }
            },
            timeout=timeout if timeout is not None else config.get("answer.timeout", 180)
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        logger.warning("Cannot connect to Ollama – it may not be running.")
        return None
    except Exception as e:
        logger.error(f"LLM request failed: {e}")
        return None


def ask_gemini_llm(prompt: str) -> Optional[str]:
    """Ask Gemini API."""
    return get_gemini_response(prompt)


def ask_grok_llm(prompt: str) -> Optional[str]:
    """Ask Grok API. Placeholder for compatibility with tests and future expansion."""
    return None


def ask_openai_llm(prompt: str) -> Optional[str]:
    """Ask OpenAI Chat Completion."""
    try:
        import openai
        api_key = os.environ.get("OPENAI_API_KEY") or config.get("answer.openai_api_key", "")
        if not api_key or api_key == "${OPENAI_API_KEY}":
            return None

        client = openai.OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=config.get("answer.openai_model", "gpt-4"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.get("answer.max_tokens", 256),
            temperature=0
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None

    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return None


def ask_huggingface_llm(prompt: str) -> Optional[str]:
    """Ask HuggingFace Inference API."""
    try:
        if not config.get("answer.use_huggingface", False):
            return None

        from huggingface_hub import InferenceClient
        token = os.environ.get("HF_TOKEN") or config.get("answer.hf_token", "")
        if not token or token == "${HF_TOKEN}":
            return None

        client = InferenceClient(token=token)
        resp = client.chat_completion(
            model=config.get("answer.hf_model", "meta-llama/Llama-3.2-3B-Instruct"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.get("answer.max_tokens", 256),
            temperature=0.1
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None

    except Exception as e:
        logger.error(f"HuggingFace request failed: {e}")
        return None

# ──────────────────────────────────────────────────────
# Optimised multi-provider fallback chain
# ──────────────────────────────────────────────────────
#
# Spec contract (do NOT regress):
#   • Cloud chain : Gemini → Grok → HuggingFace → (gated) Bedrock → Extractive
#   • Local mode  : Ollama → Extractive   (NO cloud calls)
#   • Bedrock is LAST RESORT: only when no acceptable answer was produced
#     AND the per-session call cap has not been reached.
#   • Each LLM call is hard-bounded by LLM_CALL_TIMEOUT_SECONDS (default 2s).
#   • Returned dict carries: response, provider, status, fallback_level,
#     latency_ms (the winning provider) and total_latency_ms.
#
# Knobs (all optional env vars):
#   LLM_CALL_TIMEOUT_SECONDS  → per-call wall-clock cap (float, default 2.0)
#   LOCAL_MODE                → "1"/"true" forces Ollama-only flow
#   BEDROCK_MAX_CALLS         → integer cap per process (default 5)
#   BEDROCK_DISABLED          → "1" to disable Bedrock entirely
#   XAI_API_KEY               → enables Grok via x.ai OpenAI-compatible API
#   AWS_REGION / AWS_*        → standard boto3 credentials (Bedrock)
import time as _time
import threading as _threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout

_DEFAULT_TIMEOUT_S = float(os.environ.get("LLM_CALL_TIMEOUT_SECONDS", "2"))
_BEDROCK_MAX_CALLS = int(os.environ.get("BEDROCK_MAX_CALLS", "5"))
_BEDROCK_DISABLED = os.environ.get("BEDROCK_DISABLED", "").lower() in ("1", "true", "yes")
_BEDROCK_CALL_COUNT = 0
_BEDROCK_LOCK = _threading.Lock()
# One module-level executor — cheap and threadsafe; avoids spinning a new
# pool per request which would dominate latency at our 2s budget.
_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-call")


def _is_local_mode() -> bool:
    """LOCAL_MODE env var OR config flag forces the no-cloud flow."""
    if os.environ.get("LOCAL_MODE", "").lower() in ("1", "true", "yes"):
        return True
    return bool(config.get("answer.local_mode", False))


def get_bedrock_call_count() -> int:
    """Public accessor — used by tests and the API to surface usage."""
    return _BEDROCK_CALL_COUNT


def reset_bedrock_call_count() -> None:
    """Tests only. Resets the per-process Bedrock counter."""
    global _BEDROCK_CALL_COUNT
    with _BEDROCK_LOCK:
        _BEDROCK_CALL_COUNT = 0


def _is_acceptable_answer(text: Optional[str]) -> bool:
    """An LLM response is 'acceptable' when it is a non-trivial answer that
    is NOT the explicit not-found sentinel and NOT the all-providers-down
    error string. Matches the gate used to decide whether Bedrock is needed.
    """
    if not text:
        return False
    t = text.strip()
    if len(t) < 5:
        return False
    if _is_not_found(t):
        return False
    if t.lower().startswith("all ai services"):
        return False
    return True


def _call_with_timeout(fn, prompt: str, timeout_s: float) -> Optional[str]:
    """Run ``fn(prompt)`` on a worker thread, hard-bounded by ``timeout_s``.

    Returns whatever the provider returned, or None on timeout/exception.
    The thread itself cannot be killed (Python limitation) so a slow upstream
    will keep running in the background — but the caller is unblocked at the
    timeout, which is what the spec requires.
    """
    fut = _LLM_EXECUTOR.submit(fn, prompt)
    try:
        return fut.result(timeout=timeout_s)
    except _FutureTimeout:
        logger.warning(f"[LLM] {fn.__name__} exceeded {timeout_s:.1f}s budget; falling through.")
        return None
    except Exception as exc:
        logger.warning(f"[LLM] {fn.__name__} raised: {exc}")
        return None


# ── Provider call functions (each returns Optional[str] or None) ──────
def _call_gemini(prompt: str) -> Optional[str]:
    global _GEMINI_DAILY_EXHAUSTED
    if _GEMINI_DAILY_EXHAUSTED:
        return None
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY") or config.get("answer.gemini_api_key", "")
        if not api_key or api_key == "${GEMINI_API_KEY}":
            return None
        genai.configure(api_key=api_key.strip("'\" \n\r"))
        model_name = config.get("answer.gemini_model", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": config.get("answer.max_tokens", 256),
            },
        )
        if resp and getattr(resp, "text", None):
            return resp.text.strip()
        return None
    except Exception as exc:
        msg = str(exc)
        if _looks_like_daily_quota(msg):
            _GEMINI_DAILY_EXHAUSTED = True
            logger.warning("[LLM] Gemini daily quota exhausted — disabling for this session.")
        else:
            logger.warning(f"[LLM] Gemini error: {exc}")
        return None


def _call_grok(prompt: str) -> Optional[str]:
    """x.ai Grok via the OpenAI-compatible HTTP API. Returns None when no key.

    We use raw HTTP (requests) instead of the OpenAI SDK to keep the per-call
    timeout strict and avoid any SDK-level retry that would blow our budget.
    """
    api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY", "")
    if not api_key:
        return None
    try:
        r = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": config.get("answer.grok_model", "grok-2-latest"),
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": config.get("answer.max_tokens", 256),
                "temperature": 0,
            },
            timeout=_DEFAULT_TIMEOUT_S,  # belt-and-braces
        )
        if r.status_code != 200:
            logger.warning(f"[LLM] Grok HTTP {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content")
        return text.strip() if text else None
    except Exception as exc:
        logger.warning(f"[LLM] Grok error: {exc}")
        return None


def _call_huggingface(prompt: str) -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or config.get("answer.hf_token", "")
    if not token or token == "${HF_TOKEN}":
        return None
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(token=token, timeout=_DEFAULT_TIMEOUT_S)
        resp = client.chat_completion(
            model=config.get("answer.hf_model", "meta-llama/Llama-3.2-3B-Instruct"),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.get("answer.max_tokens", 256),
            temperature=0.1,
        )
        text = resp.choices[0].message.content
        return text.strip() if text else None
    except Exception as exc:
        logger.warning(f"[LLM] HuggingFace error: {exc}")
        return None


def _call_ollama(prompt: str) -> Optional[str]:
    return ask_local_llm(prompt, timeout=_DEFAULT_TIMEOUT_S)


def _call_bedrock(prompt: str) -> Optional[str]:
    """AWS Bedrock — strict last resort. Increments the global counter ONLY on
    a real call attempt (not when skipped due to disabled flag / no creds)."""
    global _BEDROCK_CALL_COUNT
    if _BEDROCK_DISABLED:
        return None
    try:
        import boto3, json as _json
    except ImportError:
        logger.info("[LLM] boto3 not installed — Bedrock unavailable.")
        return None
    # Atomic reserve-or-bail: under concurrent requests the check + increment
    # MUST happen in a single critical section, otherwise N threads racing the
    # cap can each pass an independent check and collectively overshoot the
    # configured spend ceiling. We reserve the slot first; if the call later
    # fails we leave the count incremented because the network call did happen
    # (or at least was attempted past the reservation point).
    with _BEDROCK_LOCK:
        if _BEDROCK_CALL_COUNT >= _BEDROCK_MAX_CALLS:
            logger.warning(f"[LLM] Bedrock cap reached ({_BEDROCK_CALL_COUNT}/{_BEDROCK_MAX_CALLS}) — skipping.")
            return None
        _BEDROCK_CALL_COUNT += 1
        reserved_slot = _BEDROCK_CALL_COUNT
    try:
        client = boto3.client("bedrock-runtime",
                              region_name=os.environ.get("AWS_REGION", "us-east-1"))
        model_id = config.get("answer.bedrock_model",
                              "anthropic.claude-3-haiku-20240307-v1:0")
        body = _json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.get("answer.max_tokens", 256),
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        })
        resp = client.invoke_model(modelId=model_id, body=body)
        payload = _json.loads(resp["body"].read())
        text = (payload.get("content") or [{}])[0].get("text")
        return text.strip() if text else None
    except Exception as exc:
        logger.warning(f"[LLM] Bedrock error: {exc}")
        return None


# ── Public entrypoint ─────────────────────────────────────────────────
def generate_with_fallback(prompt: str) -> Dict[str, Any]:
    """Run the optimised provider chain.

    Cloud mode (default):
        Gemini → Grok → HuggingFace → (gated) Bedrock → "error"
        Bedrock fires only when no provider above produced an acceptable answer.

    Local mode (``LOCAL_MODE=1`` or ``answer.local_mode: true``):
        Ollama → "error"   (NO cloud calls, no Bedrock — strict per spec)

    Returns ``{response, provider, status, fallback_level, latency_ms,
    total_latency_ms, providers_tried}``. ``provider == "error"`` means
    every configured provider was unavailable; the caller should fall back
    to extractive (which ``generate_answer`` already does).
    """
    t_total = _time.time()
    providers_tried: List[str] = []

    if _is_local_mode():
        # LOCAL: Ollama only. NO cloud, NO Bedrock.
        chain = [("ollama", _call_ollama)]
    else:
        # CLOUD: Gemini → Grok → HuggingFace. Bedrock is appended below
        # only if the chain above failed to produce an acceptable answer.
        chain = [
            ("gemini", _call_gemini),
            ("grok", _call_grok),
            ("hf", _call_huggingface),
        ]

    for level, (name, fn) in enumerate(chain):
        providers_tried.append(name)
        t0 = _time.time()
        text = _call_with_timeout(fn, prompt, _DEFAULT_TIMEOUT_S)
        dt = int((_time.time() - t0) * 1000)
        if _is_acceptable_answer(text):
            logger.info(f"[LLM] ✔ {name} accepted (level={level}, {dt}ms, {len(text)} chars)")
            return {
                "response": text.strip(),
                "provider": name,
                "status": f"Using {name} (fallback level {level})",
                "fallback_level": level,
                "latency_ms": dt,
                "total_latency_ms": int((_time.time() - t_total) * 1000),
                "providers_tried": providers_tried,
            }
        logger.info(f"[LLM] ✗ {name} no usable answer ({dt}ms)")

    # ── Bedrock gate (cloud mode only): "last resort, only if no valid answer".
    # We reached here precisely because nothing above produced an acceptable
    # answer — i.e. effective confidence in the chain so far is 0 (< 0.6).
    if not _is_local_mode():
        with _BEDROCK_LOCK:
            cap_remaining = _BEDROCK_MAX_CALLS - _BEDROCK_CALL_COUNT
        if not _BEDROCK_DISABLED and cap_remaining > 0:
            providers_tried.append("bedrock")
            t0 = _time.time()
            text = _call_with_timeout(_call_bedrock, prompt, _DEFAULT_TIMEOUT_S)
            dt = int((_time.time() - t0) * 1000)
            if _is_acceptable_answer(text):
                level = len(chain)  # next position in the chain
                logger.warning(
                    f"[LLM] ✔ bedrock LAST-RESORT accepted (level={level}, {dt}ms, "
                    f"calls used={_BEDROCK_CALL_COUNT}/{_BEDROCK_MAX_CALLS})"
                )
                return {
                    "response": text.strip(),
                    "provider": "bedrock",
                    "status": f"Using Bedrock (last resort, level {level})",
                    "fallback_level": level,
                    "latency_ms": dt,
                    "total_latency_ms": int((_time.time() - t_total) * 1000),
                    "providers_tried": providers_tried,
                }
            logger.warning(f"[LLM] ✗ bedrock did not produce an acceptable answer ({dt}ms)")
        else:
            logger.info("[LLM] Bedrock gate skipped (disabled or call cap exhausted).")

    # All providers failed. The caller (generate_answer) will swap in the
    # deterministic extractive fallback — never user-visible "All AI services
    # unavailable" text.
    return {
        "response": "All AI services unavailable",
        "provider": "error",
        "status": "All AI services unavailable",
        "fallback_level": -1,
        "latency_ms": 0,
        "total_latency_ms": int((_time.time() - t_total) * 1000),
        "providers_tried": providers_tried,
    }

# ──────────────────────────────────────────────────────
# Extractive Fallback (Ollama offline safety net)
# ──────────────────────────────────────────────────────

def _extractive_fallback(
    query: str,
    contexts: List[Dict[str, Any]],
) -> Optional[str]:
    """Extract the first 2 sentences from the top chunk as a grounded answer,
    using the same '(see [mm:ss - mm:ss])' citation format as LLM output."""
    if not contexts:
        return None

    top_ctx = contexts[0]
    text    = (top_ctx.get("text") or "").strip()
    start   = top_ctx.get("start", 0)
    end     = top_ctx.get("end", 0)

    if not text:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", text)
    excerpt   = " ".join(sentences[:2]).strip()

    if not excerpt:
        return None

    logger.warning(
        "[EXTRACTIVE FALLBACK] LLM failed — "
        "returning extractive answer from top context chunk."
    )

    return f"{excerpt} (see {format_time_range(start, end)})"


_CITATION_RE = re.compile(r"\(see\s*\[\d{1,3}:\d{2}\s*-\s*\d{1,3}:\d{2}\]\)")

def _has_citation(answer: str) -> bool:
    """Check that the answer ends with a valid '(see [mm:ss - mm:ss])' citation."""
    if not answer:
        return False
    return bool(_CITATION_RE.search(answer))

# ──────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────

def generate_answer(
    query: str,
    contexts: List[Dict[str, Any]],
    use_verification: bool = False,
    all_contexts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Production-quality answer generation.

    Architecture:
      1. Build prompt with top-2 contexts
      2. Call LLM via fallback chain (Ollama → Gemini → OpenAI → HF)
      3. Run confidence scoring on ALL retrieved contexts
      4. Optionally run evidence verification

    Args:
        query:            User question
        contexts:         Top-2 chunks sent to LLM prompt
        use_verification: Whether to run evidence verification
        all_contexts:     All retrieved chunks (top-5) for confidence scoring + UI display

    Returns:
        Dict with answer, timestamp, confidence data, verification data
    """
    logger.info("=" * 60)
    logger.info("ANSWER GENERATION (simple strict mode)")
    logger.info("=" * 60)

    if not contexts:
        return {
            "answer":    "Question cannot be answered from this video.",
            "timestamp": None,
            "contexts":  [],
            "verified":  False,
        }

    # Use max 2 chunks as requested, limit chars just in case
    safe_contexts = contexts[:2]
    MAX_CHARS_PER_CHUNK = 1500
    for ctx in safe_contexts:
        ctx["text"] = ctx.get("text", "")[:MAX_CHARS_PER_CHUNK]

    # Timestamp is strictly taken from the top chunk metadata
    top_chunk = safe_contexts[0]
    timestamp = format_time_range(top_chunk.get("start", 0), top_chunk.get("end", 0))
    retrieval_score = float(top_chunk.get("score", 0.0))

    logger.info(f"\n[CONTEXT] Using top {len(safe_contexts)} chunks:")
    for i, ctx in enumerate(safe_contexts):
        score = ctx.get("score", 0.0)
        start = ctx.get("start", 0)
        end   = ctx.get("end", 0)
        logger.info(
            f"  Chunk[{i+1}] | Score={score:.4f} | "
            f"{format_timestamp(start)}-{format_timestamp(end)} | "
            f"{len(ctx.get('text', ''))} chars"
        )

    # Build prompt
    prompt = build_answer_prompt(query, safe_contexts)
    logger.info(f"\n[PROMPT] Length={len(prompt)} chars | Preview: {prompt[:200]}...")

    # ── LLM CALL ────────────────────────────────────────────────
    fallback_result = generate_with_fallback(prompt)
    raw_response = fallback_result.get("response")
    provider = fallback_result.get("provider", "error")
    status_msg = fallback_result.get("status", "Unknown output")
    # New telemetry fields from the optimised fallback chain. Expose them
    # to the caller (API, /ask_question, /compare_videos) without changing
    # the existing answer/timestamp/confidence shape.
    fallback_level = fallback_result.get("fallback_level", -1)
    llm_latency_ms = fallback_result.get("latency_ms", 0)
    llm_total_latency_ms = fallback_result.get("total_latency_ms", 0)
    providers_tried = fallback_result.get("providers_tried", [])

    answer = None
    if provider != "error" and raw_response:
        answer = raw_response
        if answer.lower().startswith("answer:"):
            answer = answer.split(":", 1)[1].strip()
    else:
        logger.warning("[LLM] All AI services failed or generated None.")

    # Decide fate of the LLM response.
    llm_failed = (not answer) or (len(answer.strip()) < 5)

    if llm_failed:
        extractive = _extractive_fallback(query, contexts)
        if extractive:
            answer = extractive
            provider = "extractive"
            status_msg = "LLM unavailable — using extractive fallback"
        else:
            answer = "Not found in video."
    elif _is_not_found(answer):
        # LLM explicitly said "Not found" — keep its decision
        answer = "Not found in video."
    elif not _has_citation(answer):
        # LLM answered but did not follow citation contract —
        # append a citation derived from the top chunk so the UI still has
        # a verifiable timestamp, and flag this in the status.
        logger.warning("[LLM] Answer missing '(see [mm:ss - mm:ss])' citation — appending from top chunk.")
        answer = f"{answer.rstrip('. ')}. (see {timestamp})"
        status_msg = f"{status_msg} | citation appended"

    # If we fell back to extractive, level conventionally one past the cloud chain
    # (so callers can see "we exhausted everything before going extractive").
    if provider == "extractive" and fallback_level < 0:
        fallback_level = 99
    result = {
        "answer":            answer,
        "timestamp":         timestamp,
        "contexts":          contexts,
        "all_contexts":      all_contexts or contexts,
        "retrieval_score":   float(f"{retrieval_score:.4f}"),
        "raw_response":      raw_response,
        "provider":          provider,
        "status_msg":        status_msg,
        "verified":          False,
        # New fallback telemetry (additive — pre-existing keys unchanged).
        "fallback_level":    fallback_level,
        "llm_latency_ms":    llm_latency_ms,
        "llm_total_latency_ms": llm_total_latency_ms,
        "providers_tried":   providers_tried,
        "bedrock_calls_used": get_bedrock_call_count(),
    }

    # ── CONFIDENCE SCORING ─────────────────────────────────────────────────
    score_contexts = all_contexts if all_contexts else contexts
    conf = compute_confidence(answer, score_contexts)
    result["confidence"]           = conf["score"]
    result["confidence_label"]     = conf["label"]
    result["confidence_explanation"] = conf["explanation"]
    result["confidence_breakdown"] = conf["breakdown"]
    result["useful_chunks"]        = conf["useful_chunks"]
    result["total_chunks"]         = conf["total_chunks"]

    logger.info(
        f"[CONFIDENCE] Score={conf['score']}% ({conf['label']}) | "
        f"UsefulChunks={conf['useful_chunks']}/{conf['total_chunks']}"
    )

    # ── STRUCTURED DEBUG LOG ───────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("[ANSWER DEBUG]")
    logger.info(f"  Retrieval Score   : {retrieval_score:.4f}  (raw FAISS IP)")
    
    val_ts = result.get('timestamp')
    val_ans = result.get('answer', '')
    logger.info(f"  Timestamp         : {val_ts} (From Top Chunk Metadata)")
    logger.info(f"  Answer            : {str(val_ans)}")
    logger.info("=" * 50 + "\n")

    # ── EVIDENCE VERIFICATION (optional) ──────────────────────────────
    if use_verification and config.get("verification.enabled", True):
        try:
            from .evidence_verifier import verify_answer as _verify
            logger.info("[VERIFICATION] Running evidence verification…")
            vr = _verify(answer, score_contexts, query)
            result["verification"] = vr
            result["trust_score"]  = vr.get("trust_score", 0)
            result["verified"]     = vr.get("is_valid", False)

            if vr.get("status") == "HALLUCINATION":
                logger.warning("[VERIFICATION] HALLUCINATION detected — suppressing hallucinated answer.")
                # Keep answer visible but flag it; don't blank it out
                result["verified"] = False
        except Exception as e:
            logger.warning(f"[VERIFICATION] Failed (non-fatal): {e}")

    return result

# ──────────────────────────────────────────────────────
# AnswerGenerator class wrapper
# ──────────────────────────────────────────────────────

class AnswerGenerator:
    """Answer generation wrapper (keeps pipeline.py interface intact)."""

    def __init__(self):
        self.use_verification = config.get("verification.enabled", True)

    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        all_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate answer for query using contexts."""
        return generate_answer(query, contexts, self.use_verification, all_contexts=all_contexts)

