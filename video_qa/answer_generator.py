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
    Build a simple prompt with combined context.
    """
    context_text = "\n\n".join(
        [f"--- Chunk {i+1} ---\n{ctx.get('text', '')}" for i, ctx in enumerate(contexts)]
    )

    return f"""You are a Video QA assistant.

Given the following context from a video:
{context_text}

Answer ONLY from the given context. If not found, say exactly: Not found in video.
Keep your answer concise (1-2 sentences). Do not hallucinate or use outside knowledge.

Question: {query}
Answer:"""

# ──────────────────────────────────────────────────────
# LLM Call
# ──────────────────────────────────────────────────────

def ask_local_llm(prompt: str, timeout: Optional[int] = None) -> Optional[str]:
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
            timeout=timeout or config.get("answer.timeout", 180)
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

def generate_with_fallback(prompt: str) -> Dict[str, str]:
    """
    Generate an answer trying multiple LLM providers.
    Sequence: local -> Gemini -> Ollama -> OpenAI -> HuggingFace -> Error
    """
    import os

    # If configured for cloud (no local), directly use Gemini first
    if not config.get("answer.use_local", True):
        gemini_answer = get_gemini_response(prompt)
        if gemini_answer:
            return {
                "response": gemini_answer,
                "provider": "gemini",
                "status": "Using Gemini (cloud configuration)"
            }
        logger.warning("Gemini fallback in cloud mode returned no response. Continuing with fallback chain.")

    # 1. Try Ollama (local mode)
    try:
        if config.get("answer.use_local", True):
            response = ask_local_llm(prompt)
            if response:
                return {
                    "response": response,
                    "provider": "ollama",
                    "status": "Using local AI (Ollama)"
                }
            logger.warning("⚠️ Ollama not available (low memory or not running)")
    except Exception as e:
        logger.error(f"Ollama fallback skipped: {e}")

    # 2. Try Gemini
    try:
        import google.generativeai as genai
        from dotenv import dotenv_values
        
        # Try finding key in normal os.environ or fallback to explicit direct disk read
        api_key = os.environ.get("GEMINI_API_KEY") or config.get("answer.gemini_api_key", "")
        if not api_key or api_key == "${GEMINI_API_KEY}":
            env_dict = dotenv_values(".env")
            api_key = env_dict.get("GEMINI_API_KEY", "")
            
        if api_key and api_key != "${GEMINI_API_KEY}":
            genai.configure(api_key=api_key.strip("'\" \n\r"))
            model = genai.GenerativeModel(config.get("answer.gemini_model", "gemini-2.5-flash"))
            resp = model.generate_content(prompt)
            if resp and hasattr(resp, 'text') and resp.text:
                return {
                    "response": resp.text.strip(),
                    "provider": "gemini",
                    "status": "Using Gemini fallback"
                }
    except Exception as e:
        logger.error(f"Gemini fallback failed: {e}")

    # 3. Try OpenAI
    try:
        if config.get("answer.use_openai", False):
            import openai
            api_key = os.environ.get("OPENAI_API_KEY") or config.get("answer.openai_api_key", "")
            if api_key and api_key != "${OPENAI_API_KEY}":
                client = openai.OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=config.get("answer.openai_model", "gpt-4"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.get("answer.max_tokens", 256),
                    temperature=0
                )
                text = resp.choices[0].message.content
                if text:
                    return {
                        "response": text.strip(),
                        "provider": "openai",
                        "status": "Using OpenAI fallback"
                    }
    except Exception as e:
        logger.error(f"OpenAI fallback failed: {e}")

    # 4. Try HuggingFace
    try:
        if config.get("answer.use_huggingface", False):
            from huggingface_hub import InferenceClient
            token = os.environ.get("HF_TOKEN") or config.get("answer.hf_token", "")
            if token and token != "${HF_TOKEN}":
                client = InferenceClient(token=token)
                resp = client.chat_completion(
                    model=config.get("answer.hf_model", "meta-llama/Llama-3.2-3B-Instruct"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.get("answer.max_tokens", 256),
                    temperature=0.1
                )
                text = resp.choices[0].message.content
                if text:
                    return {
                        "response": text.strip(),
                        "provider": "hf",
                        "status": "Using HuggingFace fallback"
                    }
    except Exception as e:
        logger.error(f"HuggingFace fallback failed: {e}")

    # 5. Final Failure
    return {
         "response": "All AI services unavailable",
         "provider": "error",
         "status": "All AI services unavailable"
    }

# ──────────────────────────────────────────────────────
# Extractive Fallback (Ollama offline safety net)
# ──────────────────────────────────────────────────────

def _extractive_fallback(
    query: str,
    contexts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Extract the first 2 sentences from the top chunk as a grounded answer."""
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

    response_text = (
        f"Answer: {excerpt}\n"
        f"Timestamp: {format_time_range(start, end)}"
    )
    return response_text

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

    answer = None
    if provider != "error" and raw_response:
        answer = raw_response
        if answer.lower().startswith("answer:"):
            answer = answer.split(":", 1)[1].strip()
    else:
        logger.warning("[LLM] All AI services failed or generated None.")

    if not answer or len(answer.strip()) < 5 or _is_not_found(answer):
        if contexts:
            answer = contexts[0].get("text", "")[:200]
        else:
            answer = "NOT FOUND IN VIDEO"

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

