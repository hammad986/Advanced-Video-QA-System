import logging
import time
import requests
from dotenv import load_dotenv

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
    import google.generativeai as genai
    import os
    from dotenv import dotenv_values
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        env_dict = dotenv_values(".env")
        api_key = env_dict.get("GEMINI_API_KEY")
        
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
        
    genai.configure(api_key=api_key.strip("'\" "))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    if response and hasattr(response, 'text') and response.text:
        return response.text.strip()
    return ""

def rewrite_query(question: str, force: bool = False) -> Tuple[str, str]:
    if not isinstance(question, str) or not question.strip():
        return question, "fallback"

    original_query = question.strip()

    prompt = f"Rewrite this question clearly. Return ONLY the rewritten question text without any extra conversational filler or options:\n{original_query}"
    logger.info(f"[QUERY REWRITE] Initiating rewrite cascade for query {original_query!r}")

    # 1. TRY GEMINI FIRST
    try:
        rewritten = call_gemini(prompt)
        print("[DEBUG] Gemini rewrite:", rewritten)

        if rewritten and len(rewritten.strip()) > 5:
            return rewritten.strip(), "gemini"

    except Exception as e:
        print("[ERROR] Gemini failed:", e)
        logger.warning(f"[QUERY REWRITE] Gemini failed: {e}")
        return original_query, f"ERROR: {type(e).__name__} - {str(e)}"

    # FINAL FALLBACK
    logger.warning("[QUERY REWRITE] All API providers failed or returned empty. Returning original query.")
    return original_query, "fallback"
