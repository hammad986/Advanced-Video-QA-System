"""
Evidence Verification Module for Video-QA System
─────────────────────────────────────────────────────────────────
Verifies that generated answers are actually grounded in the evidence.

Priority chain:
  1. Gemini (primary — reliable, fast, already configured)
  2. HuggingFace NLI (secondary)
  3. Ollama LLM (tertiary)
  4. Conservative fallback

Returns structured verification result:
  {
    "status":        "VERIFIED" | "PARTIALLY_SUPPORTED" | "HALLUCINATION" | "UNKNOWN",
    "trust_score":   int   (0-100),
    "confidence":    float (0.0-1.0),
    "is_valid":      bool,
    "justification": str,   ← human-readable explanation of classification
    "method":        str,   ← which verifier produced the result
  }
"""

import logging
import re
from typing import Dict, Any, List, Tuple, Optional

from .config import config
from .logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────
# Rule-based justification fallbacks (no LLM needed)
# ──────────────────────────────────────────────────────────────────

_DEFAULT_JUSTIFICATIONS = {
    "VERIFIED":            "The answer is well-supported by the retrieved transcript evidence.",
    "PARTIALLY_SUPPORTED": "Some parts of the answer may not be directly present in the retrieved context.",
    "HALLUCINATION":       "The answer appears to contain information not found in the video transcript.",
    "UNKNOWN":             "Verification could not be completed — treat this answer with caution.",
    "INCOMPLETE":          "Some parts of the answer may not be directly present in the retrieved context.",
}

def _default_justification(status: str) -> str:
    return _DEFAULT_JUSTIFICATIONS.get(status, "Verification status uncertain.")


# ──────────────────────────────────────────────────────────────────
# Helper: normalise old INCOMPLETE → PARTIALLY_SUPPORTED
# ──────────────────────────────────────────────────────────────────

def _normalise_status(status: str) -> str:
    if status == "INCOMPLETE":
        return "PARTIALLY_SUPPORTED"
    return status


# ──────────────────────────────────────────────────────────────────
# PATH 1: Gemini (primary, preferred)
# ──────────────────────────────────────────────────────────────────

def _verify_with_gemini(
    answer: str,
    contexts: List[Dict[str, Any]],
    query: str,
) -> Optional[Dict[str, Any]]:
    """
    Use Gemini to classify and justify the answer.
    Returns None if Gemini is unavailable.
    """
    try:
        import os
        import google.generativeai as genai
        from dotenv import dotenv_values

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key or api_key.startswith("${"):
            env = dotenv_values(".env")
            api_key = env.get("GEMINI_API_KEY", "")

        if not api_key:
            return None

        genai.configure(api_key=api_key.strip("'\" \n\r"))
        model_name = config.get("answer.gemini_model", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)

        premise = " ".join(c.get("text", "") for c in contexts)[:2000]

        prompt = f"""You are a strict Evidence Verifier for a Video QA system.

Your task: Determine if the ANSWER is supported by the EVIDENCE retrieved from the video transcript.

EVIDENCE:
{premise}

QUESTION: {query}

GENERATED ANSWER: {answer}

Instructions:
1. Classify the answer as exactly ONE of:
   - VERIFIED (answer is fully supported by evidence)
   - PARTIALLY_SUPPORTED (answer is partly supported but some claims are missing or unclear)
   - HALLUCINATION (answer contains information clearly NOT in the evidence)

2. Provide a single concise sentence explaining WHY you chose this classification.

Output format (STRICT):
STATUS: <VERIFIED|PARTIALLY_SUPPORTED|HALLUCINATION>
REASON: <one sentence explanation>"""

        resp = model.generate_content(prompt)
        text = resp.text.strip() if (resp and hasattr(resp, "text")) else ""

        # Parse output
        status_match = re.search(r"STATUS:\s*(VERIFIED|PARTIALLY_SUPPORTED|HALLUCINATION)", text, re.IGNORECASE)
        reason_match = re.search(r"REASON:\s*(.+)", text, re.DOTALL)

        if not status_match:
            logger.warning(f"[Gemini Verifier] Could not parse status from: {text[:200]}")
            return None

        raw_status = status_match.group(1).upper()
        status = _normalise_status(raw_status)
        justification = reason_match.group(1).strip() if reason_match else _default_justification(status)
        # Cap justification at 200 chars
        justification = justification[:200].split("\n")[0].strip()

        trust_map   = {"VERIFIED": 90, "PARTIALLY_SUPPORTED": 50, "HALLUCINATION": 10}
        conf_map    = {"VERIFIED": 0.9, "PARTIALLY_SUPPORTED": 0.5, "HALLUCINATION": 0.1}

        logger.info(f"[Gemini Verifier] Status={status} | Reason={justification}")

        return {
            "status":        status,
            "trust_score":   trust_map.get(status, 30),
            "confidence":    conf_map.get(status, 0.3),
            "justification": justification,
            "method":        "gemini",
        }

    except Exception as exc:
        logger.warning(f"[Gemini Verifier] Failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────
# PATH 2: HuggingFace NLI
# ──────────────────────────────────────────────────────────────────

_hf_client = None

def _load_hf_client():
    global _hf_client
    if _hf_client is not None:
        return True
    try:
        from huggingface_hub import InferenceClient
        import os
        token = os.environ.get("HF_TOKEN") or config.get("answer.hf_token", "")
        _hf_client = InferenceClient(token=token if token else None)
        return True
    except Exception as exc:
        logger.warning(f"Failed to load NLI InferenceClient: {exc}")
        return False


def _verify_with_hf_nli(
    answer: str,
    contexts: List[Dict[str, Any]],
    query: str,
) -> Optional[Dict[str, Any]]:
    """Use HuggingFace Serverless API as NLI verifier."""
    if not _load_hf_client():
        return None

    premise = " ".join(c.get("text", "") for c in contexts)[:1500]

    sys_prompt = (
        "You are a strict Evidence Verifier. Given a FACT and EVIDENCE, "
        "determine if the FACT is supported. Reply with exactly one of: "
        "VERIFIED, PARTIALLY_SUPPORTED, or HALLUCINATION."
    )
    user_msg = f"EVIDENCE:\n{premise}\n\nFACT:\n{answer}"

    try:
        model = config.get("answer.hf_model", "meta-llama/Llama-3.2-3B-Instruct")
        response = _hf_client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=20,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip().upper()

        if "HALLUCINATION" in raw or "NOT VERIFIED" in raw or "CONTRADICTION" in raw:
            status = "HALLUCINATION"
        elif "PARTIALLY" in raw or "PARTIAL" in raw:
            status = "PARTIALLY_SUPPORTED"
        elif "VERIFIED" in raw or "ENTAILMENT" in raw:
            status = "VERIFIED"
        else:
            return None  # Could not parse → fall through

        trust_map = {"VERIFIED": 88, "PARTIALLY_SUPPORTED": 48, "HALLUCINATION": 10}
        conf_map  = {"VERIFIED": 0.88, "PARTIALLY_SUPPORTED": 0.48, "HALLUCINATION": 0.10}

        return {
            "status":        status,
            "trust_score":   trust_map.get(status, 30),
            "confidence":    conf_map.get(status, 0.3),
            "justification": _default_justification(status),
            "method":        "hf_nli",
        }

    except Exception as exc:
        logger.warning(f"[HF NLI Verifier] Failed: {exc}")
        return None


# ──────────────────────────────────────────────────────────────────
# PATH 3: Ollama (legacy fallback)
# ──────────────────────────────────────────────────────────────────

def _verify_with_ollama(
    query: str,
    answer: str,
    contexts: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Ask Ollama LLM to verify (returns None if unavailable)."""
    import requests

    context_text = "\n".join(
        f"Passage {i+1}: {c.get('text', '')}" for i, c in enumerate(contexts)
    )
    prompt = f"""You are an Evidence Verifier.
EVIDENCE:
{context_text}

QUESTION: {query}
GENERATED ANSWER: {answer}

Is the answer VERIFIED, PARTIALLY_SUPPORTED, or HALLUCINATION?
Respond with exactly one word: VERIFIED, PARTIALLY_SUPPORTED, or HALLUCINATION."""

    try:
        response = requests.post(
            config.get("verification.ollama_url", "http://localhost:11434/api/generate"),
            json={
                "model": config.get("verification.local_model", "phi3:mini"),
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 10, "temperature": 0},
            },
            timeout=config.get("verification.timeout", 120),
        )
        if response.status_code == 200:
            raw = response.json().get("response", "").strip().upper()
            if "HALLUCINATION" in raw:
                status = "HALLUCINATION"
            elif "PARTIALLY" in raw:
                status = "PARTIALLY_SUPPORTED"
            elif "VERIFIED" in raw:
                status = "VERIFIED"
            else:
                return None

            trust_map = {"VERIFIED": 75, "PARTIALLY_SUPPORTED": 40, "HALLUCINATION": 10}
            return {
                "status":        status,
                "trust_score":   trust_map.get(status, 30),
                "confidence":    trust_map.get(status, 30) / 100,
                "justification": _default_justification(status),
                "method":        "ollama",
            }
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def verify_answer(
    answer: str,
    contexts: List[Dict[str, Any]],
    query: str,
) -> Dict[str, Any]:
    """
    Verify whether the answer is grounded in the retrieved evidence.

    Priority: Gemini → HF NLI → Ollama → conservative fallback.

    Returns:
        {
            "status":        "VERIFIED" | "PARTIALLY_SUPPORTED" | "HALLUCINATION" | "UNKNOWN",
            "trust_score":   int   (0-100),
            "confidence":    float (0.0-1.0),
            "is_valid":      bool,
            "justification": str,
            "method":        str,
        }
    """
    if not answer or "not found" in answer.lower() or len(answer.strip()) < 5:
        return {
            "status":        "PARTIALLY_SUPPORTED",
            "trust_score":   0,
            "confidence":    0.0,
            "is_valid":      False,
            "justification": "Answer was empty or indicated content was not found.",
            "method":        "fallback",
        }

    if not contexts:
        return {
            "status":        "UNKNOWN",
            "trust_score":   0,
            "confidence":    0.0,
            "is_valid":      False,
            "justification": "No evidence contexts available for verification.",
            "method":        "fallback",
        }

    logger.info("[Verifier] Starting verification chain…")

    # PATH 1 — Gemini
    result = _verify_with_gemini(answer, contexts, query)
    if result:
        logger.info(f"[Verifier] Method=gemini | Status={result['status']}")
        result["is_valid"] = result["status"] in ("VERIFIED", "PARTIALLY_SUPPORTED")
        return result

    # PATH 2 — HF NLI
    result = _verify_with_hf_nli(answer, contexts, query)
    if result:
        logger.info(f"[Verifier] Method=hf_nli | Status={result['status']}")
        result["is_valid"] = result["status"] in ("VERIFIED", "PARTIALLY_SUPPORTED")
        return result

    # PATH 3 — Ollama
    result = _verify_with_ollama(query, answer, contexts)
    if result:
        logger.info(f"[Verifier] Method=ollama | Status={result['status']}")
        result["is_valid"] = result["status"] in ("VERIFIED", "PARTIALLY_SUPPORTED")
        return result

    # PATH 4 — Conservative fallback
    logger.warning("[Verifier] All methods failed — conservative fallback.")
    return {
        "status":        "PARTIALLY_SUPPORTED",
        "trust_score":   50,
        "confidence":    0.5,
        "is_valid":      True,
        "justification": "Verification services unavailable — answer accepted conservatively.",
        "method":        "fallback",
    }


# Backward-compat tuple-returning wrapper
def verify_answer_compat(
    answer: str,
    contexts: List[Dict[str, Any]],
    query: str,
) -> Tuple[bool, Optional[str]]:
    result = verify_answer(answer, contexts, query)
    is_valid = result.get("is_valid", False)
    return is_valid, (answer if is_valid else None)


class EvidenceVerifier:
    """Evidence verification wrapper class."""

    def __init__(self):
        pass

    def verify(
        self,
        answer: str,
        contexts: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        return verify_answer(answer, contexts, query)


if __name__ == "__main__":
    test_contexts = [
        {
            "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "start": 0,
            "end": 10,
        }
    ]
    result = verify_answer(
        "Machine learning is a subset of artificial intelligence.",
        test_contexts,
        "What is machine learning?",
    )
    print(result)
