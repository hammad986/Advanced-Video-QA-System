"""Multi-Video Comparison endpoint.

Separate, additive module — does not touch /ask_question or the existing
pipeline. Answers a single question against multiple of the caller's videos
and returns a comparative answer plus a per-video breakdown with timestamps
and confidence.

Pipeline per request:
  1. Auth + ownership check on every video_id (cross-tenant safe).
  2. For each video_id, run the existing RetrievalSystem with the video_id
     filter (so only that tenant+video's chunks are considered).
  3. Build a single structured prompt that segments each video's evidence,
     ask the LLM for a comparative answer in JSON, and parse it.
  4. If the LLM is unavailable / quota-exhausted / returns malformed JSON,
     fall back to a deterministic extractive comparison so the endpoint never
     500s on transient LLM failures.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from . import auth, db

logger = logging.getLogger("video_qa.api.compare")

router = APIRouter(tags=["compare"])
_bearer = HTTPBearer(auto_error=False)


# ── Auth dep (local copy so this module stays standalone) ──────────────
def _current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Dict[str, Any]:
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    user = auth.user_from_token(creds.credentials)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return user


# ── Schemas ────────────────────────────────────────────────────────────
class CompareIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    video_ids: List[str] = Field(min_length=2, max_length=8,
                                 description="2–8 of the caller's processed videos")
    top_k_per_video: int = Field(default=3, ge=1, le=8)


class TimestampSpan(BaseModel):
    start: float
    end: float
    label: str  # "[mm:ss - mm:ss]"


class PerVideoOut(BaseModel):
    video_id: str
    filename: str
    explanation: str
    timestamps: List[TimestampSpan]
    confidence: int  # 0–100
    chunk_ids: List[str]
    top_score: float


class CompareOut(BaseModel):
    question: str
    answer: str
    per_video: List[PerVideoOut]
    provider: str
    latency_ms: int


# ── Helpers ────────────────────────────────────────────────────────────
def _fmt_ts(start: float, end: float) -> str:
    def mmss(s: float) -> str:
        s = max(0, int(s))
        return f"{s // 60:02d}:{s % 60:02d}"
    return f"[{mmss(start)} - {mmss(end)}]"


def _confidence_from_scores(scores: List[float]) -> int:
    """Mean of top scores → 0–100 int. FAISS IP scores on normalized BGE
    embeddings sit in roughly [0.5, 0.95] for relevant content."""
    if not scores:
        return 0
    mean = sum(scores) / len(scores)
    # Map [0.4 → 0, 0.95 → 100], clamp.
    pct = (mean - 0.4) / (0.95 - 0.4) * 100.0
    return max(0, min(100, int(round(pct))))


def _build_prompt(question: str, per_video_chunks: List[Dict[str, Any]]) -> str:
    parts = [
        "You are comparing how multiple source videos answer the same question.",
        "Use ONLY the evidence provided. Do not invent facts.",
        f"Question: {question}",
        "",
        "Evidence:",
    ]
    for entry in per_video_chunks:
        parts.append(f"--- VIDEO id={entry['video_id']} (\"{entry['filename']}\") ---")
        if not entry["chunks"]:
            parts.append("(no relevant chunks found)")
            continue
        for i, c in enumerate(entry["chunks"], 1):
            ts = _fmt_ts(c.get("start", 0), c.get("end", 0))
            text = (c.get("text") or "").strip().replace("\n", " ")
            if len(text) > 600:
                text = text[:600] + "…"
            parts.append(f"  [{i}] {ts} {text}")
    parts += [
        "",
        "Respond with STRICT JSON of the form:",
        "{",
        '  "answer": "<2-5 sentence comparative answer across all videos>",',
        '  "per_video": [',
        '    {"video_id": "<id>", "explanation": "<1-3 sentences citing only this video>"},',
        "    ...",
        "  ]",
        "}",
        "Do not output anything before or after the JSON.",
    ]
    return "\n".join(parts)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = _JSON_BLOCK.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extractive_explanation(chunks: List[Dict[str, Any]]) -> str:
    """Deterministic fallback: stitch the top 1–2 chunk texts."""
    if not chunks:
        return "No relevant content found in this video."
    bits = []
    for c in chunks[:2]:
        t = (c.get("text") or "").strip().replace("\n", " ")
        if t:
            bits.append(t if len(t) <= 280 else t[:280] + "…")
    return " ".join(bits) if bits else "No relevant content found in this video."


# ── Route ──────────────────────────────────────────────────────────────
@router.post("/compare_videos", response_model=CompareOut)
def compare_videos(
    payload: CompareIn,
    user: Dict[str, Any] = Depends(_current_user),
) -> CompareOut:
    t0 = time.time()

    # 1. Validate ownership + status (404 to hide existence of other tenants').
    requested = list(dict.fromkeys(payload.video_ids))  # de-dup, preserve order
    if len(requested) < 2:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            "Provide at least 2 distinct video_ids to compare")
    videos: List[Dict[str, Any]] = []
    for vid in requested:
        row = db.get_video(vid)
        if not row or row["user_id"] != user["id"]:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"Video '{vid}' not found")
        if row["status"] != "ready":
            raise HTTPException(
                status.HTTP_409_CONFLICT,
                f"Video '{vid}' is '{row['status']}', not ready. Process it first.",
            )
        videos.append(row)

    # 2. Per-video retrieval. Lazy-import to keep module standalone and avoid
    #    paying retrieval cost on imports.
    from video_qa.retrieval import RetrievalSystem
    retriever = RetrievalSystem()
    if not retriever.is_available():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Retrieval index is not available yet")

    per_video_chunks: List[Dict[str, Any]] = []
    for v in videos:
        chunks = retriever.retrieve(payload.question,
                                    top_k=payload.top_k_per_video,
                                    video_id=v["video_id"]) or []
        per_video_chunks.append({
            "video_id": v["video_id"],
            "filename": v["filename"],
            "chunks": chunks,
        })

    # 3. LLM call (single comparative prompt). Falls back gracefully.
    from video_qa.answer_generator import generate_with_fallback
    prompt = _build_prompt(payload.question, per_video_chunks)
    llm_result = generate_with_fallback(prompt) or {}
    # generate_with_fallback returns {"response": ..., "provider": ...}.
    # Accept "answer" too for forward-compat with any future callers.
    raw = (llm_result.get("response") or llm_result.get("answer") or "").strip()
    provider = llm_result.get("provider") or "extractive"
    parsed = _parse_llm_json(raw)

    overall_answer = ""
    explanations: Dict[str, str] = {}
    if parsed and isinstance(parsed.get("per_video"), list):
        overall_answer = (parsed.get("answer") or "").strip()
        for item in parsed["per_video"]:
            if isinstance(item, dict) and "video_id" in item:
                explanations[str(item["video_id"])] = (item.get("explanation") or "").strip()
    else:
        # Fallback: keep whatever the LLM said (or build it ourselves).
        provider = "extractive" if not raw else f"{provider}+unstructured"
        overall_answer = raw or (
            "Comparative answer unavailable from LLM; see per-video explanations."
        )

    # 4. Assemble per-video output with deterministic confidence + timestamps.
    per_video_out: List[PerVideoOut] = []
    for entry in per_video_chunks:
        chunks = entry["chunks"]
        scores = [float(c.get("score", 0.0)) for c in chunks]
        spans = [
            TimestampSpan(
                start=float(c.get("start", 0)),
                end=float(c.get("end", 0)),
                label=_fmt_ts(c.get("start", 0), c.get("end", 0)),
            )
            for c in chunks
        ]
        chunk_ids = [
            f"{entry['video_id']}:{c.get('chunk_id', i)}"
            for i, c in enumerate(chunks)
        ]
        explanation = explanations.get(entry["video_id"]) or _extractive_explanation(chunks)
        per_video_out.append(PerVideoOut(
            video_id=entry["video_id"],
            filename=entry["filename"],
            explanation=explanation or _extractive_explanation(chunks),
            timestamps=spans,
            confidence=_confidence_from_scores(scores),
            chunk_ids=chunk_ids,
            top_score=max(scores) if scores else 0.0,
        ))

    if not overall_answer:
        # Final guard: synthesise a one-liner from per-video confidences.
        ranked = sorted(per_video_out, key=lambda x: x.confidence, reverse=True)
        overall_answer = (
            f"Across the {len(per_video_out)} videos, '{ranked[0].filename}' has the "
            f"strongest evidence (confidence {ranked[0].confidence}). See per-video "
            f"explanations for details."
        )

    return CompareOut(
        question=payload.question,
        answer=overall_answer,
        per_video=per_video_out,
        provider=provider,
        latency_ms=int((time.time() - t0) * 1000),
    )
