"""Pydantic request / response models for the public API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


# ── Auth ───────────────────────────────────────────────────────────────

class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


class UserOut(BaseModel):
    id: str
    email: str
    created_at: float


# ── Videos ─────────────────────────────────────────────────────────────

class VideoOut(BaseModel):
    video_id: str
    filename: str
    status: str
    error: Optional[str] = None
    created_at: float
    updated_at: float


class UploadOut(BaseModel):
    video_id: str
    filename: str
    status: str
    next: str = "POST /process_video to index this file"


class ProcessIn(BaseModel):
    video_id: str


class ProcessOut(BaseModel):
    video_id: str
    status: str
    detail: Optional[str] = None


# ── Q&A ────────────────────────────────────────────────────────────────

class AskIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    video_id: Optional[str] = Field(
        default=None,
        description="Optional: scope the answer to a single uploaded video. "
                    "If omitted, the answer is drawn from all of your videos.",
    )
    use_cache: bool = True


class SupportOut(BaseModel):
    status: str
    semantic_similarity: float
    keyword_overlap: float
    support_score: float


class AskOut(BaseModel):
    answer: str
    confidence: int
    confidence_label: Optional[str] = None
    status: str
    timestamp: Optional[str] = None
    chunk_ids: List[str] = []
    provider: Optional[str] = None
    support: Optional[SupportOut] = None
    neighbors: List[Dict[str, Any]] = []
    cached: bool = False
    latency_ms: int = 0


# ── Misc ───────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str
    indexed_chunks: int
    db_users: int
    db_videos: int
