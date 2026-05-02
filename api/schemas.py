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
    email_verified: bool
    auth_provider: str = "local"


# ── Email Verification ─────────────────────────────────────────────────

class VerifyEmailIn(BaseModel):
    email: EmailStr
    code: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class VerifyEmailOut(BaseModel):
    message: str
    verified: bool


class ResendVerificationIn(BaseModel):
    email: EmailStr


class ResendVerificationOut(BaseModel):
    message: str


# ── OTP / Password Reset ───────────────────────────────────────────────

class RequestResetIn(BaseModel):
    email: EmailStr


class RequestResetOut(BaseModel):
    message: str


class VerifyOTPIn(BaseModel):
    email: EmailStr
    otp: str = Field(min_length=6, max_length=6, pattern=r"^\d{6}$")


class VerifyOTPOut(BaseModel):
    message: str
    verified: bool


class ResetPasswordIn(BaseModel):
    email: EmailStr
    new_password: str = Field(min_length=8, max_length=128)


class ResetPasswordOut(BaseModel):
    message: str


# ── Change Password ────────────────────────────────────────────────────

class ChangePasswordIn(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8, max_length=128)


class ChangePasswordOut(BaseModel):
    message: str


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


# ── URL Processing ─────────────────────────────────────────────────────

class ProcessURLIn(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class ProcessURLOut(BaseModel):
    video_id: str
    status: str
    detail: Optional[str] = None


# ── Q&A ────────────────────────────────────────────────────────────────

class AskIn(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    video_id: Optional[str] = Field(
        default=None,
        description="Scope answer to one video. Omit to search all your videos.",
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
    fallback_level: Optional[int] = None
    llm_latency_ms: Optional[int] = None
    providers_tried: List[str] = []
    bedrock_calls_used: Optional[int] = None


# ── Misc ───────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str
    indexed_chunks: int
    db_users: int
    db_videos: int
    db_backend: str = "sqlite"
