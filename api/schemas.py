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
    role: str = "user"


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
    job_id: Optional[str] = None
    progress: int = 0
    stage: Optional[str] = None
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    chunk_count: Optional[int] = None


class UploadOut(BaseModel):
    video_id: str
    job_id: str
    filename: str
    file_url: str
    status: str = "queued"
    next: str = ""


class JobStatusOut(BaseModel):
    job_id: str
    video_id: str
    status: str           # queued | processing | ready | failed
    progress: int         # 0–100
    stage: Optional[str] = None
    error: Optional[str] = None


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
    job_id: str
    status: str = "queued"
    next: str = ""


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


class ConfidenceBreakdownOut(BaseModel):
    """Structured confidence breakdown from the deterministic scorer."""
    avg_similarity: float = Field(description="Mean cosine similarity of retrieved chunks (0–1)")
    context_overlap: float = Field(description="Fraction of answer words found in retrieved context (0–1)")
    chunk_agreement: float = Field(description="Agreement between top-2 chunk scores (0–1)")
    useful_chunks: int = Field(description="Chunks with similarity ≥ 0.50")
    total_chunks: int = Field(description="Total chunks retrieved")
    explanation: List[str] = Field(default_factory=list, description="Human-readable bullets")


class CrossVideoLinkOut(BaseModel):
    """Another user video that contains relevant content for the same question."""
    video_id: str
    filename: str
    top_score: float = Field(description="Top FAISS similarity score (0–1)")
    timestamp_span: Optional[str] = None
    relevance_label: str = Field(description='"High" | "Medium" | "Low"')


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
    confidence_breakdown: Optional[ConfidenceBreakdownOut] = Field(
        default=None,
        description="Detailed confidence scoring breakdown (deterministic, no LLM)",
    )
    hallucination_risk: Optional[str] = Field(
        default=None,
        description='"None" | "Low" | "High" — derived from hallucination post-check',
    )
    cross_video_links: List[CrossVideoLinkOut] = Field(
        default_factory=list,
        description="Other user videos that contain relevant content for this question",
    )


# ── Misc ───────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str
    indexed_chunks: int
    db_users: int
    db_videos: int
    db_backend: str = "sqlite"


# ── Admin ───────────────────────────────────────────────────────────────

class AdminUserOut(BaseModel):
    id: str
    email: str
    created_at: float
    email_verified: bool
    role: str
    auth_provider: str
    video_count: int = 0
    ready_count: int = 0


class AdminVideoOut(BaseModel):
    video_id: str
    filename: str
    status: str
    error: Optional[str] = None
    created_at: float
    updated_at: float
    chunk_count: Optional[int] = None
    owner_email: str
    owner_id: str


class AdminStatsOut(BaseModel):
    total_users: int
    admin_count: int
    total_videos: int
    ready_videos: int
    failed_videos: int
    total_chunks: int


class SetRoleIn(BaseModel):
    role: str = Field(pattern=r"^(user|admin)$")
