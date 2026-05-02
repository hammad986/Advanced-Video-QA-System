# Video-QA RAG System

## Overview

A research-grade Video Question Answering (Video-QA) system using Retrieval-Augmented Generation (RAG). It processes video/audio files, transcribes them via Whisper, creates a searchable FAISS vector index, and answers user questions grounded strictly in the video's content — with timestamps, confidence scores, and NLI-based verification.

## Tech Stack

- **Language**: Python 3.12
- **API**: FastAPI + uvicorn (REST surface on port 5000)
- **UI**: Static HTML/JS at `api/static/index.html` served at `/ui` — SSE-first progress (EventSource), polling fallback, animated live-dot + shimmer, human-readable stage labels
- **Auth**: JWT (HS256, 7-day TTL) + pbkdf2_sha256 password hashing + OTP password reset + Google OAuth
- **App DB**: PostgreSQL (primary, `DATABASE_URL` env var) with SQLite fallback at `data/saas.db`
- **Background Jobs**: Python `threading.Queue` in-process worker with retry (default); RQ + external `rq worker` process if `REDIS_URL` set (retry via `rq.Retry`, `job_timeout=3600s`, 2 worker replicas in Docker)
- **Real-time Progress**: SSE endpoint `GET /job_stream/{job_id}` (500 ms poll, 5 min timeout, auth via `?token=`)
- **Storage**: Local filesystem at `data/media/` (default); S3 if `USE_S3=true` + AWS creds set
- **Embeddings**: `BAAI/bge-small-en` (via sentence-transformers)
- **Vector Store**: FAISS (CPU)
- **Speech-to-Text**: faster-whisper (Whisper tiny model)
- **LLM Providers**: Ollama (local), Google Gemini, OpenAI, HuggingFace, AWS Bedrock
- **Verification**: NLI-based (facebook/bart-large-mnli)
- **Media Processing**: ffmpeg, yt-dlp (YouTube URL processing)

## Entry Point

```
uvicorn api.main:app --host 0.0.0.0 --port 5000 --workers 1
```

Workflow: `Start application` in `.replit`

## Project Structure

```
api/
  main.py             # FastAPI app + all routes
  schemas.py          # Pydantic request/response models
  auth.py             # pbkdf2_sha256 + JWT + OTP utilities
  db.py               # PostgreSQL/SQLite (users, videos, rate_limit_events + job progress)
  compare.py          # Multi-video compare endpoint
  compare_gating.py   # Comparability gating logic
  compare_ranking.py  # Topic-strength ranking
  jobs.py             # threading.Queue worker (RQ+Redis if REDIS_URL set)
  storage.py          # Local / S3 file storage abstraction
  pipeline_singleton.py  # Shared pipeline instance + index_write_lock
  static/
    index.html        # Static frontend (served at /ui) — sidebar + player + progress polling

workers/
  video_worker.py          # 6-stage job: queued→validating→normalizing→transcribing→indexing→ready
  model_cache.py           # Process-level pipeline singleton + preload() for RQ worker processes
  rq_worker_entrypoint.py  # Separate-process RQ worker entry with model preload

video_qa/           # Core RAG pipeline package
  pipeline.py       # 9-stage RAG pipeline orchestrator
  video_processor.py
  speech_understanding.py
  embeddings.py
  retrieval.py
  reranker.py
  answer_generator.py
  evidence_verifier.py
  query_rewriter.py
  config.py         # Config manager (reads config.yaml)

data/               # Runtime data
  videos/           # Input videos
  audio/            # Extracted audio
  transcripts/      # JSON transcripts
  chunks/           # Semantic chunks
  cache/            # Cached processed video data
  users/            # Per-user uploaded video files
  saas.db           # SQLite database

models/             # Persistent FAISS index
  video_index.faiss
  metadata.pkl

config.yaml         # Main configuration
config_local.yaml   # Local override config
config_loader.py    # Config loader (at project root)
requirements.txt    # Python dependencies
```

## API Endpoints

### Auth
| Method | Path | Description |
|--------|------|-------------|
| POST | /auth/register | Create account (password policy enforced, rate-limited). Generates email verification code logged server-side. |
| POST | /auth/login | Get JWT bearer token. Blocked with HTTP 403 if email not verified. |
| GET  | /auth/me | Current user info including `email_verified` and `auth_provider` (auth required) |
| POST | /auth/verify_email | Verify email address with 6-digit code (24h TTL) |
| POST | /auth/resend_verification | Resend email verification code (rate-limited 5/hr per IP) |
| POST | /auth/change_password | Change password (JWT required). Blocked for OAuth accounts. Invalidates all existing sessions. |
| POST | /auth/request_reset | Request OTP for password reset (rate-limited) |
| POST | /auth/verify_otp | Verify 6-digit OTP — consumes the hash (max 5 attempts, then locked) |
| POST | /auth/reset_password | Reset password after OTP verification. Only requires email + new_password (OTP already consumed by verify_otp). |
| GET  | /auth/google | Redirect to Google OAuth consent screen. Returns 503 if GOOGLE_CLIENT_ID/SECRET not set. |
| GET  | /auth/google/callback | Handles Google OAuth callback. Creates or links user, issues JWT, redirects to `/ui#token=<jwt>`. |

### Videos
| Method | Path | Description |
|--------|------|-------------|
| POST | /upload_video | Upload a video/audio file (auth, max 300 MB). Returns immediately with `job_id` + `file_url`. |
| GET  | /job_status/{job_id} | Poll background job progress: `{status, progress, stage, file_url, error}` (auth) |
| GET  | /job_stream/{job_id} | SSE real-time progress stream — auth via `?token=<jwt>` query param (EventSource-compatible) |
| POST | /process_url | Validate + enqueue a YouTube URL for background download+index (auth, rate-limited 3/hr/user). Returns `job_id` immediately. |
| GET  | /media/{path} | Serve stored media file — requires `?token=<jwt>` query param (auth via query param) |
| POST | /process_video | Transcribe + index an uploaded video (legacy, synchronous, auth) |
| POST | /process_url | Download + index a YouTube URL (auth, rate-limited, max 200 MB / 30 min) |
| GET  | /videos | List current user's videos with `job_id`, `progress`, `stage`, `file_url` (auth) |

### Q&A
| Method | Path | Description |
|--------|------|-------------|
| POST | /ask_question | Ask a question against indexed videos (auth) |
| POST | /compare_videos | Compare a question across multiple videos (auth) |

### System
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health probe (public) |
| GET | /ui | Static frontend |
| GET | /docs | Swagger UI |

## Security Features

### Password Policy
All passwords (register + reset) must contain:
- At least 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one digit
- At least one special character

### Email Verification Flow
1. `POST /auth/register` → generates 6-digit code, SHA-256 hashed, 24h TTL, logged server-side
2. `POST /auth/verify_email` → verifies code, clears hash, sets `email_verified=1`
3. `POST /auth/login` → HTTP 403 if `email_verified=0`
4. `POST /auth/resend_verification` → generates fresh code (rate-limited 5/hr per IP)
- Existing users (pre-migration) default to `email_verified=1` — they are not affected.

### OTP Reset Flow
1. `POST /auth/request_reset` → generates 6-digit OTP, SHA-256 hashed, 10-min expiry
2. `POST /auth/verify_otp` → verifies OTP, **consumes the hash** (sets `otp_hash=NULL`, `otp_verified=1`), max 5 attempts before lockout
3. `POST /auth/reset_password` → checks `otp_verified=1` + expiry only (no OTP re-entry needed); invalidates all existing sessions

### Rate Limiting (sliding window, per-IP or per-user)
- Login: 10 requests / 15 min per IP
- Register: 5 requests / hour per IP
- OTP request: 3 requests / hour per IP + 3 per user
- URL processing: 3 requests / hour per user

### URL Processing Security
- Only `youtube.com` and `youtu.be` domains accepted (HTTP 400 otherwise)
- Max video duration: 30 minutes
- Max file size: 200 MB
- yt-dlp runs in `asyncio.to_thread()` — never blocks the FastAPI event loop

### JWT Token Invalidation
- Password reset invalidates all previously issued tokens via `tokens_invalidated_before` timestamp
- Tokens with `iat < tokens_invalidated_before` are rejected

## Environment Variables (Optional)

Set via Replit Secrets:

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Google Gemini for answer generation + query rewriting |
| `OPENAI_API_KEY` | OpenAI GPT-4 fallback |
| `HF_TOKEN` | HuggingFace inference API fallback |
| `XAI_API_KEY` | Grok (x.ai) fallback |
| `JWT_SECRET` | Override JWT signing key (auto-generated if absent) |
| `VIDEO_QA_MAX_UPLOAD_MB` | Override max upload size (default: 300) |
| `BEDROCK_DISABLED` | Set to "1" to disable AWS Bedrock |
| `LOCAL_MODE` | Set to "1" to force Ollama-only, no cloud LLM calls |

If no API keys are set, the system uses an extractive fallback (no LLM) — the pipeline still works.

## Answer Generation Provider Chain

**Cloud mode** (default): Gemini → Grok → HuggingFace → Bedrock (last resort, capped) → Extractive fallback

**Local mode** (`LOCAL_MODE=1`): Ollama → Extractive fallback

Each provider has a 2-second hard timeout (`LLM_CALL_TIMEOUT_SECONDS` env var to override).

## Configuration

- `config.yaml` — primary config (speech model size, embedding model, LLM providers, etc.)
- `config_local.yaml` — local development overrides
- Speech model set to `tiny` to avoid OOM on CPU-only instances
- Reranking disabled by default (too large for CPU-only)
- Verification (NLI) enabled by default

## Performance Notes

- FAISS index pre-loaded on first request (cold start ~2s)
- LRU query cache (256 entries) keyed on `(normalized_query, video_id)`
- Single uvicorn worker + threading locks for FAISS index safety
- For multi-replica deployments: replace `_index_write_lock` with a distributed advisory lock

## Eval Results (`evaluation/last_report.json`)

| Metric | Value |
|--------|-------|
| retrieval_precision | 1.0 |
| answer_accuracy | 0.67 |
| hallucination_rate | 0.0 |
| refusal_correctness | 1.0 |
| mean_confidence (in-scope) | 92.5 |
| cache_hits_second_pass | 9/9 |
| mean_latency | 4.0 s |
