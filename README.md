# Video-QA — Research-Grade RAG System

A production-ready Video Question-Answering system built on FastAPI. Upload videos or paste YouTube URLs, ask natural-language questions grounded in the actual spoken content. Every answer includes a timestamp, confidence score, hallucination check, and — when you have multiple videos — cross-video links and side-by-side comparisons.

---

## Architecture

```
Browser UI (index.html)
        │
        ▼
FastAPI (api/main.py)  ──  compare router (api/compare.py)
        │
        ├── Auth layer (JWT + Google OAuth + email verification)
        ├── Video ingest  ─► Worker thread (workers/video_worker.py)
        │                        │
        │                ┌───────┴──────────────────┐
        │                ▼                          ▼
        │         faster-whisper               yt-dlp (YouTube)
        │         transcription
        │                │
        │         knowledge_structuring  (semantic chunking)
        │                │
        │         BGE embeddings  ──►  FAISS index
        │
        └── Query pipeline (video_qa/pipeline.py)
                 │
         ┌───────┴──────────────────────────────────┐
         ▼                                           ▼
   Query rewriter                           Summary router
         │                            (instant for overview Qs)
   FAISS retrieval (top-5)
         │
   Cross-encoder reranker
         │
   Temporal neighbour expansion
         │
   LLM answer generation  (Gemini → extractive fallback)
         │
   Confidence scorer  (deterministic, no LLM)
         │
   Hallucination post-check  (cosine sim + keyword overlap)
         │
   Cross-video linker  (lightweight FAISS pass on other videos)
         │
   Structured JSON response
```

**Storage** — SQLite (`data/video_qa.db`) for users, videos, jobs, rate-limit events. Video files and FAISS indexes live under `data/`.

**Auth** — JWT bearer tokens (HS256, 24 h expiry). Google OAuth optional. Email verification required before login. Password reset via 6-digit OTP.

---

## Setup

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| pip | latest |
| ffmpeg | system-installed |

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Environment variables

| Variable | Required | Description |
|---|---|---|
| `JWT_SECRET` | Yes | Random secret for signing JWTs (min 32 chars) |
| `DATABASE_URL` | No | PostgreSQL URL; omit to use SQLite |
| `SMTP_EMAIL` | No | Gmail address for sending verification/OTP emails |
| `SMTP_PASSWORD` | No | Gmail App Password (not your login password) |
| `GOOGLE_CLIENT_ID` | No | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | No | Google OAuth client secret |
| `GOOGLE_REDIRECT_URI` | No | OAuth callback URL (auto-set on Replit) |
| `VIDEO_QA_MAX_UPLOAD_MB` | No | Max upload size in MB (default: 300) |

### 3 — Run

```bash
uvicorn api.main:app --host 0.0.0.0 --port 5000
```

The UI is served at `/` and the interactive API docs at `/docs`.

---

## Key Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/auth/register` | Create account |
| POST | `/auth/login` | Obtain JWT bearer token |
| POST | `/upload_video` | Upload a video file (multipart, max 300 MB) |
| POST | `/process_url` | Queue a YouTube URL for download + indexing |
| GET | `/job_status/{job_id}` | Poll processing progress (0–100) |
| GET | `/videos` | List your processed videos |
| POST | `/ask_question` | Ask a question; returns answer + confidence + hallucination check + cross-video links |
| POST | `/compare_videos` | Compare 2–8 videos on the same question |
| GET | `/health` | Public health probe |
| GET | `/docs` | Swagger UI |
| GET | `/admin` | Admin panel (admin role required) |

---

## `/ask_question` Response Contract

```json
{
  "answer": "...",
  "confidence": 82,
  "confidence_label": "High",
  "status": "SUPPORTED",
  "timestamp": "[02:14 - 02:31]",
  "hallucination_risk": "None",
  "confidence_breakdown": {
    "avg_similarity": 0.821,
    "context_overlap": 0.714,
    "chunk_agreement": 0.953,
    "useful_chunks": 4,
    "total_chunks": 5,
    "explanation": ["High similarity across retrieved chunks", "Strong keyword overlap"]
  },
  "cross_video_links": [
    {
      "video_id": "abc123",
      "filename": "lecture2.mp4",
      "top_score": 0.774,
      "timestamp_span": "[05:10 - 05:44]",
      "relevance_label": "High"
    }
  ],
  "support": {
    "status": "SUPPORTED",
    "semantic_similarity": 0.81,
    "keyword_overlap": 0.62,
    "support_score": 0.734
  },
  "latency_ms": 1240,
  "cached": false
}
```

---

## Project Structure

```
api/
  main.py                  FastAPI app, all HTTP routes
  compare.py               Multi-video comparison router
  compare_gating.py        Comparability decision logic
  compare_ranking.py       Topic-strength ranking layer
  schemas.py               Pydantic request/response models
  auth.py                  JWT + password hashing + Google OAuth
  db.py                    SQLite/PostgreSQL data access
  email.py                 SMTP email delivery (fire-and-forget)
  jobs.py                  Background job queue (threading)
  storage.py               Local file storage abstraction
  static/
    index.html             Main user UI
    admin.html             Admin panel UI

video_qa/
  pipeline.py              9-stage RAG pipeline
  retrieval.py             FAISS vector search
  embeddings.py            BGE embedding model
  confidence_scorer.py     Deterministic confidence scoring
  hallucination_detector.py  Post-check: cosine sim + keyword overlap
  answer_generator.py      LLM answer generation + fallback chain
  reranker.py              Cross-encoder reranker
  query_rewriter.py        Query optimisation
  query_router.py          Summary routing + lecture summaries
  temporal_neighbors.py    Prev/next chunk expansion

workers/
  video_worker.py          Background transcription + indexing

data/
  video_qa.db              SQLite database (auto-created)
  media/                   Uploaded video files
  cache/                   FAISS indexes per video
  summaries/               Pre-computed lecture summaries
```
