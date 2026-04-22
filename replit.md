# Video-QA RAG System

## Overview

A research-grade Video Question Answering (Video-QA) system using Retrieval-Augmented Generation (RAG). It processes video/audio files, transcribes them via Whisper, creates a searchable FAISS vector index, and answers user questions grounded strictly in the video's content — with timestamps, confidence scores, and NLI-based verification.

## Tech Stack

- **Language**: Python 3.12
- **API**: FastAPI + uvicorn (REST surface on port 5000)
- **UI**: Streamlit (legacy interface, runs separately if needed)
- **Auth**: JWT (HS256, 7-day TTL) + pbkdf2_sha256 password hashing
- **App DB**: SQLite at `data/saas.db` (users, videos)
- **Embeddings**: `BAAI/bge-small-en` (via sentence-transformers)
- **Vector Store**: FAISS (CPU)
- **Speech-to-Text**: WhisperX / faster-whisper
- **LLM Providers**: Ollama (local), Google Gemini, OpenAI, HuggingFace
- **Verification**: NLI-based (facebook/bart-large-mnli)
- **Media Processing**: ffmpeg, yt-dlp

## Project Structure

```
api/                # SaaS REST surface (uvicorn entrypoint: api.main:app)
  main.py           # FastAPI app + routes
  schemas.py        # Pydantic request/response models
  auth.py           # pbkdf2_sha256 + JWT (HS256) bearer auth
  db.py             # SQLite (users, videos)

video_qa/           # Core package
  app.py            # Streamlit UI (legacy)
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

models/             # Persistent FAISS index
  video_index.faiss
  metadata.pkl

config.yaml         # Main configuration
config_loader.py    # Config loader (at project root)
app.py              # Root launcher (delegates to video_qa/app.py)
```

## Running the App

The app runs via the "Start application" workflow:
```
streamlit run video_qa/app.py --server.port 5000 --server.address 0.0.0.0
```

## Configuration

- `config.yaml` — primary config (speech model size, embedding model, LLM providers, etc.)
- `config_local.yaml` — local override config
- `config_cloud.yaml` — Streamlit Cloud config

## Environment Variables

- `GEMINI_API_KEY` — Google Gemini API key (for query rewriting & answer generation)
- `OPENAI_API_KEY` — OpenAI API key (optional)
- `HF_TOKEN` — HuggingFace token (optional)

## Key Notes

- `video_qa/app.py` manually adds the project root to `sys.path` at startup so `config_loader` can be found
- Streamlit config in `.streamlit/config.toml` sets port 5000, address 0.0.0.0, CORS/XSRF disabled (required for Replit proxy)
- The speech model is set to `tiny` in config.yaml to avoid OOM errors on CPU-only machines

## Answer Generation

- **Provider priority**: Gemini → Ollama (local) → OpenAI → HuggingFace → extractive fallback
- **Gemini model**: `gemini-2.5-flash` with `temperature=0.1`, retries up to 3× on 429 rate limits with exponential backoff
- **Prompt**: Passes the top-2 retrieved chunks with their `[mm:ss - mm:ss]` timestamps and strict rules (answer only from context, cite timestamp, say "Not found in video." if unknown)
- **Retrieval scoping**: Strict — if `active_video_id` is provided and matches 0 chunks, an empty result is returned (no silent global fallback). When no `active_video_id` is provided, global retrieval is used.
- **Extractive fallback**: Invoked from `generate_answer()` when the LLM returns empty/too-short output — returns top-chunk excerpt with its timestamp.
- **Gemini daily-quota kill-switch**: Once a Gemini 429 with the "generate_content_free_tier_requests" daily quota marker is observed, a module-level flag disables Gemini for the rest of the process so subsequent queries do not pay retry/backoff time on unrecoverable errors.

## SaaS-Grade Extensions (this branch)

- `video_qa/cache.py` — thread-safe LRU query cache (256 entries) keyed on `(normalized_query, video_id)`. Verified 9/9 cache hits on second pass.
- `video_qa/temporal_neighbors.py` — pulls prev/next chunks of the top-1 retrieved chunk from FAISS metadata grouped by `video_id`; neighbour text is appended to the LLM prompt for context continuity.
- `video_qa/hallucination_detector.py` — deterministic post-check. Computes cosine similarity between answer embedding and concatenated-context embedding plus keyword overlap; classifies SUPPORTED / PARTIAL / UNSUPPORTED. PARTIAL caps confidence ≤ 70, UNSUPPORTED caps ≤ 35.
- **Out-of-scope (OOS) gate** (in `pipeline.ask`): after retrieval, refuses immediately with `status="UNSUPPORTED"`, `confidence=10`, `provider="oos_gate"` when the **union of original + rewritten query content words** shares **zero** tokens with the top-1 chunk AND the top retrieval score is below `0.85`. Needed because BGE cosine scores stay in ~0.76–0.81 even for totally unrelated queries, so a score threshold alone cannot separate scope.
- **Production response contract** from `pipeline.ask`:
  `{answer, confidence, confidence_label, status, timestamp, chunk_ids, provider, contexts, support, neighbors, cached}`.
- `evaluation/run_eval.py` + `evaluation/eval_dataset.json` (18 queries) / `evaluation/eval_dataset_small.json` (9 queries, fits Gemini free-tier daily quota). Metrics: retrieval_precision, answer_accuracy, hallucination_rate, refusal_correctness, mean_confidence, cache_hits, mean_latency.

### Last eval results (`evaluation/last_report.json`)

| metric | value |
|---|---|
| retrieval_precision | 1.0 |
| answer_accuracy | 0.67 |
| hallucination_rate | 0.0 |
| refusal_correctness | 1.0 |
| mean_confidence (in-scope) | 92.5 |
| cache_hits_second_pass | 9/9 |
| mean_latency | 4.0 s |

## Topic-Strength + Recommendation Layer (`api/compare_ranking.py`)

Final additive enhancement to `/compare_videos`. Pure deterministic ranking
(NO LLM calls, no network) computed from the retrieval signals already
produced by the gating step.

- **Per-video scoring:** `score = 0.5·similarity + 0.3·coverage + 0.2·clarity`
  - `similarity`: mean of top-k FAISS scores (clamped to 0..1)
  - `coverage`: returned chunks / requested top_k_per_video
  - `clarity`: text-stat heuristic (sentence length + word length anchors).
    A floor of `n_words / 30` on the sentence count keeps ASR transcripts
    without punctuation from collapsing to a single huge "sentence".
- **best_video:** argmax of score, **only** when status ∈ {COMPARABLE, PARTIAL}.
- **recommendation:** beginner = max clarity; revision = max coverage·similarity.
- **differences:** pairwise factual deltas (sentence length, vocabulary
  density, score) — emitted only when above MIN_*_DELTA thresholds.
- **Strict withholding:** for NOT_COMPARABLE / INSUFFICIENT the route returns
  `best_video=null`, `topic_strength={}`, `recommendation` with picks=null
  plus a human-readable `explanation`, and `differences=[]`.
- **No regressions:** `/ask_question` is untouched; existing CompareOut
  fields are preserved (additive only).
- **Validation:** route-level test forces all four gate decisions and
  asserts the contract; ranker unit tests cover argmax / refusal branches.
- **UI:** new per-card topic-strength pill, Recommendation card (beginner/
  revision picks + explanation), and Key-differences list rendered below
  the per-video grid. Empty/withheld states render the explanation only.

## Optimised Multi-Provider Fallback (Apr 2026)
Cost/latency-controlled answer chain in `video_qa/answer_generator.py` and
rewrite chain in `video_qa/query_rewriter.py`.
- **Query rewrite:** Gemini ONLY → deterministic `_rule_based_rewrite`
  (filler-strip, contraction expansion, whitespace collapse, no LLM, no
  network). Bedrock and other LLMs are intentionally excluded.
- **Answer cloud chain:** Gemini → Grok (x.ai, OpenAI-compat HTTP) →
  HuggingFace → **gated Bedrock** (Claude Haiku via boto3) → extractive.
  Bedrock fires only when no provider above produced an acceptable answer
  AND the per-process cap (`BEDROCK_MAX_CALLS`, default 5) is not reached.
- **Local mode** (`LOCAL_MODE=1` or `answer.local_mode: true`): Ollama only,
  no cloud calls, no Bedrock.
- **Per-call budget:** 2 s wall-clock cap via a module-level
  `ThreadPoolExecutor` (`LLM_CALL_TIMEOUT_SECONDS` overrides).
- **Acceptability gate:** non-empty, length ≥ 5, not the not-found sentinel
  and not the all-providers-down error string.
- **Telemetry:** `generate_with_fallback` returns `provider`,
  `fallback_level`, `latency_ms`, `total_latency_ms`, `providers_tried`;
  `generate_answer` adds `bedrock_calls_used`; surfaced through `AskOut`
  (additive, optional) so the UI / clients can audit cost & path.
