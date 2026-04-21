# Video-QA RAG System

## Overview

A research-grade Video Question Answering (Video-QA) system using Retrieval-Augmented Generation (RAG). It processes video/audio files, transcribes them via Whisper, creates a searchable FAISS vector index, and answers user questions grounded strictly in the video's content — with timestamps, confidence scores, and NLI-based verification.

## Tech Stack

- **Language**: Python 3.12
- **UI**: Streamlit (web interface on port 5000)
- **Embeddings**: `BAAI/bge-small-en` (via sentence-transformers)
- **Vector Store**: FAISS (CPU)
- **Speech-to-Text**: WhisperX / faster-whisper
- **LLM Providers**: Ollama (local), Google Gemini, OpenAI, HuggingFace
- **Verification**: NLI-based (facebook/bart-large-mnli)
- **Media Processing**: ffmpeg, yt-dlp

## Project Structure

```
video_qa/           # Core package
  app.py            # Streamlit UI (main entry point)
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
