"""
Pipeline Module for Video-QA System
Integrates all components: video processing, speech understanding,
knowledge structuring, embeddings, retrieval, re-ranking, and answer generation.

Fixes applied:
- Force fresh audio extraction on every new video (force=True)
- Delete stale transcript/chunk JSON files before re-processing
- Build FAISS index from scratch (reset=True) to prevent cross-video contamination
- Lecture summary generated after transcription (saved to data/summaries/)
- Query router: summary/overview questions return stored summary instantly,
  factual questions continue through the standard RAG pipeline
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from .config import config
from .logger import get_logger, setup_logger
from .query_rewriter import rewrite_query

from .video_processor import (
    take_video_input,
    convert_video_to_audio,
    check_ffmpeg
)

from .speech_understanding import transcribe_audio
from .knowledge_structuring import process_transcript_to_chunks, process_all_transcripts
from .embeddings import build_vector_index, VectorStore
from .retrieval import RetrievalSystem
from .reranker import ReRanker
from .answer_generator import AnswerGenerator
from .query_router import generate_lecture_summary, route_summary_query


logger = get_logger(__name__)


class VideoQAPipeline:
    """
    Complete Video-QA Pipeline.
    
    Integrates all 9 stages:
    1. Video Processing
    2. Speech Recognition (WhisperX)
    3. Knowledge Structuring (Semantic Segmentation)
    4. Indexing (BGE Embeddings)
    5. Retrieval (FAISS)
    6. Re-Ranking (Cross-Encoder)
    7. Answer Generation (LLM)
    8. Evidence Verification
    9. Output
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        self.retriever = None
        self.reranker = None
        self.answer_generator = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize retrieval components."""
        # Initialize retrieval system
        self.retriever = RetrievalSystem()
        
        # Initialize re-ranker
        self.reranker = ReRanker()
        
        # Initialize answer generator
        self.answer_generator = AnswerGenerator()
        
        logger.info("Pipeline components initialized")
    
    def process_video(self, video_path: str) -> Optional[str]:
        """
        Process a video through the complete pipeline.
        
        Args:
            video_path: Path to video file
            
        Returns:
            video_id string if successful, else None
        """
        import hashlib
        import shutil
        
        logger.info("="*60)
        logger.info("VIDEO-QA PIPELINE - FULL PROCESSING")
        logger.info("="*60)
        
        def get_file_md5(fp):
            hash_md5 = hashlib.md5()
            with open(fp, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
            
        video_md5 = get_file_md5(video_path)
        cache_dir = Path("data/cache") / video_md5
        video_id = Path(video_path).stem
        
        # Paths for the active pipeline
        active_transcript = Path("data/transcripts") / f"{video_id}.json"
        active_chunks = Path("data/chunks") / f"{video_id}.json"
        active_summary = Path("data/summaries") / f"{video_id}.json"
        active_index = Path(config.get("retrieval.index_path", "models/video_index.faiss"))
        active_meta = Path(config.get("retrieval.metadata_path", "models/metadata.pkl"))
        
        # Cached paths
        cached_transcript = cache_dir / "transcript.json"
        cached_chunks = cache_dir / "chunks.json"
        cached_summary = cache_dir / "summary.json"
        cached_index = cache_dir / "video_index.faiss"
        cached_meta = cache_dir / "metadata.pkl"

        # ── CACHE HIT ────────────────────────────────────────────────────────
        # We only restore the per-video artifacts (transcript / chunks / summary)
        # and then merge them into the *current* shared index via
        # build_vector_index(reset=False). We deliberately never overwrite the
        # active FAISS/metadata files with the cached snapshots — those are
        # single-video subsets that would wipe out every other tenant's data.
        # (For multi-tenant SaaS, the same source bytes uploaded by two users
        # must produce two namespaced video_ids, so we also rewrite the
        # video_id stamped inside the cached JSON before merging.)
        if cache_dir.exists() and cached_chunks.exists():
            logger.info(f"⚡ CACHE HIT for hash {video_md5} — reusing transcript/chunks, "
                        f"merging into shared index as '{video_id}'.")

            active_transcript.parent.mkdir(parents=True, exist_ok=True)
            active_chunks.parent.mkdir(parents=True, exist_ok=True)
            active_summary.parent.mkdir(parents=True, exist_ok=True)

            def _restamp_and_copy(src: Path, dst: Path) -> None:
                """Copy a cached JSON to the active path, rewriting any embedded
                video_id field to match the new active video_id."""
                if not src.exists():
                    return
                try:
                    import json as _json
                    with src.open("r", encoding="utf-8") as f:
                        data = _json.load(f)
                    if isinstance(data, dict) and "video_id" in data:
                        data["video_id"] = video_id
                    with dst.open("w", encoding="utf-8") as f:
                        _json.dump(data, f, ensure_ascii=False)
                except Exception:
                    # Fall back to raw copy if the file isn't JSON-shaped.
                    shutil.copy2(src, dst)

            _restamp_and_copy(cached_transcript, active_transcript)
            _restamp_and_copy(cached_chunks, active_chunks)
            _restamp_and_copy(cached_summary, active_summary)

            # Merge the new chunks into the shared FAISS index (no reset).
            # build_vector_index dedups by text so this is a no-op when the
            # exact chunk text is already present under another video_id.
            if not build_vector_index(reset=False):
                logger.error("Index merge from cache failed")
                return None

            logger.info("Cache restore + index merge complete.")
            self._initialize_components()
            return video_id

        # ── CACHE MISS (Normal Processing) ───────────────────────────────────
        logger.info(f"Cache miss for {video_id}. Starting full processing.")

        # Stage 1: Video Processing
        logger.info("\n[STAGE 1/5] Video Processing...")

        if not check_ffmpeg():
            logger.error("FFmpeg not available")
            return None

        # Always force re-extract audio
        audio_path = convert_video_to_audio(video_path, force=True)
        if audio_path is None:
            logger.error("Audio extraction failed")
            return None

        # Clear stale active data before processing
        for stale in [active_transcript, active_chunks, active_summary]:
            if stale.exists():
                stale.unlink()

        # Stage 2: Speech Recognition
        logger.info("\n[STAGE 2/5] Speech Recognition...")

        transcript_path = transcribe_audio(audio_path)
        if transcript_path is None:
            logger.error("Transcription failed")
            return None

        # Stage 3: Knowledge Structuring
        logger.info("\n[STAGE 3/5] Knowledge Structuring (Semantic Segmentation)...")

        chunks_path = process_transcript_to_chunks(transcript_path)
        if chunks_path is None:
            logger.error("Chunking failed")
            return None

        # Stage 4: Indexing
        logger.info("\n[STAGE 4/5] Building Vector Index (BGE Embeddings) — fresh index...")
        success = build_vector_index(reset=True)
        if not success:
            logger.error("Index building failed")
            return None

        # Stage 5: Summary Generation
        logger.info("\n[STAGE 5/5] Generating lecture summary…")
        try:
            import json as _json
            with open(transcript_path, "r", encoding="utf-8") as _f:
                _tdata = _json.load(_f)
            _full_text = _tdata.get("full_transcript", "")
            if _full_text:
                _summary = generate_lecture_summary(_full_text, video_id)
                if _summary:
                    logger.info(f"Lecture summary generated: {len(_summary.get('bullets', []))} bullets")
                else:
                    logger.warning("Lecture summary generation skipped (LLM unavailable).")
            else:
                logger.warning("No full_transcript field in transcript JSON — summary skipped.")
        except Exception as _exc:
            logger.warning(f"Summary generation failed (non-fatal): {_exc}")

        # ── SAVE TO CACHE ────────────────────────────────────────────────────
        logger.info(f"\n[CACHE] Saving results to cache directory: {video_md5}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache only per-video artifacts. We deliberately do NOT cache the
        # global FAISS/metadata files: those are shared across all videos
        # (and, in SaaS, across all users) and become stale snapshots the
        # moment any other video is indexed. The cache-hit branch above
        # rebuilds index entries from the cached chunks instead.
        if active_transcript.exists():
            shutil.copy2(active_transcript, cached_transcript)
        if active_chunks.exists():
            shutil.copy2(active_chunks, cached_chunks)
        if active_summary.exists():
            shutil.copy2(active_summary, cached_summary)

        logger.info("\n[COMPLETE] Pipeline finished!")
        logger.info("="*60)

        # Re-initialize components with new index
        self._initialize_components()

        return video_id
    
    def process_directory(self, video_dir: str = "data/videos") -> int:
        """
        Process all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            
        Returns:
            Number of successfully processed videos
        """
        video_path = Path(video_dir)
        
        if not video_path.exists():
            logger.error(f"Video directory not found: {video_dir}")
            return 0
        
        # Get all video files
        supported = config.get("video.supported_formats", [".mp4", ".mkv", ".avi", ".mov", ".webm"])
        video_files = [f for f in video_path.iterdir() 
                      if f.is_file() and f.suffix.lower() in supported]
        
        if not video_files:
            logger.warning(f"No video files found in {video_dir}")
            return 0
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        success_count = 0
        for video_file in video_files:
            logger.info(f"\nProcessing: {video_file.name}")
            
            if self.process_video(str(video_file)):
                success_count += 1
        
        logger.info(f"\nProcessed {success_count}/{len(video_files)} videos successfully")
        
        return success_count
    
    def ask(
        self,
        query: str,
        active_video_id: Optional[str] = None,
        *,
        use_cache: bool = True,
        include_neighbors: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer a question about processed videos.

        Pipeline stages:
          1. Cache lookup                 (skips LLM on repeat queries)
          2. Summary routing              (instant for summary-style queries)
          3. Query rewrite + retrieval    (top-5 FAISS + optional rerank)
          4. Temporal neighbour expansion (prev/next chunks for context)
          5. Answer generation            (Gemini → fallback chain)
          6. Confidence scoring           (FAISS + overlap + agreement)
          7. Hallucination post-check     (SUPPORTED / PARTIAL / UNSUPPORTED)
          8. Final response assembly      (answer, confidence, status, ts, chunk_ids)

        Returns a dict with the production response contract:
            {
              "answer":     str,
              "confidence": int 0-100,
              "confidence_label": "High"|"Medium"|"Low",
              "status":     "SUPPORTED"|"PARTIAL"|"UNSUPPORTED",
              "timestamp":  "[mm:ss - mm:ss]",
              "chunk_ids":  ["video_id:chunk_id", ...],
              "provider":   "gemini"|"extractive"|...,
              "contexts":   [ {text, score, start, end, video_id, chunk_id}, ... ],
              "support":    { semantic_similarity, keyword_overlap, support_score },
              "cached":     bool,
            }
        """
        # ── Cache lookup ─────────────────────────────────────────────────────
        from .cache import get_cache
        cache = get_cache()
        if use_cache:
            hit = cache.get(query, active_video_id)
            if hit is not None:
                hit["cached"] = True
                logger.info(f"[CACHE HIT] query='{query[:50]}' vid={active_video_id}")
                return hit

        logger.info("="*60)
        logger.info(f"QUERY: {query}")
        logger.info("="*60)

        # ── PATH A: Summary / overview questions ─────────────────────────────
        summary_result = route_summary_query(query, active_video_id)
        if summary_result is not None:
            logger.info("[Pipeline] Returning pre-computed lecture summary.")
            summary_result.setdefault("status", "SUPPORTED")
            summary_result.setdefault("chunk_ids", [])
            summary_result["cached"] = False
            if use_cache:
                cache.set(query, active_video_id, summary_result)
            return summary_result

        # ── PATH B: Factual questions — standard RAG ──────────────────────────
        if not self.retriever.is_available():
            logger.error("Vector index not available. Please process videos first.")
            return {
                "answer": "Question cannot be answered from this video.",
                "timestamp": None,
                "confidence": 0,
                "confidence_label": "Low",
                "status": "UNSUPPORTED",
                "chunk_ids": [],
                "contexts": [],
                "error": "No videos processed",
                "cached": False,
            }

        # Stage 5A: Query Rewrite
        logger.info("\n[QUERY REWRITE] Optimizing search query...")
        rewrite_result = rewrite_query(query)
        if isinstance(rewrite_result, tuple):
            search_query, provider = rewrite_result
        else:
            search_query = rewrite_result
            provider = "rewriter"

        if search_query != query:
            logger.info(f"Using search query: '{search_query}' (via {provider})")
            summary_result = route_summary_query(search_query, active_video_id)
            if summary_result is not None:
                logger.info("[Pipeline] Returning pre-computed lecture summary (re-routed after rewrite).")
                summary_result.setdefault("status", "SUPPORTED")
                summary_result.setdefault("chunk_ids", [])
                summary_result["cached"] = False
                if use_cache:
                    cache.set(query, active_video_id, summary_result)
                return summary_result
        else:
            logger.info("Using original query for search.")

        # Stage 5B: Retrieval (top-5)
        logger.info("\n[RETRIEVAL] Finding relevant passages...")
        retrieved = self.retriever.retrieve(search_query, top_k=5, video_id=active_video_id)

        if not retrieved:
            logger.warning("No relevant passages found")
            out = {
                "answer": "Question cannot be answered from this video.",
                "timestamp": None,
                "confidence": 0,
                "confidence_label": "Low",
                "status": "UNSUPPORTED",
                "chunk_ids": [],
                "contexts": [],
                "cached": False,
            }
            if use_cache:
                cache.set(query, active_video_id, out)
            return out

        logger.info(f"Retrieved {len(retrieved)} passages")

        # Stage 5C: Out-of-scope gate
        # BGE embeddings return superficially high cosine scores even for totally
        # unrelated queries (e.g. "lasagna" → 0.77 against ML lectures). A single
        # retrieval-score threshold therefore cannot separate in-scope from OOS.
        # Instead, require at least one content word shared between the query and
        # the top-1 retrieved chunk. If none exists, refuse immediately.
        try:
            from .hallucination_detector import _content_words
            # Use the union of original and rewritten query tokens so a well-formed
            # rewrite (e.g. "regularization" → "L2 weight decay") cannot trigger a
            # false OOS refusal just because the user's raw wording was brief.
            q_words = set(_content_words(query)) | set(_content_words(search_query))
            top_text = (retrieved[0].get("text") or "")
            top_words = set(_content_words(top_text))
            overlap = q_words & top_words
            top_score = float(retrieved[0].get("score", 0.0))
            # Refuse when the user's question shares zero content words with the
            # best chunk AND the retrieval score is not decisively high.
            if q_words and not overlap and top_score < 0.85:
                logger.warning(
                    f"[OOS GATE] Query has no content-word overlap with top chunk "
                    f"(top_score={top_score:.3f}). Refusing."
                )
                out = {
                    "answer": "Not found in video.",
                    "timestamp": None,
                    "confidence": 10,
                    "confidence_label": "Low",
                    "status": "UNSUPPORTED",
                    "chunk_ids": [],
                    "contexts": [],
                    "support": {"status": "UNSUPPORTED", "semantic_similarity": 0.0,
                                "keyword_overlap": 0.0, "support_score": 0.0},
                    "neighbors": [],
                    "provider": "oos_gate",
                    "cached": False,
                }
                if use_cache:
                    cache.set(query, active_video_id, out)
                return out
        except Exception as e:
            logger.warning(f"[OOS GATE] check skipped: {e}")

        # Stage 6: Re-Ranking
        logger.info("\n[RE-RANKING] Re-ranking passages...")
        reranked = self.reranker.rerank(query, retrieved)
        prompt_contexts = reranked[:2]
        all_contexts    = reranked

        # Stage 6B: Temporal neighbour expansion (prev/next chunks)
        neighbors: List[Dict[str, Any]] = []
        if include_neighbors:
            try:
                from .temporal_neighbors import get_neighbors
                # Neighbours only for the top-1 chunk to keep prompt small
                if prompt_contexts:
                    nb = get_neighbors(prompt_contexts[0], window=1)
                    neighbors = nb["previous"] + nb["next"]
                    logger.info(f"[TEMPORAL] Added {len(neighbors)} neighbour chunks for context.")
            except Exception as e:
                logger.warning(f"[TEMPORAL] neighbour expansion failed: {e}")

        # Stage 7: Answer Generation — include neighbours alongside top-2 for richer context
        logger.info("\n[ANSWER GENERATION] Generating answer...")
        llm_contexts = list(prompt_contexts)
        # Append neighbours at the end so retrieved top-2 remain first (prompt priority)
        for n in neighbors:
            llm_contexts.append({
                "text":     n.get("text", ""),
                "score":    0.0,
                "start":    n.get("start", 0.0),
                "end":      n.get("end", 0.0),
                "video_id": n.get("video_id"),
                "chunk_id": n.get("chunk_id"),
            })
        result = self.answer_generator.generate(query, llm_contexts, all_contexts=all_contexts)

        # Stage 8: Hallucination post-check
        try:
            from .hallucination_detector import check_answer
            support = check_answer(result.get("answer", ""), all_contexts)
        except Exception as e:
            logger.warning(f"[HALLUCINATION] post-check failed: {e}")
            support = {"status": "PARTIAL", "semantic_similarity": 0.0,
                       "keyword_overlap": 0.0, "support_score": 0.0}

        # Stage 9: Final response assembly
        chunk_ids = [
            f"{c.get('video_id')}:{c.get('chunk_id')}"
            for c in all_contexts
            if c.get("video_id") is not None and c.get("chunk_id") is not None
        ]

        # If the hallucination check says UNSUPPORTED, cap the displayed confidence
        # so the UI cannot show "High confidence" for an unsupported answer.
        if support["status"] == "UNSUPPORTED":
            result["confidence"] = min(int(result.get("confidence", 0)), 35)
            result["confidence_label"] = "Low"
        elif support["status"] == "PARTIAL":
            result["confidence"] = min(int(result.get("confidence", 0)), 70)
            if result["confidence"] < 40:
                result["confidence_label"] = "Low"
            else:
                result["confidence_label"] = "Medium"

        result["status"]    = support["status"]
        result["support"]   = support
        result["chunk_ids"] = chunk_ids
        result["neighbors"] = [
            {"video_id": n.get("video_id"), "chunk_id": n.get("chunk_id"),
             "relation": n.get("relation"), "start": n.get("start"),
             "end": n.get("end"), "text": (n.get("text") or "")[:200]}
            for n in neighbors
        ]
        result["cached"] = False

        logger.info("\n[OUTPUT]")
        logger.info(f"Answer:     {result.get('answer')}")
        logger.info(f"Timestamp:  {result.get('timestamp')}")
        logger.info(f"Confidence: {result.get('confidence')}% ({result.get('confidence_label')})")
        logger.info(f"Status:     {result.get('status')} "
                    f"(sim={support['semantic_similarity']}, overlap={support['keyword_overlap']})")
        logger.info(f"ChunkIDs:   {chunk_ids}")

        if use_cache:
            cache.set(query, active_video_id, result)
        return result

    
    def interactive_mode(self):
        """Run interactive question-answering mode."""
        print("\n" + "="*60)
        print("VIDEO-QA INTERACTIVE MODE")
        print("="*60)
        print("\nAsk questions about the video(s).")
        print("Type 'quit' or 'exit' to end session.")
        print("Type 'menu' to see options.\n")
        
        while True:
            try:
                query = input("\nYour question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if query.lower() == 'menu':
                    print("""
Options:
- Ask any question about the video content
- Type 'quit' to exit
                    """)
                    continue
                
                result = self.ask(query)
                
                print("\n" + "-"*40)
                print("ANSWER:")
                print("-"*40)
                print(result.get('answer', 'NOT FOUND IN VIDEO'))
                
                if result.get('timestamp'):
                    print(f"\nTimestamp: {result['timestamp']}")
                
                if result.get('verified'):
                    print("\n[✓] Answer verified against evidence")
                
                print("-"*40)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


def run_full_pipeline(video_path: Optional[str] = None) -> bool:
    """
    Run full pipeline: process video and answer questions.
    
    Args:
        video_path: Optional path to video file
        
    Returns:
        True if successful
    """
    # Initialize pipeline
    pipeline = VideoQAPipeline()
    
    # Get video path
    if video_path is None:
        video_path = take_video_input()
    
    if video_path is None:
        print("No video selected")
        return False
    
    # Process video
    success = pipeline.process_video(video_path)
    
    if not success:
        print("Video processing failed")
        return False
    
    # Run interactive mode
    pipeline.interactive_mode()
    
    return True


def run_query_only() -> Dict[str, Any]:
    """
    Run query-only mode (assumes videos already processed).
    
    Returns:
        Query result dictionary
    """
    pipeline = VideoQAPipeline()
    
    if not pipeline.retriever.is_available():
        print("No vector index found. Please process videos first.")
        return {"answer": "NOT FOUND IN VIDEO", "error": "No index"}
    
    query = input("\nAsk a question: ").strip()
    
    if not query:
        return {"answer": "NOT FOUND IN VIDEO", "error": "Empty query"}
    
    return pipeline.ask(query)


if __name__ == "__main__":
    # Setup logging
    setup_logger()
    
    # Run pipeline
    print("""
=================================================
   VIDEO KNOWLEDGE RETRIEVAL AI SYSTEM
   (Research-Level RAG Pipeline)
=================================================

1 → Process New Video
2 → Process All Videos in Directory
3 → Ask Question (Interactive Mode)
4 → Exit
    """)
    
    choice = input("Enter choice: ").strip()
    
    pipeline = VideoQAPipeline()
    
    if choice == "1":
        video_path = take_video_input()
        if video_path:
            pipeline.process_video(video_path)
    
    elif choice == "2":
        pipeline.process_directory()
    
    elif choice == "3":
        if pipeline.retriever.is_available():
            pipeline.interactive_mode()
        else:
            print("No videos processed yet. Please process videos first.")
    
    else:
        print("Exiting.")
