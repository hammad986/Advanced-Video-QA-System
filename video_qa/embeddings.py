"""
Embeddings Module for Video-QA System
Uses retrieval-optimized BGE/E5 embeddings instead of MiniLM.
"""

import os
import json
import numpy as np
import faiss
import joblib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .config import config
from .logger import get_logger


logger = get_logger(__name__)

# Global model cache
_embedding_model = None


def load_embedding_model() -> Any:
    """
    Load retrieval-optimized embedding model (BGE or E5).
    
    Returns:
        SentenceTransformer model
    """
    global _embedding_model
    
    if _embedding_model is None:
        embed_config = config.get_section("embeddings")
        model_name = embed_config.get("model_name", "BAAI/bge-small-en")
        device = embed_config.get("device", "cpu")
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name, device=device)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.error("sentence-transformers not installed")
            return None
        except Exception as e:
            logger.warning(f"Failed to load {model_name}, trying fallback: {e}")
            # Fallback chain: bge-base → MiniLM
            for fallback in ["BAAI/bge-base-en-v1.5", "all-MiniLM-L6-v2"]:
                try:
                    _embedding_model = SentenceTransformer(fallback, device=device)
                    logger.info(f"Using fallback model: {fallback}")
                    break
                except Exception:
                    continue
            if _embedding_model is None:
                logger.error("Failed to load any embedding model")
                return None
    
    return _embedding_model


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize embeddings for cosine similarity.
    
    Args:
        vectors: Raw embedding vectors
        
    Returns:
        L2-normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    return vectors / norms


def create_embeddings(texts: List[str], batch_size: int = 32) -> Optional[np.ndarray]:
    """
    Create embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for encoding
        
    Returns:
        Numpy array of embeddings
    """
    model = load_embedding_model()
    if model is None:
        return None
    
    embed_config = config.get_section("embeddings")
    normalize = embed_config.get("normalize", True)
    
    logger.info(f"Creating embeddings for {len(texts)} texts...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size or embed_config.get("batch_size", 32),
        show_progress_bar=True,
        normalize_embeddings=normalize
    )
    
    if not normalize:
        embeddings = normalize_embeddings(embeddings)
    
    logger.info(f"Embeddings created: {embeddings.shape}")
    return embeddings


def build_vector_index(
    chunks_dir: str = "data/chunks",
    model_dir: str = "models",
    reset: bool = False
) -> bool:
    """
    Build FAISS vector index from chunk files.
    
    Args:
        chunks_dir: Directory containing chunk JSON files
        model_dir:  Directory for saving index and metadata
        reset:      When True, discard any existing index and build from scratch.
                    Pass reset=True when processing a new video to prevent
                    cross-video contamination.
        
    Returns:
        True if successful
    """
    logger.info("="*50)
    logger.info("EMBEDDING & INDEXING PIPELINE")
    logger.info("="*50)

    if reset:
        logger.info("reset=True — discarding existing index and starting fresh.")
    
    # Setup paths
    chunks_path = Path(chunks_dir)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    # Use config paths directly (they already contain the full relative path like
    # "models/video_index.faiss"). Only join with model_path when config gives just
    # a filename (no directory component).
    _cfg_index = config.get("retrieval.index_path", "video_index.faiss")
    _cfg_meta  = config.get("retrieval.metadata_path", "metadata.pkl")

    index_path = Path(_cfg_index) if Path(_cfg_index).parent != Path(".") else model_path / _cfg_index
    meta_path  = Path(_cfg_meta)  if Path(_cfg_meta).parent  != Path(".") else model_path / _cfg_meta
    
    # ── CROSS-VIDEO CONTAMINATION FIX ──────────────────────────────────────
    # When reset=True, skip loading the old index entirely so we build a
    # clean index containing only the newly uploaded video's chunks.
    existing_texts = []
    existing_metadata = []
    existing_embeddings = None
    
    if not reset and index_path.exists() and meta_path.exists():
        logger.info("Loading existing vector database...")
        try:
            existing_index = faiss.read_index(str(index_path))
            existing_metadata = joblib.load(str(meta_path))
            
            # Extract existing texts
            for meta in existing_metadata:
                existing_texts.append(meta.get("text", ""))
            
            logger.info(f"Found {len(existing_metadata)} existing knowledge segments")
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")
            existing_metadata = []
    
    # Load all chunk files
    if not chunks_path.exists():
        logger.error(f"Chunks directory not found: {chunks_dir}")
        return False
    
    chunk_files = list(chunks_path.glob("*.json"))
    if not chunk_files:
        logger.error("No chunk files found")
        return False
    
    # Collect new texts and metadata
    new_texts = []
    new_metadata = []
    
    for chunk_file in chunk_files:
        with open(chunk_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_id = data.get("video_id", chunk_file.stem)

        # Support both 'chunks' (new format) and 'segments' (existing format)
        chunks = data.get("chunks") or data.get("segments", [])

        for chunk in chunks:
            # Support 'text' or 'content' as the text field
            text = (chunk.get("text") or chunk.get("content") or "").strip()

            # If no text but has title, use title as text (some formats)
            if not text:
                title = chunk.get("title", "")
                if title:
                    text = title.strip()

            if not text:
                continue

            # Support both start/end and start_time/end_time timestamps
            start = chunk.get("start", chunk.get("start_time", 0)) or 0
            end   = chunk.get("end",   chunk.get("end_time",   0)) or 0

            # Check if this text already exists in the database
            if text not in existing_texts:
                new_texts.append(text)
                new_metadata.append({
                    "video_id":   video_id,
                    "chunk_id":   chunk.get("chunk_id", chunk.get("number", len(new_metadata))),
                    "start":      float(start),
                    "end":        float(end),
                    "text":       text,
                    "title":      chunk.get("title", ""),
                    "word_count": chunk.get("word_count", len(text.split())),
                })

    
    logger.info(f"Found {len(new_texts)} new segments to index")
    
    # If no new texts, we're done
    if len(new_texts) == 0:
        logger.info("No new segments to index. Using existing database.")
        return True
    
    # Create embeddings for new texts
    logger.info("Creating embeddings for new segments...")
    new_embeddings = create_embeddings(new_texts)
    
    if new_embeddings is None:
        logger.error("Failed to create embeddings")
        return False
    
    # Combine with existing if any
    if len(existing_metadata) > 0:
        # Re-embed existing texts for consistency
        logger.info("Re-embedding existing segments for consistency...")
        existing_embeddings = create_embeddings(existing_texts)
        
        if existing_embeddings is not None:
            combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
            combined_metadata = existing_metadata + new_metadata
        else:
            combined_embeddings = new_embeddings
            combined_metadata = new_metadata
    else:
        combined_embeddings = new_embeddings
        combined_metadata = new_metadata
    
    # Build FAISS index
    logger.info("Building FAISS index...")
    dimension = combined_embeddings.shape[1]
    
    # Use Inner Product for cosine similarity (with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(combined_embeddings.astype("float32"))
    
    # Save index and metadata
    logger.info("Saving index and metadata...")
    faiss.write_index(index, str(index_path))
    joblib.dump(combined_metadata, str(meta_path))
    
    logger.info("="*50)
    logger.info("Index build complete!")
    logger.info(f"Total segments indexed: {len(combined_metadata)}")
    logger.info(f"Index saved to: {index_path}")
    
    return True


class VectorStore:
    """Vector store for retrieval."""
    
    def __init__(self, index_path: Optional[str] = None, metadata_path: Optional[str] = None):
        """
        Initialize vector store.
        
        Args:
            index_path: Path to FAISS index
            metadata_path: Path to metadata pickle
        """
        self.index = None
        self.metadata = []
        self.embedding_dimension = None
        
        # Load from config if not provided
        if index_path is None:
            index_path = config.get("retrieval.index_path", "models/video_index.faiss")
        if metadata_path is None:
            metadata_path = config.get("retrieval.metadata_path", "models/metadata.pkl")
        
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        
        self._load()
    
    def _load(self) -> bool:
        """Load index and metadata from disk."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("Index or metadata not found")
            return False
        
        try:
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = joblib.load(str(self.metadata_path))
            self.embedding_dimension = self.index.d
            logger.info(f"Loaded index with {len(self.metadata)} segments, dimension={self.embedding_dimension}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def _check_dimension_compatibility(self) -> bool:
        """Check if the index dimension matches the embedding model."""
        # Get expected dimension from embedding model
        model = load_embedding_model()
        if model is None:
            return False
        
        # Get dimension from model's embedding size
        expected_dim = model.get_sentence_embedding_dimension()
        
        if self.embedding_dimension != expected_dim:
            logger.warning(f"Dimension mismatch: index={self.embedding_dimension}, model={expected_dim}")
            logger.info("Rebuilding index with new embedding dimension...")
            return False
        
        return True
    
    def _rebuild_index(self) -> bool:
        """Rebuild the index with the correct embedding dimension."""
        if not self.metadata:
            logger.error("No metadata available to rebuild index")
            return False

        # Re-embed all existing texts
        texts = [meta.get("text", "") for meta in self.metadata]
        embeddings = create_embeddings(texts)

        if embeddings is None:
            logger.error("Failed to create embeddings for rebuild")
            return False

        # Create new index with correct dimension
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))

        # ── METADATA SAVE FIX ──────────────────────────────────────────────
        # Previously only the FAISS index was persisted here; the metadata
        # file was left stale/absent, causing retrieval crashes on next load.
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        joblib.dump(self.metadata, str(self.metadata_path))   # ← was missing!

        self.embedding_dimension = dimension
        logger.info(f"Index rebuilt successfully with dimension={dimension}, metadata entries={len(self.metadata)}")

        return True
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0,
        video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        RETRIEVAL FIX (v2):
        - Always fetch top_k*10 candidates from FAISS (no hard threshold).
        - Normalise query embedding the same way as index embeddings (cosine).
        - If video_id scoping is requested, try filtered results first.
          If that yields 0 results (ID mismatch / pre-existing index), fall
          back to un-filtered top-k — avoids the "0 relevant chunks" bug.

        Args:
            query:     Query string
            top_k:     Number of results to return
            min_score: Minimum cosine similarity (0.0 = return everything)
            video_id:  Optional video scope filter

        Returns:
            List of matching chunks with scores
        """
        if self.index is None:
            logger.error("Index not loaded")
            return []

        # ── Dimension compatibility check ──────────────────────────────────
        if not self._check_dimension_compatibility():
            logger.warning("Dimension mismatch detected, rebuilding index…")
            if not self._rebuild_index():
                logger.error("Failed to rebuild index")
                return []

        # ── Create query embedding (normalised = cosine similarity) ─────────
        query_embeddings = create_embeddings([query])
        if query_embeddings is None:
            logger.error("Failed to create query embedding")
            return []

        # ── FAISS search — fetch candidates ─────────────
        n_total = self.index.ntotal
        if n_total == 0:
            logger.warning("Index is empty — no vectors indexed yet")
            return []

        # If a video filter is strictly requested, fetch all vectors so we don't truncate 
        # legitimate matches from this video just because other videos score higher globally.
        fetch_k = n_total if video_id else min(n_total, max(top_k * 10, 100))
        logger.info(f"FAISS search: fetch_k={fetch_k}, total_vectors={n_total}")

        scores, indices = self.index.search(query_embeddings.astype("float32"), fetch_k)

        # ── Build results — two-pass if video_id scoping is requested ───────
        def _collect(apply_filter: bool) -> List[Dict[str, Any]]:
            out = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                if score < min_score:
                    continue
                if idx >= len(self.metadata):
                    continue

                chunk_meta = self.metadata[idx]

                if apply_filter and video_id:
                    if chunk_meta.get("video_id") != video_id:
                        continue

                item = chunk_meta.copy()
                item["score"] = float(score)
                out.append(item)

                if len(out) >= top_k:
                    break
            return out

        if video_id:
            # Strict scoping: only return chunks belonging to this video.
            # Returning empty when there are no matches preserves grounding
            # integrity (caller must not silently get answers from other videos).
            results = _collect(apply_filter=True)
            if not results:
                logger.warning(
                    f"video_id filter '{video_id}' matched 0 chunks — "
                    "preserving scope: returning empty result."
                )
        else:
            results = _collect(apply_filter=False)

        logger.info(f"FAISS returned {len(results)} chunks for query: {query[:60]!r}")
        return results
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Get all indexed chunks."""
        return self.metadata


if __name__ == "__main__":
    # Test embedding creation
    test_texts = [
        "This is a test sentence.",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    embeddings = create_embeddings(test_texts)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Test indexing
    build_vector_index()
