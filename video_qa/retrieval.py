"""
Retrieval Module for Video-QA System
Handles vector-based retrieval from FAISS index.
"""

from typing import List, Dict, Any, Optional

from .config import config
from .logger import get_logger
from .embeddings import VectorStore, create_embeddings


logger = get_logger(__name__)


class RetrievalSystem:
    """Vector retrieval system."""
    
    def __init__(self):
        """Initialize retrieval system."""
        self.vector_store = None
        self._initialize()
    
    def _initialize(self):
        """Initialize vector store."""
        index_path = config.get("retrieval.index_path", "models/video_index.faiss")
        metadata_path = config.get("retrieval.metadata_path", "models/metadata.pkl")
        
        self.vector_store = VectorStore(index_path, metadata_path)
        
        if self.vector_store.index is None:
            logger.warning("Vector store not initialized. Please process videos first.")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        video_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User question
            top_k: Number of results (default from config)
            min_score: Minimum similarity score (default from config)
            
        Returns:
            List of relevant chunks with scores
        """
        if self.vector_store is None or self.vector_store.index is None:
            logger.error("Vector store not available")
            return []
        
        # Get config values
        # FIX: Set default top_k=5 for improved answer quality
        if top_k is None:
            top_k = config.get("retrieval.top_k", 5)
        if min_score is None:
            min_score = config.get("retrieval.min_score", 0.0)
        
        logger.info(f"Retrieving top-{top_k} chunks for query: {query[:50]}...")
        
        # Define a helper function to perform the search and apply filters
        def _collect(apply_filter: bool) -> List[Dict[str, Any]]:
            if apply_filter and video_id:
                return self.vector_store.search(query, top_k, min_score, video_id=video_id)
            else:
                # If apply_filter is False, or video_id is None, search without video_id filter
                return self.vector_store.search(query, top_k, min_score)

        # ── FORENSIC DEBUG: Stage 1 - Retrieval Details ────────────────────────────
        # The initial call to search is now part of the conditional logic below
        # to prevent fallback.
        
        if video_id:
            results = _collect(apply_filter=True)
            if not results:
                logger.warning(
                    f"[RETRIEVAL] video_id filter '{video_id}' matched 0 chunks — "
                    "falling back to global retrieval."
                )
                results = _collect(apply_filter=False)
        else:
            results = _collect(apply_filter=False)
        
        logger.info(f"\n[RETRIEVAL DEBUG] Total chunks retrieved: {len(results)}")
        for i, chunk in enumerate(results, 1):
            score = chunk.get('score', 0)
            start = chunk.get('start', 0)
            end = chunk.get('end', 0)
            text = chunk.get('text', '')[:80]
            logger.info(
                f"  [Chunk {i}] Score: {score:.4f} | "
                f"Timestamp: {start:.1f}s-{end:.1f}s | "
                f"Text: {text}..."
            )
        # ──────────────────────────────────────────────────────────────────────────
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        return results
    
    def is_available(self) -> bool:
        """Check if retrieval system is ready."""
        return self.vector_store is not None and self.vector_store.index is not None


def retrieve_chunks(
    query: str,
    top_k: Optional[int] = None,
    video_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve chunks.
    
    Args:
        query: User question
        top_k: Number of results
        
    Returns:
        List of relevant chunks
    """
    retriever = RetrievalSystem()
    return retriever.retrieve(query, top_k, video_id=video_id)


if __name__ == "__main__":
    # Test retrieval
    query = "What is machine learning?"
    results = retrieve_chunks(query)
    
    print(f"\nRetrieved {len(results)} results:")
    for i, r in enumerate(results[:5], 1):
        print(f"\n{i}. Score: {r.get('score', 0):.3f}")
        print(f"   Text: {r.get('text', '')[:100]}...")
