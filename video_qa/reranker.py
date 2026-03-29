"""
Re-Ranking Module for Video-QA System
Uses Cross-Encoder Re-Ranker (BGE-Reranker) to improve retrieval quality.
This is the "game changer" that converts student project → research-quality RAG.
"""

import os
from typing import List, Dict, Any, Optional

from .config import config
from .logger import get_logger


logger = get_logger(__name__)

# Global reranker model cache
_reranker_model = None


def load_reranker_model():
    """
    Load BGE Cross-Encoder Re-Ranker model.
    
    Returns:
        CrossEncoder model or None
    """
    global _reranker_model
    
    if _reranker_model is None:
        rerank_config = config.get_section("reranking")
        
        if not rerank_config.get("enabled", True):
            logger.info("Re-ranking is disabled")
            return None
        
        model_name = rerank_config.get("model_name", "BAAI/bge-reranker-large")
        device = rerank_config.get("device", "cpu")
        
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading re-ranker model: {model_name}")
            _reranker_model = CrossEncoder(model_name, max_length=512)
            logger.info("Re-ranker model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not available for re-ranking")
            return None
        except Exception as e:
            logger.warning(f"Failed to load re-ranker: {e}")
            # Try smaller model as fallback
            try:
                _reranker_model = CrossEncoder("BAAI/bge-reranker-base")
                logger.info("Using fallback: bge-reranker-base")
            except:
                logger.warning("Re-ranking disabled - no suitable model found")
                return None
    
    return _reranker_model


def rerank_passages(
    query: str,
    passages: List[Dict[str, Any]],
    top_k: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Re-rank passages using cross-encoder.
    
    The re-ranker directly scores how well each passage answers the question,
    which is more accurate than vector similarity.
    
    Args:
        query: User question
        passages: List of retrieved passages
        top_k: Number of top passages to keep
        
    Returns:
        Re-ranked list of passages
    """
    if not passages:
        return []
    
    # Get config
    rerank_config = config.get_section("reranking")
    
    if not rerank_config.get("enabled", True):
        logger.info("Re-ranking disabled, returning original order")
        return passages
    
    if top_k is None:
        top_k = rerank_config.get("top_k", 3)
    
    model = load_reranker_model()
    
    if model is None:
        logger.warning("Re-ranker not available, returning original passages")
        return passages[:top_k]
    
    logger.info(f"Re-ranking {len(passages)} passages...")
    
    # Prepare query-passage pairs for cross-encoder
    pairs = [[query, passage.get("text", "")] for passage in passages]
    
    try:
        # Get relevance scores
        scores = model.predict(pairs)
        
        # Add scores to passages
        for i, (passage, score) in enumerate(zip(passages, scores)):
            passages[i]["rerank_score"] = float(score)
        
        # Sort by re-rank score (descending)
        reranked = sorted(passages, key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        # Keep top-k
        results = reranked[:top_k]
        
        logger.info(f"Re-ranked to top-{len(results)} passages")
        
        return results
        
    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")
        return passages[:top_k]


class ReRanker:
    """Re-ranking wrapper class."""
    
    def __init__(self):
        """Initialize re-ranker."""
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the re-ranker model."""
        self.model = load_reranker_model()
    
    def rerank(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Re-rank passages for a query.
        
        Args:
            query: User question
            passages: Retrieved passages
            top_k: Number to keep
            
        Returns:
            Re-ranked passages
        """
        return rerank_passages(query, passages, top_k)
    
    def is_available(self) -> bool:
        """Check if re-ranker is available."""
        return self.model is not None


if __name__ == "__main__":
    # Test re-ranking
    test_passages = [
        {
            "text": "Machine learning is a type of artificial intelligence.",
            "start": 0,
            "end": 10
        },
        {
            "text": "The weather today is sunny and warm.",
            "start": 10,
            "end": 20
        },
        {
            "text": "Deep learning uses neural networks with multiple layers.",
            "start": 20,
            "end": 30
        }
    ]
    
    query = "What is machine learning?"
    
    results = rerank_passages(query, test_passages, top_k=2)
    
    print(f"\nRe-ranked results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. Score: {r.get('rerank_score', 0):.3f}")
        print(f"   Text: {r.get('text')}")
