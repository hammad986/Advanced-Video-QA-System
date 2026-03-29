"""
Knowledge Structuring Module for Video-QA System
Handles semantic segmentation using embedding similarity.
Splits transcripts into topic-level chunks (NOT fixed word chunking).
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

from .config import config
from .logger import get_logger


logger = get_logger(__name__)

# Global embedding model cache
_embedding_model = None


def load_embedding_model():
    """Load embedding model for semantic chunking."""
    global _embedding_model
    
    if _embedding_model is None:
        chunking_config = config.get_section("chunking")
        semantic_config = chunking_config.get("semantic", {})
        model_name = semantic_config.get(
            "embedding_model", 
            "BAAI/bge-small-en-v1.5"
        )
        
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {model_name}")
            _embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded")
        except ImportError:
            logger.error("sentence-transformers not installed")
            return None
    
    return _embedding_model


def get_sentence_embeddings(sentences: List[str]) -> np.ndarray:
    """Get embeddings for a list of sentences."""
    model = load_embedding_model()
    if model is None:
        return None
    
    embeddings = model.encode(sentences, show_progress_bar=False)
    return np.array(embeddings)


def calculate_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings."""
    # Normalize
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    return float(np.dot(emb1, emb2))


def semantic_segmentation(
    segments: List[Dict[str, Any]],
    similarity_threshold: float = 0.3,
    min_chunk_words: int = 50,
    max_chunk_words: int = 300
) -> List[Dict[str, Any]]:
    """
    Perform semantic segmentation on transcript segments.
    
    Uses embedding similarity to detect topic changes:
    - If similarity between consecutive sentences drops below threshold → new chunk
    - This preserves topic coherence vs. arbitrary word count splitting
    
    Args:
        segments: List of transcript segments
        similarity_threshold: Threshold for starting new chunk
        min_chunk_words: Minimum words per chunk
        max_chunk_words: Maximum words per chunk
        
    Returns:
        List of semantic chunks
    """
    if not segments:
        return []
    
    logger.info("Performing semantic segmentation...")
    
    # Extract sentences from segments
    sentences = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            sentences.append({
                "text": text,
                "start": seg.get("start", 0),
                "end": seg.get("end", 0)
            })
    
    if not sentences:
        return []
    
    # Get embeddings
    text_list = [s["text"] for s in sentences]
    embeddings = get_sentence_embeddings(text_list)
    
    if embeddings is None:
        logger.warning("Falling back to fixed chunking")
        return fixed_chunking(segments)
    
    # Find topic boundaries
    chunk_boundaries = [0]  # Start with first sentence
    
    for i in range(1, len(sentences)):
        similarity = calculate_similarity(embeddings[i-1], embeddings[i])
        
        # Check if similarity drops below threshold
        if similarity < similarity_threshold:
            # Check if current chunk has minimum words
            current_start = chunk_boundaries[-1]
            current_text = " ".join(s["text"] for s in sentences[current_start:i])
            word_count = len(current_text.split())
            
            if word_count >= min_chunk_words:
                chunk_boundaries.append(i)
    
    # Always include the last segment
    if chunk_boundaries[-1] < len(sentences):
        chunk_boundaries.append(len(sentences))
    
    # Create chunks from boundaries
    chunks = []
    for i in range(len(chunk_boundaries) - 1):
        start_idx = chunk_boundaries[i]
        end_idx = chunk_boundaries[i + 1]
        
        chunk_segments = sentences[start_idx:end_idx]
        
        # Calculate chunk metadata
        chunk_text = " ".join(s["text"] for s in chunk_segments)
        word_count = len(chunk_text.split())
        
        # Enforce max chunk size
        if word_count > max_chunk_words:
            # Split into smaller chunks
            sub_chunks = split_large_chunk(chunk_segments, max_chunk_words)
            chunks.extend(sub_chunks)
        else:
            chunks.append({
                "start": chunk_segments[0]["start"],
                "end": chunk_segments[-1]["end"],
                "text": chunk_text,
                "word_count": word_count
            })
    
    logger.info(f"Created {len(chunks)} semantic chunks")
    return chunks


def split_large_chunk(
    segments: List[Dict[str, Any]],
    max_words: int
) -> List[Dict[str, Any]]:
    """Split a large chunk into smaller ones."""
    chunks = []
    current_chunk = []
    current_words = 0
    
    for seg in segments:
        seg_words = len(seg["text"].split())
        
        if current_words + seg_words <= max_words:
            current_chunk.append(seg)
            current_words += seg_words
        else:
            if current_chunk:
                chunks.append(create_chunk_from_segments(current_chunk))
            current_chunk = [seg]
            current_words = seg_words
    
    if current_chunk:
        chunks.append(create_chunk_from_segments(current_chunk))
    
    return chunks


def create_chunk_from_segments(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create a chunk dictionary from segments."""
    return {
        "start": segments[0]["start"],
        "end": segments[-1]["end"],
        "text": " ".join(s["text"] for s in segments),
        "word_count": len(" ".join(s["text"] for s in segments).split())
    }


def fixed_chunking(
    segments: List[Dict[str, Any]],
    chunk_size: int = None,
    overlap: int = None
) -> List[Dict[str, Any]]:
    """
    Fallback fixed word-count chunking.
    
    Args:
        segments: List of transcript segments
        chunk_size: Words per chunk
        overlap: Overlap words between chunks
        
    Returns:
        List of chunks
    """
    # FIX: Make chunk_size and overlap configurable from config
    if chunk_size is None or overlap is None:
        from .config import config
        chunking_config = config.get_section("chunking")
        fixed_config = chunking_config.get("fixed", {})
        if chunk_size is None:
            chunk_size = fixed_config.get("chunk_size_words", 150)
        if overlap is None:
            overlap = fixed_config.get("overlap_words", 40)

    chunks = []
    current_chunk = []
    word_count = 0

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        words = text.split()
        word_len = len(words)

        if word_count + word_len <= chunk_size:
            current_chunk.append(seg)
            word_count += word_len
        else:
            if current_chunk:
                chunks.append(create_chunk_from_segments(current_chunk))

            # Create overlap from previous chunk
            overlap_chunk = []
            overlap_words = 0

            for prev in reversed(current_chunk):
                overlap_chunk.insert(0, prev)
                overlap_words += len(prev["text"].split())
                if overlap_words >= overlap:
                    break

            current_chunk = overlap_chunk + [seg]
            word_count = sum(len(x["text"].split()) for x in current_chunk)

    if current_chunk:
        chunks.append(create_chunk_from_segments(current_chunk))

    return chunks


def process_transcript_to_chunks(
    transcript_path: str,
    output_dir: Optional[str] = None
) -> Optional[str]:
    """
    Convert transcript to semantic chunks.
    
    Args:
        transcript_path: Path to transcript JSON
        output_dir: Output directory for chunks
        
    Returns:
        Path to chunks JSON file
    """
    logger.info("="*50)
    logger.info("KNOWLEDGE STRUCTURING PIPELINE")
    logger.info("="*50)
    
    # Setup paths
    transcript_path_obj = Path(transcript_path)
    output_dir = output_dir or "data/chunks"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    chunks_file = output_path / f"{transcript_path_obj.stem}.json"
    
    # Check if already processed
    if chunks_file.exists():
        logger.info(f"Chunks already exist: {chunks_file.name}")
        return str(chunks_file)
    
    # Load transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)
    
    segments = transcript_data.get("segments", [])
    video_name = transcript_data.get("video_name", transcript_path_obj.stem)
    
    logger.info(f"Processing transcript: {video_name}")
    logger.info(f"Total segments: {len(segments)}")
    
    # Get chunking settings
    chunking_config = config.get_section("chunking")
    method = chunking_config.get("method", "semantic")
    
    if method == "semantic":
        semantic_config = chunking_config.get("semantic", {})
        chunks = semantic_segmentation(
            segments,
            similarity_threshold=semantic_config.get("similarity_threshold", 0.3),
            min_chunk_words=semantic_config.get("min_chunk_words", 50),
            max_chunk_words=semantic_config.get("max_chunk_words", 300)
        )
    else:
        fixed_config = chunking_config.get("fixed", {})
        chunks = fixed_chunking(
            segments,
            chunk_size=fixed_config.get("chunk_size_words", 150),
            overlap=fixed_config.get("overlap_words", 40)
        )
    
    # Add metadata
    final_chunks = []
    for i, chunk in enumerate(chunks):
        final_chunks.append({
            "chunk_id": i,
            "video_id": video_name,
            "start": chunk["start"],
            "end": chunk["end"],
            "duration": round(chunk["end"] - chunk["start"], 2),
            "word_count": chunk.get("word_count", len(chunk["text"].split())),
            "text": chunk["text"]
        })
    
    # Save chunks
    output_data = {
        "video_id": video_name,
        "total_chunks": len(final_chunks),
        "chunks": final_chunks
    }
    
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Chunks saved: {chunks_file.name}")
    logger.info(f"Created {len(final_chunks)} chunks")
    
    return str(chunks_file)


def process_all_transcripts(
    transcript_dir: str = "data/transcripts",
    output_dir: str = "data/chunks"
) -> List[str]:
    """
    Process all transcripts in a directory to chunks.
    
    Args:
        transcript_dir: Directory containing transcripts
        output_dir: Output directory for chunks
        
    Returns:
        List of processed chunk file paths
    """
    logger.info("Processing all transcripts...")
    
    transcript_path = Path(transcript_dir)
    if not transcript_path.exists():
        logger.error(f"Transcript directory not found: {transcript_dir}")
        return []
    
    processed = []
    for transcript_file in transcript_path.glob("*.json"):
        result = process_transcript_to_chunks(
            str(transcript_file),
            output_dir
        )
        if result:
            processed.append(result)
    
    logger.info(f"Processed {len(processed)} transcripts")
    return processed


if __name__ == "__main__":
    # Test chunking
    import sys
    
    if len(sys.argv) > 1:
        transcript = sys.argv[1]
    else:
        transcript = input("Enter transcript path: ")
    
    result = process_transcript_to_chunks(transcript)
    print(f"\nOutput: {result}")
