"""
Rebuild FAISS index using all-MiniLM-L6-v2 from existing chunk files.
Run this from the project root: python rebuild_index.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Remove old index so build_vector_index() rebuilds from scratch
import os
old_idx = Path("models/video_index.faiss")
old_meta = Path("models/metadata.pkl")
if old_idx.exists():
    os.remove(old_idx)
    print(f"Removed old index: {old_idx}")
if old_meta.exists():
    os.remove(old_meta)
    print(f"Removed old metadata: {old_meta}")

# Rebuild using the pipeline's own function (reads config automatically)
from video_qa.embeddings import build_vector_index
from video_qa.logger import setup_logger

setup_logger()
print("\nBuilding new FAISS index with all-MiniLM-L6-v2 (384-dim)...")
success = build_vector_index(chunks_dir="data/chunks", model_dir="models")

if success:
    print("\n=== SUCCESS: Index rebuilt! ===")
    # Sanity test
    import faiss, joblib
    import numpy as np
    from sentence_transformers import SentenceTransformer
    
    idx = faiss.read_index("models/video_index.faiss")
    meta = joblib.load("models/metadata.pkl")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_sentence_embedding_dimension()
    print(f"Index dim={idx.d}, vectors={idx.ntotal}, model_dim={dim}, match={idx.d == dim}")

    q = "what is machine learning"
    qvec = model.encode([q], normalize_embeddings=True).astype("float32")
    scores, ids = idx.search(qvec, 3)
    print("\nTop 3 results for 'what is machine learning':")
    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            print(f"  score={s:.4f}  |  {meta[i]['text'][:120]}")
else:
    print("\n=== FAILED: Check logs above ===")
    sys.exit(1)
