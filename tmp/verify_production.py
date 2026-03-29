"""
End-to-end verification of the simplified accurate Video QA pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from video_qa.answer_generator import generate_answer, format_time_range
from video_qa.query_rewriter import rewrite_query

MOCK_CONTEXTS = [
    {
        "id": 0,
        "text": "Transformers are a type of neural network architecture that rely on self-attention mechanisms to process sequential data.",
        "start": 60.0,
        "end": 90.0,
        "score": 0.82,
    },
    {
        "id": 1,
        "text": "Deep learning requires large amounts of labeled training data to achieve high accuracy.",
        "start": 91.0,
        "end": 120.0,
        "score": 0.61,
    }
]

QUERY = "What architecture do Transformers use?"

print("\n=== TEST 1: Query Rewrite ===")
rewritten = rewrite_query(QUERY, force=True)
print(f"Original: {QUERY}")
print(f"Rewritten: {rewritten}")

print("\n=== TEST 2: Answer Generation ===")
result = generate_answer(rewritten, MOCK_CONTEXTS)
print(f"Answer          : {result.get('answer')}")
print(f"Timestamp       : {result.get('timestamp')}")
print(f"Retrieval Score : {result.get('retrieval_score')}")

top_chunk_ts = format_time_range(MOCK_CONTEXTS[0]["start"], MOCK_CONTEXTS[0]["end"])
assert result.get('timestamp') == top_chunk_ts, f"Expected {top_chunk_ts}, got {result.get('timestamp')}"
print("PASS -- Timestamp correctly matches top chunk metadata.")
