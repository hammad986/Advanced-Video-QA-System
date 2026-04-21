from pathlib import Path
import json
import joblib
import faiss

from video_qa.embeddings import load_embedding_model, create_embeddings, VectorStore
from video_qa.pipeline import VideoQAPipeline
from video_qa.retrieval import RetrievalSystem
from video_qa.config import config

root = Path('.')
index_path = Path(config.get('retrieval.index_path', 'models/video_index.faiss'))
meta_path = Path(config.get('retrieval.metadata_path', 'models/metadata.pkl'))

print('EMBEDDINGS_INSTALLED=', end='')
try:
    from sentence_transformers import SentenceTransformer
    print('yes')
except Exception as e:
    print(f'no: {e}')

model = load_embedding_model()
print('EMBEDDING_MODEL_LOADED=', bool(model))
if model is not None:
    sample = create_embeddings(['The lecture explains gradient descent and optimization.'])
    print('SAMPLE_EMBEDDING_SHAPE=', getattr(sample, 'shape', None))

print('INDEX_EXISTS=', index_path.exists())
print('META_EXISTS=', meta_path.exists())

if index_path.exists() and meta_path.exists():
    index = faiss.read_index(str(index_path))
    metadata = joblib.load(str(meta_path))
    print('VECTOR_COUNT=', index.ntotal)
    print('METADATA_COUNT=', len(metadata))
    print('SAMPLE_METADATA=')
    for i, item in enumerate(metadata[:3]):
        print(json.dumps({k: item.get(k) for k in ['video_id', 'chunk_id', 'start', 'end', 'score', 'text']}, ensure_ascii=False)[:1000])

    retriever = RetrievalSystem()
    query = 'What is the main topic of the lecture?'
    results = retriever.retrieve(query, top_k=5, video_id=None)
    print('RETRIEVAL_QUERY=', query)
    print('RETRIEVAL_COUNT=', len(results))
    for i, r in enumerate(results[:5], 1):
        print(json.dumps({
            'rank': i,
            'score': r.get('score'),
            'video_id': r.get('video_id'),
            'start': r.get('start'),
            'end': r.get('end'),
            'text': r.get('text', '')[:220]
        }, ensure_ascii=False))

    pipeline = VideoQAPipeline()
    qa = pipeline.ask('What is the main topic of the lecture?', active_video_id=None)
    print('FULL_QA=')
    print(json.dumps({
        'answer': qa.get('answer'),
        'confidence': qa.get('confidence'),
        'confidence_label': qa.get('confidence_label'),
        'provider': qa.get('provider'),
        'timestamp': qa.get('timestamp'),
        'contexts': [
            {'score': c.get('score'), 'start': c.get('start'), 'end': c.get('end'), 'text': c.get('text', '')[:160]}
            for c in (qa.get('contexts') or [])[:3]
        ]
    }, ensure_ascii=False, indent=2))
else:
    print('PIPELINE NOT FUNCTIONAL')
