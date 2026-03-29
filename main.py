"""
Video Knowledge Retrieval AI System
Research-Level RAG Pipeline with Hallucination-Free Answer Generation

This system implements all 5 key upgrades:
1. WhisperX forced alignment + transcript correction
2. Semantic segmentation (topic-based chunking)
3. BGE retrieval-optimized embeddings
4. Cross-encoder re-ranking (bge-reranker-large)
5. Evidence verification (anti-hallucination guard)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from video_qa.config import config
from video_qa.logger import setup_logger, get_logger
from video_qa.video_processor import take_video_input, convert_video_to_audio
from video_qa.speech_understanding import transcribe_audio
from video_qa.knowledge_structuring import process_transcript_to_chunks
from video_qa.embeddings import build_vector_index
from video_qa.retrieval import RetrievalSystem
from video_qa.reranker import ReRanker
from video_qa.answer_generator import AnswerGenerator


def print_header():
    print("""
=================================================
   Video Knowledge Retrieval AI System
   (Research-Level RAG Pipeline)
   Hallucination-Free & Evidence-Grounded
=================================================
""")


def process_pipeline():
    """Process video through the complete pipeline."""
    print("\n" + "="*50)
    print("STEP 1: Video Input")
    print("="*50)
    
    video_path = take_video_input()
    
    if video_path is None:
        print("Video import failed.")
        return False
    
    print("\n" + "="*50)
    print("STEP 2: Audio Extraction")
    print("="*50)
    audio_path = convert_video_to_audio(video_path)
    
    if audio_path is None:
        print("Audio extraction failed.")
        return False
    
    print("\n" + "="*50)
    print("STEP 3: Speech Recognition (WhisperX)")
    print("="*50)
    transcript_path = transcribe_audio(audio_path)
    
    if transcript_path is None:
        print("Transcription failed.")
        return False
    
    print("\n" + "="*50)
    print("STEP 4: Semantic Chunking")
    print("="*50)
    chunks_path = process_transcript_to_chunks(transcript_path)
    
    if chunks_path is None:
        print("Chunking failed.")
        return False
    
    print("\n" + "="*50)
    print("STEP 5: Building Vector Index (BGE)")
    print("="*50)
    success = build_vector_index()
    
    if not success:
        print("Index building failed.")
        return False
    
    print("\n" + "="*50)
    print("System Ready!")
    print("="*50)
    return True


def run_query_pipeline():
    """Run the query pipeline: retrieval → re-ranking → answer → verification."""
    print("\n" + "="*50)
    print("QUERY PIPELINE")
    print("="*50)
    
    # Initialize components
    retriever = RetrievalSystem()
    reranker = ReRanker()
    answer_gen = AnswerGenerator()
    
    # Check if index exists
    if not retriever.is_available():
        print("\nNo vector index found.")
        print("Please process videos first (Option 1 or 4).")
        return
    
    query = input("\nAsk a question about the video: ").strip()
    
    if not query:
        print("Empty question.")
        return
    
    print(f"\nQuery: {query}")
    print("-"*40)
    
    # Stage 1: Retrieval (top-20)
    print("\n[1] Retrieving relevant passages...")
    retrieved = retriever.retrieve(query, top_k=20)
    
    if not retrieved:
        print("\n===== ANSWER =====\n")
        print("NOT FOUND IN VIDEO")
        return
    
    print(f"Retrieved {len(retrieved)} passages")
    
    # Stage 2: Re-Ranking (top-3)
    print("\n[2] Re-ranking passages...")
    reranked = reranker.rerank(query, retrieved, top_k=3)
    print(f"Re-ranked to top {len(reranked)} passages")
    
    # Stage 3: Answer Generation
    print("\n[3] Generating answer...")
    result = answer_gen.generate(query, reranked)
    
    # Stage 4: Output
    print("\n" + "="*40)
    print("ANSWER")
    print("="*40)
    
    answer = result.get('answer', 'NOT FOUND IN VIDEO')
    timestamp = result.get('timestamp')
    verified = result.get('verified', False)
    
    print(answer)
    
    if timestamp:
        print(f"\nTimestamp: {timestamp}")
    
    if verified:
        print("\n[✓] Answer verified against evidence")
    
    print("="*40)


def show_suggestions():
    print("""
Try asking:

• What is the main topic of the lecture?
• Explain the key concept discussed.
• Where does the speaker define the topic?
• What example is given in the lecture?
• Summarize the beginning of the video.

(Ask only about the video content)
""")


def main_menu():
    """Main menu loop."""
    while True:
        print("""
=================================================
MAIN MENU
=================================================

1 → Process New Video (Full Pipeline)
2 → Ask Question (Interactive)
3 → Full Pipeline (Process + Ask)
4 → Exit
""")
        
        choice = input("Enter choice: ").strip()
        
        if choice == "1":
            process_pipeline()
        
        elif choice == "2":
            show_suggestions()
            run_query_pipeline()
        
        elif choice == "3":
            success = process_pipeline()
            if success:
                show_suggestions()
                run_query_pipeline()
        
        elif choice == "4":
            print("Exiting system.")
            sys.exit()
        
        else:
            print("Invalid choice.\n")


if __name__ == "__main__":
    # Setup logging
    setup_logger()
    
    # Print header
    print_header()
    
    # Run main menu
    main_menu()
