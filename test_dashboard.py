import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from video_qa.pipeline import VideoQAPipeline
from video_qa.logger import setup_logger

setup_logger()

def main():
    pipe = VideoQAPipeline()
    
    if not pipe.retriever.is_available():
        print("ERROR: Vector index not available. Need to process a video first.")
        # Let's process the directory to be sure
        print("Processing directory to build index...")
        pipe.process_directory()
        
    if not pipe.retriever.is_available():
        print("STILL NO INDEX. Exiting.")
        return
        
    print("\n--- Running Test Query ---")
    query = "What is machine learning?"
    result = pipe.ask(query)
    
    print("\n\n=== FINAL RESULT DICT KEYS ===")
    print(list(result.keys()))
    
    print("\n=== CONFIDENCE DATA ===")
    print("Score:", result.get("confidence"))
    print("Label:", result.get("confidence_label"))
    print("Explanation:", result.get("confidence_explanation"))
    
    print("\n=== VERIFICATION DATA ===")
    v_data = result.get("verification", {})
    print("Status:", v_data.get("status"))
    print("Justification:", v_data.get("justification"))
    print("Method:", v_data.get("method"))
    
    print("\n=== EVIDENCE ===")
    print("Contexts count:", len(result.get("contexts", [])))
    print("All contexts count:", len(result.get("all_contexts", [])))
    print("Useful chunks:", result.get("useful_chunks"))

if __name__ == "__main__":
    main()
