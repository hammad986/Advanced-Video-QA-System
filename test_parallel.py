import os
import json
import time
import pprint
from pathlib import Path
from video_qa.pipeline import VideoQAPipeline

def test_system():
    pipeline = VideoQAPipeline()
    test_video = "data/videos/01_index.mp4"
    
    # ── 1. TEST PROCESSING ──
    print(f"\n[TEST 1] Processing Video: {test_video}")
    t0 = time.time()
    vid_id = pipeline.process_video(test_video)
    t1 = time.time()
    print(f"✅ Processing finished in {t1 - t0:.2f} seconds")
    
    # ── 2. TEST TEXT QUALITY ──
    transcript_file = f"data/transcripts/{vid_id}.json"
    with open(transcript_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("\n[TEST 2] Transcript segments check:")
    segments = data.get("segments", [])
    if segments:
        print(f"Total segments: {len(segments)}")
        print(f"First 3 segments:\n{pprint.pformat(segments[:3])}")
        print(f"Last 3 segments:\n{pprint.pformat(segments[-3:])}")
    
    # ── 3. TEST CACHING ──
    print("\n[TEST 3] Running processing AGAIN to test MD5 Cache Hit...")
    t2 = time.time()
    vid_id_cached = pipeline.process_video(test_video)
    t3 = time.time()
    print(f"✅ Cache processing finished in {t3 - t2:.2f} seconds")
    assert (t3 - t2) < 5.0, "Cache hit should be near instant!"
    
    # ── 4. TEST SUMMARY MODE ──
    print("\n[TEST 4] Requesting Lecture Summary...")
    summary_result = pipeline.ask("Summarize the main topics", active_video_id=vid_id)
    print("Summary Output Keys:", summary_result.keys())
    print("Mode:", "Summary" if summary_result.get("is_summary") else "QA")
    print("Confidence:", summary_result.get("confidence"))
    print("Verification Status:", summary_result.get("verification", {}).get("status"))
    
    # ── 5. TEST QA MODE ──
    print("\n[TEST 5] Requesting Factual RAG Q&A...")
    qa_result = pipeline.ask("What is feature scaling?", active_video_id=vid_id)
    print("QA Output Keys:", qa_result.keys())
    print("Mode:", "Summary" if qa_result.get("is_summary") else "QA")
    print("Confidence:", qa_result.get("confidence"))
    print("Verification Status:", qa_result.get("verification", {}).get("status"))

if __name__ == "__main__":
    test_system()
