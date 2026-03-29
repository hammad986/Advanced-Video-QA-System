import os
import json
import re
import tempfile
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
import concurrent.futures

from .config import config
from .logger import get_logger

logger = get_logger(__name__)

# Global model cache and lock for singleton
_whisper_model = None
_model_lock = threading.Lock()
_align_model = None


def load_whisper_model():
    """Load Whisper model (singleton, thread-safe)."""
    global _whisper_model
    with _model_lock:
        if _whisper_model is None:
            speech_config = config.get_section("speech")
            # Force small for performance optimization
            model_size = speech_config.get("model_size", "small")
            device = speech_config.get("device", "cpu")
            compute_type = speech_config.get("compute_type", "int8")
            
            try:
                from faster_whisper import WhisperModel
                logger.info(f"Loading Whisper model: {model_size}")
                _whisper_model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type
                )
                logger.info("Whisper model loaded successfully")
            except ImportError:
                logger.warning("faster-whisper not installed, falling back to whisper")
                import whisper
                _whisper_model = whisper.load_model(model_size, device=device)
    
    return _whisper_model


def load_alignment_model():
    """Load WhisperX alignment model."""
    global _align_model
    if _align_model is None:
        try:
            import whisperx
            logger.info("Loading WhisperX alignment model...")
            _align_model = whisperx.load_align_model(language_code="en")
            logger.info("Alignment model loaded")
        except ImportError:
            logger.warning("whisperx not available, skipping alignment")
    
    return _align_model


def _transcribe_chunk(chunk_path: str, offset: float, chunk_idx: int) -> Dict[str, Any]:
    """Transcribe a single audio chunk and return offset-adjusted segments."""
    model = load_whisper_model()
    speech_config = config.get_section("speech")
    
    try:
        is_faster = "faster_whisper" in str(type(model)).lower()
        segment_list = []
        full_text = ""
        language = "en"
        language_probability = 1.0

        if is_faster:
            segments, info = model.transcribe(
                chunk_path,
                language="en",
                beam_size=speech_config.get("beam_size", 1),
                best_of=speech_config.get("best_of", 1),
                temperature=speech_config.get("temperature", 0),
                vad_filter=speech_config.get("vad_filter", True),
                vad_parameters=dict(min_silence_duration_ms=speech_config.get("vad_min_silence_ms", 700))
            )
            language = info.language
            language_probability = float(info.language_probability)
            
            for seg in segments:
                text = getattr(seg, "text", "").strip()
                start = float(getattr(seg, "start", 0)) + offset
                end = float(getattr(seg, "end", 0)) + offset
                
                # ── USER'S EXPLICIT MANDATORY MERGE LOGIC ──
                # Chunk A (0-40s): Keep end <= 40s (which is offset + 40).
                # Chunk B (30-70s) etc: Keep start >= overlap_start (which is offset + 10).
                overlap_start = offset + 10.0
                overlap_end = offset + 40.0
                
                keep = True
                if chunk_idx == 0:
                    if end > overlap_end:
                        keep = False
                else:
                    if start < overlap_start:
                        keep = False

                if keep:
                    segment_list.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })
        else:
            out = model.transcribe(
                chunk_path,
                language="en",
                beam_size=speech_config.get("beam_size", 1),
                best_of=speech_config.get("best_of", 1),
                temperature=speech_config.get("temperature", 0)
            )
            language = out.get("language", "en")
            for seg in out.get("segments", []):
                text = seg.get("text", "").strip()
                start = float(seg.get("start", 0)) + offset
                end = float(seg.get("end", 0)) + offset
                
                overlap_start = offset + 10.0
                overlap_end = offset + 40.0
                
                keep = True
                if chunk_idx == 0:
                    if end > overlap_end:
                        keep = False
                else:
                    if start < overlap_start:
                        keep = False

                if keep:
                    segment_list.append({
                        "start": start,
                        "end": end,
                        "text": text
                    })
        
        return {
            "language": str(language),
            "language_probability": float(language_probability),
            "segments": segment_list
        }
    except Exception as e:
        logger.error(f"Chunk transcription failed at offset {offset}: {e}")
        return {"language": "en", "language_probability": 0.0, "segments": []}


def transcribe_with_whisper(audio_path: str) -> Optional[Dict[str, Any]]:
    """
    Transcribe audio using parallel overlapping Whisper chunks.
    Chunk size: 40s, Overlap: 10s.
    """
    logger.info(f"Parallel chunking and transcribing: {audio_path}")
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
    except ImportError:
        logger.error("pydub not installed. Please install it (pip install pydub) for parallel chunking.")
        return None
    except Exception as e:
        logger.error(f"Failed to load audio for chunking: {e}")
        return None

    duration_ms = len(audio)
    chunk_size_ms = 40000
    stride_ms = 30000
    
    chunks_info = []
    chunk_idx = 0
    for start_ms in range(0, duration_ms, stride_ms):
        end_ms = start_ms + chunk_size_ms
        chunk = audio[start_ms:end_ms]
        
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        chunk.export(temp_path, format="wav")
        
        chunks_info.append({
            "idx": chunk_idx,
            "path": temp_path,
            "offset": start_ms / 1000.0
        })
        chunk_idx += 1
        
        # If this chunk reached the end, stop
        if end_ms >= duration_ms:
            break

    logger.info(f"Created {len(chunks_info)} audio chunks for parallel processing.")
    
    # Process in parallel
    all_segments = []
    language_probs = []
    default_lang = "en"
    
    # Start thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_chunk = {
            executor.submit(_transcribe_chunk, ci["path"], ci["offset"], ci["idx"]): ci 
            for ci in chunks_info
        }
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            ci = future_to_chunk[future]
            try:
                res = future.result()
                all_segments.extend(res["segments"])
                if res["language_probability"] > 0:
                    language_probs.append((res["language"], res["language_probability"]))
                
                # Cleanup temp file
                os.unlink(ci["path"])
            except Exception as e:
                logger.error(f"Error processing chunk {ci['idx']}: {e}")
                if os.path.exists(ci["path"]):
                    os.unlink(ci["path"])

    # Sort segments by start time to maintain global continuity
    all_segments.sort(key=lambda x: x["start"])
    
    # Compute full text
    full_text = " ".join([s["text"] for s in all_segments]).strip()
    
    # Determine majority language
    final_lang = "en"
    final_prob = 1.0
    if language_probs:
        langs = {}
        for l, p in language_probs:
            langs[l] = langs.get(l, 0) + p
        final_lang = max(langs.items(), key=lambda x: x[1])[0]
        final_prob = sum(p for l, p in language_probs if l == final_lang) / len(language_probs)

    result = {
        "language": str(final_lang),
        "language_probability": float(final_prob),
        "full_text": full_text,
        "segments": all_segments
    }
    
    logger.info(f"Transcription complete: {len(all_segments)} merged segments")
    return result


def align_transcript(audio_path: str, transcript: Dict[str, Any]) -> Dict[str, Any]:
    align_model = load_alignment_model()
    if align_model is None:
        logger.info("Skipping alignment (whisperx not available)")
        return transcript
    
    try:
        import whisperx
        aligned = whisperx.align(transcript["segments"], align_model, "en", audio_path)
        if aligned and "segments" in aligned:
            transcript["segments"] = aligned["segments"]
            logger.info("Forced alignment complete")
    except Exception as e:
        logger.warning(f"Alignment failed: {e}")
    
    return transcript


def restore_punctuation(text: str) -> str:
    """Robust regex-based punctuation restoration for Whisper transcripts."""
    if not text:
        return text
    
    # 1. Capitalize first letter
    text = text[0].upper() + text[1:]
    
    # 2. Add period between a lowercase word and an uppercase word
    # e.g., "learning is fun And this is" -> "learning is fun. And this is"
    text = re.sub(r'([a-z])\s+([A-Z])', r'\1. \2', text)
    
    # 3. Ensure sentence ends with punctuation
    if text[-1] not in '.!?':
        text = text + '.'
        
    # 4. Cleanup trailing or duplicate spaces/periods
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    return text.strip()


def correct_transcript_with_llm(transcript: str) -> str:
    try:
        import requests
        correction_prompt = f"Correct any grammar errors in the transcript. Only output corrected text.\n{transcript}\nCorrected:"
        response = requests.post(
            config.get("answer.ollama_url", "http://localhost:11434/api/generate"),
            json={"model": config.get("answer.local_model", "phi3:mini"), "prompt": correction_prompt, "stream": False},
            timeout=60
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"LLM correction failed: {e}")
    return transcript


def transcribe_audio(audio_path: str, output_dir: Optional[str] = None) -> Optional[str]:
    logger.info("="*50)
    logger.info("SPEECH UNDERSTANDING PIPELINE (PARALLEL)")
    logger.info("="*50)
    
    audio_path_obj = Path(audio_path)
    output_dir = output_dir or "data/transcripts"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    transcript_file = output_path / f"{audio_path_obj.stem}.json"
    
    if transcript_file.exists():
        logger.info(f"Transcript already exists: {transcript_file.name}")
        return str(transcript_file)
    
    logger.info("Step 1: Whisper parallel transcription...")
    transcript = transcribe_with_whisper(audio_path)
    
    if not transcript:
        logger.error("Transcription failed")
        return None
    
    if config.get("speech.alignment", True):
        logger.info("Step 2: WhisperX forced alignment...")
        transcript = align_transcript(audio_path, transcript)
    
    if config.get("speech.restore_punctuation", True):
        logger.info("Step 3: Restoring punctuation...")
        for seg in transcript["segments"]:
            seg["text"] = restore_punctuation(seg["text"])
        transcript["full_text"] = restore_punctuation(transcript["full_text"])
    
    if config.get("speech.llm_correction", False):
        logger.info("Step 4: LLM correction pass...")
        transcript["full_text"] = correct_transcript_with_llm(transcript["full_text"])
    
    total_duration = 0
    if transcript["segments"]:
        total_duration = transcript["segments"][-1]["end"]
    word_count = len(transcript["full_text"].split())
    
    output_data = {
        "video_name": audio_path_obj.stem,
        "language": transcript.get("language", "en"),
        "language_confidence": transcript.get("language_probability", 0),
        "total_segments": len(transcript["segments"]),
        "total_duration": round(total_duration, 2),
        "word_count": word_count,
        "full_transcript": transcript["full_text"],
        "segments": transcript["segments"]
    }
    
    with open(transcript_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Transcript saved: {transcript_file.name}")
    return str(transcript_file)
