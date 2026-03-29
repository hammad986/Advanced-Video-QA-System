"""
Video Processing Module for Video-QA System
Handles video input and audio extraction using ffmpeg.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from .config import config
from .logger import get_logger


logger = get_logger(__name__)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available in system PATH."""
    if shutil.which("ffmpeg") is None:
        logger.error("FFmpeg not found. Please install FFmpeg and add it to system PATH.")
        return False
    return True


def get_video_files(directory: str) -> list:
    """Get all supported video files from a directory."""
    supported = config.get("video.supported_formats", [".mp4", ".mkv", ".avi", ".mov", ".webm"])
    videos = []
    
    dir_path = Path(directory)
    if dir_path.exists():
        for f in dir_path.iterdir():
            if f.is_file() and f.suffix.lower() in supported:
                videos.append(str(f))
    
    return sorted(videos)


def take_video_input(video_dir: Optional[str] = None) -> Optional[str]:
    """
    Prompt user for video file path.
    
    Args:
        video_dir: Optional directory to look for videos
        
    Returns:
        Path to video file or None if cancelled
    """
    video_dir = video_dir or config.get("video.input_dir", "data/videos")
    supported = config.get("video.supported_formats", [".mp4", ".mkv", ".avi", ".mov", ".webm"])
    
    print("\n" + "="*50)
    print("VIDEO INPUT")
    print("="*50)
    print(f"\nPaste full video path (example: D:\\lecture.mp4)")
    print(f"Or drag & drop the video file into terminal and press Enter")
    print(f"Supported formats: {', '.join(supported)}")
    
    if video_dir and os.path.exists(video_dir):
        existing = get_video_files(video_dir)
        if existing:
            print(f"\nOr choose from existing videos in {video_dir}:")
            for i, v in enumerate(existing, 1):
                print(f"  {i} - {os.path.basename(v)}")
            print(f"  0 - Enter new path")
    
    video_path = input("\nVideo path: ").strip().strip('"')
    
    # Check if user selected existing video
    if video_dir and os.path.exists(video_dir):
        existing = get_video_files(video_dir)
        try:
            idx = int(video_path)
            if 1 <= idx <= len(existing):
                video_path = existing[idx - 1]
        except ValueError:
            pass
    
    if not os.path.exists(video_path):
        logger.error(f"File not found: {video_path}")
        return None
    
    if not video_path.lower().endswith(tuple(supported)):
        logger.error(f"Unsupported video format: {video_path}")
        return None
    
    # Copy to input directory
    input_dir = Path(video_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    
    filename = os.path.basename(video_path)
    dest_path = input_dir / filename
    
    if not dest_path.exists():
        shutil.copy(video_path, dest_path)
        logger.info(f"Video imported: {filename}")
    else:
        logger.info(f"Video already exists: {filename}")
    
    return str(dest_path)


def convert_video_to_audio(
    video_path: str,
    audio_dir: Optional[str] = None,
    force: bool = False
) -> Optional[str]:
    """
    Convert video file to audio (16kHz mono WAV).
    
    Args:
        video_path: Path to video file
        audio_dir: Output directory for audio
        force: Force re-conversion even if file exists
        
    Returns:
        Path to audio file or None on failure
    """
    if not check_ffmpeg():
        return None
    
    audio_dir = audio_dir or config.get("video.output_dir", "data/audio")
    audio_settings = config.get("video.audio_settings", {})
    
    audio_path = Path(audio_dir)
    audio_path.mkdir(parents=True, exist_ok=True)
    
    filename = Path(video_path).stem
    output_path = audio_path / f"{filename}.wav"
    
    # Skip if already converted (unless force)
    if output_path.exists() and not force:
        logger.info(f"Audio already exists: {output_path.name}")
        return str(output_path)
    
    logger.info(f"Extracting audio from: {Path(video_path).name}")
    
    # Build ffmpeg command
    command = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-i", video_path,
        "-vn",  # No video
        "-ac", str(audio_settings.get("channels", 1)),  # Mono
        "-ar", str(audio_settings.get("sample_rate", 16000)),  # 16kHz
        "-acodec", audio_settings.get("codec", "pcm_s16le"),  # PCM
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Audio saved: {output_path.name}")
        return str(output_path)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Audio extraction failed: {e.stderr}")
        return None


def process_video_pipeline(video_path: str) -> Optional[str]:
    """
    Full pipeline: video input → audio extraction.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to extracted audio file or None on failure
    """
    logger.info("="*50)
    logger.info("VIDEO PROCESSING PIPELINE")
    logger.info("="*50)
    
    # Step 1: Validate video
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return None
    
    # Step 2: Convert to audio
    audio_path = convert_video_to_audio(video_path)
    
    if audio_path:
        logger.info("Video processing complete!")
    else:
        logger.error("Video processing failed!")
    
    return audio_path


if __name__ == "__main__":
    # Test video processing
    video = take_video_input()
    if video:
        audio = convert_video_to_audio(video)
        print(f"\nOutput: {audio}")
