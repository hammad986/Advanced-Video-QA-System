"""
Video-QA: Research-Level Video Question Answering System

A modular RAG pipeline for answering questions about lecture videos
with hallucination-resistant, evidence-grounded responses.
"""

__version__ = "1.0.0"
__author__ = "Video-QA Research Team"

from .config import Config
from .logger import setup_logger
from .pipeline import VideoQAPipeline

__all__ = [
    "Config",
    "setup_logger", 
    "VideoQAPipeline",
]
