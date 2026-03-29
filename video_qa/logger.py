"""
Logging module for Video-QA system.
Provides structured logging with file and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from .config import config


def setup_logger(
    name: str = "video_qa",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    # Get settings from config
    log_level = level or config.get("logging.level", "INFO")
    log_format = config.get(
        "logging.format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file_path = log_file or config.get("logging.file", "video_qa.log")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file_path:
        try:
            file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_formatter = logging.Formatter(log_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Could not create file handler: {e}")
    
    return logger


# Default logger instance
default_logger = setup_logger()


def get_logger(name: str = "video_qa") -> logging.Logger:
    """Get or create a logger with the given name"""
    return logging.getLogger(name)
