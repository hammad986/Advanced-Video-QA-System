"""
Configuration module for Video-QA system.
Loads settings from config.yaml.

Fix: Config.get() now checks the file modification time on every call.
If config.yaml has been saved since the last load, it is reloaded
automatically — no process restart needed.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# BASE_DIR is the project root (parent of video_qa folder)
BASE_DIR = Path(__file__).parent.parent


def running_on_streamlit_cloud():
    """Check if the app is running on Streamlit Cloud."""
    try:
        import streamlit as st
        # System variable set by Streamlit Cloud
        return st.secrets.get("CLOUD", False) or os.getenv("STREAMLIT_CLOUD_DEPLOY") == "true"
    except Exception:
        return False


class Config:
    """Configuration manager for Video-QA system."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}
    _config_path: Optional[Path] = None
    _last_mtime: float = 0.0          # filesystem mtime at last load

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _expand_env_vars(self, data: Any) -> Any:
        """Recursively expand environment variables in config data.

        Checks Streamlit secrets first (for cloud), then environment variables.
        """
        import re
        if isinstance(data, str):
            # Check if string matches ${VAR_NAME} pattern
            match = re.match(r'^\$\{([^}]+)\}$', data)
            if match:
                var_name = match.group(1)
                
                # 1. Try Streamlit Secrets (Cloud priority)
                try:
                    import streamlit as st
                    if var_name in st.secrets:
                        return st.secrets[var_name]
                except Exception:
                    pass

                # 2. Try OS environment
                value = os.getenv(var_name)
                if value is None:
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        f"Env var '{var_name}' not set — provider will be skipped."
                    )
                    return ""
                return value
            else:
                return data
        elif isinstance(data, dict):
            return {key: self._expand_env_vars(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]
        else:
            return data


    def _load_config(self):
        """Load configuration from config.yaml"""
        # Only load .env if NOT on cloud
        if not running_on_streamlit_cloud():
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except ImportError:
                pass
        
        config_path = BASE_DIR / "config.yaml"
        self._config_path = config_path

        if not config_path.exists():
            # Create default config if not exists
            self._config = self._get_default_config()
            self._save_config(config_path)
            self._last_mtime = config_path.stat().st_mtime
        else:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f) or {}
            # Expand environment variables in the loaded config
            self._config = self._expand_env_vars(loaded_config)
            self._last_mtime = config_path.stat().st_mtime

    def reload(self) -> None:
        """Force a fresh reload of config.yaml from disk."""
        if self._config_path and self._config_path.exists():
            with open(self._config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
            self._last_mtime = self._config_path.stat().st_mtime

    def _check_reload(self) -> None:
        """Silently reload config.yaml if the file has been modified."""
        if self._config_path and self._config_path.exists():
            try:
                mtime = self._config_path.stat().st_mtime
                if mtime != self._last_mtime:
                    self.reload()
            except OSError:
                pass
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration with absolute paths."""
        return {
            "video": {
                "input_dir": str(BASE_DIR / "data/videos"),
                "output_dir": str(BASE_DIR / "data/audio"),
                "supported_formats": [".mp4", ".mkv", ".avi", ".mov", ".webm"],
                "audio_settings": {
                    "sample_rate": 16000,
                    "channels": 1,
                    "codec": "pcm_s16le"
                }
            },
            "speech": {
                "model_size": "small",
                "device": "cpu",
                "compute_type": "int8",
                "beam_size": 1,
                "best_of": 1,
                "temperature": 0,
                "use_whisperx": True,
                "alignment": True,
                "vad_filter": True,
                "vad_min_silence_ms": 700,
                "restore_punctuation": True,
                "llm_correction": True
            },
            "chunking": {
                "method": "semantic",
                "semantic": {
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "similarity_threshold": 0.3,
                    "min_chunk_words": 50,
                    "max_chunk_words": 300
                },
                "fixed": {
                    "chunk_size_words": 150,
                    "overlap_words": 40
                }
            },
            "embeddings": {
                "model_name": "BAAI/bge-large-en-v1.5",
                "normalize": True,
                "batch_size": 32,
                "device": "cpu"
            },
            "retrieval": {
                "vector_db": "faiss",
                "top_k": 20,
                "min_score": 0.30,
                "index_path": str(BASE_DIR / "models/video_index.faiss"),
                "metadata_path": str(BASE_DIR / "models/metadata.pkl")
            },
            "reranking": {
                "enabled": True,
                "model_name": "BAAI/bge-reranker-large",
                "top_k": 3,
                "device": "cpu"
            },
            "answer": {
                "use_local": True,
                "local_model": "phi3:mini",
                "ollama_url": "http://localhost:11434/api/generate",
                "timeout": 180,
                "use_openai": True,
                "openai_model": "gpt-4",
                "openai_api_key": "${OPENAI_API_KEY}",
                "use_gemini": True,
                "gemini_api_key": "${GEMINI_API_KEY}",
                "gemini_model": "gemini-1.5-flash",
                "use_huggingface": True,
                "hf_model": "meta-llama/Llama-3.2-3B-Instruct",
                "hf_token": "${HF_TOKEN}",
                "strict_mode": True,
                "extractive": True
            },
            "verification": {
                "enabled": True,
                "local_model": "phi3:mini",
                "ollama_url": "http://localhost:11434/api/generate",
                "timeout": 120
            },
            "output": {
                "format": "structured",
                "include_citations": True,
                "include_timestamps": True
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "video_qa.log"
            }
        }
    
    def _save_config(self, path: Path):
        """Save default configuration to file"""
        with open(path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'video.input_dir').

        Automatically reloads config.yaml if it has been modified on disk
        since the last read — no process restart required.
        """
        self._check_reload()

        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        # Treat an empty / whitespace-only string as "not set" (return default)
        if isinstance(value, str) and not value.strip():
            return default

        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self._config.get(section, {})
    
    @property
    def video(self) -> Dict[str, Any]:
        return self._config.get("video", {})
    
    @property
    def speech(self) -> Dict[str, Any]:
        return self._config.get("speech", {})
    
    @property
    def chunking(self) -> Dict[str, Any]:
        return self._config.get("chunking", {})
    
    @property
    def embeddings(self) -> Dict[str, Any]:
        return self._config.get("embeddings", {})
    
    @property
    def retrieval(self) -> Dict[str, Any]:
        return self._config.get("retrieval", {})
    
    @property
    def reranking(self) -> Dict[str, Any]:
        return self._config.get("reranking", {})
    
    @property
    def answer(self) -> Dict[str, Any]:
        return self._config.get("answer", {})
    
    @property
    def verification(self) -> Dict[str, Any]:
        return self._config.get("verification", {})
    
    @property
    def output(self) -> Dict[str, Any]:
        return self._config.get("output", {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        return self._config.get("logging", {})


# Global config instance
config = Config()

