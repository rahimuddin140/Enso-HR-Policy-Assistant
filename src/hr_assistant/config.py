"""Configuration management for the Enso HR Policy Assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except ImportError:
        env_path = Path(".env")
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "google"
    google_api_key: str = ""
    chat_model: str = "gemini-3.8-flash"
    embedding_model: str = "gemini-embedding-2-preview"
    chroma_persist_dir: str = "./data/chroma"
    pdf_path: str = "./data/hr_manual.pdf"
    retrieval_k: int = 4
    relevance_threshold: float = 0.32
    chunk_size: int = 900
    chunk_overlap: int = 150

    def validate_for_live_api(self) -> None:
        if self.llm_provider.lower() == "google" and not self.google_api_key.strip():
            raise ValueError(
                "Missing API key configuration: set GOOGLE_API_KEY or GEMINI_API_KEY in your environment."
            )
        if not self.chat_model.strip():
            raise ValueError("Missing CHAT_MODEL configuration.")
        if not self.embedding_model.strip():
            raise ValueError("Missing EMBEDDING_MODEL configuration.")


def get_settings() -> Settings:
    _load_dotenv_if_available()
    if os.environ.get("OFFLINE_MODE") == "1":
        api_key = ""
    else:
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    return Settings(
        llm_provider=os.environ.get("LLM_PROVIDER", "google"),
        google_api_key=api_key,
        chat_model=os.environ.get("CHAT_MODEL", "gemini-3.8-flash"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "gemini-embedding-2-preview"),
        chroma_persist_dir=os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma"),
        pdf_path=os.environ.get("PDF_PATH", "./data/hr_manual.pdf"),
        retrieval_k=int(os.environ.get("RETRIEVAL_K", "4")),
        relevance_threshold=float(os.environ.get("RELEVANCE_THRESHOLD", "0.32")),
        chunk_size=int(os.environ.get("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "150")),
    )
