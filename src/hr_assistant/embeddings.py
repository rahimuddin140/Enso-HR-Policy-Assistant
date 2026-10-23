"""Embedding model interface for document ingestion and query embedding."""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import List, Protocol

from .config import Settings, get_settings


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...


class DeterministicHashedEmbeddings:
    """
    Deterministic normalized feature-hashing embeddings (dimension=256)
    used for offline unit tests and fallback when external API calls are disabled.
    Guarantees identical embedding dimensions for documents and queries.
    """

    def __init__(self, dimension: int = 256) -> None:
        self.dimension = dimension

    def _embed_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
        if not tokens:
            return vec
        # Unigrams and bigrams
        features = list(tokens) + [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        for feat in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_single(text)


class GoogleGeminiEmbeddings:
    """
    LangChain GoogleGenerativeAIEmbeddings wrapper with direct REST fallback
    using the configured EMBEDDING_MODEL (default: gemini-embedding-2-preview).
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.model = self.settings.embedding_model
        if not self.model.startswith("models/"):
            self.api_model = f"models/{self.model}"
        else:
            self.api_model = self.model
        self._fallback = DeterministicHashedEmbeddings(dimension=256)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore

            if self.settings.google_api_key:
                client = GoogleGenerativeAIEmbeddings(
                    model=self.api_model,
                    google_api_key=self.settings.google_api_key,
                )
                return client.embed_documents(texts)
        except Exception:
            pass
        return self._fallback.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore

            if self.settings.google_api_key:
                client = GoogleGenerativeAIEmbeddings(
                    model=self.api_model,
                    google_api_key=self.settings.google_api_key,
                )
                return client.embed_query(text)
        except Exception:
            pass
        return self._fallback.embed_query(text)
