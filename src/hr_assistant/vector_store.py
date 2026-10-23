"""Persistent ChromaDB vector store management with idempotent re-ingestion."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import DeterministicHashedEmbeddings, EmbeddingBackend
from .schemas import DocumentChunk


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Embedding dimension mismatch: query has {len(vec_a)} dims, document has {len(vec_b)} dims."
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class HRVectorStore:
    """
    Persistent vector store backed by ChromaDB (when installed) and a synchronized
    on-disk JSON collection inside CHROMA_PERSIST_DIR so persistence and reload
    are deterministic across restarts and environments.
    """

    def __init__(
        self,
        persist_dir: str | Path = "./data/chroma",
        collection_name: str = "enso_hr_policy_manual",
        embedding_backend: Optional[EmbeddingBackend] = None,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_backend: EmbeddingBackend = embedding_backend or DeterministicHashedEmbeddings()
        self._store_file = self.persist_dir / f"{self.collection_name}.json"
        self._records: Dict[str, Dict[str, Any]] = {}
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if self._store_file.exists():
            try:
                payload = json.loads(self._store_file.read_text(encoding="utf-8"))
                records = payload.get("records", [])
                self._records = {str(r["chunk_id"]): r for r in records if "chunk_id" in r}
            except Exception:
                self._records = {}

    def _save_to_disk(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection_name": self.collection_name,
            "count": len(self._records),
            "records": list(self._records.values()),
        }
        self._store_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
        if self._store_file.exists():
            self._store_file.unlink()

    def index_chunks(self, chunks: List[DocumentChunk], replace_existing: bool = True) -> int:
        """
        Embed and persist chunks into ChromaDB + persistent store.
        When replace_existing=True or chunk_ids match, avoids duplicate records.
        """
        if not chunks:
            raise ValueError("Cannot index an empty list of chunks.")

        if replace_existing:
            self._records.clear()

        texts = [c.text for c in chunks]
        vectors = self.embedding_backend.embed_documents(texts)

        # Also persist into native chromadb if installed in environment
        try:
            import chromadb  # type: ignore

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_dir))
            if replace_existing:
                try:
                    client.delete_collection(self.collection_name)
                except Exception:
                    pass
            collection = client.get_or_create_collection(name=self.collection_name)
            collection.upsert(
                ids=[c.chunk_id for c in chunks],
                documents=texts,
                embeddings=vectors,
                metadatas=[
                    {
                        "source": c.source,
                        "page": int(c.page),
                        "chunk_id": c.chunk_id,
                        "section": c.section or "General HR Policy",
                    }
                    for c in chunks
                ],
            )
        except ImportError:
            pass

        for chunk, vector in zip(chunks, vectors):
            self._records[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": int(chunk.page),
                "section": chunk.section,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": vector,
            }

        self._save_to_disk()
        return len(self._records)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[DocumentChunk, float]]:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        if not self._records:
            return []

        query_vec = self.embedding_backend.embed_query(query)
        scored: List[Tuple[DocumentChunk, float]] = []

        for rec in self._records.values():
            emb = rec.get("embedding", [])
            sim = _cosine_similarity(query_vec, emb)
            chunk = DocumentChunk(
                chunk_id=rec["chunk_id"],
                source=rec["source"],
                page=int(rec["page"]),
                section=rec.get("section"),
                text=rec["text"],
                metadata=rec.get("metadata", {}),
            )
            scored.append((chunk, sim))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, k)]

    def get_all_chunks(self) -> List[DocumentChunk]:
        return [
            DocumentChunk(
                chunk_id=rec["chunk_id"],
                source=rec["source"],
                page=int(rec["page"]),
                section=rec.get("section"),
                text=rec["text"],
                metadata=rec.get("metadata", {}),
            )
            for rec in self._records.values()
        ]
