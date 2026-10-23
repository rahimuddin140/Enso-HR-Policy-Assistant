"""Strict data schemas and response contracts for the HR Policy Assistant."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DocumentChunk:
    chunk_id: str
    source: str
    page: int
    section: Optional[str]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CitationSource:
    source: str
    page: int
    section: Optional[str]
    excerpt: str
    chunk_id: Optional[str] = None
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "source": self.source,
            "page": int(self.page),
            "section": self.section,
            "excerpt": self.excerpt,
        }
        if self.chunk_id is not None:
            payload["chunk_id"] = self.chunk_id
        if self.score is not None:
            payload["score"] = round(float(self.score), 4)
        return payload


@dataclass
class AnswerResponse:
    answer: str
    found: bool
    sources: List[CitationSource] = field(default_factory=list)
    message: str = ""

    @classmethod
    def not_found(cls, message: str = "The manual does not contain enough evidence to answer this question.") -> "AnswerResponse":
        return cls(
            answer="Not found",
            found=False,
            sources=[],
            message=message,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer if self.found else "Not found",
            "found": bool(self.found and self.answer != "Not found" and len(self.sources) > 0),
            "sources": [s.to_dict() for s in self.sources] if self.found else [],
            "message": self.message,
        }


@dataclass
class IngestionReport:
    source: str
    page_count: int
    non_empty_pages: int
    empty_pages: List[int]
    chunk_count: int
    persisted_dir: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
