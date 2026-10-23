"""Hybrid semantic + keyword retrieval and evidence relevance filtering."""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from .schemas import DocumentChunk
from .vector_store import HRVectorStore

STOPWORDS: Set[str] = {
    "what",
    "how",
    "many",
    "much",
    "does",
    "do",
    "is",
    "are",
    "the",
    "a",
    "an",
    "in",
    "of",
    "for",
    "to",
    "up",
    "get",
    "per",
    "year",
    "given",
    "company",
    "companys",
    "employee",
    "employees",
    "current",
    "policy",
    "enso",
    "group",
}


def extract_query_keywords(query: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [t for t in tokens if len(t) >= 2 and t not in STOPWORDS]


def compute_lexical_overlap_score(query: str, chunk: DocumentChunk) -> float:
    keywords = extract_query_keywords(query)
    if not keywords:
        return 0.0
    haystack = f"{chunk.section or ''} {chunk.text}".lower()
    haystack_tokens = set(re.findall(r"[a-z0-9]+", haystack))

    matched = 0.0
    for kw in keywords:
        if kw in haystack_tokens:
            matched += 1.0
        elif kw in haystack:
            matched += 0.7

    ratio = matched / len(keywords)
    # Phrase boost for exact multi-word policy terms
    q_lower = query.lower()
    for phrase in ("privilege leave", "probation", "senior manager", "sr. manager", "maternity leave", "health insurance", "gst"):
        if phrase in q_lower and phrase in haystack:
            ratio = min(1.0, ratio + 0.35)
    return min(1.0, ratio)


def retrieve_relevant_chunks(
    vector_store: HRVectorStore,
    question: str,
    k: int = 4,
    relevance_threshold: float = 0.32,
) -> List[Tuple[DocumentChunk, float]]:
    """
    Retrieve top-k chunks combining vector similarity and lexical keyword verification.
    Filters out chunks below relevance_threshold so irrelevant excerpts are not passed as proof.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if vector_store.count() == 0:
        return []

    # Candidate pool from vector store
    raw_candidates = vector_store.similarity_search_with_score(question, k=max(k * 6, 20))
    all_chunks = vector_store.get_all_chunks()

    by_id = {chunk.chunk_id: (chunk, vec_score) for chunk, vec_score in raw_candidates}
    for chunk in all_chunks:
        if chunk.chunk_id not in by_id:
            by_id[chunk.chunk_id] = (chunk, 0.0)

    combined: List[Tuple[DocumentChunk, float]] = []
    for chunk, vec_score in by_id.values():
        lex_score = compute_lexical_overlap_score(question, chunk)
        # Require meaningful lexical or semantic support
        hybrid_score = (0.45 * max(0.0, vec_score)) + (0.55 * lex_score)
        if lex_score == 0.0 and vec_score < 0.65:
            hybrid_score *= 0.4
        if hybrid_score >= relevance_threshold:
            combined.append((chunk, round(hybrid_score, 4)))

    combined.sort(key=lambda item: item[1], reverse=True)
    return combined[: max(1, k)]
