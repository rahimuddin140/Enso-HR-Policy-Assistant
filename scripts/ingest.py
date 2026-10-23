#!/usr/bin/env python3
"""CLI script to ingest the Enso Group HR Policy Manual PDF into persistent ChromaDB storage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from hr_assistant.config import get_settings  # noqa: E402
from hr_assistant.embeddings import GoogleGeminiEmbeddings  # noqa: E402
from hr_assistant.ingestion import ingest_pdf_to_chunks, load_pdf_pages  # noqa: E402
from hr_assistant.vector_store import HRVectorStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest HR Policy Manual PDF into ChromaDB.")
    parser.add_argument("--pdf-path", type=str, default=None, help="Path to the HR Policy Manual PDF.")
    parser.add_argument("--persist-dir", type=str, default=None, help="ChromaDB persistence directory.")
    args = parser.parse_args()

    settings = get_settings()
    pdf_path = Path(args.pdf_path or settings.pdf_path)
    persist_dir = Path(args.persist_dir or settings.chroma_persist_dir)

    print(f"[1/4] Loading PDF from: {pdf_path}")
    pages = load_pdf_pages(pdf_path)
    print(f"[2/4] Extracted {len(pages)} pages. Cleaning and chunking text...")

    chunks, report = ingest_pdf_to_chunks(
        pages,
        source_name=pdf_path.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(
        f"[3/4] Created {report.chunk_count} chunks across {report.non_empty_pages} non-empty pages "
        f"(empty pages: {report.empty_pages})."
    )

    embeddings = GoogleGeminiEmbeddings(settings=settings)
    store = HRVectorStore(persist_dir=persist_dir, embedding_backend=embeddings)
    indexed_count = store.index_chunks(chunks, replace_existing=True)
    report.persisted_dir = str(persist_dir)

    print(f"[4/4] Indexed {indexed_count} chunks into persistent store: {persist_dir}")
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
