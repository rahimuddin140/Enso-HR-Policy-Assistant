#!/usr/bin/env python3
"""
Entrypoint for the Python Enso HR Policy Assistant.
Supports both Streamlit UI mode (`streamlit run app.py`) and CLI/HTTP check mode (`python3 app.py`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from hr_assistant.answer_service import AnswerService  # noqa: E402
from hr_assistant.config import get_settings  # noqa: E402
from hr_assistant.embeddings import GoogleGeminiEmbeddings  # noqa: E402
from hr_assistant.ingestion import ingest_pdf_to_chunks, load_pdf_pages  # noqa: E402
from hr_assistant.vector_store import HRVectorStore  # noqa: E402


def create_assistant_service(auto_ingest: bool = True) -> AnswerService:
    settings = get_settings()
    embeddings = GoogleGeminiEmbeddings(settings=settings)
    store = HRVectorStore(persist_dir=settings.chroma_persist_dir, embedding_backend=embeddings)

    if auto_ingest and store.count() == 0 and Path(settings.pdf_path).exists():
        pages = load_pdf_pages(settings.pdf_path)
        chunks, _ = ingest_pdf_to_chunks(
            pages,
            source_name=Path(settings.pdf_path).name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        store.index_chunks(chunks, replace_existing=True)

    return AnswerService(vector_store=store, settings=settings)


def main() -> int:
    parser = argparse.ArgumentParser(description="Enso HR Policy Assistant CLI / App runner.")
    parser.add_argument("--question", "-q", type=str, default=None, help="Ask a single HR policy question.")
    parser.add_argument("--status", action="store_true", help="Print current vector store and PDF status.")
    args = parser.parse_args()

    settings = get_settings()
    service = create_assistant_service(auto_ingest=True)

    if args.status:
        print(
            json.dumps(
                {
                    "pdf_exists": Path(settings.pdf_path).exists(),
                    "pdf_path": settings.pdf_path,
                    "indexed_chunks": service.vector_store.count(),
                    "persist_dir": settings.chroma_persist_dir,
                },
                indent=2,
            )
        )
        return 0

    if args.question:
        result = service.answer_question(args.question)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print("Enso HR Policy Assistant ready. Use --question '...' or run `npm run dev` for the web interface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
