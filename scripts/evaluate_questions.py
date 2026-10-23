#!/usr/bin/env python3
"""Evaluate the 5 required LMS questions against the ingested Enso Group HR Policy Manual."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from hr_assistant.answer_service import AnswerService  # noqa: E402
from hr_assistant.config import get_settings  # noqa: E402
from hr_assistant.embeddings import GoogleGeminiEmbeddings  # noqa: E402
from hr_assistant.ingestion import ingest_pdf_to_chunks, load_pdf_pages  # noqa: E402
from hr_assistant.vector_store import HRVectorStore  # noqa: E402

EVALUATION_QUESTIONS: List[str] = [
    "How many days of Privilege Leave (PL) does a permanent employee get per year?",
    "What is the probation period for employees up to Senior Manager?",
    "What is the company's health-insurance premium amount?",
    "What is the current GST rate in India?",
    "How many days of maternity leave are given?",
]


def run_evaluation() -> List[Dict[str, Any]]:
    settings = get_settings()
    embeddings = GoogleGeminiEmbeddings(settings=settings)
    store = HRVectorStore(persist_dir=settings.chroma_persist_dir, embedding_backend=embeddings)

    if store.count() == 0:
        pages = load_pdf_pages(settings.pdf_path)
        chunks, _ = ingest_pdf_to_chunks(
            pages,
            source_name=Path(settings.pdf_path).name,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        store.index_chunks(chunks, replace_existing=True)

    service = AnswerService(vector_store=store, settings=settings)
    results: List[Dict[str, Any]] = []

    for idx, question in enumerate(EVALUATION_QUESTIONS, start=1):
        try:
            response = service.answer_question(question)
            res_dict = response.to_dict()
            results.append(
                {
                    "id": idx,
                    "question": question,
                    "answer": res_dict["answer"],
                    "found": res_dict["found"],
                    "sources": res_dict["sources"],
                    "supported_by_evidence": bool(res_dict["found"] and len(res_dict["sources"]) > 0),
                    "message": res_dict["message"],
                    "error": None,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "id": idx,
                    "question": question,
                    "answer": "Not found",
                    "found": False,
                    "sources": [],
                    "supported_by_evidence": False,
                    "message": "",
                    "error": str(exc),
                }
            )
    return results


def main() -> int:
    results = run_evaluation()
    output_path = ROOT_DIR / "data" / "evaluation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 80)
    print("ENSO GROUP HR POLICY MANUAL — 5-QUESTION RAG EVALUATION REPORT")
    print("=" * 80)
    for item in results:
        print(f"\nQ{item['id']}: {item['question']}")
        print(f"  Answer             : {item['answer']}")
        print(f"  Supported Evidence : {item['supported_by_evidence']}")
        if item["sources"]:
            for src in item["sources"]:
                print(f"  Citation           : {src['source']} | Page {src['page']} | Section: {src.get('section')}")
                excerpt_preview = src["excerpt"].replace("\n", " ")[:180]
                print(f"  Excerpt            : {excerpt_preview}...")
        else:
            print(f"  Citation           : None ({item['message']})")
        if item["error"]:
            print(f"  Error              : {item['error']}")
    print("\nSaved JSON evaluation report to:", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
