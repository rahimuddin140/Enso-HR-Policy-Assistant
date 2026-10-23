"""Unit tests for Chroma vector store persistence, reload, duplicate prevention, and retrieval."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from hr_assistant.embeddings import DeterministicHashedEmbeddings  # noqa: E402
from hr_assistant.ingestion import ingest_pdf_to_chunks  # noqa: E402
from hr_assistant.retrieval import retrieve_relevant_chunks  # noqa: E402
from hr_assistant.vector_store import HRVectorStore  # noqa: E402


class TestVectorStoreAndRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.synthetic_pages = [
            (
                10,
                "HR Manual - 10\n[B] Probation Policy\nSynthetic Fixture: All employees up to Senior Manager are placed on probation for 3 months.",
            ),
            (
                20,
                "HR Manual - 20\n[G] Leave Policy\nSynthetic Fixture: Permanent employees receive 12 days of Privilege Leave (PL) per calendar year.",
            ),
        ]
        self.chunks, _ = ingest_pdf_to_chunks(
            self.synthetic_pages,
            source_name="fixture_manual.pdf",
            chunk_size=500,
            chunk_overlap=50,
        )

    def test_chroma_persistence_reload_and_no_duplicates_on_reingestion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            embeddings = DeterministicHashedEmbeddings(dimension=128)
            store = HRVectorStore(persist_dir=tmpdir, embedding_backend=embeddings)
            count_first = store.index_chunks(self.chunks, replace_existing=True)
            self.assertEqual(count_first, 2)

            # Re-ingest same chunks and verify no duplicate records
            count_second = store.index_chunks(self.chunks, replace_existing=True)
            self.assertEqual(count_second, 2)

            # Reopen persisted store from disk
            reloaded_store = HRVectorStore(persist_dir=tmpdir, embedding_backend=embeddings)
            self.assertEqual(reloaded_store.count(), 2)

            # Verify query embedding dimensions match document embedding dimensions
            results = retrieve_relevant_chunks(
                reloaded_store,
                "How many days of Privilege Leave (PL) does a permanent employee get?",
                k=2,
                relevance_threshold=0.25,
            )
            self.assertGreaterEqual(len(results), 1)
            top_chunk, score = results[0]
            self.assertEqual(top_chunk.page, 20)
            self.assertIn("12 days", top_chunk.text)
            self.assertGreaterEqual(score, 0.25)

    def test_unrelated_question_returns_empty_above_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HRVectorStore(persist_dir=tmpdir)
            store.index_chunks(self.chunks, replace_existing=True)

            results = retrieve_relevant_chunks(
                store,
                "What is the current GST rate in India?",
                k=4,
                relevance_threshold=0.35,
            )
            self.assertEqual(len(results), 0)

    def test_empty_query_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = HRVectorStore(persist_dir=tmpdir)
            store.index_chunks(self.chunks, replace_existing=True)
            with self.assertRaises(ValueError):
                retrieve_relevant_chunks(store, "   ")


if __name__ == "__main__":
    unittest.main()
