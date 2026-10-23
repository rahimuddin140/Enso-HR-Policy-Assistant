"""Unit tests for grounded AnswerService, abstention ('Not found'), malformed LLM outputs, and config errors."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from hr_assistant.answer_service import AnswerService  # noqa: E402
from hr_assistant.config import Settings  # noqa: E402
from hr_assistant.ingestion import ingest_pdf_to_chunks  # noqa: E402
from hr_assistant.vector_store import HRVectorStore  # noqa: E402


class TestAnswerService(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        pages = [
            (
                60,
                "HR Manual - 60\n[G] Leave Policy\nPrivilege Leave (PL) – 12 days. All permanent employees are eligible to avail PL.",
            ),
            (
                42,
                "HR Manual - 42\n[B] Probation Policy\nAll employees up to Sr. Manager would be placed in Probation for a period of 3 months from the date of their joining.",
            ),
        ]
        self.chunks, _ = ingest_pdf_to_chunks(pages, source_name="fixture.pdf")
        self.store = HRVectorStore(persist_dir=self.tmpdir.name)
        self.store.index_chunks(self.chunks, replace_existing=True)
        self.offline_settings = Settings(google_api_key="")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_supported_question_returns_grounded_answer_and_citation(self) -> None:
        service = AnswerService(vector_store=self.store, settings=self.offline_settings)
        resp = service.answer_question(
            "How many days of Privilege Leave (PL) does a permanent employee get per year?"
        )
        self.assertTrue(resp.found)
        self.assertIn("12 days", resp.answer)
        self.assertEqual(len(resp.sources), 1)
        self.assertEqual(resp.sources[0].page, 60)
        self.assertEqual(resp.sources[0].source, "fixture.pdf")

    def test_missing_answer_returns_exact_not_found(self) -> None:
        service = AnswerService(vector_store=self.store, settings=self.offline_settings)
        resp = service.answer_question("What is the current GST rate in India?")
        self.assertFalse(resp.found)
        self.assertEqual(resp.answer, "Not found")
        self.assertEqual(resp.sources, [])

    def test_malformed_llm_output_safely_falls_back_to_not_found(self) -> None:
        service = AnswerService(
            vector_store=self.store,
            llm_callable=lambda _prompt: "THIS IS NOT VALID JSON {{{",
        )
        resp = service.answer_question("How many days of Privilege Leave does an employee get?")
        self.assertFalse(resp.found)
        self.assertEqual(resp.answer, "Not found")
        self.assertEqual(resp.sources, [])

    def test_contradictory_evidence_returns_not_found(self) -> None:
        contradictory_pages = [
            (10, "HR Manual - 10\n[B] Probation Policy\nProbation period for staff is 3 months."),
            (11, "HR Manual - 11\n[B] Probation Policy\nProbation period for staff is 9 months."),
        ]
        chunks, _ = ingest_pdf_to_chunks(contradictory_pages, source_name="contradict.pdf")
        with tempfile.TemporaryDirectory() as tmp:
            store = HRVectorStore(persist_dir=tmp)
            store.index_chunks(chunks, replace_existing=True)
            service = AnswerService(vector_store=store, settings=self.offline_settings)
            resp = service.answer_question("What is the probation period for staff?")
            self.assertFalse(resp.found)
            self.assertEqual(resp.answer, "Not found")

    def test_empty_question_raises_validation_error(self) -> None:
        service = AnswerService(vector_store=self.store)
        with self.assertRaises(ValueError):
            service.answer_question("")

    def test_missing_api_key_configuration_validation(self) -> None:
        bad_settings = Settings(llm_provider="google", google_api_key="", chat_model="gemini-3.8-flash")
        with self.assertRaises(ValueError):
            bad_settings.validate_for_live_api()


if __name__ == "__main__":
    unittest.main()
