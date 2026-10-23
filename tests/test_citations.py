"""Unit tests ensuring citations cannot be fabricated or reference non-existent pages/chunks."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from hr_assistant.answer_service import validate_llm_json_and_citations  # noqa: E402
from hr_assistant.schemas import DocumentChunk  # noqa: E402


class TestCitationIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.retrieved = [
            (
                DocumentChunk(
                    chunk_id="hr_manual_p060_c01",
                    source="hr_manual.pdf",
                    page=60,
                    section="[G] Leave Policy",
                    text="Privilege Leave (PL) – 12 days.",
                ),
                0.88,
            )
        ]

    def test_fabricated_chunk_id_is_rejected_and_returns_not_found(self) -> None:
        fake_llm_json = json.dumps(
            {
                "answer": "Employees get 25 days of leave.",
                "found": True,
                "cited_chunk_ids": ["invented_chunk_p999_c01"],
                "message": "Invented citation.",
            }
        )
        result = validate_llm_json_and_citations(fake_llm_json, self.retrieved)
        self.assertFalse(result.found)
        self.assertEqual(result.answer, "Not found")
        self.assertEqual(result.sources, [])

    def test_missing_citations_for_positive_answer_returns_not_found(self) -> None:
        no_citation_json = json.dumps(
            {
                "answer": "12 days of Privilege Leave.",
                "found": True,
                "cited_chunk_ids": [],
                "message": "No chunk ID provided.",
            }
        )
        result = validate_llm_json_and_citations(no_citation_json, self.retrieved)
        self.assertFalse(result.found)
        self.assertEqual(result.answer, "Not found")
        self.assertEqual(result.sources, [])

    def test_valid_citation_inherits_only_source_document_page_and_excerpt(self) -> None:
        valid_json = json.dumps(
            {
                "answer": "Permanent employees get 12 days of Privilege Leave (PL) per calendar year.",
                "found": True,
                "cited_chunk_ids": ["hr_manual_p060_c01"],
                "message": "Supported by Leave Policy.",
            }
        )
        result = validate_llm_json_and_citations(valid_json, self.retrieved)
        self.assertTrue(result.found)
        self.assertEqual(len(result.sources), 1)
        self.assertEqual(result.sources[0].page, 60)
        self.assertEqual(result.sources[0].source, "hr_manual.pdf")
        self.assertEqual(result.sources[0].section, "[G] Leave Policy")
        self.assertEqual(result.sources[0].excerpt, "Privilege Leave (PL) – 12 days.")


if __name__ == "__main__":
    unittest.main()
