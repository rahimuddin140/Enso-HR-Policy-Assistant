"""Unit tests for PDF loading, cleaning, chunk creation, metadata preservation, and error handling."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from hr_assistant.ingestion import (  # noqa: E402
    clean_extracted_text,
    detect_section_heading,
    ingest_pdf_to_chunks,
    load_pdf_pages,
)


class TestIngestionPipeline(unittest.TestCase):
    def test_missing_pdf_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_pdf_pages("/tmp/non_existent_hr_manual_12345.pdf")

    def test_corrupt_or_empty_pdf_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_pdf = Path(tmpdir) / "empty.pdf"
            empty_pdf.write_bytes(b"")
            with self.assertRaises(ValueError):
                load_pdf_pages(empty_pdf)

            corrupt_pdf = Path(tmpdir) / "corrupt.pdf"
            corrupt_pdf.write_bytes(b"NOT_A_PDF_HEADER")
            with self.assertRaises(ValueError):
                load_pdf_pages(corrupt_pdf)

    def test_clean_extracted_text_preserves_numbers_and_fixes_split_words(self) -> None:
        raw = "HR Manual - 60\n[G] Leave Policy\ncalculat - ed for 12 days( - 90days)"
        cleaned = clean_extracted_text(raw)
        self.assertIn("calculated", cleaned)
        self.assertIn("12 days", cleaned)
        self.assertIn("(90 days)", cleaned)
        self.assertIn("[G] Leave Policy", cleaned)

    def test_chunk_creation_and_metadata_preservation_with_synthetic_fixture(self) -> None:
        # Synthetic test fixture only (clearly labelled)
        synthetic_pages = [
            (1, "HR Manual - 1\nTable of Contents"),
            (2, "HR Manual - 2\n   "),  # Empty body page
            (
                42,
                "HR Manual - 42\n[B] Probation Policy\nAll employees up to Sr. Manager would be placed in Probation for a period of 3 months from the date of their joining.",
            ),
        ]
        chunks, report = ingest_pdf_to_chunks(
            synthetic_pages,
            source_name="synthetic_fixture.pdf",
            chunk_size=300,
            chunk_overlap=50,
        )
        self.assertEqual(report.page_count, 3)
        self.assertEqual(report.non_empty_pages, 2)
        self.assertEqual(report.empty_pages, [2])
        self.assertGreaterEqual(len(chunks), 2)

        probation_chunk = [c for c in chunks if c.page == 42][0]
        self.assertEqual(probation_chunk.source, "synthetic_fixture.pdf")
        self.assertEqual(probation_chunk.page, 42)
        self.assertEqual(probation_chunk.section, "[B] Probation Policy")
        self.assertTrue(probation_chunk.chunk_id.startswith("synthetic_fixture_p042_"))

    def test_real_pdf_loads_and_preserves_79_pages_if_present(self) -> None:
        real_pdf = ROOT_DIR / "data" / "hr_manual.pdf"
        if real_pdf.exists():
            pages = load_pdf_pages(real_pdf)
            self.assertEqual(len(pages), 79)
            self.assertEqual(pages[0][0], 1)
            self.assertEqual(pages[-1][0], 79)


if __name__ == "__main__":
    unittest.main()
