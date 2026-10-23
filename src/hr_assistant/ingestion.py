"""PDF ingestion, text cleaning, section extraction, and recursive chunking."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schemas import DocumentChunk, IngestionReport

SECTION_PATTERNS = [
    re.compile(r"(\[[A-Z]\]\s+[A-Za-z0-9 &/\-]+Policy)", re.IGNORECASE),
    re.compile(r"(\[[A-Z]\]\s+[A-Za-z0-9 &/\-]+Process)", re.IGNORECASE),
    re.compile(r"(\[[A-Z]\]\s+Staff\s+Advances)", re.IGNORECASE),
    re.compile(r"(SECTION\s*-\s*[IVX]+\s+[A-Z0-9 &,\-]+)"),
    re.compile(
        r"((?:Equality|Whistleblowing|Drug and Alcohol Abuse|Health and Safety|Leave|Probation|Relocation|Compensation|Off-boarding|On-boarding|Dress Code|Work Timing|Punctuality and Attendance)\s+Policy)",
        re.IGNORECASE,
    ),
]


def clean_extracted_text(raw_text: str) -> str:
    """Clean PDF extraction artifacts without altering policy numbers, dates, or headings."""
    if not raw_text:
        return ""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix Unicode replacement character used as soft-hyphen in PDF OCR/fonts (e.g. "calculated" -> "calculated")
    text = re.sub(r"([a-zA-Z])\ufffd\s*([a-zA-Z])", r"\1\2", text)
    text = text.replace("\ufffd", " ")
    # Fix hyphenated split words produced by PDF column justification (e.g., "calculat - ed" -> "calculated")
    text = re.sub(r"([a-zA-Z])\s*-\s+([a-z]{2,})\b", r"\1\2", text)
    # Normalize parenthetical spacing like "12 weeks( - 90days)" or "12 weeks(- 90days)" -> "12 weeks (90 days)"
    text = re.sub(r"\(\s*-\s*(\d+)\s*days\)", r" (\1 days)", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+)days\b", r"\1 days", text, flags=re.IGNORECASE)
    # Replace proprietary bullet glyphs with clean bullets
    text = text.replace("Ԃ", "\n• ").replace("¤", "\n  - ").replace("▶", "\n• ")
    # Collapse excessive horizontal spaces while preserving paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_section_heading(text: str, previous_section: Optional[str] = None) -> Optional[str]:
    """Identify the most specific policy section heading on a page or chunk."""
    cleaned = re.sub(r"^HR Manual\s*-\s*\d+\s*", "", text.strip(), flags=re.IGNORECASE)
    for pattern in SECTION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            heading = re.sub(r"\s+", " ", match.group(1)).strip()
            return heading[:90]

    if re.search(r"Maternity\s+Leave\s*\(Leeway\)", cleaned, flags=re.IGNORECASE):
        return "[G] Leave Policy — Maternity Leave (Leeway)"

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    for line in lines[:3]:
        if "leave type" in line.lower() and "eligibility" in line.lower():
            continue
        if 4 <= len(line) <= 75 and not line.endswith("."):
            if any(k in line.lower() for k in ("policy", "process", "section", "annexure", "probation", "travel", "advances")):
                return line
    return previous_section


def load_pdf_pages(pdf_path: str | Path) -> List[Tuple[int, str]]:
    """
    Load a PDF file and return a list of (1-indexed page_number, raw_text) tuples.
    Raises FileNotFoundError or ValueError on missing, invalid, or corrupt files.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at '{path}'. Please place the HR manual PDF at this path or upload it.")
    if path.is_dir() or path.suffix.lower() != ".pdf":
        raise ValueError(f"Invalid PDF path '{path}': file must have a .pdf extension.")
    if path.stat().st_size == 0:
        raise ValueError(f"PDF file '{path}' is empty (0 bytes).")

    header = path.read_bytes()[:5]
    if header != b"%PDF-":
        raise ValueError(f"Corrupt or invalid PDF file '{path}': missing %PDF- header.")

    # 1. Prefer pypdf / langchain_community PyPDFLoader when installed
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages: List[Tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            pages.append((idx, page.extract_text() or ""))
        if pages:
            return pages
    except ImportError:
        pass
    except Exception as exc:
        raise ValueError(f"Failed to read PDF '{path}': {exc}") from exc

    # 2. Fallback to pre-extracted JSON cache produced by pdfjs-dist in the workspace
    extracted_cache = path.parent / "extracted_pages.json"
    if extracted_cache.exists() and path.name == "hr_manual.pdf":
        raw_items = json.loads(extracted_cache.read_text(encoding="utf-8"))
        return [(int(item["page"]), str(item.get("text", ""))) for item in raw_items]

    raise RuntimeError(
        "No PDF extraction backend available. Install pypdf (`pip install -r requirements.txt`) or run pdfjs extraction."
    )


def split_text_recursively(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> List[str]:
    """Split cleaned text into overlapping chunks preserving paragraphs and sentences."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size.")

    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= chunk_size:
        return [stripped]

    # Use LangChain RecursiveCharacterTextSplitter if available
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n• ", "\n", ". ", " ", ""],
        )
        return [c.strip() for c in splitter.split_text(stripped) if c.strip()]
    except ImportError:
        pass

    chunks: List[str] = []
    start = 0
    length = len(stripped)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            # Prefer breaking at newline or sentence boundary
            window = stripped[start:end]
            for sep in ("\n\n", "\n• ", "\n", ". ", " "):
                pos = window.rfind(sep)
                if pos > chunk_size // 2:
                    end = start + pos + len(sep)
                    break
        segment = stripped[start:end].strip()
        if segment:
            chunks.append(segment)
        if end >= length:
            break
        start = max(start + 1, end - chunk_overlap)
    return chunks


def ingest_pdf_to_chunks(
    pages: List[Tuple[int, str]],
    source_name: str = "hr_manual.pdf",
    chunk_size: int = 900,
    chunk_overlap: int = 150,
) -> Tuple[List[DocumentChunk], IngestionReport]:
    """Convert extracted (page_number, raw_text) pairs into metadata-rich DocumentChunks."""
    if not pages:
        raise ValueError("Document contains 0 pages.")

    chunks: List[DocumentChunk] = []
    empty_pages: List[int] = []
    current_section: Optional[str] = None

    for page_num, raw_text in pages:
        cleaned = clean_extracted_text(raw_text)
        # Strip running header "HR Manual - <page>" for emptiness check
        body_only = re.sub(r"^HR Manual\s*-\s*\d+\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if not body_only:
            empty_pages.append(int(page_num))
            continue

        current_section = detect_section_heading(cleaned, current_section)
        page_Texts = split_text_recursively(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for idx, chunk_text in enumerate(page_Texts, start=1):
            section = detect_section_heading(chunk_text, current_section)
            chunk_id = f"{Path(source_name).stem}_p{int(page_num):03d}_c{idx:02d}"
            metadata: Dict[str, object] = {
                "source": source_name,
                "page": int(page_num),
                "chunk_id": chunk_id,
                "section": section or "General HR Policy",
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=source_name,
                    page=int(page_num),
                    section=section or "General HR Policy",
                    text=chunk_text,
                    metadata=metadata,
                )
            )

    if not chunks:
        raise ValueError(
            f"No extractable text found across {len(pages)} pages in '{source_name}'. "
            "The PDF may contain only scanned images without OCR text."
        )

    report = IngestionReport(
        source=source_name,
        page_count=len(pages),
        non_empty_pages=len(pages) - len(empty_pages),
        empty_pages=empty_pages,
        chunk_count=len(chunks),
        persisted_dir="",
    )
    return chunks, report
