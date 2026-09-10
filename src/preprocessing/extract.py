"""PDF discovery and page-level text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from .clean import clean_text, is_low_text
except ImportError:  # Allows running this module from the filesystem directly.
    from clean import clean_text, is_low_text


ORGANIZATION = "AFAD"


@dataclass(frozen=True)
class PageRecord:
    source_file: str
    relative_path: str
    page: int
    text: str
    organization: str
    is_low_text: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": self.source_file,
            "relative_path": self.relative_path,
            "page": self.page,
            "text": self.text,
            "organization": self.organization,
            "is_low_text": self.is_low_text,
        }


def find_pdfs(raw_root: Path) -> list[Path]:
    """Find PDF files under raw_root recursively in deterministic order."""
    return sorted(path for path in raw_root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")


def output_path_for(pdf_path: Path, raw_root: Path, output_root: Path) -> Path:
    """Map a raw PDF path to its processed JSON path while preserving folders."""
    relative_pdf_path = pdf_path.relative_to(raw_root)
    return output_root / relative_pdf_path.with_suffix(".json")


def extract_pdf_pages(pdf_path: Path, raw_root: Path) -> list[PageRecord]:
    """Extract clean text from every page of a PDF using PyMuPDF."""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is not installed. Install dependencies with: pip install -r requirements.txt") from exc

    relative_path = pdf_path.relative_to(raw_root).as_posix()
    records: list[PageRecord] = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = clean_text(page.get_text("text"))
            records.append(
                PageRecord(
                    source_file=pdf_path.name,
                    relative_path=relative_path,
                    page=page_index + 1,
                    text=text,
                    organization=ORGANIZATION,
                    is_low_text=is_low_text(text),
                )
            )

    return records
