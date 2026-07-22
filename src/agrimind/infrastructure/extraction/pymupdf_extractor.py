"""PDF text extraction using PyMuPDF."""

from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    page_number: int  # 1-based, as pages are conventionally numbered from 1
    text: str


def extract_pages(pdf_path: Path) -> list[ExtractedPage]:
    """Return the text of each page in a PDF with its 1-based page number.

    Text is stripped of surrounding whitespace; pages without extractable text
    (e.g. scanned images) yield an empty string. Deciding what to do about empty
    pages is the caller's concern, not the extractor's.
    """
    with fitz.open(pdf_path) as document:
        return [
            ExtractedPage(page_number=index + 1, text=page.get_text("text").strip())
            for index, page in enumerate(document)
        ]
