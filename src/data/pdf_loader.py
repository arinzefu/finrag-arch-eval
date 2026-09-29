"""
Low-level PDF page extraction for FinanceBench.

Critical invariant
------------------
FinanceBench's ``evidence_page_num`` is ZERO-indexed. This module therefore
enumerates PDF pages starting at 0 and never drops pages, including pages with
no extractable text. Dropping blank pages would shift every subsequent page
identifier and corrupt retrieval evaluation.

This module performs page extraction only. It deliberately avoids OCR and
aggressive text cleaning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError


@dataclass(frozen=True)
class ExtractedPage:
    """One physical PDF page with zero-indexed page identity."""

    doc_name: str
    page: int
    text: str
    company: Optional[str] = None
    doc_type: Optional[str] = None
    doc_period: Optional[int | str] = None
    page_number_1based: Optional[int] = None
    text_length: int = 0
    extraction_status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def clean_extracted_text(text: str | None) -> str:
    """
    Apply only conservative normalization to extracted text.

    We preserve textual content and page boundaries because Stage 6 needs to
    compare these pages with FinanceBench gold evidence. No lowercasing,
    stemming, punctuation removal, de-hyphenation, or paragraph merging is
    performed here.
    """
    if not text:
        return ""

    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing horizontal whitespace line-by-line while preserving
    # explicit line structure.
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    return "\n".join(lines).strip()


def open_pdf(pdf_path: str | Path) -> PdfReader:
    """Open a PDF with pypdf and fail clearly if it cannot be parsed."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file, received: {path}")

    try:
        reader = PdfReader(str(path), strict=False)
    except (PdfReadError, OSError, ValueError) as exc:
        raise RuntimeError(f"Could not parse PDF {path}: {exc}") from exc

    if len(reader.pages) == 0:
        raise RuntimeError(f"PDF contains zero pages: {path}")

    return reader


def iter_pdf_pages(
    pdf_path: str | Path,
    *,
    doc_name: str,
    company: str | None = None,
    doc_type: str | None = None,
    doc_period: int | str | None = None,
    fail_on_page_error: bool = True,
) -> Iterator[ExtractedPage]:
    """
    Yield every physical PDF page in order.

    Page indices are zero-based by design to match FinanceBench
    ``evidence_page_num``.

    Empty-text pages are retained with ``extraction_status='empty'``. A true
    extraction exception raises by default so it cannot silently contaminate
    the corpus.
    """
    reader = open_pdf(pdf_path)

    for page_index, page_object in enumerate(reader.pages):
        try:
            raw_text = page_object.extract_text()
            text = clean_extracted_text(raw_text)
            status = "ok" if text else "empty"
        except Exception as exc:
            if fail_on_page_error:
                raise RuntimeError(
                    f"Text extraction failed for {doc_name}, "
                    f"zero-indexed page {page_index}: {exc}"
                ) from exc
            text = ""
            status = f"error:{type(exc).__name__}"

        yield ExtractedPage(
            doc_name=doc_name,
            page=page_index,
            text=text,
            company=company,
            doc_type=doc_type,
            doc_period=doc_period,
            page_number_1based=page_index + 1,
            text_length=len(text),
            extraction_status=status,
        )


