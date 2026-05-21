"""PDF text extraction via PyMuPDF.

We return per-page text rather than a single blob so the chunker can preserve
page numbers — essential for legal citations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class PdfPage:
    page_number: int  # 1-indexed
    text: str


@dataclass(slots=True, frozen=True)
class ParsedPdf:
    pages: list[PdfPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class PdfParser:
    def parse(self, path: str | Path) -> ParsedPdf:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        pages: list[PdfPage] = []
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                pages.append(PdfPage(page_number=i, text=_normalize(text)))

        logger.info("pdf.parsed", path=str(path), pages=len(pages))
        return ParsedPdf(pages=pages)


def _normalize(text: str) -> str:
    # Collapse runs of whitespace but keep paragraph breaks.
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip() != "" or True).strip()
