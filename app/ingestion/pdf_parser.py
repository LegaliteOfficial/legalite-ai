"""PDF text extraction via PyMuPDF, with Tesseract OCR fallback for scans.

Two-stage strategy (Phase 5.1 / audit Item 5):

1. Try PyMuPDF's text extractor (``page.get_text("text")``). For
   text-layer PDFs this returns the canonical text in seconds.
2. If a page returns fewer than ``OCR_MIN_TEXT_CHARS`` characters AND
   OCR is available, render the page to a pixmap and run pytesseract
   over it. The OCR'd text replaces the empty PyMuPDF result for that
   page only — other pages are unaffected.

OCR is opt-in by configuration:

- ``OCR_ENABLED=true``  → fail loudly if Tesseract isn't reachable.
- ``OCR_ENABLED=false`` → never OCR (useful for tests).
- ``OCR_ENABLED`` unset → auto-detect (try the import once, fall
  back to text-only if it fails). This is the default.

Each page emits exactly one observation on
``legalite_ocr_pages_total{outcome=…}``:

- ``text_extracted`` — PyMuPDF found enough text; OCR not attempted.
- ``ocr_extracted``  — OCR ran and returned text.
- ``ocr_failed``     — OCR was attempted but raised / returned empty.
- ``ocr_disabled``   — page was scan-like but OCR is off / unavailable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import fitz  # PyMuPDF

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.telemetry import ocr_latency, record_ocr_page

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


class _OcrCallable(Protocol):
    def __call__(self, image: object, *, lang: str, timeout: int) -> str: ...


class PdfParser:
    """Parser with optional Tesseract OCR fallback.

    ``ocr_runner`` lets tests inject a deterministic stub instead of
    actually shelling out to Tesseract. When not supplied the parser
    lazily resolves pytesseract on first use.
    """

    def __init__(
        self,
        *,
        ocr_runner: _OcrCallable | None = None,
        ocr_enabled: bool | None = None,
    ) -> None:
        self._injected_ocr = ocr_runner
        settings = get_settings()
        # Explicit override beats settings.
        if ocr_enabled is None:
            ocr_enabled = settings.ocr_enabled
        self._explicit_enable = ocr_enabled
        # Lazy probe — None means "haven't tried yet".
        self._ocr_resolved: _OcrCallable | None = None
        self._ocr_resolution_done = False

    # ---- public API --------------------------------------------------------

    def parse(self, path: str | Path) -> ParsedPdf:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        settings = get_settings()
        min_chars = settings.ocr_min_text_chars
        pages: list[PdfPage] = []
        ocr_attempted = 0
        ocr_succeeded = 0

        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                raw_text = page.get_text("text") or ""
                normalised = _normalize(raw_text)

                if len(normalised) >= min_chars:
                    record_ocr_page("text_extracted")
                    pages.append(PdfPage(page_number=i, text=normalised))
                    continue

                # Scan-like page: try OCR.
                runner = self._get_ocr_runner()
                if runner is None:
                    record_ocr_page("ocr_disabled")
                    pages.append(PdfPage(page_number=i, text=normalised))
                    continue

                ocr_attempted += 1
                ocr_text = self._ocr_page(page, runner)
                if ocr_text:
                    ocr_succeeded += 1
                    record_ocr_page("ocr_extracted")
                    pages.append(PdfPage(page_number=i, text=_normalize(ocr_text)))
                else:
                    record_ocr_page("ocr_failed")
                    pages.append(PdfPage(page_number=i, text=normalised))

        logger.info(
            "pdf.parsed",
            path=str(path),
            pages=len(pages),
            ocr_attempted=ocr_attempted,
            ocr_succeeded=ocr_succeeded,
        )
        return ParsedPdf(pages=pages)

    # ---- internals ---------------------------------------------------------

    def _get_ocr_runner(self) -> _OcrCallable | None:
        """Resolve the OCR callable once and cache. Returns None if disabled.

        Explicit ``ocr_enabled=False`` wins over an injected runner — tests
        rely on this to assert the disable path is honored even when a
        callable is wired up.
        """
        if self._explicit_enable is False:
            return None
        if self._injected_ocr is not None:
            return self._injected_ocr
        if self._ocr_resolution_done:
            return self._ocr_resolved

        self._ocr_resolution_done = True

        try:
            import pytesseract  # noqa: F401  (probe)
        except ImportError:
            if self._explicit_enable is True:
                # Operator asked for OCR explicitly; surface the missing dep.
                raise
            logger.warning("ocr.pytesseract_missing", note="OCR fallback disabled")
            self._ocr_resolved = None
            return None

        # Build a thin adapter so the parser can stay independent of pytesseract types.
        from pytesseract import image_to_string

        def _runner(image: object, *, lang: str, timeout: int) -> str:
            return image_to_string(image, lang=lang, timeout=timeout)  # type: ignore[arg-type]

        self._ocr_resolved = _runner
        return _runner

    def _ocr_page(self, page: fitz.Page, runner: _OcrCallable) -> str:
        """Render the page to a PIL image and run the OCR callable on it.

        On any exception (Tesseract not installed, timeout, ImportError on
        Pillow) we log + return empty; the caller records ``ocr_failed``.
        Ingestion must never abort because of OCR — text-only is still
        useful for hybrid + structured paths.
        """
        settings = get_settings()
        try:
            from PIL import Image  # local import so tests can run without Pillow
        except ImportError:
            logger.warning("ocr.pillow_missing")
            return ""

        try:
            # render at configured DPI; matrix = DPI/72 because PDF uses points
            zoom = settings.ocr_render_dpi / 72.0
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        except Exception:
            logger.exception("ocr.render_failed", page=page.number)
            return ""

        t0 = time.perf_counter()
        try:
            text = runner(image, lang=settings.ocr_lang, timeout=settings.ocr_timeout_seconds)
        except Exception:
            logger.exception("ocr.tesseract_failed", page=page.number)
            return ""
        finally:
            ocr_latency.observe(time.perf_counter() - t0)
        return text or ""


def _normalize(text: str) -> str:
    # Collapse runs of whitespace but keep paragraph breaks.
    lines = [ln.rstrip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln.strip() != "" or True).strip()
