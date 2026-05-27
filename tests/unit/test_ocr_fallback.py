"""OCR fallback in PdfParser — Phase 5.1 (audit Item 5).

We exercise the decision path with an injected OCR callable so the tests
don't depend on the Tesseract binary or Pillow. The actual rendering
(``page.get_pixmap``) is the only piece that needs PyMuPDF; we use a real
in-memory PDF written via PyMuPDF so the test stays end-to-end on the
"low-text page → OCR fallback → recorded outcome" path.
"""

from __future__ import annotations

import fitz
import pytest

from app.ingestion.pdf_parser import PdfParser


def _make_pdf(tmp_path, pages: list[str]):
    """Build a multi-page PDF where each entry in ``pages`` is the page text.

    Empty strings produce a page with no text at all — exactly the scan
    shape we want to trigger the OCR fallback.
    """
    doc = fitz.open()
    for body in pages:
        page = doc.new_page()
        if body:
            page.insert_text((72, 72), body)
    path = tmp_path / "test.pdf"
    doc.save(path)
    doc.close()
    return path


def test_text_page_skips_ocr(tmp_path) -> None:
    """A page with abundant text should not invoke the OCR runner at all."""
    body = "This is a Ghana legal document with plenty of recognisable text. " * 5
    pdf = _make_pdf(tmp_path, [body])
    calls = {"n": 0}

    def _ocr_fail(_img, *, lang, timeout):
        calls["n"] += 1
        return "OCR-FALLBACK-MARKER"

    parser = PdfParser(ocr_runner=_ocr_fail)
    parsed = parser.parse(pdf)

    assert calls["n"] == 0, "OCR must not run when PyMuPDF returned enough text"
    assert parsed.page_count == 1
    assert "OCR-FALLBACK-MARKER" not in parsed.full_text


def test_low_text_page_triggers_ocr(tmp_path) -> None:
    """A page with no text layer should be OCR'd."""
    pdf = _make_pdf(tmp_path, [""])  # blank page → scan-like

    def _ocr_runner(_img, *, lang, timeout):
        # Tesseract would return whatever it sees; we stub it.
        assert lang == "eng"
        assert timeout > 0
        return "OCR-EXTRACTED-MARKER-XYZ"

    parser = PdfParser(ocr_runner=_ocr_runner)
    parsed = parser.parse(pdf)

    assert "OCR-EXTRACTED-MARKER-XYZ" in parsed.full_text


def test_ocr_failure_falls_back_to_empty(tmp_path) -> None:
    """OCR raising must NOT abort parsing — page just stays empty."""
    pdf = _make_pdf(tmp_path, ["", "good page with enough text " * 5])

    def _ocr_boom(_img, *, lang, timeout):
        raise RuntimeError("tesseract unavailable")

    parser = PdfParser(ocr_runner=_ocr_boom)
    parsed = parser.parse(pdf)

    assert parsed.page_count == 2
    # Page 2's text-layer content must still be present.
    assert "good page with enough text" in parsed.full_text


def test_ocr_disabled_via_explicit_false(tmp_path) -> None:
    """ocr_enabled=False → never call OCR even on blank pages."""
    pdf = _make_pdf(tmp_path, [""])

    def _ocr_should_not_run(*_a, **_kw):
        pytest.fail("OCR runner invoked when ocr_enabled=False")

    parser = PdfParser(ocr_runner=_ocr_should_not_run, ocr_enabled=False)
    parsed = parser.parse(pdf)
    # Page returned, empty text — no crash, no OCR.
    assert parsed.page_count == 1
    assert parsed.full_text == ""


def test_ocr_metrics_recorded(tmp_path) -> None:
    """Each page emits exactly one ocr_pages_total observation."""
    from app.core.telemetry import ocr_pages_total

    pdf = _make_pdf(tmp_path, ["", "Plenty of text on this page " * 10])

    def _ocr_runner(_img, *, lang, timeout):
        return "OCR-OK"

    # Capture before/after counter snapshots.
    def _snapshot() -> dict[str, float]:
        out: dict[str, float] = {}
        for sample in ocr_pages_total.collect()[0].samples:
            if sample.name.endswith("_total"):
                out[sample.labels["outcome"]] = sample.value
        return out

    before = _snapshot()
    PdfParser(ocr_runner=_ocr_runner).parse(pdf)
    after = _snapshot()

    def _delta(key: str) -> float:
        return after.get(key, 0.0) - before.get(key, 0.0)

    assert _delta("ocr_extracted") >= 1, "blank page must emit ocr_extracted"
    assert _delta("text_extracted") >= 1, "text page must emit text_extracted"
