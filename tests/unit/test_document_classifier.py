"""Upload-time doc_type classification (``POST /upload-law/bulk`` with no
``doc_type``).

The heuristic is the floor the classifier always falls back to, so it is
pinned against the header shapes Ghana instruments actually carry. The
LLM path is exercised with a fake client: a valid label + summary is
taken, anything off-list or an API failure (e.g. the 429 quota error)
must quietly degrade to the heuristic, never fail the upload.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.ingestion.document_classifier import (
    FALLBACK_DOC_TYPE,
    DocumentClassifier,
    classify_heuristic,
)
from app.ingestion.pdf_parser import ParsedPdf, PdfPage


@pytest.mark.parametrize(
    "title,head,expected",
    [
        ("Republic v. Ibrahim", "", "Case Law"),
        (
            "Judgment 2019",
            "IN THE SUPREME COURT OF GHANA\nCORAM: ATUGUBA JSC\n"
            "Article 2 of the Constitution of the Republic of Ghana",
            "Case Law",
        ),
        (
            "1992 Constitution",
            "THE CONSTITUTION OF THE REPUBLIC OF GHANA\nCHAPTER ONE",
            "Constitution",
        ),
        (
            "Labour Regulations",
            "LABOUR REGULATIONS, 2007\nL.I. 1833\nmade under the Labour Act, 2003 (Act 651)",
            "Regulation",
        ),
        ("Labour Act, 2003 (Act 651)", "", "Act"),
        ("courts", "AN ACT to amend...\nBE IT ENACTED by Parliament", "Act"),
        ("pd", "PRACTICE DIRECTION ON DISCLOSURE", "Practice Direction"),
        ("scan_0001", "", FALLBACK_DOC_TYPE),
    ],
)
def test_classify_heuristic(title: str, head: str, expected: str) -> None:
    assert classify_heuristic(head, title) == expected


def _parsed(text: str) -> ParsedPdf:
    return ParsedPdf(pages=[PdfPage(page_number=1, text=text)])


class _FakeLlm:
    def __init__(self, *, content: str | None = None, error: Exception | None = None) -> None:
        self._content = content
        self._error = error
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs: object) -> object:
        self.calls += 1
        if self._error:
            raise self._error
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_llm_label_and_summary_are_used() -> None:
    llm = _FakeLlm(
        content=json.dumps({"doc_type": "case law", "summary": "Supreme Court ruling on bail."})
    )
    result = DocumentClassifier(llm_client=llm, llm_enabled=True).classify(  # type: ignore[arg-type]
        _parsed("BE IT ENACTED"), "anything"
    )
    assert (result.doc_type, result.summary, result.source) == (
        "Case Law",
        "Supreme Court ruling on bail.",
        "llm",
    )


@pytest.mark.parametrize(
    "llm",
    [
        _FakeLlm(content=json.dumps({"doc_type": "Statute", "summary": "x"})),
        _FakeLlm(content="not json"),
        _FakeLlm(error=RuntimeError("429 insufficient_quota")),
    ],
)
def test_llm_problems_fall_back_to_heuristic(llm: _FakeLlm) -> None:
    result = DocumentClassifier(llm_client=llm, llm_enabled=True).classify(  # type: ignore[arg-type]
        _parsed("BE IT ENACTED by Parliament"), "courts"
    )
    assert (result.doc_type, result.summary, result.source) == ("Act", None, "heuristic")


def test_blank_text_skips_llm() -> None:
    """A scan with no text layer gives the LLM nothing to read."""
    llm = _FakeLlm(content=json.dumps({"doc_type": "Act", "summary": "x"}))
    result = DocumentClassifier(llm_client=llm, llm_enabled=True).classify(  # type: ignore[arg-type]
        _parsed("   "), "Republic v. Ibrahim"
    )
    assert llm.calls == 0
    assert (result.doc_type, result.source) == ("Case Law", "heuristic")
