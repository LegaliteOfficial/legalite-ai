"""Firm archive: extraction, analysis, the /archive/analyze route, and the
agent's archive tools."""

from __future__ import annotations

import io
import json
import zipfile
from types import SimpleNamespace
from typing import Any

import fitz
import pytest
from fastapi.testclient import TestClient

from app.rag.workspace_tools import WorkspaceToolkit
from app.services.archive_analyzer import (
    ArchiveAnalyzer,
    UnsupportedDocumentError,
    docx_text,
    epub_text,
    normalize_analysis,
)
from evaluation.agent_chat.fixtures import FakeWorkspaceClient


class _FakeOpenAI:
    """Records requests; vision calls get a transcription, analysis calls JSON."""

    def __init__(self, analysis: dict[str, Any] | str | None = None, vision: str = "") -> None:
        self.requests: list[dict[str, Any]] = []
        self._analysis = analysis if analysis is not None else {}
        self._vision = vision
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.requests.append(kw)
        if kw.get("response_format"):
            body = self._analysis if isinstance(self._analysis, str) else json.dumps(self._analysis)
        else:
            body = self._vision
        msg = SimpleNamespace(content=body)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    @property
    def vision_calls(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if not r.get("response_format")]


def _analyzer(client: _FakeOpenAI) -> ArchiveAnalyzer:
    return ArchiveAnalyzer(client, model="gpt-test", ocr=False)  # type: ignore[arg-type]


def _docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    if table:
        rows = "".join(
            "<w:tr>"
            + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row)
            + "</w:tr>"
            for row in table
        )
        body += f"<w:tbl>{rows}</w:tbl>"
    xml = f'<?xml version="1.0"?><w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>'
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def _pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data: bytes = doc.tobytes()
    doc.close()
    return data


# -- extraction ---------------------------------------------------------------


def test_docx_paragraphs_and_tables() -> None:
    text = docx_text(_docx(["MEMO", "Re: Limitation"], [["Item", "Amount"], ["Fees", "GH₵ 5,000"]]))
    assert "MEMO\nRe: Limitation" in text
    assert "Fees | GH₵ 5,000" in text


def test_epub_follows_spine_order() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OEBPS/content.opf"/>'
            "</rootfiles></container>",
        )
        z.writestr(
            "OEBPS/content.opf",
            '<package><manifest><item id="b" href="b.xhtml"/>'
            '<item id="a" href="a.xhtml"/></manifest>'
            '<spine><itemref idref="a"/><itemref idref="b"/></spine></package>',
        )
        z.writestr(
            "OEBPS/a.xhtml", "<html><body><h1>Chapter One</h1><p>Land law.</p></body></html>"
        )
        z.writestr(
            "OEBPS/b.xhtml", "<html><body><p>Chapter Two</p><script>x()</script></body></html>"
        )
    text = epub_text(buf.getvalue())
    assert text.index("Chapter One") < text.index("Chapter Two")
    assert "x()" not in text


def test_text_pdf_uses_text_layer_without_vision() -> None:
    client = _FakeOpenAI({"title": "Writ", "document_type": "pleading"})
    ex = _analyzer(client).extract(_pdf(["WRIT OF SUMMONS between Mensah and Owusu " * 3]), "w.pdf")
    assert ex.method == "text"
    assert "WRIT OF SUMMONS" in ex.text
    assert client.vision_calls == []


def test_scanned_pdf_pages_are_transcribed_by_vision() -> None:
    client = _FakeOpenAI(vision="Page one text\n===== PAGE BREAK =====\nPage two text")
    ex = _analyzer(client).extract(_pdf(["", ""]), "scan.pdf")
    assert ex.method == "vision"
    assert "Page one text" in ex.text and "Page two text" in ex.text
    assert len(client.vision_calls) == 1
    images = [
        c for c in client.vision_calls[0]["messages"][0]["content"] if c["type"] == "image_url"
    ]
    assert len(images) == 2


def test_image_goes_to_vision() -> None:
    client = _FakeOpenAI(vision="RECEIPT No. 44")
    ex = _analyzer(client).extract(b"\x89PNG fake", "receipt.png")
    assert ex.method == "vision" and ex.text == "RECEIPT No. 44"


@pytest.mark.parametrize("name", ["old.doc", "sheet.xlsx", "noext"])
def test_unsupported_types_are_rejected_clearly(name: str) -> None:
    with pytest.raises(UnsupportedDocumentError):
        _analyzer(_FakeOpenAI()).extract(b"data", name)


# -- analysis -------------------------------------------------------------------


def test_analyze_returns_content_and_normalised_fields() -> None:
    client = _FakeOpenAI(
        {
            "title": "Memo - Limitation",
            "document_type": "Memo",
            "summary": "Advises on s.10 NRCD 54.",
            "parties": [{"name": "Ama Owusu", "role": "author"}, {"role": "no name"}],
            "document_date": "4 March 2025",
            "tags": ["Land", "land", "Limitation"],
            "confidence": 7,
            "legibility": "excellent",
        }
    )
    out = _analyzer(client).analyze(b"MEMO\nSection 10 bars recovery after 12 years.", "memo.txt")
    assert out["document_type"] == "memo"
    assert out["parties"] == [{"name": "Ama Owusu", "role": "author"}]
    assert out["document_date"] is None  # not ISO -> dropped, not guessed
    assert out["tags"] == ["land", "limitation"]
    assert out["confidence"] == 1.0
    assert out["legibility"] == "good"
    assert out["content"].startswith("MEMO")
    assert out["word_count"] == 8
    assert out["extraction_method"] == "text"
    # The document is fenced as data in the analysis call.
    user = client.requests[0]["messages"][1]["content"]
    assert "<<<DOCUMENT" in user


def test_analysis_handles_bad_json_and_empty_text() -> None:
    out = _analyzer(_FakeOpenAI("not json")).analyze(b"Some text here", "x.txt")
    assert out["title"] == "x" and out["document_type"] == "other"
    empty = _analyzer(_FakeOpenAI()).analyze(b"   ", "blank.txt")
    assert empty["legibility"] == "poor" and empty["confidence"] == 0.0


def test_long_documents_are_sampled_head_and_tail() -> None:
    client = _FakeOpenAI({"title": "Book"})
    analyzer = _analyzer(client)
    analyzer._analysis_chars = 1000
    analyzer._max_content_chars = 5000
    text = ("START " + "word " * 3000 + "END").encode()
    out = analyzer.analyze(text, "book.txt")
    sent = client.requests[0]["messages"][1]["content"]
    assert "START" in sent and "END" in sent and "omitted" in sent
    assert len(out["content"]) <= 5000
    assert any("very long" in w for w in out["warnings"])


def test_normalize_unknown_type_falls_back() -> None:
    assert normalize_analysis({"document_type": "spaceship"}, "f.pdf")["document_type"] == "other"


# -- route ----------------------------------------------------------------------


_ORG = "11111111-1111-1111-1111-111111111111"
_USER = "22222222-2222-2222-2222-222222222222"


def _tenant_headers(role: str = "lawyer") -> dict[str, str]:
    """Tenant headers, signed when the environment configures a gateway secret."""
    import time

    from app.core.service_auth import compute_signature

    ts = str(int(time.time()))
    return {
        "x-organization-id": _ORG,
        "x-user-id": _USER,
        "x-role": role,
        "x-legalite-timestamp": ts,
        "x-legalite-signature": compute_signature(
            timestamp=ts, organization_id=_ORG, user_id=_USER, role=role
        ),
    }


def _app_with(analyzer: ArchiveAnalyzer) -> TestClient:
    from app.api.routes.archive import get_archive_analyzer
    from app.main import create_app

    app = create_app()
    app.dependency_overrides[get_archive_analyzer] = lambda: analyzer
    return TestClient(app)


def test_route_requires_tenant_and_returns_analysis() -> None:
    client = _app_with(_analyzer(_FakeOpenAI({"title": "Lease", "document_type": "lease"})))
    files = {"file": ("lease.txt", b"LEASE between A and B", "text/plain")}
    assert client.post("/archive/analyze", files=files).status_code == 401
    res = client.post(
        "/archive/analyze",
        files=files,
        data={"category": "document"},
        headers=_tenant_headers(),
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["document_type"] == "lease" and body["content"].startswith("LEASE")


def test_route_rejects_unsupported_file() -> None:
    client = _app_with(_analyzer(_FakeOpenAI()))
    res = client.post(
        "/archive/analyze",
        files={"file": ("a.doc", b"x", "application/msword")},
        headers=_tenant_headers(),
    )
    assert res.status_code == 422
    assert ".docx" in res.json()["detail"]


# -- agent tools --------------------------------------------------------------------


def test_search_archive_lists_ready_hits_with_ids() -> None:
    kit = WorkspaceToolkit(FakeWorkspaceClient())  # type: ignore[arg-type]
    out = kit.execute("search_archive", {"query": "head of family consent"})
    assert "archive_id=arc-1" in out.content
    assert "LD/0412/2019" in out.content
    assert out.count == 1


def test_search_archive_retries_without_bad_type_filter() -> None:
    kit = WorkspaceToolkit(FakeWorkspaceClient())  # type: ignore[arg-type]
    out = kit.execute("search_archive", {"query": "limitation", "document_type": "judgment"})
    assert "arc-2" in out.content


def test_read_archive_entry_includes_notes_and_text() -> None:
    kit = WorkspaceToolkit(FakeWorkspaceClient())  # type: ignore[arg-type]
    out = kit.execute("read_archive_entry", {"archive_id": "arc-1"})
    assert "GH₵ 40,000" in out.content
    assert "signature page) is missing" in out.content
    assert "<<<DOCUMENT" in out.content
    assert (
        "No archived document" in kit.execute("read_archive_entry", {"archive_id": "zzz"}).content
    )
