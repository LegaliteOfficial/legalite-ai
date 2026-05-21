"""Structure-aware chunker for Ghana legal documents (v2).

Why custom: generic recursive splitters break citations across article/section
boundaries. This version:

1. Detects a document-level header (Constitution / Act No. X / Regulation /
   Judgment) and propagates law metadata to every emitted chunk.
2. Scans each page for legal-hierarchy markers and emits one chunk per leaf
   unit. Higher levels (Title → Chapter → Part → Article → Section → Clause →
   Subsection) carry forward; setting a higher level clears the lower ones.
3. Falls back to size-based splitting (with overlap) for prose blocks without
   explicit markers.

Hierarchy detected:
    Title  →  Chapter  →  Part  →  Article  →  Section  →  Clause  →  Subsection

Works for Constitution, Acts, Regulations, and (best-effort) Judgments.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.ingestion.pdf_parser import ParsedPdf

# ---------------------------------------------------------------------------
# Document-level metadata detection
# ---------------------------------------------------------------------------

_ACT_NO_RE = re.compile(
    r"\bAct(?:\s+No\.?)?\s*(\d{1,4})\b", re.IGNORECASE
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_REGULATION_RE = re.compile(r"\b(L\.?I\.?\s*\d+|Legislative\s+Instrument)\b", re.IGNORECASE)
_CONSTITUTION_RE = re.compile(r"\bCONSTITUTION\s+OF\s+THE\s+REPUBLIC\s+OF\s+GHANA\b", re.IGNORECASE)
_JUDGMENT_RE = re.compile(
    r"\b(JUDGMENT|RULING|HOLDING|SUPREME\s+COURT|HIGH\s+COURT|COURT\s+OF\s+APPEAL)\b",
    re.IGNORECASE,
)


def _detect_doc_metadata(parsed: ParsedPdf, declared_title: str | None) -> dict[str, object]:
    """Best-effort extraction of (law_type, act_number, effective_year) from the head of the doc.

    Inspects the first ~3 pages. We deliberately keep this conservative: if a
    signal isn't clearly present, we leave the field None so the chunker
    doesn't fabricate metadata.
    """
    head = "\n".join(p.text for p in parsed.pages[:3])
    haystack = f"{declared_title or ''}\n{head}"

    law_type: str | None = None
    act_number: str | None = None
    effective_year: int | None = None

    if _CONSTITUTION_RE.search(haystack):
        law_type = "constitution"
    elif _JUDGMENT_RE.search(haystack):
        law_type = "case_law"
    elif _REGULATION_RE.search(haystack):
        law_type = "regulation"
    elif _ACT_NO_RE.search(haystack):
        law_type = "act"

    m = _ACT_NO_RE.search(haystack)
    if m:
        act_number = m.group(1)

    years = [int(m.group(0)) for m in _YEAR_RE.finditer(haystack)]
    if years:
        # Prefer the latest year in the header — usually the year of enactment/publication.
        effective_year = max(years)

    return {
        "law_type": law_type,
        "act_number": act_number,
        "effective_year": effective_year,
    }


# ---------------------------------------------------------------------------
# Hierarchy markers
# ---------------------------------------------------------------------------

# Order matters — first match wins per line.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("title", re.compile(r"^\s*TITLE\s+([IVXLC\d]+)\b.*$", re.IGNORECASE)),
    ("chapter", re.compile(r"^\s*CHAPTER\s+([IVXLC\d]+)\b.*$", re.IGNORECASE)),
    ("part", re.compile(r"^\s*PART\s+([IVXLC\d]+)\b.*$", re.IGNORECASE)),
    ("article", re.compile(r"^\s*ARTICLE\s+(\d+[A-Z]?)\b.*$", re.IGNORECASE)),
    ("section", re.compile(r"^\s*SECTION\s+(\d+[A-Z]?(?:\(\d+\))?)\b.*$", re.IGNORECASE)),
    # Numeric section markers like "12." or "12(2)"
    ("section", re.compile(r"^\s*(\d{1,3}[A-Z]?(?:\(\d+\))?)\.\s+")),
    ("clause", re.compile(r"^\s*\(([a-z])\)\s+")),
    ("subsection", re.compile(r"^\s*\(([ivxlcdm]+)\)\s+", re.IGNORECASE)),
]

# Heuristic legal topic mapping based on keywords in the chunk content.
_TOPIC_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("employment", re.compile(r"\b(employer|employee|dismissal|wages?|labou?r|maternity)\b", re.IGNORECASE)),
    ("landlord_tenant", re.compile(r"\b(landlord|tenant|tenancy|eviction|rent)\b", re.IGNORECASE)),
    ("criminal", re.compile(r"\b(offen[cs]e|imprisonment|conviction|warrant|arrest|bail)\b", re.IGNORECASE)),
    ("constitutional", re.compile(r"\b(fundamental\s+rights?|liberty|equality|due\s+process)\b", re.IGNORECASE)),
    ("family", re.compile(r"\b(marriage|divorce|custody|guardian|matrimonial)\b", re.IGNORECASE)),
    ("contract", re.compile(r"\b(contract|breach|consideration|offer|acceptance)\b", re.IGNORECASE)),
    ("property", re.compile(r"\b(conveyance|easement|deed|title|mortgage|land)\b", re.IGNORECASE)),
]


def _detect_topic(text: str) -> str | None:
    for topic, pattern in _TOPIC_KEYWORDS:
        if pattern.search(text):
            return topic
    return None


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class LegalChunk:
    content: str
    title: str | None = None
    chapter: str | None = None
    part: str | None = None
    article: str | None = None
    section: str | None = None
    clause: str | None = None
    subsection: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    # Extended metadata (from doc-level detection + topic heuristic)
    law_type: str | None = None
    act_number: str | None = None
    effective_year: int | None = None
    legal_topic: str | None = None


@dataclass(slots=True)
class _Cursor:
    title: str | None = None
    chapter: str | None = None
    part: str | None = None
    article: str | None = None
    section: str | None = None
    clause: str | None = None
    subsection: str | None = None
    buffer: list[str] = field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None

    def snapshot(self) -> LegalChunk:
        return LegalChunk(
            content="\n".join(self.buffer).strip(),
            title=self.title,
            chapter=self.chapter,
            part=self.part,
            article=self.article,
            section=self.section,
            clause=self.clause,
            subsection=self.subsection,
            page_start=self.page_start,
            page_end=self.page_end,
        )

    def reset_buffer(self) -> None:
        self.buffer = []
        self.page_start = None
        self.page_end = None


class LegalChunker:
    """Splits a parsed PDF into hierarchy-aware chunks with rich metadata."""

    def __init__(
        self,
        *,
        max_chunk_chars: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self._max_chunk_chars = max_chunk_chars or settings.rag_chunk_size * 4
        self._overlap_chars = chunk_overlap or settings.rag_chunk_overlap * 4

    def chunk(self, parsed: ParsedPdf, doc_title: str | None = None) -> list[LegalChunk]:
        doc_meta = _detect_doc_metadata(parsed, doc_title)
        cursor = _Cursor(title=doc_title)
        chunks: list[LegalChunk] = []

        def flush() -> None:
            if cursor.buffer and "".join(cursor.buffer).strip():
                chunk = cursor.snapshot()
                for part in _split_oversized(chunk, self._max_chunk_chars, self._overlap_chars):
                    _attach_doc_meta(part, doc_meta)
                    part.legal_topic = _detect_topic(part.content)
                    chunks.append(part)
            cursor.reset_buffer()

        for page in parsed.pages:
            for raw_line in page.text.splitlines():
                line = raw_line.rstrip()
                if not line.strip():
                    cursor.buffer.append("")
                    continue

                level = _detect_level(line)
                if level is None:
                    if cursor.page_start is None:
                        cursor.page_start = page.page_number
                    cursor.page_end = page.page_number
                    cursor.buffer.append(line)
                    continue

                flush()
                _apply_marker(cursor, level, line)
                cursor.page_start = page.page_number
                cursor.page_end = page.page_number
                cursor.buffer.append(line)

        flush()
        return [c for c in chunks if c.content]


def _attach_doc_meta(chunk: LegalChunk, meta: dict[str, object]) -> None:
    chunk.law_type = meta.get("law_type")  # type: ignore[assignment]
    chunk.act_number = meta.get("act_number")  # type: ignore[assignment]
    chunk.effective_year = meta.get("effective_year")  # type: ignore[assignment]


def _detect_level(line: str) -> tuple[str, str] | None:
    for level, pattern in _PATTERNS:
        m = pattern.match(line)
        if m:
            return level, m.group(1)
    return None


def _apply_marker(cursor: _Cursor, level_value: tuple[str, str], line: str) -> None:
    level, value = level_value
    # Setting a higher level clears lower levels so citations stay accurate.
    if level == "title":
        cursor.title = line.strip()
        cursor.chapter = cursor.part = cursor.article = cursor.section = None
        cursor.clause = cursor.subsection = None
    elif level == "chapter":
        cursor.chapter = value
        cursor.part = cursor.article = cursor.section = cursor.clause = cursor.subsection = None
    elif level == "part":
        cursor.part = value
        cursor.article = cursor.section = cursor.clause = cursor.subsection = None
    elif level == "article":
        cursor.article = value
        cursor.section = cursor.clause = cursor.subsection = None
    elif level == "section":
        cursor.section = value
        cursor.clause = cursor.subsection = None
    elif level == "clause":
        cursor.clause = value
        cursor.subsection = None
    elif level == "subsection":
        cursor.subsection = value


def _split_oversized(chunk: LegalChunk, max_chars: int, overlap: int) -> list[LegalChunk]:
    """If a single hierarchy unit is huge, split with overlap to preserve context."""
    text = chunk.content
    if len(text) <= max_chars:
        return [chunk]

    parts: list[LegalChunk] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        parts.append(
            LegalChunk(
                content=text[start:end].strip(),
                title=chunk.title,
                chapter=chunk.chapter,
                part=chunk.part,
                article=chunk.article,
                section=chunk.section,
                clause=chunk.clause,
                subsection=chunk.subsection,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
        )
        if end == len(text):
            break
        start = end - overlap if overlap < max_chars else end
    return parts
