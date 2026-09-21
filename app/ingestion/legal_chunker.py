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
    # Phase 5 M9 — character offsets within page_start / page_end.
    # Populated best-effort by locating the chunk content in the source
    # pages after emission. NULL when the chunk text cannot be located
    # (e.g. heavy whitespace normalization).
    start_char: int | None = None
    end_char: int | None = None


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
        self._max_chunk_chars = max_chunk_chars or settings.rag_max_chunk_chars
        self._overlap_chars = chunk_overlap or settings.rag_chunk_overlap_chars

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
        emitted = [c for c in chunks if c.content]
        _attach_spans(emitted, parsed)
        return emitted


def _attach_doc_meta(chunk: LegalChunk, meta: dict[str, object]) -> None:
    chunk.law_type = meta.get("law_type")  # type: ignore[assignment]
    chunk.act_number = meta.get("act_number")  # type: ignore[assignment]
    chunk.effective_year = meta.get("effective_year")  # type: ignore[assignment]


def _attach_spans(chunks: list[LegalChunk], parsed: ParsedPdf) -> None:
    """Locate each emitted chunk's text inside the source page text.

    The chunker normalizes line whitespace, so the chunk content does not
    always match the raw page byte-for-byte. We try two strategies in order:

    1. Exact ``str.find`` of the leading & trailing characters within the
       chunk's ``page_start`` / ``page_end`` text.
    2. Locate the first non-whitespace word of the chunk in the start
       page; locate the last non-whitespace word in the end page.

    On failure we leave the offsets None — the UI falls back to page-only
    highlighting. We track a per-page cursor so two chunks that share the
    same opening words still resolve to distinct offsets.
    """
    pages_by_number: dict[int, str] = {p.page_number: p.text for p in parsed.pages}
    cursor_by_page: dict[int, int] = {}

    for chunk in chunks:
        if not chunk.content:
            continue
        start_text = _leading_anchor(chunk.content)
        end_text = _trailing_anchor(chunk.content)
        if chunk.page_start is not None:
            page_text = pages_by_number.get(chunk.page_start, "")
            start = _find_from(page_text, start_text, cursor_by_page.get(chunk.page_start, 0))
            if start >= 0:
                chunk.start_char = start
                cursor_by_page[chunk.page_start] = start + len(start_text)
        if chunk.page_end is not None:
            page_text = pages_by_number.get(chunk.page_end, "")
            # End offset = (position of end_text) + len(end_text). If the
            # chunk fits on one page and end_text isn't found again, fall
            # back to start_char + len(content) capped to the page length.
            search_from = (
                chunk.start_char
                if chunk.start_char is not None and chunk.page_end == chunk.page_start
                else cursor_by_page.get(chunk.page_end, 0)
            )
            idx = _find_from(page_text, end_text, search_from)
            if idx >= 0:
                chunk.end_char = idx + len(end_text)
            elif chunk.start_char is not None and chunk.page_end == chunk.page_start:
                chunk.end_char = min(
                    chunk.start_char + len(chunk.content), len(page_text)
                )


def _leading_anchor(content: str, length: int = 40) -> str:
    """First N visible characters of the chunk — used as a forward-search anchor."""
    stripped = content.lstrip()
    return stripped[:length]


def _trailing_anchor(content: str, length: int = 40) -> str:
    stripped = content.rstrip()
    return stripped[-length:]


def _find_from(haystack: str, needle: str, start: int) -> int:
    """``str.find`` with a minimum start cursor; -1 if not found at or after."""
    if not haystack or not needle:
        return -1
    return haystack.find(needle, max(0, start))


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


_SENTENCE_END_RE = re.compile(r"[.!?][\"'\)\]]?(?=\s|\n|$)")
# How far back from a hard character cutoff to search for a natural break
# (sentence end, then line break, then any whitespace) before giving up
# and accepting the hard cutoff. A fraction of max_chars rather than a
# fixed constant so it scales sensibly whether max_chars is 500 or 50000.
_BREAK_SEARCH_FRACTION = 0.2


def _find_break_point(text: str, hard_end: int, *, min_end: int) -> int:
    """Find the best place at or before ``hard_end`` to end a slice.

    Searches backward within a window for, in priority order: a sentence
    ending, a line break, any whitespace. Never returns a value <= the
    slice's start (``min_end``) — callers must fall back to ``hard_end``
    in that case to guarantee forward progress on text with no usable
    break in range (e.g. one very long unbroken token).
    """
    if hard_end >= len(text):
        return len(text)

    lookback = max(1, int((hard_end - min_end) * _BREAK_SEARCH_FRACTION))
    window_start = max(min_end, hard_end - lookback)
    window = text[window_start:hard_end]

    last_sentence_end = None
    for m in _SENTENCE_END_RE.finditer(window):
        last_sentence_end = m.end()
    if last_sentence_end is not None:
        return window_start + last_sentence_end

    last_newline = window.rfind("\n")
    if last_newline != -1:
        return window_start + last_newline + 1

    last_space = window.rfind(" ")
    if last_space != -1:
        return window_start + last_space + 1

    return hard_end


def _snap_forward_past_partial_word(text: str, pos: int, *, limit: int) -> int:
    """If ``pos`` lands inside a word, advance to the start of the next
    whole word (skip to the end of the current whitespace run following
    it). Choosing a clean END boundary for one slice is not enough — the
    NEXT slice's start is ``end - overlap``, an arithmetic offset with no
    guarantee it lands on a boundary too. Without this, the overlap
    region can begin mid-word (e.g. a slice starting with a stray "s"
    left over from "This").

    Returns ``limit`` unchanged if no whitespace exists before it (e.g.
    the overlap window sits entirely inside one long unbroken token) —
    the caller then skips overlap for that split rather than risk a
    fragment.
    """
    if pos <= 0 or pos >= limit or text[pos - 1].isspace():
        return pos  # already at (or past) a clean boundary
    i = pos
    while i < limit and not text[i].isspace():
        i += 1
    while i < limit and text[i].isspace():
        i += 1
    return i


def _split_oversized(chunk: LegalChunk, max_chars: int, overlap: int) -> list[LegalChunk]:
    """If a single hierarchy unit is huge, split with overlap to preserve context.

    Prefers a sentence or whitespace boundary near the character limit over
    a hard cut, so this never slices a word in half when a reasonable break
    point exists nearby. Falls back to a hard cut only when no break is
    found within the search window (e.g. a single run of text with no
    spaces at all) — that guarantees the loop always makes forward
    progress and terminates.
    """
    text = chunk.content
    if len(text) <= max_chars:
        return [chunk]

    parts: list[LegalChunk] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = _find_break_point(text, hard_end, min_end=start)
        if end <= start:
            end = hard_end
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
        next_start = end - overlap
        if next_start > start:
            next_start = _snap_forward_past_partial_word(text, next_start, limit=end)
        start = next_start if next_start > start else end
    return parts
