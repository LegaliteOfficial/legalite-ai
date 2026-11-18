"""Re-derive hierarchy labels for chunks that are already stored.

Documents ingested before the section plan (see
``app/ingestion/legal_chunker.py``) carry labels taken from any line that
started with a number: judgment paragraphs became "sections", Schedule
items became sections near the end of an Act, and the Constitution's
articles were labelled sections. This replays the chunker's rules over the
stored chunks of one document, in page order, and returns only the labels
that must change. Content, boundaries and vectors are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.ingestion.legal_chunker import (
    _SCHEDULE_RE,
    _detect_level,
    _doc_mode,
    _Event,
    _event_for,
    plan_sections,
)

LABELS = ("article", "section", "clause", "subsection")


@dataclass(slots=True)
class StoredChunk:
    id: Any
    vector_id: str | None
    page_start: int | None
    start_char: int | None
    content: str
    article: str | None
    section: str | None
    clause: str | None
    subsection: str | None


@dataclass(slots=True)
class LabelChange:
    chunk: StoredChunk
    new: dict[str, str | None]

    def diff(self) -> dict[str, tuple[str | None, str | None]]:
        return {k: (getattr(self.chunk, k), v) for k, v in self.new.items()}


def _lines(text: str) -> tuple[str, list[str]]:
    """First non-empty line, and the non-empty lines after it."""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return (lines[0], lines[1:]) if lines else ("", [])


def relabel(
    chunks: list[StoredChunk],
    *,
    doc_type: str | None,
    law_type: str | None,
    total_pages: int | None,
    title: str | None = None,
) -> list[LabelChange]:
    """Labels that differ from what the current chunker would assign."""
    mode = _doc_mode(doc_type, law_type, title)
    changes: list[LabelChange] = []

    if mode == "judgment":
        for c in chunks:
            new = {k: None for k in LABELS if getattr(c, k) is not None}
            if new:
                changes.append(LabelChange(c, new))
        return changes

    # Chunks without a position inside their page cannot be ordered
    # reliably: they neither drive the replay nor get changed.
    ordered = sorted(
        (c for c in chunks if c.start_char is not None),
        key=lambda c: (c.page_start or 0, c.start_char or 0),
    )
    pages = total_pages or max((c.page_start or 0 for c in ordered), default=1)

    # Each stored chunk starts at the marker that opened it; a Schedule
    # heading may sit anywhere inside it.
    per_chunk: list[tuple[_Event | None, int]] = []
    events: list[_Event] = []
    for c in ordered:
        page = c.page_start or 0
        first, rest = _lines(c.content)
        ev = _event_for(first, mode, page)
        if ev is not None:
            events.append(ev)
        if ev is None and _SCHEDULE_RE.match(first):
            events.append(_Event("schedule", page))
        n_sched = sum(1 for ln in rest if _SCHEDULE_RE.match(ln))
        events.extend(_Event("schedule", page) for _ in range(n_sched))
        per_chunk.append((ev, n_sched + (1 if ev is None and _SCHEDULE_RE.match(first) else 0)))
    verdicts = iter(plan_sections(events, pages))

    article: str | None = None
    section: str | None = None
    closed = False
    for c, (ev, n_sched) in zip(ordered, per_chunk, strict=True):
        first, _ = _lines(c.content)
        level = _detect_level(first)
        if ev is not None and ev.kind == "article":
            next(verdicts)
            article, section = level[1] if level else None, None
        elif ev is not None:
            verdict = next(verdicts)
            if verdict == "close":
                closed = True
            elif verdict == "gap":
                section = None
            elif verdict == "accept" and level is not None:
                if mode == "constitution":
                    article, section = level[1], None
                else:
                    section = level[1]
        elif level is not None:
            kind, value = level
            if kind in ("title", "chapter", "part"):
                article = section = None
            elif kind == "article" and mode != "constitution":
                article, section = value, None
        want: dict[str, str | None] = {"article": article, "section": section}
        if closed:
            want = {
                "article": None if mode == "constitution" else article,
                "section": None,
                "clause": None,
                "subsection": None,
            }
        new = {k: v for k, v in want.items() if getattr(c, k) != v}
        if new:
            changes.append(LabelChange(c, new))
        for _ in range(n_sched):
            if next(verdicts) == "close":
                closed = True
    return changes


__all__ = ["LABELS", "LabelChange", "StoredChunk", "relabel"]
