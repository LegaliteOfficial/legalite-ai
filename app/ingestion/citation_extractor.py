"""Extract inline case-to-case citations from a judgment body.

We only attempt the patterns that Ghana judgments use in practice. Resolving
the citation to a ``LegalCase`` in our corpus is deferred to a separate
matcher pass (`CaseCitationMatcher`) that runs after ingestion of a batch,
because Case B may cite Case A even when A hasn't been ingested yet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(slots=True)
class ExtractedCitation:
    raw_text: str
    case_title: str | None = None
    reporter: str | None = None


# "[2020] GHASC 5" / "[2018] GHACA 22" / "[1996-97] SCGLR 804" — the canonical
# reporter form. We anchor on this and look BEHIND it for an optional case title,
# rather than letting the title regex run greedy across sentence prefixes.
_REPORTER_RE = re.compile(
    r"\[(?P<year>\d{4}(?:-\d{2,4})?)\]\s+(?P<reporter>GHASC|GHACA|GHAHC|SCGLR|GLR|GHAR|GHA[A-Z]{2,3})\s+(?P<num>\d+)",
    re.IGNORECASE,
)

# Looks backwards from the reporter for a "Party A v Party B" caption.
# Bounded to ~200 chars and requires the title to start at a likely caption
# boundary (sentence start, comma, or after "see"/"in"/"following").
_TITLE_BEFORE_REPORTER_RE = re.compile(
    r"(?:^|[\s\.,;\(]|\b(?:in|see|see\s+also|following|citing|per|cf\.?)\s+)"
    r"(?P<title>[A-Z][A-Za-z0-9'\-\.&]+(?:\s+[A-Za-z0-9'\-\.&,()]+){0,15}?"
    r"\s+v\.?\s+"
    r"[A-Z][A-Za-z0-9'\-\.&]+(?:\s+[A-Za-z0-9'\-\.&,()]+){0,15}?)\s*$",
    re.IGNORECASE,
)


class CitationExtractor:
    def extract(self, text: str) -> list[ExtractedCitation]:
        out: list[ExtractedCitation] = []
        seen_reporter_keys: set[str] = set()

        for m in _REPORTER_RE.finditer(text):
            reporter_key = f"{m.group('year')}|{m.group('reporter').upper()}|{m.group('num')}"
            if reporter_key in seen_reporter_keys:
                continue
            seen_reporter_keys.add(reporter_key)

            # Look up to 200 chars behind for a case title attached to this reporter.
            window_start = max(0, m.start() - 200)
            window = text[window_start : m.start()]
            title = _title_before(window)
            raw_start = window_start + (window.rfind(title) if title else len(window))
            raw_end = m.end()
            raw = _normalize(text[raw_start:raw_end])

            out.append(
                ExtractedCitation(
                    raw_text=raw,
                    case_title=title,
                    reporter=m.group("reporter").upper(),
                )
            )

        return out


def _title_before(window: str) -> str | None:
    """Return a likely case-caption title that appears just before a reporter."""
    if " v" not in window.lower():
        return None
    m = _TITLE_BEFORE_REPORTER_RE.search(window.strip())
    if not m:
        return None
    return _clean_title(m.group("title"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_title(text: str) -> str:
    title = re.sub(r"\s+", " ", text).strip(" .,;:'\"-")
    return title[:512]
