"""Decide a PDF's ``doc_type`` from its own content.

Lets ``POST /upload-law/bulk`` accept files without the caller picking a
type: the worker parses the PDF, classifies it here, then chunks and
embeds it under the detected type.

Two-stage design, mirroring ``CaseMetadataExtractor``:

1. **Heuristic pass (always runs).** Title + first pages matched against
   the phrases Ghana instruments carry on their face ("BE IT ENACTED",
   "Legislative Instrument", "CORAM", ...). Free and deterministic, and
   the result whenever the LLM is off or fails.

2. **LLM pass (optional).** One JSON-mode call on the head of the
   document returns a label from :data:`DOC_TYPES` plus a one-to-two
   sentence summary. Gated by ``DOC_CLASSIFIER_LLM_ENABLED`` (auto: on
   iff a real-looking OpenAI key is set). An off-list label from the
   model is discarded in favour of the heuristic rather than trusted.

Classification never blocks ingestion — every failure path degrades to
the heuristic result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.pdf_parser import ParsedPdf

if TYPE_CHECKING:
    from openai import OpenAI

logger = get_logger(__name__)

# Sentinel ``doc_type`` stored on a job/document until classification runs.
AUTO_DOC_TYPE = "auto"

# Canonical labels. Kept in step with the curated list the upload UI used
# to offer, so auto-classified and hand-picked documents group together
# in corpus inventory / suggestion pools.
DOC_TYPES: tuple[str, ...] = (
    "Constitution",
    "Act",
    "Regulation",
    "Case Law",
    "Practice Direction",
    "Legal Notice",
)
FALLBACK_DOC_TYPE = "Other"

ClassificationSource = Literal["llm", "heuristic"]


@dataclass(slots=True)
class DocumentClassification:
    doc_type: str
    summary: str | None
    source: ClassificationSource


# ---------------------------------------------------------------------------
# Heuristic patterns — first match wins, most specific first. Case law
# precedes Constitution because constitutional judgments quote its full name.
# ---------------------------------------------------------------------------

_HEURISTIC_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("Practice Direction", re.compile(r"\bPRACTICE\s+DIRECTIONS?\b", re.IGNORECASE)),
    ("Legal Notice", re.compile(r"\bLEGAL\s+NOTICE\b", re.IGNORECASE)),
    (
        "Case Law",
        re.compile(
            r"\bCORAM\b|\b(?:Suit|Civil\s+Appeal|Criminal\s+Appeal)\s+No\b"
            r"|\[\d{4}\]\s+GH[A-Z]{2,4}\s+\d+|^\s*(?:JUDGMENT|RULING)\s*$",
            re.IGNORECASE | re.MULTILINE,
        ),
    ),
    (
        "Constitution",
        re.compile(r"\bCONSTITUTION\s+OF\s+THE\s+REPUBLIC\s+OF\s+GHANA\b", re.IGNORECASE),
    ),
    (
        "Regulation",
        re.compile(
            r"\bLegislative\s+Instrument\b|\bL\.\s?I\.\s?\d+|\bExecutive\s+Instrument\b"
            r"|\bREGULATIONS,?\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Act",
        re.compile(
            r"\bBE\s+IT\s+ENACTED\b|\bACT\s+\d{1,4}\b|\bACT,\s+\d{4}\b",
            re.IGNORECASE,
        ),
    ),
]

# Party separator in a filename-derived title ("Republic v. Ibrahim").
# Only trusted on the title — body text is full of "v." citations.
_TITLE_PARTIES_RE = re.compile(r"\s(?:v|vs)\.?\s", re.IGNORECASE)

_HEAD_PAGES = 3
_LLM_MAX_CHARS = 6_000


class DocumentClassifier:
    """Classifies a parsed PDF into one of :data:`DOC_TYPES`.

    Pass ``llm_enabled=False`` to force heuristic-only (used in unit tests).
    """

    def __init__(
        self,
        *,
        llm_client: OpenAI | None = None,
        llm_enabled: bool | None = None,
    ) -> None:
        settings = get_settings()
        if llm_enabled is None:
            llm_enabled = settings.doc_classifier_llm_enabled
        if llm_enabled is None:
            llm_enabled = (
                bool(settings.openai_api_key)
                and settings.openai_api_key != "test-key"
            )
        self._llm_enabled = llm_enabled
        self._llm_client = llm_client
        self._llm_model = settings.openai_llm_model

    def classify(self, parsed: ParsedPdf, title: str) -> DocumentClassification:
        head = "\n".join(p.text for p in parsed.pages[:_HEAD_PAGES])
        heuristic = classify_heuristic(head, title)

        if not self._llm_enabled or not head.strip():
            return DocumentClassification(heuristic, None, "heuristic")

        llm = self._classify_with_llm(head, title)
        if llm is None:
            return DocumentClassification(heuristic, None, "heuristic")
        return llm

    def _classify_with_llm(self, head: str, title: str) -> DocumentClassification | None:
        try:
            client = self._llm_client or self._lazy_openai_client()
            completion = client.chat.completions.create(
                model=self._llm_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _USER_PROMPT.format(
                            labels=", ".join(DOC_TYPES + (FALLBACK_DOC_TYPE,)),
                            title=title,
                            text=head[:_LLM_MAX_CHARS],
                        ),
                    },
                ],
            )
            data = json.loads((completion.choices[0].message.content or "{}").strip())
        except Exception:
            logger.exception("doc_classifier.llm_failed", title=title)
            return None

        doc_type = _canonical_label(str(data.get("doc_type") or ""))
        if doc_type is None:
            logger.warning(
                "doc_classifier.llm_off_list_label",
                title=title,
                label=data.get("doc_type"),
            )
            return None
        summary = str(data.get("summary") or "").strip() or None
        return DocumentClassification(doc_type, summary, "llm")

    def _lazy_openai_client(self) -> OpenAI:
        from openai import OpenAI as _OpenAI

        return _OpenAI(api_key=get_settings().openai_api_key)


def classify_heuristic(head_text: str, title: str) -> str:
    """Rule-based label from the title and first pages.

    The title is checked on its own first: a filename like
    ``Labour Act, 2003 (Act 651).pdf`` is a stronger signal than body
    text, which routinely cites other instruments.
    """
    if _TITLE_PARTIES_RE.search(f" {title} "):
        return "Case Law"
    for haystack in (title, head_text):
        for label, pattern in _HEURISTIC_RULES:
            if pattern.search(haystack):
                return label
    return FALLBACK_DOC_TYPE


def _canonical_label(raw: str) -> str | None:
    wanted = raw.strip().lower()
    for label in DOC_TYPES + (FALLBACK_DOC_TYPE,):
        if label.lower() == wanted:
            return label
    return None


_SYSTEM_PROMPT = (
    "You classify Ghanaian legal documents. Base the answer only on the "
    "text provided. The summary restates what the document is and covers; "
    "never invent facts, parties, dates or holdings."
)

_USER_PROMPT = (
    "Classify this document and summarise it.\n\n"
    "Return a strict JSON object with keys:\n"
    '  "doc_type": exactly one of: {labels}\n'
    '  "summary": one or two plain sentences on what the document is '
    "(e.g. which Act, which court and parties) and its subject matter.\n\n"
    "Filename-derived title: {title}\n\n"
    "Opening pages:\n{text}"
)
