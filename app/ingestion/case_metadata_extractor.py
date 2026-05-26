"""Extract structured case-law metadata from a parsed PDF.

Two-stage design:

1. **Regex / heuristic pass (always runs).** Deterministic, free, and good
   for the structured fields Ghana judgments expose reliably: case_number,
   court, decision_date, judges, parties, basic outcome verbs ("dismissed",
   "appeal allowed"). Topics reuse the keyword table the chunker already has
   so a case and a statute landing on the same topic agree.

2. **LLM enrichment pass (optional).** For narrative fields the regex pass
   cannot reliably produce — ``holding`` and ``facts_summary`` — a single
   structured-output call on the cleaned text. Gated by
   ``CASE_LLM_ENRICHMENT_ENABLED`` (default off when ``OPENAI_API_KEY`` is the
   test sentinel; on otherwise) so tests stay deterministic and we don't pay
   for tokens during fixture-based unit tests.

The extractor never mutates the existing chunker's output. It runs in
parallel on the same ``ParsedPdf`` and emits a ``CaseMetadata`` dataclass that
``IngestionService`` persists via ``CaseRepository``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.pdf_parser import ParsedPdf

if TYPE_CHECKING:
    from openai import OpenAI

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CasePartyExtract:
    role: str
    party_name: str


@dataclass(slots=True)
class CaseMetadata:
    case_title: str
    case_number: str | None = None
    court: str | None = None
    judges: list[str] = field(default_factory=list)
    decision_date: date | None = None
    outcome: str | None = None
    holding: str | None = None
    facts_summary: str | None = None
    legal_topics: list[str] = field(default_factory=list)
    parties: list[CasePartyExtract] = field(default_factory=list)
    # diagnostic: which fields came from LLM vs regex (helps debugging upstream)
    enrichment_used: bool = False


# ---------------------------------------------------------------------------
# Regex patterns — calibrated against Ghana judgment formats
# ---------------------------------------------------------------------------

# "Suit No. J4/12/2020", "Civil Appeal No. J4/55/2018", "Writ No. JR/12/2020"
_CASE_NUMBER_RE = re.compile(
    r"\b(?:Suit|Civil\s+Appeal|Criminal\s+Appeal|Writ|Application|Motion)\s+No\.?\s*"
    r"([A-Z]{1,4}/?\d+/?[A-Z]?\d*(?:/\d{2,4})?)",
    re.IGNORECASE,
)

# Ghana reporter citations: "[2020] GHASC 5", "[2018] GHACA 22", "[2019] GHAHC 14"
_REPORTER_CITATION_RE = re.compile(
    r"\[(\d{4})\]\s+(GHASC|GHACA|GHAHC|SCGLR|GLR|GH[A-Z]{0,3})\s+(\d+)",
    re.IGNORECASE,
)

_COURT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Supreme Court", re.compile(r"\bSUPREME\s+COURT\b", re.IGNORECASE)),
    ("Court of Appeal", re.compile(r"\bCOURT\s+OF\s+APPEAL\b", re.IGNORECASE)),
    ("High Court", re.compile(r"\bHIGH\s+COURT\b", re.IGNORECASE)),
    ("Circuit Court", re.compile(r"\bCIRCUIT\s+COURT\b", re.IGNORECASE)),
    ("District Court", re.compile(r"\bDISTRICT\s+COURT\b", re.IGNORECASE)),
]

# "JUSTICE", "JSC" (Justice of the Supreme Court), "JA" (Justice of Appeal), "J."
# Match a name in title case, allow particles ("Dr.", "Mrs.").
_JUDGE_RE = re.compile(
    r"(?:JUSTICE|HON\.?(?:OURABLE)?\s+(?:JUSTICE|MR\.?\s+JUSTICE|MRS?\.?\s+JUSTICE))\s+"
    r"([A-Z][A-Za-z'\-\.]+(?:\s+[A-Z][A-Za-z'\-\.]+){0,3})",
    re.IGNORECASE,
)
# Trailing post-nominal form: "KOFI ADUMUAH-BOSSMAN, JSC" / "AKAMBA, JSC"
_JUDGE_POSTNOMINAL_RE = re.compile(
    r"\b([A-Z][A-Z'\-]+(?:\s+[A-Z][A-Z'\-]+){0,3}),?\s+(JSC|JA|J\.|JSC\.|J\.A\.)\b"
)

# Parties: "BETWEEN: A v. B" / "REPUBLIC v TOMMY THOMPSON BOOKS LTD" / "A v. B (No. 2)"
_PARTIES_VS_RE = re.compile(
    r"([A-Z][A-Z0-9 ,&'\-\.()]{2,200}?)\s+(?:v\.?|vs\.?|versus)\s+([A-Z][A-Z0-9 ,&'\-\.()]{2,200})",
    re.IGNORECASE,
)

# Decision date — months long-form. We also try ISO and DD/MM/YYYY.
_DATE_LONG_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)
_DATE_ISO_RE = re.compile(r"\b(20\d{2}|19\d{2})-(\d{2})-(\d{2})\b")
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

_OUTCOME_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("appeal allowed", re.compile(r"\bappeal\s+(?:is\s+)?allowed\b", re.IGNORECASE)),
    ("appeal dismissed", re.compile(r"\bappeal\s+(?:is\s+)?dismissed\b", re.IGNORECASE)),
    ("application granted", re.compile(r"\bapplication\s+(?:is\s+)?granted\b", re.IGNORECASE)),
    ("application dismissed", re.compile(r"\bapplication\s+(?:is\s+)?dismissed\b", re.IGNORECASE)),
    ("judgment for plaintiff", re.compile(r"\bjudgment\s+(?:is\s+)?(?:entered\s+)?for\s+the\s+plaintiff\b", re.IGNORECASE)),
    ("judgment for defendant", re.compile(r"\bjudgment\s+(?:is\s+)?(?:entered\s+)?for\s+the\s+defendant\b", re.IGNORECASE)),
    ("acquitted", re.compile(r"\bacquitted\b", re.IGNORECASE)),
    ("convicted", re.compile(r"\bconvicted\b", re.IGNORECASE)),
]

# Shared topic vocabulary with the chunker. Keep this list as the canonical
# legal-topic taxonomy so chunks and cases tag identically.
_TOPIC_KEYWORDS: list[tuple[str, re.Pattern[str]]] = [
    ("employment", re.compile(r"\b(employer|employee|dismissal|wages?|labou?r|maternity)\b", re.IGNORECASE)),
    ("landlord_tenant", re.compile(r"\b(landlord|tenant|tenancy|eviction|rent)\b", re.IGNORECASE)),
    ("criminal", re.compile(r"\b(offen[cs]e|imprisonment|conviction|warrant|arrest|bail)\b", re.IGNORECASE)),
    ("constitutional", re.compile(r"\b(fundamental\s+rights?|liberty|equality|due\s+process|article\s+\d+\s+of\s+the\s+Constitution)\b", re.IGNORECASE)),
    ("family", re.compile(r"\b(marriage|divorce|custody|guardian|matrimonial)\b", re.IGNORECASE)),
    ("contract", re.compile(r"\b(contract|breach|consideration|offer|acceptance|repudiation)\b", re.IGNORECASE)),
    ("property", re.compile(r"\b(conveyance|easement|deed|title|mortgage|land|trespass)\b", re.IGNORECASE)),
    ("tort", re.compile(r"\b(negligence|nuisance|defamation|duty\s+of\s+care)\b", re.IGNORECASE)),
    ("corporate", re.compile(r"\b(company|directors?|shareholders?|winding\s+up|corporate)\b", re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class CaseMetadataExtractor:
    """Extracts CaseMetadata from a parsed judgment PDF.

    Pass ``llm_client=None`` to disable enrichment (used in unit tests).
    """

    def __init__(
        self,
        *,
        llm_client: OpenAI | None = None,
        llm_enrichment_enabled: bool | None = None,
    ) -> None:
        settings = get_settings()
        if llm_enrichment_enabled is None:
            llm_enrichment_enabled = settings.case_llm_enrichment_enabled
        if llm_enrichment_enabled is None:
            # Auto: on iff we have a real-looking API key (not the test sentinel).
            llm_enrichment_enabled = (
                bool(settings.openai_api_key)
                and settings.openai_api_key != "test-key"
            )
        self._llm_enrichment_enabled = llm_enrichment_enabled
        self._llm_client = llm_client
        self._llm_model = settings.openai_llm_model

    # ---- public API --------------------------------------------------------

    def extract(self, parsed: ParsedPdf, declared_title: str | None = None) -> CaseMetadata:
        # The first ~5 pages of a judgment hold the header (court, parties,
        # case number, date, judges). The body of the judgment holds the
        # holding/facts. We cap each to keep prompts cheap.
        head_pages = parsed.pages[:5]
        head_text = "\n".join(p.text for p in head_pages)
        body_text = "\n".join(p.text for p in parsed.pages[:30])  # cap for LLM cost

        meta = self._extract_regex(head_text, body_text, declared_title)

        if self._llm_enrichment_enabled and (meta.holding is None or meta.facts_summary is None):
            self._enrich_with_llm(meta, body_text)

        return meta

    # ---- regex pass --------------------------------------------------------

    def _extract_regex(
        self, head_text: str, body_text: str, declared_title: str | None
    ) -> CaseMetadata:
        parties = self._extract_parties(head_text)
        case_title = self._build_case_title(parties, declared_title, head_text)
        return CaseMetadata(
            case_title=case_title,
            case_number=self._extract_case_number(head_text),
            court=self._extract_court(head_text),
            judges=self._extract_judges(head_text),
            decision_date=self._extract_decision_date(head_text),
            outcome=self._extract_outcome(body_text),
            legal_topics=self._extract_topics(body_text),
            parties=parties,
        )

    @staticmethod
    def _extract_case_number(text: str) -> str | None:
        m = _CASE_NUMBER_RE.search(text)
        if m:
            return m.group(0).strip()
        m = _REPORTER_CITATION_RE.search(text)
        if m:
            return m.group(0).strip()
        return None

    @staticmethod
    def _extract_court(text: str) -> str | None:
        for label, pattern in _COURT_PATTERNS:
            if pattern.search(text):
                return label
        return None

    @staticmethod
    def _extract_judges(text: str) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for m in _JUDGE_RE.finditer(text):
            name = _clean_name(m.group(1))
            if name and name.lower() not in seen:
                names.append(name)
                seen.add(name.lower())
        for m in _JUDGE_POSTNOMINAL_RE.finditer(text):
            name = _clean_name(m.group(1))
            if name and name.lower() not in seen and len(name.split()) <= 4:
                names.append(name)
                seen.add(name.lower())
        return names[:7]  # cap to avoid swallowing party names that look like judges

    @staticmethod
    def _extract_decision_date(text: str) -> date | None:
        # Prefer the explicit long form found near "JUDGMENT" / "DELIVERED ON".
        focus = text
        anchor = re.search(
            r"\b(?:DELIVERED\s+ON|DATE|JUDGMENT\s+DATE)[:\s]*",
            text,
            re.IGNORECASE,
        )
        if anchor:
            focus = text[anchor.end(): anchor.end() + 200]

        for candidate in (focus, text):
            m = _DATE_LONG_RE.search(candidate)
            if m:
                day, month, year = m.group(1), m.group(2), m.group(3)
                try:
                    return datetime.strptime(
                        f"{day} {month} {year}", "%d %B %Y"
                    ).date()
                except ValueError:
                    pass
            m = _DATE_ISO_RE.search(candidate)
            if m:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            m = _DATE_SLASH_RE.search(candidate)
            if m:
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                try:
                    return date(y, mo, d)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_outcome(text: str) -> str | None:
        for label, pattern in _OUTCOME_PATTERNS:
            if pattern.search(text):
                return label
        return None

    @staticmethod
    def _extract_topics(text: str) -> list[str]:
        topics: list[str] = []
        for topic, pattern in _TOPIC_KEYWORDS:
            if pattern.search(text):
                topics.append(topic)
        return topics

    @staticmethod
    def _extract_parties(text: str) -> list[CasePartyExtract]:
        """Heuristic party extraction.

        We take the FIRST ``X v Y`` match found in the head (which is almost
        always the case caption). Whichever side appears second is assumed
        respondent/defendant — we leave naming nuance to the LLM pass if
        ambiguity matters.
        """
        m = _PARTIES_VS_RE.search(text)
        if not m:
            return []
        left = _clean_party(m.group(1))
        right = _clean_party(m.group(2))
        if not left or not right:
            return []
        # Guess role pairs from court context — Supreme/Appellate cases use
        # "appellant/respondent"; first-instance uses "plaintiff/defendant".
        upper = text.upper()
        if "COURT OF APPEAL" in upper or "SUPREME COURT" in upper or "APPELLANT" in upper:
            return [
                CasePartyExtract(role="appellant", party_name=left),
                CasePartyExtract(role="respondent", party_name=right),
            ]
        return [
            CasePartyExtract(role="plaintiff", party_name=left),
            CasePartyExtract(role="defendant", party_name=right),
        ]

    @staticmethod
    def _build_case_title(
        parties: list[CasePartyExtract],
        declared_title: str | None,
        head_text: str,
    ) -> str:
        if len(parties) >= 2:
            return f"{parties[0].party_name} v {parties[1].party_name}"
        if declared_title:
            return declared_title.strip()
        # Last resort: first non-empty line of the document.
        for line in head_text.splitlines():
            if line.strip():
                return line.strip()[:512]
        return "Untitled Case"

    # ---- LLM enrichment ----------------------------------------------------

    def _enrich_with_llm(self, meta: CaseMetadata, body_text: str) -> None:
        """Single structured call to fill ``holding`` and ``facts_summary``.

        Strict instructions: only restate what's in the text, never invent.
        On any failure we silently keep the regex-only result — extraction
        must never block ingestion.
        """
        try:
            client = self._llm_client or self._lazy_openai_client()
        except Exception:
            logger.exception("case.llm_enrichment.client_init_failed")
            return

        prompt = _ENRICHMENT_PROMPT.format(
            text=_truncate(body_text, max_chars=12_000)
        )
        try:
            completion = client.chat.completions.create(
                model=self._llm_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _ENRICHMENT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (completion.choices[0].message.content or "{}").strip()
            data = json.loads(raw)
        except Exception:
            logger.exception("case.llm_enrichment.failed")
            return

        holding = (data.get("holding") or "").strip() or None
        facts = (data.get("facts_summary") or "").strip() or None
        if holding and meta.holding is None:
            meta.holding = holding
        if facts and meta.facts_summary is None:
            meta.facts_summary = facts
        meta.enrichment_used = bool(holding or facts)

    def _lazy_openai_client(self) -> OpenAI:
        from openai import OpenAI as _OpenAI

        settings = get_settings()
        return _OpenAI(api_key=settings.openai_api_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_ENRICHMENT_SYSTEM = (
    "You extract concise structured fields from Ghana case-law judgments. "
    "You only restate what is in the text. You never invent facts or holdings. "
    "If a field is not explicitly supported by the text, return an empty string."
)

_ENRICHMENT_PROMPT = (
    "Extract two fields from the following judgment text.\n\n"
    "Return a strict JSON object with keys:\n"
    "  - \"holding\": the legal principle the court decided, 1-3 sentences.\n"
    "  - \"facts_summary\": the material facts, 2-4 sentences.\n\n"
    "Both must be drawn DIRECTLY from the text below. If you cannot find them,\n"
    "return an empty string for that field. Do not add commentary.\n\n"
    "TEXT:\n{text}"
)


def _clean_name(raw: str) -> str:
    name = re.sub(r"\s+", " ", raw).strip(" .,'-")
    # Reject all-caps single words and obvious non-names.
    if len(name) < 3:
        return ""
    if name.upper() in {"THE", "AND", "FOR", "WITH", "BEFORE", "BETWEEN"}:
        return ""
    return name.title() if name.isupper() else name


def _clean_party(raw: str) -> str:
    s = re.sub(r"\s+", " ", raw).strip()
    # Drop leading noise like "BETWEEN:" or "PLAINTIFFS:".
    s = re.sub(r"^(BETWEEN|PLAINTIFFS?|DEFENDANTS?|APPELLANTS?|RESPONDENTS?)[:\s]+", "", s, flags=re.IGNORECASE)
    return s.strip(" .,;:'\"")[:256]


def _truncate(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"
