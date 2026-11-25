"""Query planner — classify intent before retrieval (Phase 5 — M2).

The QA pipeline used to dispatch every /ask question to the same hybrid
retriever. Some questions are better served by other pipelines: "find
similar tenancy cases" should hit :class:`CaseSimilarityEngine`,
"compare these two judgments" belongs in `/cases/compare`, etc.

The planner returns a :class:`QueryPlan` indicating which pipeline should
run, the intent label, and a confidence in [0, 1]. The integration layer
(:class:`QAService`) decides whether to act on it (route the request) or
just record it (so the planner can be observed and tuned in production
without immediately changing behavior).

Strategy
--------
- **Heuristic pass first** — deterministic regex + keyword rules. Cheap,
  testable, and produces a confidence score.
- **LLM fallback (optional)** — gated by ``query_planner_llm_fallback_enabled``.
  Called only when the heuristic confidence is below
  ``query_planner_min_heuristic_confidence``. Strict JSON schema; failures
  collapse back to the heuristic result.

The planner NEVER bypasses safety: at most it picks the retrieval path,
the actual answer generation still runs through the existing prompt /
refusal contract.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from openai import OpenAI

logger = get_logger(__name__)


class QueryIntent(StrEnum):
    STATUTE_QUESTION = "statute_question"
    CASE_SIMILARITY = "case_similarity"
    CASE_COMPARISON = "case_comparison"
    CASE_SUMMARY = "case_summary"
    PROCEDURAL_QUESTION = "procedural_question"
    TENANT_CONTEXTUAL = "tenant_contextual"
    FOLLOW_UP = "follow_up"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    GENERAL_LEGAL_QA = "general_legal_qa"
    # Conversational intents — handled by ConversationalResponder
    # WITHOUT retrieval. The pipeline short-circuits before hybrid search.
    CONVERSATIONAL_GREETING = "conversational_greeting"
    CONVERSATIONAL_FAREWELL = "conversational_farewell"
    CONVERSATIONAL_META = "conversational_meta"
    CONVERSATIONAL_SMALL_TALK = "conversational_small_talk"
    # Catch-all for short, non-legal queries that don't match a specific
    # conversational pattern — typos, ambiguous fragments, "Am i goo?".
    # Routes to a capability-aware response instead of a canned refusal.
    CONVERSATIONAL_UNCLEAR = "conversational_unclear"


CONVERSATIONAL_INTENTS: frozenset[QueryIntent] = frozenset(
    {
        QueryIntent.CONVERSATIONAL_GREETING,
        QueryIntent.CONVERSATIONAL_FAREWELL,
        QueryIntent.CONVERSATIONAL_META,
        QueryIntent.CONVERSATIONAL_SMALL_TALK,
        QueryIntent.CONVERSATIONAL_UNCLEAR,
    }
)


# Below this character count, a query without legal substance or a
# specific conversational pattern match is treated as ambiguous noise
# rather than a real legal question. Tuned empirically: most genuine
# legal questions are well over 40 characters; most typos / fragments
# are well under.
_UNCLEAR_MAX_CHARS = 40


@dataclass(slots=True, frozen=True)
class QueryPlan:
    intent: QueryIntent
    confidence: float
    source: str  # "heuristic" | "llm"
    rationale: str
    matched_signals: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Heuristic rules
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _Rule:
    intent: QueryIntent
    label: str
    pattern: re.Pattern[str]
    weight: float  # contribution to confidence when matched


_RULES: tuple[_Rule, ...] = (
    _Rule(
        QueryIntent.CASE_COMPARISON,
        "compare_keyword",
        re.compile(
            r"\b(compare|comparison|differen(?:t|ce|ces)|contrast|"
            r"versus\s+(?:case|judgment|judgement))\b",
            re.IGNORECASE,
        ),
        0.7,
    ),
    _Rule(
        QueryIntent.CASE_SIMILARITY,
        "similar_keyword",
        re.compile(
            r"\b(similar\s+cases?|similar\s+precedents?|precedents?|"
            r"cases?\s+like|find\s+(?:similar\s+)?(?:cases?|judgements?)|"
            r"analogous\s+cases?|on\s+all\s+fours)\b",
            re.IGNORECASE,
        ),
        0.75,
    ),
    _Rule(
        QueryIntent.CASE_SUMMARY,
        "summary_keyword",
        re.compile(
            r"\b(summari[sz]e|summary\s+of\s+(?:this|that)\s+(?:case|judgment|judgement)|"
            r"explain\s+(?:this|that)\s+(?:case|judgment|judgement))\b",
            re.IGNORECASE,
        ),
        0.75,
    ),
    _Rule(
        QueryIntent.TENANT_CONTEXTUAL,
        "tenant_pronoun",
        re.compile(
            r"\b(our\s+(?:firm|memo|case|precedent|playbook|template|"
            r"strategy|file)|my\s+firm|internal\s+(?:memo|note|policy)|"
            r"uploaded\s+(?:case|memo|document)|firm[''']s|our\s+private)\b",
            re.IGNORECASE,
        ),
        0.7,
    ),
    _Rule(
        QueryIntent.STATUTE_QUESTION,
        "statute_keyword",
        re.compile(
            r"\b(act\s+\d+|article\s+\d+|section\s+\d+|"
            r"constitution|labour\s+act|rent\s+act|land\s+act|"
            r"regulation|statute|legislative\s+instrument)\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
    _Rule(
        QueryIntent.PROCEDURAL_QUESTION,
        "procedural_keyword",
        re.compile(
            r"\b(how\s+do\s+I|what\s+is\s+the\s+process|procedure\s+for|"
            r"steps\s+to|file\s+a\s+(?:claim|suit|motion)|"
            r"requirements?\s+for\s+filing)\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
    _Rule(
        QueryIntent.FOLLOW_UP,
        "follow_up_pronoun",
        re.compile(
            r"^\s*(what\s+if|what\s+about|in\s+that\s+(?:case|situation)|"
            r"and\s+(?:if|when)|then\s+what)\b",
            re.IGNORECASE,
        ),
        0.65,
    ),
)

# Ambiguous-pronoun phrases that trigger insufficient_context when no memory.
_AMBIGUOUS_PHRASES = (
    re.compile(r"^\s*(this|that|it)\s+(case|law|situation|matter)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*(more|further|tell\s+me\s+more)\s*\.?\s*$", re.IGNORECASE),
)


# Conversational rules. These match BEFORE legal rules but only "win" when
# the question carries no legal substance — e.g. "Hi, what does the
# constitution say about land?" still routes to STATUTE_QUESTION because
# legal rules also match. See ``_classify_heuristic``.
_CONVERSATIONAL_RULES: tuple[_Rule, ...] = (
    _Rule(
        QueryIntent.CONVERSATIONAL_GREETING,
        "greeting",
        re.compile(
            r"^\s*(hi+|hello+|hey+|yo|"
            r"good\s+(morning|afternoon|evening|day)|"
            r"greetings|howdy|sup|wassup|what'?s\s+up|"
            r"how\s+are\s+(you|things|you\s+doing)|"
            r"how('?s|\s+is)\s+(it\s+going|your\s+day))\b",
            re.IGNORECASE,
        ),
        0.85,
    ),
    _Rule(
        QueryIntent.CONVERSATIONAL_FAREWELL,
        "farewell_or_thanks",
        re.compile(
            r"^\s*(bye+|goodbye|see\s+(?:you|ya)|cya|later|"
            r"talk\s+(?:to\s+you\s+)?later|peace(?:\s+out)?|take\s+care|"
            r"thanks?(?:\s+a\s+lot|\s+so\s+much)?|thank\s+you|thx|ty|"
            r"appreciate\s+it|much\s+appreciated|"
            r"cool|got\s+it|understood|"
            r"ok(?:ay)?|alright|sounds\s+good|perfect|great|awesome)\s*[!.\s]*$",
            re.IGNORECASE,
        ),
        0.8,
    ),
    _Rule(
        QueryIntent.CONVERSATIONAL_META,
        "meta_question",
        re.compile(
            r"\b(who\s+are\s+you|what\s+(?:can\s+you\s+do|are\s+you|do\s+you\s+do)|"
            r"how\s+do\s+you\s+work|what\s+is\s+(?:this|legalite)|"
            r"introduce\s+yourself|tell\s+me\s+about\s+yourself|"
            r"are\s+you\s+(?:an?\s+)?(?:ai|bot|human|chatbot|robot)|"
            r"what'?s\s+your\s+name|"
            r"how\s+(?:can|do)\s+you\s+help|"
            r"what\s+do\s+you\s+know|what\s+can\s+I\s+ask\s+you)\b",
            re.IGNORECASE,
        ),
        0.75,
    ),
    _Rule(
        QueryIntent.CONVERSATIONAL_SMALL_TALK,
        "small_talk",
        re.compile(
            r"\b(nice\s+to\s+(?:meet|see)|good\s+to\s+(?:see|talk)|"
            r"hope\s+you('?re|\s+are)\s+(well|good|doing)|"
            r"how('?s|\s+is)\s+the\s+(weather|day)|"
            r"having\s+a\s+(good|nice)\s+day)\b",
            re.IGNORECASE,
        ),
        0.55,
    ),
)


# Legal keywords that, if present, override conversational classification.
# When a question contains both a greeting and any legal substance, the
# legal intent wins — we don't want "Hi, what does the constitution say"
# to be answered with a greeting and no retrieval.
_LEGAL_SUBSTANCE_RE = re.compile(
    r"\b(constitution|act|statute|regulation|law|legal|court|case|judge|"
    r"judgment|judgement|ruling|precedent|article|section|chapter|clause|"
    r"contract|agreement|breach|tort|negligence|liability|damages|"
    r"crime|criminal|civil|tenancy|tenant|landlord|employment|labour|"
    r"labor|dismissal|maternity|land|property|estate|inherit|will|"
    r"divorce|marriage|child|custody|citizenship|immigration|"
    r"parliament|government|minister|president|right|freedom|speech|"
    r"liberty|due\s+process|police|arrest|bail|trial|sentence|"
    r"prosecut|defend|plaintiff|appeal|jurisdiction|tribunal|"
    r"taxation|tax|revenue|company|business|incorporate)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class QueryPlanner:
    """Stateless classifier. Inject an OpenAI client only when LLM fallback
    is enabled — by default the planner is fully offline + deterministic.
    """

    def __init__(
        self,
        *,
        llm_client: OpenAI | None = None,
        llm_fallback_enabled: bool | None = None,
    ) -> None:
        settings = get_settings()
        if llm_fallback_enabled is None:
            llm_fallback_enabled = settings.query_planner_llm_fallback_enabled
        self._llm_fallback_enabled = llm_fallback_enabled
        self._llm_client = llm_client
        self._llm_model = settings.openai_llm_model
        self._min_confidence = settings.query_planner_min_heuristic_confidence

    # ---- public API --------------------------------------------------------

    def plan(
        self,
        question: str,
        *,
        has_memory: bool = False,
    ) -> QueryPlan:
        heuristic = self._classify_heuristic(question, has_memory=has_memory)
        if heuristic.confidence >= self._min_confidence or not self._llm_fallback_enabled:
            return heuristic

        llm = self._classify_llm(question)
        if llm is None:
            return heuristic
        return llm

    # ---- heuristic ---------------------------------------------------------

    def _classify_heuristic(self, question: str, *, has_memory: bool) -> QueryPlan:
        text = (question or "").strip()
        if not text:
            return QueryPlan(
                intent=QueryIntent.INSUFFICIENT_CONTEXT,
                confidence=1.0,
                source="heuristic",
                rationale="empty question",
            )

        # Ambiguous pronoun without memory — refuse to retrieve until a
        # concrete reference exists. This protects the model from being
        # tricked into "what about THAT case?" with no prior turn.
        if not has_memory:
            for pat in _AMBIGUOUS_PHRASES:
                if pat.search(text):
                    return QueryPlan(
                        intent=QueryIntent.INSUFFICIENT_CONTEXT,
                        confidence=0.9,
                        source="heuristic",
                        rationale="ambiguous pronoun without conversational memory",
                        matched_signals=("ambiguous_pronoun",),
                    )

        # Conversational pass — only "wins" when there is no legal substance
        # in the rest of the message. "Hi" or "thanks!" → conversational;
        # "Hi, what does the constitution say about land?" → legal.
        legal_present = bool(_LEGAL_SUBSTANCE_RE.search(text))
        if not legal_present:
            conv_scored: dict[QueryIntent, tuple[float, list[str]]] = {}
            for rule in _CONVERSATIONAL_RULES:
                if rule.pattern.search(text):
                    prev_score, signals = conv_scored.get(rule.intent, (0.0, []))
                    new_score = min(0.95, prev_score + rule.weight)
                    conv_scored[rule.intent] = (new_score, [*signals, rule.label])
            if conv_scored:
                intent, (confidence, signals) = max(conv_scored.items(), key=lambda x: x[1][0])
                return QueryPlan(
                    intent=intent,
                    confidence=confidence,
                    source="heuristic",
                    rationale=f"conversational match: {', '.join(signals)}",
                    matched_signals=tuple(signals),
                )

        scored: dict[QueryIntent, tuple[float, list[str]]] = {}
        for rule in _RULES:
            if rule.pattern.search(text):
                prev_score, signals = scored.get(rule.intent, (0.0, []))
                # Multiple rules for the same intent stack but saturate at 0.95.
                new_score = min(0.95, prev_score + rule.weight)
                scored[rule.intent] = (new_score, [*signals, rule.label])

        if not scored:
            # Short, non-legal queries that didn't match a specific
            # conversational rule (typos like "Am i goo?", fragments
            # like "?", ambiguous noise) are routed to the unclear
            # responder instead of being thrown at the legal LLM,
            # which would just produce the canned refusal.
            if not legal_present and len(text) < _UNCLEAR_MAX_CHARS:
                return QueryPlan(
                    intent=QueryIntent.CONVERSATIONAL_UNCLEAR,
                    confidence=0.6,
                    source="heuristic",
                    rationale=(
                        f"short non-legal text ({len(text)} chars); "
                        "routing to capability-aware response"
                    ),
                )
            return QueryPlan(
                intent=QueryIntent.GENERAL_LEGAL_QA,
                confidence=0.4,
                source="heuristic",
                rationale="no rule matched; falling back to general RAG",
            )

        # Resolve to the strongest matched intent.
        intent, (confidence, signals) = max(scored.items(), key=lambda x: x[1][0])

        # Boost case_similarity / case_summary if the question explicitly
        # references "case" / "judgment" near the trigger keyword — keeps
        # statute questions that just mention "case" out of those buckets.
        if intent in (QueryIntent.CASE_SIMILARITY, QueryIntent.CASE_SUMMARY) and not re.search(
            r"\b(case|judgement|judgment|precedent|ruling)\b", text, re.IGNORECASE
        ):
            confidence = max(0.0, confidence - 0.2)

        return QueryPlan(
            intent=intent,
            confidence=confidence,
            source="heuristic",
            rationale=f"matched: {', '.join(signals)}",
            matched_signals=tuple(signals),
        )

    # ---- LLM fallback ------------------------------------------------------

    def _classify_llm(self, question: str) -> QueryPlan | None:
        try:
            client = self._llm_client or self._lazy_client()
        except Exception:
            logger.exception("query_planner.llm.client_init_failed")
            return None

        try:
            completion = client.chat.completions.create(
                model=self._llm_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _LLM_SYSTEM},
                    {"role": "user", "content": _LLM_USER_TEMPLATE.format(q=question)},
                ],
            )
            raw = (completion.choices[0].message.content or "").strip()
            data = json.loads(raw)
        except Exception:
            logger.exception("query_planner.llm.failed")
            return None

        try:
            intent = QueryIntent(str(data.get("intent", "")).strip().lower())
        except ValueError:
            return None
        confidence = float(data.get("confidence", 0.5))
        rationale = str(data.get("rationale") or "llm classification")[:256]
        return QueryPlan(
            intent=intent,
            confidence=max(0.0, min(1.0, confidence)),
            source="llm",
            rationale=rationale,
        )

    def _lazy_client(self) -> OpenAI:
        from openai import OpenAI as _OpenAI

        settings = get_settings()
        return _OpenAI(api_key=settings.openai_api_key)


_LLM_SYSTEM = (
    "You classify Ghana-legal user questions into one of these intents:\n"
    "statute_question, case_similarity, case_comparison, case_summary, "
    "procedural_question, tenant_contextual, follow_up, insufficient_context, "
    "general_legal_qa.\n"
    'Return strict JSON: {"intent": <one>, "confidence": <0..1>, '
    '"rationale": <short>}.\n'
    "Never invent intents outside the list. Never answer the question itself."
)

_LLM_USER_TEMPLATE = "Question:\n{q}\n\nReturn the JSON object now."
