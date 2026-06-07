"""Conversational responder for non-legal turns (Phase A — interactive layer).

The QA pipeline used to send every user turn through retrieval. That made
the assistant feel rigid: a friendly "Hi!" returned an "I could not
confidently determine this" refusal because no Ghana legal passage
matched a greeting. This responder handles those turns directly — no
embedding, no Qdrant call, no LLM round-trip — so the conversation feels
natural while the legal pipeline stays unmodified for substantive
questions.

Routing happens upstream: :class:`QueryPlanner` classifies the turn as
one of the ``CONVERSATIONAL_*`` intents and :class:`QAService` short-
circuits to this responder. If the planner detects ANY legal substance
(see ``_LEGAL_SUBSTANCE_RE`` in query_planner), the turn skips this
module entirely and runs through the regular grounded path.

Design choices
--------------
- **Templated, not LLM-generated.** Determinism + zero cost + sub-ms
  latency. The cost of "more natural" generated wording is not worth
  blocking a /ask request on an OpenAI call for "hi".
- **Multiple variants per intent.** A naive "always reply with the same
  sentence" felt robotic in user testing. We select a variant
  deterministically from the question (``hash`` modulo variants), so a
  given input always returns the same reply but the bot feels less
  scripted across a session.
- **Always steers toward legal help.** Every reply ends with an invitation
  to ask a legal question. This keeps the conversational layer from
  becoming an open-ended chatbot.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.rag.confidence_engine import Confidence
from app.rag.prompt_builder import DISCLAIMER
from app.rag.query_planner import QueryIntent
from app.schemas.qa import StructuredAnswer


@dataclass(slots=True)
class ConversationalAnswer:
    """The conversational-pipeline counterpart of GeneratedAnswer.

    Carries everything QAService needs to build an AskResponse without
    touching retrieval, the reranker, the answer generator, or
    confidence engine. ``confidence`` is always ``"high"`` because
    conversational replies are deterministic templates, not synthesized
    claims about the law.
    """

    answer: str
    structured: StructuredAnswer
    intent: QueryIntent
    confidence: Confidence = "high"


# Greeting — opens a session, advertises capabilities.
_GREETING_VARIANTS: tuple[str, ...] = (
    (
        "Hello! I'm Legalite, your Ghana legal research assistant. I can help "
        "you understand statutes, case law, and constitutional provisions, and "
        "I always cite the source materials I draw from. What legal question "
        "can I help you with today?"
    ),
    (
        "Hi there! I'm Legalite — I search Ghana legal materials "
        "(Constitution, Acts, Regulations, case law) and explain what they "
        "mean in plain language, with citations. What would you like to "
        "explore?"
    ),
    (
        "Hey! I'm here to help with Ghana legal questions. You can ask me "
        "about specific statutes, cases, procedures, or general legal "
        "concepts — I'll ground every answer in the actual source materials. "
        "What's on your mind?"
    ),
)

# Farewell / thanks / acknowledgment — brief, warm, leaves the door open.
_FAREWELL_VARIANTS: tuple[str, ...] = (
    "You're welcome! Come back any time you have a Ghana legal question.",
    "Glad I could help. Reach out whenever you need legal research support.",
    "Anytime! I'm here whenever you need to dig into Ghana law.",
    "Happy to help. Let me know if anything else comes up.",
)

# Meta — "who are you / what can you do" — a small capabilities tour.
_META_VARIANTS: tuple[str, ...] = (
    (
        "I'm Legalite, an AI legal research assistant focused on Ghana law. "
        "I can:\n"
        "• Explain constitutional provisions and statutes in plain language\n"
        "• Find and summarise relevant case law\n"
        "• Walk you through legal procedures and requirements\n"
        "• Compare or analyse specific cases\n\n"
        "Every answer I give is grounded in actual Ghana legal materials and "
        "includes citations so you can verify the source. I'm not a lawyer and "
        "I don't give legal advice — I help you research the law. What would "
        "you like to look into?"
    ),
    (
        "I'm Legalite — an AI built to help you research Ghana law. I draw "
        "from the Constitution, Acts of Parliament, Regulations, and case "
        "law that have been indexed into my knowledge base, and I cite "
        "every passage I rely on. Ask me things like:\n\n"
        "• \"What does the Constitution say about citizenship?\"\n"
        "• \"Explain the Labour Act's rules on dismissal.\"\n"
        "• \"Find cases similar to a tenancy dispute in Accra.\"\n\n"
        "I'm a research tool, not a substitute for a lawyer. What can I "
        "look into for you?"
    ),
)

# Small talk — brief acknowledgment then redirect.
_SMALL_TALK_VARIANTS: tuple[str, ...] = (
    (
        "I'm doing well, thanks for asking! Ready to help with any Ghana "
        "legal question you have — what would you like to explore?"
    ),
    (
        "All good on my end! What Ghana legal topic can I help you with?"
    ),
    (
        "Doing great, thank you! Let me know what legal question I can dig "
        "into for you."
    ),
)


def _pick(question: str, variants: tuple[str, ...]) -> str:
    """Deterministic variant selection. Same question → same reply.

    Uses md5 because Python's ``hash()`` is randomized per process — we
    want stability so two replicas serving the same user-input return the
    same text (helps debugging and avoids look-mismatch in screenshots).
    """
    digest = hashlib.md5(question.strip().lower().encode("utf-8")).digest()
    return variants[digest[0] % len(variants)]


class ConversationalResponder:
    """Translates a conversational intent into a complete answer.

    No external dependencies — no LLM, no Qdrant, no DB. Constructable
    anywhere in a test without mocks.
    """

    def respond(self, question: str, intent: QueryIntent) -> ConversationalAnswer:
        if intent == QueryIntent.CONVERSATIONAL_GREETING:
            body = _pick(question, _GREETING_VARIANTS)
        elif intent == QueryIntent.CONVERSATIONAL_FAREWELL:
            body = _pick(question, _FAREWELL_VARIANTS)
        elif intent == QueryIntent.CONVERSATIONAL_META:
            body = _pick(question, _META_VARIANTS)
        elif intent == QueryIntent.CONVERSATIONAL_SMALL_TALK:
            body = _pick(question, _SMALL_TALK_VARIANTS)
        else:
            # Defensive: caller routed wrong intent here. Reply neutrally
            # rather than crash — this path should be unreachable in
            # production.
            body = _pick(question, _GREETING_VARIANTS)

        # The disclaimer applies to legal claims, not "hello". We
        # deliberately omit it from the conversational ``answer`` string
        # because it sounds disingenuous attached to a greeting. The
        # disclaimer remains a constant available to clients that want
        # to display it persistently in the UI.
        structured = StructuredAnswer(
            direct_answer=body,
            legal_reasoning=(
                "Conversational turn — no Ghana legal materials retrieved."
            ),
            confidence_assessment="Templated reply; not a legal claim.",
        )
        return ConversationalAnswer(
            answer=body,
            structured=structured,
            intent=intent,
        )


__all__ = [
    "ConversationalAnswer",
    "ConversationalResponder",
    "DISCLAIMER",
]
