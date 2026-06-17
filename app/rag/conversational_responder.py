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
from app.services.corpus_inventory import CorpusSnapshot


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

# UNCLEAR opener — gently acknowledges we didn't parse the input. The
# meat of the response (capability tour + seed questions) is generated
# dynamically from the corpus snapshot, so these are short prefaces.
_UNCLEAR_OPENERS: tuple[str, ...] = (
    "I'm not quite sure what you're asking — could you rephrase?",
    "I didn't catch that. Want to try rephrasing your question?",
    "Hmm, I'm not sure how to take that. Could you say a bit more?",
)

# Fallback used when the corpus snapshot is empty (nothing's been
# uploaded yet) — happens in fresh dev environments. We can't suggest
# real titles, so we steer toward common entry-point questions.
_EMPTY_CORPUS_HINT = (
    "Once Ghana legal materials are uploaded (Constitution, Acts, "
    "Regulations, case law), I'll be able to answer questions grounded "
    "in those sources. Until then I can chat or describe what I do."
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
    anywhere in a test without mocks. The optional ``inventory`` arg on
    :meth:`respond` lets the UNCLEAR branch tour what the corpus
    actually contains; pass ``None`` to fall back to a generic
    "what I can do" reply.
    """

    def respond(
        self,
        question: str,
        intent: QueryIntent,
        *,
        inventory: CorpusSnapshot | None = None,
    ) -> ConversationalAnswer:
        if intent == QueryIntent.CONVERSATIONAL_GREETING:
            body = _pick(question, _GREETING_VARIANTS)
        elif intent == QueryIntent.CONVERSATIONAL_FAREWELL:
            body = _pick(question, _FAREWELL_VARIANTS)
        elif intent == QueryIntent.CONVERSATIONAL_META:
            body = _pick(question, _META_VARIANTS)
        elif intent == QueryIntent.CONVERSATIONAL_SMALL_TALK:
            body = _pick(question, _SMALL_TALK_VARIANTS)
        elif intent == QueryIntent.CONVERSATIONAL_UNCLEAR:
            body = _build_unclear_response(question, inventory)
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


# ── Capability-aware UNCLEAR builder ─────────────────────────────────────


def _build_unclear_response(
    question: str, inventory: CorpusSnapshot | None
) -> str:
    """Compose a capability-aware reply for an UNCLEAR intent.

    Structure:

        <opener>

        Here's what I currently have access to:
          • 1,961 Constitution sections
          • 24 Case Law judgments
          ...

        You could try asking:
          • What does the Constitution say about citizenship?
          • Summarize Republic v. Danquah (2024 GHACC 285)
          • ...

    Falls back gracefully when the inventory is empty or missing.
    """
    opener = _pick(question, _UNCLEAR_OPENERS)
    if inventory is None or inventory.is_empty:
        return f"{opener}\n\n{_EMPTY_CORPUS_HINT}"

    tour = _format_capability_tour(inventory)
    seeds = _format_seed_questions(inventory, question_seed=question)
    parts = [opener, tour]
    if seeds:
        parts.append(seeds)
    return "\n\n".join(parts)


def _format_capability_tour(inventory: CorpusSnapshot) -> str:
    """Bulleted summary of doc_type counts.

    Sorted by count descending so the headline coverage leads. Keeps
    the user oriented around what the assistant is *actually* good
    at, rather than what it might theoretically know.
    """
    lines = ["Here's what I currently have access to:"]
    pairs = sorted(
        inventory.counts_by_doc_type.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for doc_type, count in pairs:
        plural = "" if count == 1 else "s"
        lines.append(f"• {count:,} {doc_type} document{plural}")
    return "\n".join(lines)


def _format_seed_questions(
    inventory: CorpusSnapshot, *, question_seed: str
) -> str:
    """Build 2–3 example questions grounded in real corpus titles.

    Templates are keyed off ``doc_type`` (case-insensitive). When a
    type has at least one sample title we use it verbatim — the user
    sees "Summarize Republic v. Danquah (2024 GHACC 285)" rather than
    a generic placeholder, which makes the suggestion feel real.

    ``question_seed`` is fed to the deterministic picker so the same
    user question always renders the same suggestions — useful for
    debugging "why didn't the example I just saw appear again" and
    avoids the chatbot feeling rotating-billboard-y.
    """
    suggestions: list[str] = []
    counts = inventory.counts_by_doc_type
    samples = inventory.sample_titles_by_doc_type

    # ─ Constitution
    if _has(counts, "constitution"):
        suggestions.append(
            "What does the Constitution say about citizenship?"
        )

    # ─ Case Law — use a real title when available so the suggestion
    #   feels grounded. Fall back to a generic shape when titles are
    #   empty (shouldn't happen in practice but safe to guard).
    case_titles = _samples_for(samples, ("case law", "case_law", "judgement", "judgment"))
    if case_titles:
        chosen = case_titles[_index_for(question_seed, len(case_titles))]
        # Truncate excessively long judgment titles so the suggestion
        # is one readable line.
        if len(chosen) > 80:
            chosen = chosen[:78].rstrip(" ,.-") + "…"
        suggestions.append(f"Summarize {chosen}")
    elif _has(counts, "case law", "case_law", "judgement", "judgment"):
        suggestions.append("Find cases similar to a land dispute in Accra")

    # ─ Acts / Regulations
    if _has(counts, "act", "acts"):
        # Use a real Act title when we have one — same pattern as cases.
        act_titles = _samples_for(samples, ("act", "acts"))
        if act_titles:
            chosen = act_titles[_index_for(question_seed, len(act_titles))]
            if len(chosen) > 80:
                chosen = chosen[:78].rstrip(" ,.-") + "…"
            suggestions.append(f"What does {chosen} say about dismissal?")
        else:
            suggestions.append("Explain the Labour Act's rules on dismissal.")

    if not suggestions:
        # Edge case: corpus exists but doesn't match any of our known
        # templates. Offer the catch-all so the user still gets a hint.
        suggestions.append("Ask about any Ghana law, case, or legal procedure.")

    # Cap at 3 to keep the reply short.
    suggestions = suggestions[:3]
    lines = ["You could try asking:"]
    for s in suggestions:
        lines.append(f"• {s}")
    return "\n".join(lines)


def _has(counts: dict[str, int], *keys: str) -> bool:
    """Case-insensitive substring match against any of the keys."""
    lc_keys = [k.lower() for k in keys]
    for doc_type, n in counts.items():
        if n <= 0:
            continue
        dt = doc_type.lower()
        for key in lc_keys:
            if key in dt:
                return True
    return False


def _samples_for(
    samples: dict[str, list[str]], keys: tuple[str, ...]
) -> list[str]:
    """Flatten sample titles whose doc_type matches any key (case-insensitive)."""
    lc_keys = tuple(k.lower() for k in keys)
    out: list[str] = []
    for doc_type, titles in samples.items():
        dt = doc_type.lower()
        if any(k in dt for k in lc_keys):
            out.extend(titles)
    return out


def _index_for(seed: str, n: int) -> int:
    """Deterministic index into a list of size ``n``.

    Same seed → same index, but two different user questions land on
    different sample titles. Avoids the assistant feeling like a
    broken record without going random (random would break tests and
    debug screenshots).
    """
    if n <= 0:
        return 0
    digest = hashlib.md5(seed.strip().lower().encode("utf-8")).digest()
    return digest[0] % n


__all__ = [
    "ConversationalAnswer",
    "ConversationalResponder",
    "DISCLAIMER",
]
