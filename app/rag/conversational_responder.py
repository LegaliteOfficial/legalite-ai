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
# CommonMark requires a BLANK LINE between a paragraph and a list, and
# ``-`` prefix on each item — without those, ReactMarkdown collapses
# the bullets onto one inline line.
_META_VARIANTS: tuple[str, ...] = (
    (
        "I'm Legalite, an AI legal research assistant focused on Ghana law. "
        "I can:\n\n"
        "- Explain constitutional provisions and statutes in plain language\n"
        "- Find and summarise relevant case law\n"
        "- Walk you through legal procedures and requirements\n"
        "- Compare or analyse specific cases\n\n"
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
        "- \"What does the Constitution say about citizenship?\"\n"
        "- \"Explain the Labour Act's rules on dismissal.\"\n"
        "- \"Find cases similar to a tenancy dispute in Accra.\"\n\n"
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

    Sorted by count descending so the headline coverage leads.
    Renders as a CommonMark list — note the BLANK LINE between the
    header and the bullets, and the ``-`` prefix on each item.
    Without those two details ReactMarkdown collapses everything onto
    one line.
    """
    pairs = sorted(
        inventory.counts_by_doc_type.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    bullets = []
    for doc_type, count in pairs:
        plural = "" if count == 1 else "s"
        bullets.append(f"- {count:,} {doc_type} document{plural}")
    return "Here's what I currently have access to:\n\n" + "\n".join(bullets)


# Concept-based seed questions per doc_type. We deliberately AVOID
# title-specific shapes like "Summarize <title>" because:
#
#   (a) Vector search retrieves on chunk *content*, not metadata. A
#       judgment titled "Republic v. Takyi" contains chunks about
#       burden of proof, damages, etc. — not the parties' names. So
#       "Summarize Republic v. Takyi" embeds to a query that has no
#       reliable cosine match against the case's own chunks, and
#       retrieval returns weak results from other cases.
#   (b) The assistant then refuses, which makes the AI's *own*
#       suggestion look broken to the user. Worse than no suggestion.
#
# Concept questions ("What's the burden of proof…?") match the
# language judges actually USE, so retrieval finds relevant chunks
# regardless of which specific case they're in.
#
# Adding title-aware retrieval (Postgres FTS on legal_documents.title
# before falling through to vector search) is the right long-term
# fix; once that lands we can safely surface "Summarize <title>"
# again. Tracked separately.
_CONCEPT_QUESTIONS: dict[str, tuple[str, ...]] = {
    "constitution": (
        "What does the Constitution say about citizenship?",
        "Explain freedom of speech under Ghana's Constitution.",
        "What fundamental rights does the Constitution protect?",
    ),
    "case_law": (
        "What's the burden of proof in Ghana civil cases?",
        "What principles apply to declaration of title to land?",
        "What does Ghana case law say about damages in trespass?",
        "How do courts in Ghana treat substituted service?",
    ),
    "act": (
        "What does the Labour Act say about dismissal?",
        "Explain the Companies Act's rules on incorporation.",
        "What does the Rent Act provide on landlord obligations?",
    ),
    "regulation": (
        "What regulations govern business registration in Ghana?",
    ),
    "sample_case": (
        "What firm precedents discuss employment disputes?",
        "Find similar internal cases on contract enforcement.",
    ),
    "memo": (
        "What did our firm advise on the latest tenancy matter?",
        "Summarize our recent memos on land disputes.",
    ),
}


def _format_seed_questions(
    inventory: CorpusSnapshot, *, question_seed: str
) -> str:
    """Build 2–3 example questions the corpus can plausibly answer.

    Strategy:
      1. Walk doc_types in descending count order — the user's
         biggest categories get suggestions first.
      2. For each type we recognise, pick a concept question
         deterministically from a small per-type pool. Same
         user-question seed → same suggestions, so debugging is
         repeatable.
      3. Cap at 3 suggestions to keep the reply short.

    Returns a CommonMark-formatted block (header + blank line +
    ``-`` bullets) or empty string if no doc_type matched any pool.
    """
    suggestions: list[str] = []
    seen_pools: set[str] = set()  # avoid suggesting from same pool twice

    pairs = sorted(
        inventory.counts_by_doc_type.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for doc_type, count in pairs:
        if count <= 0 or len(suggestions) >= 3:
            continue
        pool_key = _match_concept_key(doc_type)
        if pool_key is None or pool_key in seen_pools:
            continue
        pool = _CONCEPT_QUESTIONS.get(pool_key)
        if not pool:
            continue
        seen_pools.add(pool_key)
        # Seed by (question + pool_key) so different pools rotate
        # independently when the same user input fires multiple
        # suggestion blocks.
        idx = _index_for(f"{question_seed}|{pool_key}", len(pool))
        suggestions.append(pool[idx])

    if not suggestions:
        # Edge case: corpus exists but every doc_type is something we
        # don't have a concept pool for. Offer the catch-all so the
        # user still gets a hint rather than just the capability tour.
        return ""

    bullets = "\n".join(f"- {s}" for s in suggestions)
    return "You could try asking:\n\n" + bullets


def _match_concept_key(doc_type: str) -> str | None:
    """Map a doc_type string to a ``_CONCEPT_QUESTIONS`` key.

    The corpus stores ``doc_type`` as a free-text user choice
    ("Constitution", "Case Law", "case_law", "Acts of Parliament",
    etc.). We canonicalise here so the suggestion pools stay tiny.
    """
    lc = doc_type.lower().strip()
    if "constitution" in lc:
        return "constitution"
    # Sample/firm precedents must match BEFORE the general "case"
    # bucket — a Sample Case is firm-private context, not a published
    # judgment, so a "Summarize" suggestion needs a different tone.
    if "sample" in lc:
        return "sample_case"
    if "memo" in lc:
        return "memo"
    if "case" in lc or "judgement" in lc or "judgment" in lc or "precedent" in lc:
        return "case_law"
    if "regulation" in lc:
        return "regulation"
    if "act" in lc:
        return "act"
    return None


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
