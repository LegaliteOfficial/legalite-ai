"""Semantic query rewriting (Phase A — query understanding layer).

Problem the rewriter solves
---------------------------
Embedding-based retrieval is sensitive to phrasing. A user who asks
"Tell me about the constitution" and one who asks "What is the
Constitution of Ghana?" should get the same answer, but their query
embeddings land in slightly different regions of vector space — close
enough to be related, far enough that a cosine threshold can refuse one
and accept the other.

The rewriter turns one user question into a small set of semantically
equivalent reformulations. Each variant is then embedded and searched
independently. We union the results before reranking, so a chunk that
only one phrasing surfaces still has a chance to win.

What it does NOT do
-------------------
- It does NOT change the user's intent. The system prompt forbids the
  LLM from adding entities, narrowing scope, or shifting the legal
  question.
- It does NOT translate or paraphrase precise legal vocabulary
  ("certiorari", "mens rea") — those words carry retrieval signal.
- It does NOT return the original; callers add the original themselves.
  That way the original is always retried even if the rewriter fails or
  returns the same string.

Failure mode
------------
Any exception (OpenAI down, malformed JSON, empty list, etc.) returns an
empty list. The caller defaults to ``[original_question]`` and the
pipeline degrades to single-query retrieval — same behavior as before
this module existed.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from openai import OpenAI

logger = get_logger(__name__)


_REWRITE_SYSTEM_PROMPT = """\
You are a query rewriter for a Ghana legal research assistant.

Given a user's question, return 2-3 alternative phrasings of the SAME
question that preserve its meaning and intent. The goal is to improve
retrieval recall by covering different ways the same concept might
appear in legal text.

STRICT RULES:
- Preserve the user's intent exactly. Do NOT add new entities, narrow
  the topic, or shift to a related question.
- Use legal-encyclopedic phrasing in at least one variant (e.g. "What
  is the constitution" rather than "tell me about the constitution") —
  legal documents read formally and benefit from formal queries.
- Keep precise legal terminology unchanged when present (Act names,
  section numbers, latin terms, case names).
- Each variant should be a complete, standalone question or noun phrase
  — NOT a keyword list.
- Return STRICTLY this JSON: {"variants": ["...", "...", "..."]}
- 2-3 variants. NEVER more than 3. NEVER fewer than 2.
- No prose outside the JSON. No markdown.

EXAMPLES:
Input: "tell me about the constitution"
Output: {"variants": ["What is the Constitution of Ghana and what is its role?", "Overview and key provisions of the 1992 Ghana Constitution", "Purpose and structure of the Constitution of Ghana"]}

Input: "explain freedom of speech"
Output: {"variants": ["What rights protect freedom of speech in Ghana?", "Constitutional and statutory provisions on freedom of expression in Ghana", "Scope and limits of freedom of speech under Ghanaian law"]}

Input: "what does section 12 of the Land Act say"
Output: {"variants": ["Provisions of section 12 of the Land Act of Ghana", "Section 12 Land Act — text and meaning"]}
"""


_REWRITE_USER_TEMPLATE = "Question:\n{q}\n\nReturn the JSON object now."


class QueryRewriter:
    """Expand a question into a small set of semantic variants.

    Stateless. Safe to construct once and share across requests. Holds
    a small in-process cache so repeated identical questions in the
    same session don't pay for a second LLM call.
    """

    def __init__(
        self,
        *,
        llm_client: OpenAI | None = None,
        enabled: bool | None = None,
        max_variants: int = 3,
        cache_size: int = 512,
    ) -> None:
        settings = get_settings()
        self._enabled = enabled if enabled is not None else settings.query_rewriter_enabled
        self._client = llm_client
        self._model = settings.openai_llm_model
        self._max_variants = max_variants
        # FIFO-bounded dict cache. Tuple values are immutable so callers
        # can't mutate cached entries.
        self._cache: dict[str, tuple[str, ...]] = {}
        self._cache_size = cache_size

    def rewrite(self, question: str) -> list[str]:
        """Return up to ``max_variants`` rewrites EXCLUDING the original.

        Callers prepend the original themselves; this keeps the contract
        explicit — a failure here doesn't accidentally drop the original.
        """
        if not self._enabled:
            return []
        q = (question or "").strip()
        if len(q) < 3:
            return []

        cache_key = q.lower()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return list(cached)

        variants = self._call_llm(q)
        # Cache successes AND failures (empty list) to avoid retry storms
        # on a sustained outage — the rewriter is best-effort anyway.
        if len(self._cache) >= self._cache_size:
            # Drop oldest entry. dict in Python 3.7+ is insertion-ordered.
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key, None)
        self._cache[cache_key] = tuple(variants)
        return variants

    # ---- internals ---------------------------------------------------------

    def _call_llm(self, question: str) -> list[str]:
        try:
            client = self._client or self._lazy_client()
        except Exception:
            logger.exception("query_rewriter.client_init_failed")
            return []

        t0 = time.perf_counter()
        try:
            completion = client.chat.completions.create(
                model=self._model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _REWRITE_USER_TEMPLATE.format(q=question),
                    },
                ],
            )
            raw = (completion.choices[0].message.content or "").strip()
            data = json.loads(raw)
        except Exception:
            logger.exception("query_rewriter.llm_failed")
            return []

        variants = data.get("variants") if isinstance(data, dict) else None
        if not isinstance(variants, list):
            logger.warning("query_rewriter.malformed", preview=str(data)[:200])
            return []

        clean: list[str] = []
        seen: set[str] = {question.strip().lower()}
        for v in variants:
            if not isinstance(v, str):
                continue
            v = v.strip()
            if not v or len(v) > 512:
                continue
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            clean.append(v)
            if len(clean) >= self._max_variants:
                break

        logger.info(
            "query_rewriter.completed",
            n_variants=len(clean),
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        return clean

    def _lazy_client(self) -> OpenAI:
        from openai import OpenAI as _OpenAI

        settings = get_settings()
        return _OpenAI(api_key=settings.openai_api_key)
