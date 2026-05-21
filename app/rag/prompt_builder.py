"""Prompt construction for the Ghana Legal Assistant (v2).

The system prompt is the safety contract. It enforces:
- Strict grounding (answer only from the provided passages).
- Source citation by passage id, inline.
- A structured answer style (direct answer → explanation → citations →
  confidence → disclaimer).
- Explicit fact/interpretation distinction.
- Acknowledgement of conflicts between laws when retrieved passages disagree.
- Refusal with an exact sentence when evidence is insufficient.

`build_user_prompt` renders numbered passages with their hierarchy so the
LLM can cite them as [1], [2], … which the citation formatter then maps
back to structured Citation/SourceUsed objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.retriever import RetrievedPassage


SYSTEM_PROMPT = """You are Legalite, a Ghana Legal Research Assistant.

Core rules (non-negotiable):
1. Answer ONLY using the Ghana legal passages provided in the user message.
   They are the sole source of truth.
2. NEVER fabricate statutes, articles, sections, case names, page numbers,
   or years.
3. If the passages do not contain enough information to answer the question,
   reply EXACTLY this sentence and nothing else before the disclaimer:
   "I could not confidently determine this from the provided Ghana legal materials."
4. Cite the passages you used by their numeric ids — like [1] or [2, 3] —
   inline at the end of each clause they support.
5. Be conservative, precise, and neutral. Do not give legal advice.
6. Differentiate legal facts (what the law says) from interpretation (how it
   has typically been applied). When you interpret, say so plainly.
7. If two or more passages conflict, surface the conflict explicitly. Name
   the laws/sections in disagreement and do not pick a side without textual
   support.
8. Do not speculate about how a court might rule beyond what the passages
   state.

Answer style (use this structure):
1. **Direct answer** — one or two sentences, plain English.
2. **Legal explanation** — short paragraph explaining the reasoning, with
   inline [n] citations.
3. **Supporting citations** — a bullet list of the passage ids you relied on
   (e.g. "[1] Labour Act, s.15(2)(a)"). Use the metadata from the passage
   header verbatim — do not invent numbers or page references.
4. **Confidence note** — one sentence stating any caveats (missing context,
   single source, conflicting passages).
5. Always end with the disclaimer below.

Disclaimer (verbatim, last line):
"This information is for educational purposes and not legal advice."
"""

REFUSAL_TEXT = "I could not confidently determine this from the provided Ghana legal materials."
DISCLAIMER = "This information is for educational purposes and not legal advice."


@dataclass(slots=True)
class MemoryTurn:
    role: str  # "user" | "assistant"
    content: str


def build_user_prompt(
    question: str,
    passages: list[RetrievedPassage],
    *,
    memory: list[MemoryTurn] | None = None,
) -> str:
    if not passages:
        return (
            "No legal passages were retrieved.\n\n"
            f"Question: {question}\n\n"
            f'Respond exactly: "{REFUSAL_TEXT}" then the disclaimer on a new line.'
        )

    memory_block = _render_memory(memory) if memory else ""
    context_block = _render_passages(passages)
    return (
        f"{memory_block}"
        f"Question:\n{question}\n\n"
        f"Ghana legal passages (use only these — they are numbered):\n{context_block}\n\n"
        "Write the answer in the required 5-part structure. Cite passages inline as [n]. "
        "If the passages disagree, surface the conflict. If they do not answer the question, "
        f'reply with the exact refusal sentence: "{REFUSAL_TEXT}"'
    )


def _render_memory(memory: list[MemoryTurn]) -> str:
    if not memory:
        return ""
    lines = ["Previous conversation (for context only — do not cite as legal text):"]
    for t in memory:
        prefix = "User" if t.role == "user" else "Assistant"
        snippet = t.content.strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(f"  {prefix}: {snippet}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _render_passages(passages: list[RetrievedPassage]) -> str:
    blocks: list[str] = []
    for i, p in enumerate(passages, start=1):
        header_parts: list[str] = []
        if p.law:
            header_parts.append(f"Law: {p.law}")
        if p.law_type:
            header_parts.append(f"Type: {p.law_type}")
        if p.act_number:
            header_parts.append(f"Act No. {p.act_number}")
        if p.effective_year:
            header_parts.append(f"Year {p.effective_year}")
        if p.chapter:
            header_parts.append(f"Chapter {p.chapter}")
        if p.part:
            header_parts.append(f"Part {p.part}")
        if p.article:
            header_parts.append(f"Article {p.article}")
        if p.section:
            header_parts.append(f"Section {p.section}")
        if p.clause:
            header_parts.append(f"Clause ({p.clause})")
        if p.subsection:
            header_parts.append(f"Subsection ({p.subsection})")
        if p.page_start:
            page_ref = (
                f"p. {p.page_start}"
                if not p.page_end or p.page_end == p.page_start
                else f"pp. {p.page_start}-{p.page_end}"
            )
            header_parts.append(page_ref)
        header = " | ".join(header_parts) if header_parts else "Source"
        blocks.append(f"[{i}] {header}\n{p.content.strip()}")
    return "\n\n---\n\n".join(blocks)
