"""Prompt safety re-audit (Milestone 8) + structured-output rendering (M1).

We test the assembled SYSTEM_PROMPT for every guarantee the product makes,
and verify that ``build_user_prompt`` groups passages by authority tier
with explicit SCOPE markers on tenant-private chunks.
"""

from __future__ import annotations

from app.rag.authority import TIER_LABEL, annotate
from app.rag.prompt_builder import (
    DISCLAIMER,
    REFUSAL_TEXT,
    STRUCTURED_SCHEMA_EXAMPLE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from app.rag.retriever import RetrievedPassage


def _passage(**overrides: object) -> RetrievedPassage:
    base = dict(
        score=0.6,
        content="Passage content here.",
        law="Test Law",
        article=None,
        section=None,
        chapter=None,
        clause=None,
        subsection=None,
        page_start=10,
        page_end=10,
        vector_id="vid",
    )
    base.update(overrides)
    return RetrievedPassage(**base)  # type: ignore[arg-type]


# ---- SYSTEM_PROMPT contract -----------------------------------------------


def test_system_prompt_advertises_the_json_schema() -> None:
    assert STRUCTURED_SCHEMA_EXAMPLE in SYSTEM_PROMPT
    # All seven section names must be advertised.
    for section in (
        "direct_answer",
        "applicable_law",
        "relevant_public_cases",
        "firm_similar_cases",
        "legal_reasoning",
        "confidence_assessment",
        "citations",
    ):
        assert section in SYSTEM_PROMPT, f"system prompt missing {section}"


def test_system_prompt_states_authority_hierarchy() -> None:
    for label in (
        TIER_LABEL["constitution"],
        TIER_LABEL["act"],
        TIER_LABEL["regulation"],
        TIER_LABEL["public_case_law"],
    ):
        assert label in SYSTEM_PROMPT
    assert "NEVER override" in SYSTEM_PROMPT


def test_system_prompt_forbids_invention_and_cross_tenant_leakage() -> None:
    # Fabrication block.
    assert "NEVER fabricate" in SYSTEM_PROMPT
    # Cross-tenant block.
    assert "another firm" in SYSTEM_PROMPT.lower() or "another organization" in SYSTEM_PROMPT.lower()
    assert "tenant-private materials belonging to other organizations" in SYSTEM_PROMPT.lower() \
        or "NEVER reveal, summarize, or hint" in SYSTEM_PROMPT


def test_system_prompt_pins_the_exact_refusal_sentence() -> None:
    assert REFUSAL_TEXT in SYSTEM_PROMPT


def test_system_prompt_refuses_jailbreak_phrases() -> None:
    """The non-negotiable rules must mention the adversarial patterns we
    want to refuse: 'pretend tenancy doesn't exist', 'guess', 'likely
    content', role-play."""
    body = SYSTEM_PROMPT.lower()
    assert "pretend" in body
    assert "guess" in body or "speculate" in body
    assert "likely" in body
    assert "role-play" in body or "role play" in body


def test_disclaimer_is_appended_by_application_not_model() -> None:
    """The disclaimer should be reserved for the application layer (so we
    can guarantee it appears). The system prompt should say so."""
    assert DISCLAIMER not in SYSTEM_PROMPT  # not embedded as an instruction example
    assert "disclaimer is appended by the application" in SYSTEM_PROMPT.lower()


# ---- build_user_prompt — tier rendering -----------------------------------


def test_user_prompt_groups_passages_by_authority_tier() -> None:
    act = _passage(law="Act 1", document_origin="act", visibility_scope="global")
    case = _passage(law="Case 1", document_origin="case_law", visibility_scope="global")
    memo = _passage(
        law="Firm Memo",
        document_origin="internal_memo",
        visibility_scope="tenant_private",
        organization_id="org-A",
    )
    annotate([act, case, memo])
    prompt = build_user_prompt("question", [act, case, memo])

    # The tier headers must be present in canonical order.
    act_idx = prompt.find(TIER_LABEL["act"])
    case_idx = prompt.find(TIER_LABEL["public_case_law"])
    memo_idx = prompt.find(TIER_LABEL["tenant_internal_memo"])
    assert -1 < act_idx < case_idx < memo_idx


def test_user_prompt_numbers_passages_globally_across_tiers() -> None:
    """Passage ids should be 1, 2, 3 in tier order so the highest-authority
    passage is always [1]."""
    act = _passage(law="Act-X", document_origin="act", visibility_scope="global")
    memo = _passage(
        law="Memo-Y", document_origin="internal_memo", visibility_scope="tenant_private"
    )
    annotate([memo, act])  # input order shuffled
    prompt = build_user_prompt("q", [memo, act])
    # The Act tier comes before the memo tier, so the Act gets [1].
    assert prompt.find("[1]") < prompt.find("[2]")
    # And [1] should be in the Act section, not the memo section.
    one_idx = prompt.find("[1]")
    act_section_idx = prompt.find(TIER_LABEL["act"])
    memo_section_idx = prompt.find(TIER_LABEL["tenant_internal_memo"])
    assert act_section_idx < one_idx < memo_section_idx


def test_user_prompt_marks_tenant_private_passages_with_scope_tag() -> None:
    memo = _passage(
        law="Acme Memo",
        document_origin="internal_memo",
        visibility_scope="tenant_private",
        organization_id="org-A",
    )
    annotate([memo])
    prompt = build_user_prompt("q", [memo])
    assert "SCOPE: tenant-private" in prompt


def test_user_prompt_does_not_mark_global_passages_as_tenant() -> None:
    act = _passage(law="Act A", document_origin="act", visibility_scope="global")
    annotate([act])
    prompt = build_user_prompt("q", [act])
    assert "SCOPE: tenant-private" not in prompt


def test_empty_passages_emits_refusal_instruction_and_json_contract() -> None:
    prompt = build_user_prompt("anything", [])
    assert REFUSAL_TEXT in prompt
    # Must still instruct the model to return JSON (not free text).
    assert "JSON" in prompt or "json" in prompt
