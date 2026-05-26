"""Authority classification — the deterministic mapping of a passage to its
tier in the Ghana legal hierarchy. The bucketing here drives both the
prompt's tier ordering and the structured-answer's section assignment, so
it has to be predictable and resistant to malformed metadata.
"""

from __future__ import annotations

from app.rag.authority import (
    TIER_ORDER,
    annotate,
    classify,
    ordered_tiers,
    partition_by_tier,
)
from app.rag.retriever import RetrievedPassage


def _passage(**overrides: object) -> RetrievedPassage:
    base = dict(
        score=0.5,
        content="...",
        law="X",
        article=None,
        section=None,
        chapter=None,
        clause=None,
        subsection=None,
        page_start=1,
        page_end=1,
        vector_id="vid",
    )
    base.update(overrides)
    return RetrievedPassage(**base)  # type: ignore[arg-type]


# ---- classify() ------------------------------------------------------------


def test_constitution_global_classifies_to_constitution_tier() -> None:
    cls = classify(_passage(document_origin="constitution", visibility_scope="global"))
    assert cls.tier == "constitution"
    assert cls.source_scope == "global"
    assert cls.rank == 100


def test_act_outranks_regulation_outranks_case_law() -> None:
    act = classify(_passage(document_origin="act", visibility_scope="global"))
    reg = classify(_passage(document_origin="regulation", visibility_scope="global"))
    case = classify(_passage(document_origin="case_law", visibility_scope="global"))
    assert act.rank > reg.rank > case.rank


def test_tenant_origins_classify_under_their_specific_tier() -> None:
    sample = classify(
        _passage(document_origin="sample_case", visibility_scope="tenant_private")
    )
    memo = classify(
        _passage(document_origin="internal_memo", visibility_scope="tenant_private")
    )
    tpl = classify(_passage(document_origin="template", visibility_scope="tenant_private"))
    assert sample.tier == "tenant_sample_case"
    assert memo.tier == "tenant_internal_memo"
    assert tpl.tier == "tenant_template"
    # All tenant tiers must be ranked BELOW every public tier.
    case = classify(_passage(document_origin="case_law", visibility_scope="global"))
    assert sample.rank < case.rank
    assert memo.rank < case.rank
    assert tpl.rank < case.rank


def test_law_type_fallback_when_origin_missing() -> None:
    cls = classify(_passage(law_type="act", visibility_scope="global"))
    assert cls.tier == "act"
    cls = classify(_passage(law_type="case_law", visibility_scope="global"))
    assert cls.tier == "public_case_law"


def test_tenant_chunk_without_origin_downgrades_to_private_reference() -> None:
    cls = classify(
        _passage(visibility_scope="tenant_private", law_type="act")
    )
    # Even when the chunker says "act", a tenant-private chunk must NEVER be
    # treated as authoritative law. It collapses to tenant_private_reference.
    assert cls.tier == "tenant_private_reference"
    assert cls.source_scope == "tenant_private"


def test_malformed_origin_falls_back_to_unknown() -> None:
    cls = classify(_passage(document_origin="not_a_real_origin", visibility_scope="global"))
    assert cls.tier == "unknown"


def test_misplaced_tenant_origin_under_global_scope_downgrades_to_unknown() -> None:
    """Belt-and-braces: if a sample_case somehow lands with GLOBAL scope, we
    refuse to promote it to a public tier."""
    cls = classify(_passage(document_origin="sample_case", visibility_scope="global"))
    assert cls.tier == "unknown"


# ---- annotate() / partition / ordered_tiers --------------------------------


def test_annotate_stamps_rank_and_scope() -> None:
    p = _passage(document_origin="act", visibility_scope="global")
    annotate([p])
    assert p.legal_authority_rank == 90
    assert p.source_scope == "global"
    assert p.authority_tier == "act"


def test_partition_preserves_within_tier_order() -> None:
    """Rerank order inside each tier must be preserved (rerank_score desc is
    what the caller hands us)."""
    a1 = _passage(law="Act A", document_origin="act", visibility_scope="global", rerank_score=0.9)
    a2 = _passage(law="Act B", document_origin="act", visibility_scope="global", rerank_score=0.6)
    case = _passage(law="Case X", document_origin="case_law", visibility_scope="global", rerank_score=0.7)
    buckets = partition_by_tier([a1, a2, case])
    assert [p.law for p in buckets["act"]] == ["Act A", "Act B"]
    assert [p.law for p in buckets["public_case_law"]] == ["Case X"]


def test_ordered_tiers_follows_canonical_authority_order() -> None:
    """Even if we shuffle the input, the tier order is constitution → act →
    regulation → public_case_law → tenant tiers."""
    passages = [
        _passage(document_origin="sample_case", visibility_scope="tenant_private", law="firm-S"),
        _passage(document_origin="case_law", visibility_scope="global", law="case-X"),
        _passage(document_origin="constitution", visibility_scope="global", law="const-A"),
        _passage(document_origin="regulation", visibility_scope="global", law="reg-R"),
        _passage(document_origin="act", visibility_scope="global", law="act-Y"),
    ]
    buckets = partition_by_tier(passages)
    ordered = ordered_tiers(buckets)
    tier_sequence = [t for t, _ in ordered]
    # constitution before act before regulation before public_case_law before any tenant tier.
    assert tier_sequence.index("constitution") < tier_sequence.index("act")
    assert tier_sequence.index("act") < tier_sequence.index("regulation")
    assert tier_sequence.index("regulation") < tier_sequence.index("public_case_law")
    assert tier_sequence.index("public_case_law") < tier_sequence.index("tenant_sample_case")


def test_ordered_tiers_drops_empty_tiers() -> None:
    """ordered_tiers should only return tiers that actually have passages."""
    p = _passage(document_origin="act", visibility_scope="global")
    buckets = partition_by_tier([p])
    ordered = ordered_tiers(buckets)
    assert len(ordered) == 1
    assert ordered[0][0] == "act"


def test_tier_order_is_strictly_descending_by_rank() -> None:
    """Sanity-check the constant: no tier may outrank a tier listed before it."""
    last_rank = 10**9
    for tier in TIER_ORDER:
        cls = classify(_passage(document_origin="act"))  # any seed; we look up rank below
        from app.rag.authority import _TIER_RANK

        rank = _TIER_RANK[tier]
        assert rank <= last_rank, f"Tier order broken at {tier}: {rank} > {last_rank}"
        last_rank = rank
        del cls  # unused
