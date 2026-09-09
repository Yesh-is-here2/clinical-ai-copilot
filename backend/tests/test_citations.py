"""Unit tests for app.clinical.citations (Milestone E, section 9)."""

from __future__ import annotations

import pydantic
import pytest

from app.clinical.citations import (
    Citation,
    build_citations,
    citations_by_id,
    filter_valid_citation_ids,
)
from app.retrieval.fusion import RetrievedEvidence


def make_evidence(chunk_id: str, text: str, *, vector_score=0.8, keyword_score=1.2) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=f"Guideline for {chunk_id}",
        organization="Example Org",
        source_url="https://example.org/guideline",
        section_id="s1",
        section_heading="Overview",
        text=text,
        keyword_score=keyword_score,
        keyword_rank=1,
        vector_score=vector_score,
        vector_rank=1,
        rrf_score=0.5,
        fused_rank=1,
        reranker_score=0.75,
    )


def test_citation_ids_are_assigned_positionally() -> None:
    evidence = [make_evidence("c1", "text one"), make_evidence("c2", "text two")]

    citations = build_citations(evidence)

    assert [c.citation_id for c in citations] == ["C1", "C2"]
    assert citations[0].chunk_id == "c1"
    assert citations[1].chunk_id == "c2"


def test_quoted_support_is_a_verbatim_excerpt_of_chunk_text() -> None:
    evidence = [make_evidence("c1", "  Blood   pressure targets  are below 130/80. ")]

    citations = build_citations(evidence)

    assert citations[0].quoted_support == "Blood pressure targets are below 130/80."


def test_long_chunk_text_is_truncated_with_ellipsis() -> None:
    long_text = " ".join(["word"] * 100)
    evidence = [make_evidence("c1", long_text)]

    citations = build_citations(evidence)

    assert len(citations[0].quoted_support) <= 244
    assert citations[0].quoted_support.endswith("...")


def test_retrieval_score_prefers_vector_over_keyword() -> None:
    evidence = [make_evidence("c1", "text", vector_score=0.77, keyword_score=3.5)]

    citations = build_citations(evidence)

    assert citations[0].retrieval_score == 0.77


def test_retrieval_score_falls_back_to_keyword_when_no_vector_score() -> None:
    evidence = [make_evidence("c1", "text", vector_score=None, keyword_score=3.5)]

    citations = build_citations(evidence)

    assert citations[0].retrieval_score == 3.5


def test_empty_evidence_produces_no_citations() -> None:
    assert build_citations([]) == []


def test_citations_by_id_maps_by_citation_id() -> None:
    evidence = [make_evidence("c1", "one"), make_evidence("c2", "two")]
    citations = build_citations(evidence)

    by_id = citations_by_id(citations)

    assert set(by_id.keys()) == {"C1", "C2"}
    assert by_id["C1"].chunk_id == "c1"


def test_filter_valid_citation_ids_splits_valid_and_fabricated() -> None:
    known = citations_by_id(build_citations([make_evidence("c1", "one")]))

    valid, fabricated = filter_valid_citation_ids(["C1", "C99", "C1", "C42"], known)

    assert valid == ["C1"]
    assert fabricated == ["C99", "C42"]


def test_filter_valid_citation_ids_deduplicates_while_preserving_order() -> None:
    known = citations_by_id(
        build_citations([make_evidence("c1", "one"), make_evidence("c2", "two")])
    )

    valid, fabricated = filter_valid_citation_ids(["C2", "C1", "C2", "C1"], known)

    assert valid == ["C2", "C1"]
    assert fabricated == []


def test_citation_model_is_frozen_and_forbids_extra_fields() -> None:
    citation = Citation(
        citation_id="C1",
        chunk_id="c1",
        document_id="doc-1",
        document_title="Title",
        organization="Org",
        source_url="https://example.org",
        quoted_support="text",
    )

    with pytest.raises((TypeError, pydantic.ValidationError)):
        citation.citation_id = "C2"  # type: ignore[misc]
