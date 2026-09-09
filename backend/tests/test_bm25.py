from datetime import date

import pytest

from app.retrieval.bm25 import BM25Config, BM25Index, tokenize
from app.schemas.document import ClinicalDocumentCategory, DocumentChunk


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    document_id: str = "doc-1",
    chunk_number: int = 0,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title="Test Guideline",
        organization="Example Org",
        source_url="https://example.org/guideline",
        category=ClinicalDocumentCategory.HYPERTENSION,
        access_date=date(2026, 9, 9),
        chunk_number=chunk_number,
        text=text,
    )


# ---------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_non_alphanumeric() -> None:
    assert tokenize("Blood-Pressure, Monitoring!") == [
        "blood",
        "pressure",
        "monitoring",
    ]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


# ---------------------------------------------------------------------
# BM25Config validation
# ---------------------------------------------------------------------


def test_config_rejects_negative_k1() -> None:
    with pytest.raises(ValueError):
        BM25Config(k1=-1.0)


def test_config_rejects_b_out_of_range() -> None:
    with pytest.raises(ValueError):
        BM25Config(b=1.5)

    with pytest.raises(ValueError):
        BM25Config(b=-0.1)


# ---------------------------------------------------------------------
# Empty corpus / empty query
# ---------------------------------------------------------------------


def test_empty_corpus_search_returns_no_results() -> None:
    index = BM25Index([])

    assert index.corpus_size == 0
    assert index.search("hypertension") == []


def test_empty_query_returns_no_results() -> None:
    index = BM25Index([make_chunk("c1", "hypertension diagnosis evidence")])

    assert index.search("") == []
    assert index.search("   ") == []


def test_query_with_no_matching_terms_returns_no_results() -> None:
    index = BM25Index([make_chunk("c1", "hypertension diagnosis evidence")])

    assert index.search("diabetes management") == []


def test_query_with_only_punctuation_returns_no_results() -> None:
    index = BM25Index([make_chunk("c1", "hypertension diagnosis evidence")])

    assert index.search("!!!") == []


# ---------------------------------------------------------------------
# Ranking / scoring behavior
# ---------------------------------------------------------------------


def test_more_relevant_document_ranks_first() -> None:
    chunks = [
        make_chunk(
            "c1",
            "Hypertension hypertension hypertension is a chronic condition.",
        ),
        make_chunk("c2", "Diabetes is a chronic metabolic condition."),
    ]
    index = BM25Index(chunks)

    results = index.search("hypertension")

    assert results[0].chunk_id == "c1"
    assert results[0].rank == 1


def test_top_k_limits_result_count() -> None:
    chunks = [
        make_chunk(f"c{i}", "hypertension evidence and guidance", chunk_number=i)
        for i in range(10)
    ]
    index = BM25Index(chunks)

    results = index.search("hypertension", top_k=3)

    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]


def test_rejects_non_positive_top_k() -> None:
    index = BM25Index([make_chunk("c1", "hypertension evidence")])

    with pytest.raises(ValueError):
        index.search("hypertension", top_k=0)


def test_results_carry_provenance_fields() -> None:
    index = BM25Index([make_chunk("c1", "hypertension diagnosis evidence")])

    result = index.search("hypertension")[0]

    assert result.chunk_id == "c1"
    assert result.document_id == "doc-1"
    assert result.text == "hypertension diagnosis evidence"
    assert result.score > 0


def test_results_carry_document_and_section_metadata_for_fusion() -> None:
    # Milestone D's RRF fusion (app.retrieval.fusion) needs title/
    # organization/source_url/section directly off BM25Result so a
    # BM25-only hit can be turned into a fully-populated RetrievedEvidence
    # without a separate lookup.
    index = BM25Index([make_chunk("c1", "hypertension diagnosis evidence")])

    result = index.search("hypertension")[0]

    assert result.document_title == "Test Guideline"
    assert result.organization == "Example Org"
    assert result.source_url == "https://example.org/guideline"
    # make_chunk() doesn't set section_id/section_heading - both should
    # come through as None rather than raising.
    assert result.section_id is None
    assert result.section_heading is None


# ---------------------------------------------------------------------
# Determinism / stable tie-breaking
# ---------------------------------------------------------------------


def test_search_is_deterministic_across_calls() -> None:
    chunks = [
        make_chunk("c1", "hypertension evidence"),
        make_chunk("c2", "hypertension guidance"),
    ]
    index = BM25Index(chunks)

    first = index.search("hypertension")
    second = index.search("hypertension")

    assert first == second


def test_tied_scores_break_ties_by_chunk_id() -> None:
    # Identical text -> identical BM25 score. Tie must break
    # deterministically by chunk_id, not by insertion/hash order.
    chunks = [
        make_chunk("z-chunk", "hypertension evidence text"),
        make_chunk("a-chunk", "hypertension evidence text"),
    ]
    index = BM25Index(chunks)

    results = index.search("hypertension")

    assert [r.chunk_id for r in results] == ["a-chunk", "z-chunk"]


# ---------------------------------------------------------------------
# No global mutable state / independent indices
# ---------------------------------------------------------------------


def test_two_indices_do_not_share_state() -> None:
    index_a = BM25Index([make_chunk("c1", "hypertension evidence")])
    index_b = BM25Index([make_chunk("c1", "diabetes evidence")])

    assert index_a.search("hypertension")
    assert not index_b.search("hypertension")
    assert index_b.search("diabetes")


def test_duplicate_chunk_id_is_rejected() -> None:
    chunks = [
        make_chunk("c1", "hypertension evidence"),
        make_chunk("c1", "duplicate id, different text"),
    ]

    with pytest.raises(ValueError, match="Duplicate chunk_id"):
        BM25Index(chunks)
