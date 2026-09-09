"""Unit tests for Reciprocal Rank Fusion (app.retrieval.fusion).

No database, no network - VectorSearchResult/BM25Result are plain
dataclasses, constructed directly here.
"""

from __future__ import annotations

import pytest

from app.retrieval.bm25 import BM25Result
from app.retrieval.fusion import FusionError, reciprocal_rank_fusion
from app.retrieval.vector_search import VectorSearchResult


def make_vector_result(
    chunk_id: str,
    *,
    rank: int,
    score: float = 0.9,
    document_id: str = "doc-1",
    title: str = "Vector Title",
) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=title,
        organization="Example Org",
        source_url="https://example.org/guideline",
        section_id="s1",
        section_heading="Overview",
        text=f"text for {chunk_id}",
        score=score,
        rank=rank,
    )


def make_bm25_result(
    chunk_id: str,
    *,
    rank: int,
    score: float = 5.0,
    document_id: str = "doc-1",
    title: str = "Keyword Title",
) -> BM25Result:
    return BM25Result(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=title,
        organization="Example Org",
        source_url="https://example.org/guideline",
        section_id="s1",
        section_heading="Overview",
        text=f"text for {chunk_id}",
        score=score,
        rank=rank,
    )


# ---------------------------------------------------------------------
# RRF math
# ---------------------------------------------------------------------


def test_rrf_score_for_a_chunk_in_both_lists_sums_both_contributions() -> None:
    vector = [make_vector_result("c1", rank=1)]
    bm25 = [make_bm25_result("c1", rank=1)]

    fused = reciprocal_rank_fusion(vector, bm25, k=60)

    assert len(fused) == 1
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 1)
    assert fused[0].rrf_score == pytest.approx(expected)


def test_rrf_score_for_a_chunk_in_only_one_list_uses_only_that_term() -> None:
    vector = [make_vector_result("c1", rank=3)]

    fused = reciprocal_rank_fusion(vector, [], k=60)

    assert fused[0].rrf_score == pytest.approx(1.0 / (60 + 3))


def test_configurable_k_changes_the_score() -> None:
    vector = [make_vector_result("c1", rank=1)]

    fused_k60 = reciprocal_rank_fusion(vector, [], k=60)
    fused_k10 = reciprocal_rank_fusion(vector, [], k=10)

    assert fused_k60[0].rrf_score == pytest.approx(1.0 / 61)
    assert fused_k10[0].rrf_score == pytest.approx(1.0 / 11)
    assert fused_k10[0].rrf_score > fused_k60[0].rrf_score


def test_rejects_non_positive_k() -> None:
    with pytest.raises(FusionError):
        reciprocal_rank_fusion([], [], k=0)

    with pytest.raises(FusionError):
        reciprocal_rank_fusion([], [], k=-5)


# ---------------------------------------------------------------------
# Deduplication by chunk_id
# ---------------------------------------------------------------------


def test_chunk_in_both_lists_produces_exactly_one_fused_entry() -> None:
    vector = [make_vector_result("c1", rank=1)]
    bm25 = [make_bm25_result("c1", rank=1)]

    fused = reciprocal_rank_fusion(vector, bm25, k=60)

    assert len(fused) == 1
    assert fused[0].chunk_id == "c1"


def test_provenance_from_both_sources_is_retained_on_the_dedup_merge() -> None:
    vector = [make_vector_result("c1", rank=2, score=0.75)]
    bm25 = [make_bm25_result("c1", rank=1, score=4.2)]

    fused = reciprocal_rank_fusion(vector, bm25, k=60)[0]

    assert fused.vector_rank == 2
    assert fused.vector_score == pytest.approx(0.75)
    assert fused.keyword_rank == 1
    assert fused.keyword_score == pytest.approx(4.2)


# ---------------------------------------------------------------------
# Empty lists / one source only
# ---------------------------------------------------------------------


def test_both_lists_empty_returns_empty_pool() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_vector_only_still_produces_fused_evidence() -> None:
    vector = [make_vector_result("c1", rank=1), make_vector_result("c2", rank=2)]

    fused = reciprocal_rank_fusion(vector, [])

    assert [e.chunk_id for e in fused] == ["c1", "c2"]
    assert all(e.keyword_score is None and e.keyword_rank is None for e in fused)


def test_bm25_only_still_produces_fused_evidence() -> None:
    bm25 = [make_bm25_result("c1", rank=1), make_bm25_result("c2", rank=2)]

    fused = reciprocal_rank_fusion([], bm25)

    assert [e.chunk_id for e in fused] == ["c1", "c2"]
    assert all(e.vector_score is None and e.vector_rank is None for e in fused)


# ---------------------------------------------------------------------
# Different rank ordering between retrievers
# ---------------------------------------------------------------------


def test_a_chunk_ranked_well_by_both_retrievers_wins_over_ranked_well_by_one() -> None:
    # c1: vector rank 1 only. c2: bm25 rank 1 only. c3: rank 2 in both -
    # c3's combined RRF score should beat either single-source rank-1
    # result, since two moderate contributions can outweigh one large one.
    vector = [make_vector_result("c1", rank=1), make_vector_result("c3", rank=2)]
    bm25 = [make_bm25_result("c2", rank=1), make_bm25_result("c3", rank=2)]

    fused = reciprocal_rank_fusion(vector, bm25, k=1)

    scores = {e.chunk_id: e.rrf_score for e in fused}
    assert scores["c3"] > scores["c1"]
    assert scores["c3"] > scores["c2"]


def test_fused_rank_is_assigned_in_final_sorted_order() -> None:
    vector = [make_vector_result("low", rank=5), make_vector_result("high", rank=1)]

    fused = reciprocal_rank_fusion(vector, [])

    assert fused[0].chunk_id == "high"
    assert fused[0].fused_rank == 1
    assert fused[1].chunk_id == "low"
    assert fused[1].fused_rank == 2


# ---------------------------------------------------------------------
# Deterministic tie-breaking
# ---------------------------------------------------------------------


def test_tied_rrf_scores_break_ties_by_chunk_id() -> None:
    vector = [
        make_vector_result("z-chunk", rank=1),
        make_vector_result("a-chunk", rank=1),
    ]

    fused = reciprocal_rank_fusion(vector, [])

    assert [e.chunk_id for e in fused] == ["a-chunk", "z-chunk"]


def test_fusion_is_deterministic_across_calls_and_input_order() -> None:
    vector = [make_vector_result("c1", rank=1), make_vector_result("c2", rank=2)]
    bm25 = [make_bm25_result("c2", rank=1), make_bm25_result("c1", rank=2)]

    first = reciprocal_rank_fusion(vector, bm25)
    second = reciprocal_rank_fusion(list(reversed(vector)), list(reversed(bm25)))

    assert first == second


# ---------------------------------------------------------------------
# Raw-score independence: RRF must not be swayed by score magnitude,
# only by rank.
# ---------------------------------------------------------------------


def test_fusion_ignores_raw_score_magnitude_and_uses_rank_only() -> None:
    # c1 has a huge BM25 score but is ranked #2; c2 has a tiny BM25 score
    # but is ranked #1. RRF must favor c2 (better rank), regardless of
    # how large c1's raw score is.
    bm25 = [
        make_bm25_result("c2", rank=1, score=0.0001),
        make_bm25_result("c1", rank=2, score=999999.0),
    ]

    fused = reciprocal_rank_fusion([], bm25, k=60)

    assert fused[0].chunk_id == "c2"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_fusion_score_identical_for_identical_ranks_despite_different_raw_scores() -> None:
    vector_a = [make_vector_result("c1", rank=1, score=0.99)]
    vector_b = [make_vector_result("c1", rank=1, score=0.01)]

    fused_a = reciprocal_rank_fusion(vector_a, [])
    fused_b = reciprocal_rank_fusion(vector_b, [])

    assert fused_a[0].rrf_score == fused_b[0].rrf_score
