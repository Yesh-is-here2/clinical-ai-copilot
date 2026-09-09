"""Reciprocal Rank Fusion (RRF) for combining semantic (pgvector) and
lexical (BM25) retrieval results (Milestone D, section 2/4 of the spec).

RRF is deliberately rank-based only: a chunk's fused score is
``1 / (k + rank)`` summed across whichever retrievers returned it, never
a function of the retrievers' raw scores. Cosine similarity (bounded,
roughly 0-1) and BM25 score (unbounded, corpus-dependent) are not on
comparable scales, so combining them directly would silently let
whichever retriever happens to produce larger numbers dominate. Raw
scores are still carried through on ``RetrievedEvidence`` (as
``keyword_score``/``vector_score``) purely as provenance/debugging
metadata - never read by the fusion math itself.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

from app.core.constants import DEFAULT_RRF_K
from app.retrieval.bm25 import BM25Result
from app.retrieval.vector_search import VectorSearchResult


class FusionError(ValueError):
    pass


@dataclass(frozen=True)
class RetrievedEvidence:
    """One fused candidate, carrying provenance from every stage of the
    hybrid retrieval pipeline that touched it (section 4 of the spec).

    Fields are populated progressively:
    - chunk identity/text/provenance and keyword_*/vector_* - set by
      ``reciprocal_rank_fusion`` (this module) from whichever of
      BM25/vector search returned the chunk (either or both).
    - rrf_score/fused_rank - set by ``reciprocal_rank_fusion``, always.
    - reranker_score/reranker_rank - set later, only if a reranker
      actually ran successfully (see app.retrieval.hybrid); left as
      ``None`` when reranking was skipped/disabled/failed, so a caller
      can tell "not reranked" apart from "reranked, scored low".
    """

    chunk_id: str
    document_id: str
    title: str
    organization: str
    source_url: str
    section_id: str | None
    section_heading: str | None
    text: str

    keyword_score: float | None
    keyword_rank: int | None

    vector_score: float | None
    vector_rank: int | None

    rrf_score: float
    fused_rank: int

    reranker_score: float | None = None
    reranker_rank: int | None = None


def reciprocal_rank_fusion(
    vector_results: Sequence[VectorSearchResult],
    bm25_results: Sequence[BM25Result],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievedEvidence]:
    """Fuse semantic and lexical result lists into one ranked candidate
    pool, deduplicated by ``chunk_id``.

    Either input may be empty (one retrieval source only) or both
    (empty candidate pool out). A chunk present in both lists gets both
    its keyword and vector provenance recorded on a single
    ``RetrievedEvidence``, not two separate entries.
    """

    if k <= 0:
        raise FusionError("k must be greater than zero.")

    vector_by_chunk: dict[str, VectorSearchResult] = {}
    bm25_by_chunk: dict[str, BM25Result] = {}

    for result in vector_results:
        vector_by_chunk[result.chunk_id] = result

    for result in bm25_results:
        bm25_by_chunk[result.chunk_id] = result

    # dict.fromkeys preserves first-seen order, but that order is
    # irrelevant here - the final list is always explicitly re-sorted
    # below (rrf_score desc, chunk_id asc), so the fused ranking never
    # depends on which input list happened to list a shared chunk_id
    # first.
    all_chunk_ids = dict.fromkeys(
        [result.chunk_id for result in vector_results]
        + [result.chunk_id for result in bm25_results]
    )

    unranked: list[RetrievedEvidence] = []

    for chunk_id in all_chunk_ids:
        vector_result = vector_by_chunk.get(chunk_id)
        bm25_result = bm25_by_chunk.get(chunk_id)

        rrf_score = 0.0
        if vector_result is not None:
            rrf_score += 1.0 / (k + vector_result.rank)
        if bm25_result is not None:
            rrf_score += 1.0 / (k + bm25_result.rank)

        # Metadata (title/organization/source_url/section/document_id/
        # text): both sources index the same underlying chunk, so in
        # practice they agree - prefer the vector-search result when both
        # are present simply because it's read straight from the live
        # `clinical_documents`/`document_chunks` tables, but the choice
        # only matters for which object wins, not for correctness.
        primary = vector_result if vector_result is not None else bm25_result
        assert primary is not None  # every chunk_id here came from at least one list

        unranked.append(
            RetrievedEvidence(
                chunk_id=chunk_id,
                document_id=primary.document_id,
                title=(
                    vector_result.document_title
                    if vector_result is not None
                    else bm25_result.document_title  # type: ignore[union-attr]
                ),
                organization=primary.organization,
                source_url=primary.source_url,
                section_id=primary.section_id,
                section_heading=primary.section_heading,
                text=primary.text,
                keyword_score=bm25_result.score if bm25_result is not None else None,
                keyword_rank=bm25_result.rank if bm25_result is not None else None,
                vector_score=vector_result.score if vector_result is not None else None,
                vector_rank=vector_result.rank if vector_result is not None else None,
                rrf_score=rrf_score,
                fused_rank=0,  # placeholder, assigned below after sorting
            )
        )

    # Deterministic ordering: rrf_score descending, chunk_id ascending as
    # a stable tie-break (same convention as BM25Index.search and
    # build_vector_search_statement).
    unranked.sort(key=lambda evidence: (-evidence.rrf_score, evidence.chunk_id))

    return [
        replace(evidence, fused_rank=rank)
        for rank, evidence in enumerate(unranked, start=1)
    ]
