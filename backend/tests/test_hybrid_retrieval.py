"""Unit tests for app.retrieval.hybrid.HybridRetrievalService.

No database, no network, no ML model: vector search and the embedding
provider are faked (matching the real classes' async interface without
subclassing them - the same duck-typing pattern used by
InMemoryEmbeddingRepository in tests/test_embedding_persistence.py),
BM25 uses the real in-memory BM25Index, and reranking uses
FakeReranker/FailingReranker from app.retrieval.reranker.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import (
    HybridRetrievalError,
    HybridRetrievalService,
    RetrievalComponentStatus,
)
from app.retrieval.reranker import FailingReranker, FakeReranker, RerankerScoringError
from app.retrieval.vector_search import VectorSearchResult
from app.schemas.document import ClinicalDocumentCategory, DocumentChunk


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------


class FakeVectorSearchService:
    def __init__(self, results: list[VectorSearchResult]) -> None:
        self._results = results
        self.calls: list[tuple[list[float], int]] = []

    async def search(self, query_embedding, *, top_k: int = 15):
        self.calls.append((list(query_embedding), top_k))
        return self._results[:top_k]


class FailingVectorSearchService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def search(self, query_embedding, *, top_k: int = 15):
        raise self._error


class FakeEmbeddingProvider:
    model = "fake-embedding"
    dimensions = 8

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_batch(self, texts):
        self.calls.append(list(texts))
        return [[0.1] * self.dimensions for _ in texts]


class FailingBM25Index:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def search(self, query: str, *, top_k: int = 15):
        self.calls += 1
        raise self._error


def make_chunk(chunk_id: str, text: str, *, document_id: str = "doc-1") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title="Keyword-indexed Guideline",
        organization="Example Org",
        source_url="https://example.org/guideline",
        category=ClinicalDocumentCategory.HYPERTENSION,
        access_date=date(2026, 9, 9),
        chunk_number=0,
        text=text,
    )


def make_vector_result(chunk_id: str, *, rank: int, score: float = 0.9) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        document_title="Vector-indexed Guideline",
        organization="Example Org",
        source_url="https://example.org/guideline",
        section_id="s1",
        section_heading="Overview",
        text=f"vector text for {chunk_id}",
        score=score,
        rank=rank,
    )


def make_bm25_index() -> BM25Index:
    return BM25Index(
        [
            make_chunk("c1", "hypertension diagnosis requires ambulatory monitoring"),
            make_chunk("c2", "diabetes management involves metformin"),
        ]
    )


# ---------------------------------------------------------------------
# Configuration / constructor validation
# ---------------------------------------------------------------------


def test_requires_at_least_one_retrieval_source() -> None:
    with pytest.raises(ValueError, match="at least one"):
        HybridRetrievalService(vector_search=None, bm25_index=None)


def test_vector_search_requires_an_embedding_provider() -> None:
    with pytest.raises(ValueError, match="embedding_provider"):
        HybridRetrievalService(
            vector_search=FakeVectorSearchService([]),
            embedding_provider=None,
            bm25_index=None,
        )


def test_rejects_non_positive_rrf_k() -> None:
    with pytest.raises(ValueError):
        HybridRetrievalService(vector_search=None, bm25_index=make_bm25_index(), rrf_k=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"semantic_top_k": 0},
        {"bm25_top_k": 0},
        {"reranker_top_k": 0},
    ],
)
def test_rejects_non_positive_top_k_settings(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        HybridRetrievalService(vector_search=None, bm25_index=make_bm25_index(), **kwargs)


def test_configuration_defaults_match_spec() -> None:
    service = HybridRetrievalService(vector_search=None, bm25_index=make_bm25_index())

    assert service._rrf_k == 60
    assert service._semantic_top_k == 15
    assert service._bm25_top_k == 15
    assert service._reranker_top_k == 5
    assert service._reranker_enabled is True


# ---------------------------------------------------------------------
# Empty query
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_blank_query_is_rejected() -> None:
    service = HybridRetrievalService(vector_search=None, bm25_index=make_bm25_index())

    with pytest.raises(ValueError):
        await service.retrieve("")

    with pytest.raises(ValueError):
        await service.retrieve("   ")


# ---------------------------------------------------------------------
# Full hybrid orchestration - happy path
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_hybrid_orchestration_happy_path_runs_all_components() -> None:
    vector_service = FakeVectorSearchService([make_vector_result("c1", rank=1)])
    embedding_provider = FakeEmbeddingProvider()
    bm25_index = make_bm25_index()
    reranker = FakeReranker()

    service = HybridRetrievalService(
        vector_search=vector_service,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
        reranker=reranker,
        reranker_top_k=5,
    )

    result = await service.retrieve("hypertension diagnosis")

    assert result.vector.status is RetrievalComponentStatus.SUCCESS
    assert result.bm25.status is RetrievalComponentStatus.SUCCESS
    assert result.reranker.status is RetrievalComponentStatus.SUCCESS
    assert result.reranked is True
    assert result.degraded is False
    assert result.evidence
    assert result.total_elapsed_seconds >= 0
    # Both components were actually invoked (proves the orchestrator
    # really dispatched to both, not just one).
    assert len(vector_service.calls) == 1
    assert len(embedding_provider.calls) == 1


# ---------------------------------------------------------------------
# Independent failure handling
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_vector_failure_falls_back_to_bm25_only_with_degradation_metadata() -> None:
    vector_service = FailingVectorSearchService(RuntimeError("pgvector down"))
    embedding_provider = FakeEmbeddingProvider()
    bm25_index = make_bm25_index()

    service = HybridRetrievalService(
        vector_search=vector_service,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
        reranker_enabled=False,
    )

    result = await service.retrieve("hypertension diagnosis")

    assert result.vector.status is RetrievalComponentStatus.FAILED
    assert "pgvector down" in result.vector.error_message
    assert result.bm25.status is RetrievalComponentStatus.SUCCESS
    assert result.degraded is True
    assert result.evidence
    assert all(e.vector_score is None for e in result.evidence)


@pytest.mark.anyio
async def test_bm25_failure_falls_back_to_vector_only_with_degradation_metadata() -> None:
    vector_service = FakeVectorSearchService(
        [make_vector_result("c1", rank=1), make_vector_result("c2", rank=2)]
    )
    embedding_provider = FakeEmbeddingProvider()
    bm25_index = FailingBM25Index(RuntimeError("bm25 corrupted index"))

    service = HybridRetrievalService(
        vector_search=vector_service,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
        reranker_enabled=False,
    )

    result = await service.retrieve("hypertension diagnosis")

    assert result.bm25.status is RetrievalComponentStatus.FAILED
    assert "bm25 corrupted index" in result.bm25.error_message
    assert result.vector.status is RetrievalComponentStatus.SUCCESS
    assert result.degraded is True
    assert result.evidence
    assert all(e.keyword_score is None for e in result.evidence)
    assert bm25_index.calls == 1


@pytest.mark.anyio
async def test_both_retrieval_sources_failing_raises_typed_error() -> None:
    vector_service = FailingVectorSearchService(RuntimeError("pgvector down"))
    embedding_provider = FakeEmbeddingProvider()
    bm25_index = FailingBM25Index(RuntimeError("bm25 corrupted index"))

    service = HybridRetrievalService(
        vector_search=vector_service,
        embedding_provider=embedding_provider,
        bm25_index=bm25_index,
    )

    with pytest.raises(HybridRetrievalError):
        await service.retrieve("hypertension diagnosis")


@pytest.mark.anyio
async def test_embedding_provider_failure_counts_as_vector_component_failure() -> None:
    class FailingEmbeddingProvider:
        model = "failing-embedding"
        dimensions = 8

        async def embed_batch(self, texts):
            raise RuntimeError("OpenAI unavailable")

    service = HybridRetrievalService(
        vector_search=FakeVectorSearchService([make_vector_result("c1", rank=1)]),
        embedding_provider=FailingEmbeddingProvider(),
        bm25_index=make_bm25_index(),
        reranker_enabled=False,
    )

    result = await service.retrieve("hypertension diagnosis")

    assert result.vector.status is RetrievalComponentStatus.FAILED
    assert "OpenAI unavailable" in result.vector.error_message
    assert result.bm25.status is RetrievalComponentStatus.SUCCESS


# ---------------------------------------------------------------------
# One retrieval source only (by configuration, not failure)
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_bm25_only_configuration_marks_vector_disabled_not_failed() -> None:
    service = HybridRetrievalService(
        vector_search=None,
        bm25_index=make_bm25_index(),
        reranker_enabled=False,
    )

    result = await service.retrieve("hypertension diagnosis")

    assert result.vector.status is RetrievalComponentStatus.DISABLED
    assert result.bm25.status is RetrievalComponentStatus.SUCCESS
    assert result.degraded is False  # disabled-by-design, not a failure
    assert result.evidence


@pytest.mark.anyio
async def test_vector_only_configuration_marks_bm25_disabled_not_failed() -> None:
    service = HybridRetrievalService(
        vector_search=FakeVectorSearchService([make_vector_result("c1", rank=1)]),
        embedding_provider=FakeEmbeddingProvider(),
        bm25_index=None,
        reranker_enabled=False,
    )

    result = await service.retrieve("hypertension diagnosis")

    assert result.bm25.status is RetrievalComponentStatus.DISABLED
    assert result.vector.status is RetrievalComponentStatus.SUCCESS
    assert result.degraded is False
    assert result.evidence


# ---------------------------------------------------------------------
# Reranker success / failure / disabled
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_reranker_success_reorders_evidence_by_reranker_score() -> None:
    # Both chunks must actually match the BM25 query, or the weaker one
    # never enters the candidate pool at all and there is nothing for the
    # reranker to reorder. c1 has heavy term repetition (BM25 ranks it
    # first); c2 mentions the term once (BM25 ranks it second).
    bm25_index = BM25Index(
        [
            make_chunk(
                "c1", "hypertension hypertension hypertension screening protocol"
            ),
            make_chunk("c2", "hypertension noted briefly during diabetes follow-up"),
        ]
    )

    def scorer(query: str, text: str) -> float:
        # Force c2 to outrank c1 despite BM25 ranking c1 first - proves
        # the reranker's ordering, not BM25's, wins when reranking
        # succeeds.
        return 10.0 if "diabetes" in text else 1.0

    service = HybridRetrievalService(
        vector_search=None,
        bm25_index=bm25_index,
        reranker=FakeReranker(scorer=scorer),
        reranker_top_k=5,
    )

    result = await service.retrieve("hypertension")

    # Sanity check on the premise: BM25 alone would have ranked c1 first.
    assert result.evidence
    unranked_by_reranker = [e for e in result.evidence if e.keyword_rank == 1]
    assert unranked_by_reranker[0].chunk_id == "c1"

    assert result.reranked is True
    assert result.reranker.status is RetrievalComponentStatus.SUCCESS
    assert result.evidence[0].chunk_id == "c2"
    assert result.evidence[0].reranker_rank == 1
    assert result.evidence[0].reranker_score == pytest.approx(10.0)


@pytest.mark.anyio
async def test_reranker_failure_falls_back_to_fused_order() -> None:
    bm25_index = make_bm25_index()

    baseline = HybridRetrievalService(
        vector_search=None, bm25_index=bm25_index, reranker_enabled=False
    )
    fused_only = await baseline.retrieve("hypertension")

    service = HybridRetrievalService(
        vector_search=None,
        bm25_index=make_bm25_index(),
        reranker=FailingReranker(error=RerankerScoringError("model exploded")),
    )

    result = await service.retrieve("hypertension")

    assert result.reranker.status is RetrievalComponentStatus.FAILED
    assert "model exploded" in result.reranker.error_message
    assert result.reranked is False
    assert result.degraded is True
    assert [e.chunk_id for e in result.evidence] == [
        e.chunk_id for e in fused_only.evidence
    ]
    assert all(e.reranker_score is None for e in result.evidence)


@pytest.mark.anyio
async def test_reranker_disabled_skips_it_without_marking_degraded() -> None:
    service = HybridRetrievalService(
        vector_search=None,
        bm25_index=make_bm25_index(),
        reranker=FakeReranker(),
        reranker_enabled=False,
    )

    result = await service.retrieve("hypertension")

    assert result.reranker.status is RetrievalComponentStatus.DISABLED
    assert result.reranked is False
    assert result.degraded is False


@pytest.mark.anyio
async def test_reranker_enabled_but_not_configured_is_disabled_not_failed() -> None:
    service = HybridRetrievalService(
        vector_search=None,
        bm25_index=make_bm25_index(),
        reranker=None,
        reranker_enabled=True,
    )

    result = await service.retrieve("hypertension")

    assert result.reranker.status is RetrievalComponentStatus.DISABLED
    assert result.degraded is False


# ---------------------------------------------------------------------
# top-k behavior
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_reranker_top_k_truncates_final_evidence() -> None:
    chunks = [
        make_chunk(f"c{i}", "hypertension evidence and guidance", document_id="doc-1")
        for i in range(10)
    ]
    bm25_index = BM25Index(chunks)

    service = HybridRetrievalService(
        vector_search=None,
        bm25_index=bm25_index,
        reranker_enabled=False,
        reranker_top_k=3,
    )

    result = await service.retrieve("hypertension")

    assert len(result.evidence) == 3


# ---------------------------------------------------------------------
# Provenance preservation through the whole pipeline
# ---------------------------------------------------------------------


@pytest.mark.anyio
async def test_provenance_fields_survive_fusion_and_reranking() -> None:
    service = HybridRetrievalService(
        vector_search=None,
        bm25_index=make_bm25_index(),
        reranker=FakeReranker(),
    )

    result = await service.retrieve("hypertension")
    evidence = result.evidence[0]

    assert evidence.chunk_id
    assert evidence.document_id == "doc-1"
    assert evidence.title == "Keyword-indexed Guideline"
    assert evidence.organization == "Example Org"
    assert evidence.source_url == "https://example.org/guideline"
    assert evidence.text
    assert evidence.fused_rank is not None
    assert evidence.rrf_score is not None
