"""Milestone D: hybrid retrieval orchestration (section 7 of the spec).

    query
    -> vector retrieval + BM25 retrieval (concurrent, independent)
    -> Reciprocal Rank Fusion (app.retrieval.fusion)
    -> candidate pool
    -> reranker (app.retrieval.reranker)
    -> top evidence

Every stage degrades independently and explicitly rather than silently -
see ``ComponentOutcome``/``RetrievalComponentStatus`` on
``HybridRetrievalResult``. Only when *both* vector and BM25 retrieval
fail does this raise (``HybridRetrievalError``); a single component
failing still returns the other component's results, clearly marked as
degraded. Nothing here ever fabricates a successful component or
relabels a fallback as though it were the real thing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from enum import StrEnum

from app.core.constants import (
    DEFAULT_RERANKER_TOP_K,
    DEFAULT_RRF_K,
)
from app.retrieval.bm25 import BM25Index, BM25Result
from app.retrieval.fusion import RetrievedEvidence, reciprocal_rank_fusion
from app.retrieval.reranker import Reranker, RerankerError
from app.retrieval.vector_search import VectorSearchResult, VectorSearchService
from app.services.embeddings import EmbeddingProvider


class HybridRetrievalError(RuntimeError):
    """Raised only when every configured retrieval component failed for
    this query - there is nothing safe to return. Callers should treat
    this as a hard failure (surface a clear error to the caller), never
    retry silently with fabricated results."""


class RetrievalComponentStatus(StrEnum):
    SUCCESS = "success"
    FAILED = "failed"
    # Not attempted by design: the component wasn't configured
    # (e.g. no bm25_index given), or reranker_enabled=False. Distinct
    # from FAILED, which means it was attempted and raised.
    DISABLED = "disabled"


@dataclass(frozen=True)
class ComponentOutcome:
    status: RetrievalComponentStatus
    result_count: int = 0
    elapsed_seconds: float | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class HybridRetrievalResult:
    query: str
    evidence: list[RetrievedEvidence]
    vector: ComponentOutcome
    bm25: ComponentOutcome
    reranker: ComponentOutcome
    # True only if the reranker actually ran and its scores were applied
    # (evidence is reranker-ordered). False for disabled/failed/empty-
    # candidate-pool cases, in which case `evidence` is fused-order.
    reranked: bool
    # True if any configured component (vector, bm25, or reranker) was
    # attempted and failed. Does NOT get set for a component that was
    # simply not configured/disabled by design - that's an intentional
    # configuration, not a degradation.
    degraded: bool
    total_elapsed_seconds: float


class HybridRetrievalService:
    """Wires together vector search, BM25, RRF, and an optional
    reranker. Holds no database/model state of its own beyond what's
    injected - a thin orchestrator over the components in this package.
    """

    def __init__(
        self,
        *,
        vector_search: VectorSearchService | None,
        embedding_provider: EmbeddingProvider | None = None,
        bm25_index: BM25Index | None,
        reranker: Reranker | None = None,
        reranker_enabled: bool = True,
        rrf_k: int = DEFAULT_RRF_K,
        semantic_top_k: int = 15,
        bm25_top_k: int = 15,
        reranker_top_k: int = DEFAULT_RERANKER_TOP_K,
    ) -> None:
        if vector_search is None and bm25_index is None:
            raise ValueError(
                "HybridRetrievalService needs at least one of "
                "vector_search or bm25_index configured - with neither, "
                "there is nothing to retrieve from."
            )

        if vector_search is not None and embedding_provider is None:
            raise ValueError(
                "vector_search was provided without an embedding_provider "
                "- vector retrieval needs both to embed the query text."
            )

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if semantic_top_k <= 0 or bm25_top_k <= 0 or reranker_top_k <= 0:
            raise ValueError("top_k settings must be greater than zero.")

        self._vector_search = vector_search
        self._embedding_provider = embedding_provider
        self._bm25_index = bm25_index
        self._reranker = reranker
        self._reranker_enabled = reranker_enabled
        self._rrf_k = rrf_k
        self._semantic_top_k = semantic_top_k
        self._bm25_top_k = bm25_top_k
        self._reranker_top_k = reranker_top_k

    async def retrieve(self, query: str) -> HybridRetrievalResult:
        if not query or not query.strip():
            raise ValueError("query must not be blank.")

        overall_start = time.perf_counter()

        (vector_outcome, vector_results), (bm25_outcome, bm25_results) = (
            await asyncio.gather(
                self._vector_component(query), self._bm25_component(query)
            )
        )

        if (
            vector_outcome.status is RetrievalComponentStatus.FAILED
            and bm25_outcome.status is RetrievalComponentStatus.FAILED
        ):
            raise HybridRetrievalError(
                "Both vector and BM25 retrieval failed for this query - "
                f"vector: {vector_outcome.error_message!r}, "
                f"bm25: {bm25_outcome.error_message!r}"
            )

        fused = reciprocal_rank_fusion(vector_results, bm25_results, k=self._rrf_k)

        evidence, reranker_outcome, reranked = await self._apply_reranker(query, fused)

        top_evidence = evidence[: self._reranker_top_k]

        degraded = (
            vector_outcome.status is RetrievalComponentStatus.FAILED
            or bm25_outcome.status is RetrievalComponentStatus.FAILED
            or reranker_outcome.status is RetrievalComponentStatus.FAILED
        )

        return HybridRetrievalResult(
            query=query,
            evidence=top_evidence,
            vector=vector_outcome,
            bm25=bm25_outcome,
            reranker=reranker_outcome,
            reranked=reranked,
            degraded=degraded,
            total_elapsed_seconds=time.perf_counter() - overall_start,
        )

    async def _vector_component(
        self, query: str
    ) -> tuple[ComponentOutcome, list[VectorSearchResult]]:
        if self._vector_search is None:
            return ComponentOutcome(status=RetrievalComponentStatus.DISABLED), []

        start = time.perf_counter()

        try:
            query_embedding = (
                await self._embedding_provider.embed_batch([query])
            )[0]
            results = await self._vector_search.search(
                query_embedding, top_k=self._semantic_top_k
            )
        except Exception as exc:
            # Deliberately broad: an embedding-provider failure, a DB
            # error, a timeout - anything here means "vector retrieval
            # didn't work for this query", which is exactly the
            # degradation this boundary exists to catch and report, not
            # swallow. The error is still surfaced (error_message), just
            # not raised past this component.
            return (
                ComponentOutcome(
                    status=RetrievalComponentStatus.FAILED,
                    error_message=str(exc),
                    elapsed_seconds=time.perf_counter() - start,
                ),
                [],
            )

        return (
            ComponentOutcome(
                status=RetrievalComponentStatus.SUCCESS,
                result_count=len(results),
                elapsed_seconds=time.perf_counter() - start,
            ),
            results,
        )

    async def _bm25_component(
        self, query: str
    ) -> tuple[ComponentOutcome, list[BM25Result]]:
        if self._bm25_index is None:
            return ComponentOutcome(status=RetrievalComponentStatus.DISABLED), []

        start = time.perf_counter()

        try:
            # BM25Index.search is synchronous/CPU-bound - run it off the
            # event loop so it executes genuinely concurrently with the
            # vector component's network/DB awaits, rather than blocking
            # them.
            results = await asyncio.to_thread(
                self._bm25_index.search, query, top_k=self._bm25_top_k
            )
        except Exception as exc:
            return (
                ComponentOutcome(
                    status=RetrievalComponentStatus.FAILED,
                    error_message=str(exc),
                    elapsed_seconds=time.perf_counter() - start,
                ),
                [],
            )

        return (
            ComponentOutcome(
                status=RetrievalComponentStatus.SUCCESS,
                result_count=len(results),
                elapsed_seconds=time.perf_counter() - start,
            ),
            results,
        )

    async def _apply_reranker(
        self, query: str, fused: list[RetrievedEvidence]
    ) -> tuple[list[RetrievedEvidence], ComponentOutcome, bool]:
        if not self._reranker_enabled:
            return fused, ComponentOutcome(status=RetrievalComponentStatus.DISABLED), False

        if self._reranker is None:
            return (
                fused,
                ComponentOutcome(
                    status=RetrievalComponentStatus.DISABLED,
                    error_message=(
                        "reranker_enabled is True but no reranker was "
                        "configured on this HybridRetrievalService."
                    ),
                ),
                False,
            )

        if not fused:
            return (
                fused,
                ComponentOutcome(status=RetrievalComponentStatus.SUCCESS, result_count=0),
                False,
            )

        start = time.perf_counter()

        try:
            scores = await self._reranker.score(
                query, [evidence.text for evidence in fused]
            )
        except RerankerError as exc:
            # Typed failure, per the reranker protocol's contract - fall
            # back to fused ordering, mark it explicitly. A reranker
            # implementation raising something other than RerankerError
            # is a bug in that implementation and is deliberately NOT
            # caught here.
            return (
                fused,
                ComponentOutcome(
                    status=RetrievalComponentStatus.FAILED,
                    error_message=str(exc),
                    elapsed_seconds=time.perf_counter() - start,
                ),
                False,
            )

        if len(scores) != len(fused):
            return (
                fused,
                ComponentOutcome(
                    status=RetrievalComponentStatus.FAILED,
                    error_message=(
                        f"Reranker '{self._reranker.model_name}' returned "
                        f"{len(scores)} scores for {len(fused)} candidates."
                    ),
                    elapsed_seconds=time.perf_counter() - start,
                ),
                False,
            )

        scored = [
            dataclass_replace(evidence, reranker_score=score)
            for evidence, score in zip(fused, scores, strict=True)
        ]
        scored.sort(key=lambda evidence: (-evidence.reranker_score, evidence.chunk_id))
        reranked_evidence = [
            dataclass_replace(evidence, reranker_rank=rank)
            for rank, evidence in enumerate(scored, start=1)
        ]

        return (
            reranked_evidence,
            ComponentOutcome(
                status=RetrievalComponentStatus.SUCCESS,
                result_count=len(reranked_evidence),
                elapsed_seconds=time.perf_counter() - start,
            ),
            True,
        )
