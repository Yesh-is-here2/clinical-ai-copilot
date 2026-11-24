"""Orchestrates one ``POST /clinical/query`` request end to end
(Milestone F, section 2): resolve the optional patient, run the already-
independently-tested Milestone E graph, then persist the full query
lifecycle in one atomic transaction (section 5).

This module is the only place that bridges ``app.clinical.graph``'s
``ClinicalState`` to the persistence layer - ``app.api.routes.clinical``
never touches ``ClinicalState``, SQLAlchemy, or the graph directly
(section 6's repository/service boundary). Errors are raised as the
typed exceptions below; only ``app.api.routes.clinical`` decides how
each maps to an HTTP status (section 14) - this module never imports
FastAPI.
"""

from __future__ import annotations

import uuid

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.clinical.graph import ClinicalGraphDependencies, build_clinical_graph
from app.clinical.state import ClinicalState
from app.db.repositories.query_repository import (
    ClinicalQueryRepository,
    PersistedQueryInput,
    RetrievedEvidenceInput,
    build_citation_status_map,
)
from app.retrieval.fusion import RetrievedEvidence
from app.schemas.clinical import ClinicalQueryRequest, ClinicalQueryResponse
from app.schemas.patient import PatientContext
from app.services.patient import PatientNotFoundError, PatientService


class ClinicalQueryServiceError(RuntimeError):
    """Base class for all typed failures this service can raise."""


class PatientNotFoundForQueryError(ClinicalQueryServiceError):
    """The request named a ``patient_id`` that does not resolve to a
    known synthetic patient. Maps to 404 - never a 500."""


class ClinicalRetrievalUnavailableError(ClinicalQueryServiceError):
    """No retrieval backend is configured at all (no ingested documents
    and/or no embedding provider) - there is nothing safe to retrieve
    evidence from. Maps to 503: a genuine "this deployment isn't ready"
    condition, not a clinical-safety decision the graph itself should
    make silently."""


class ClinicalQueryPersistenceError(ClinicalQueryServiceError):
    """The clinical response was produced but could not be durably
    recorded (e.g. the database is unreachable). Maps to 503 - the
    caller must not receive a response this system cannot confirm it
    audited/persisted, matching the project's auditability requirements.
    """


class ClinicalQueryService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        patient_service: PatientService,
        graph_dependencies: ClinicalGraphDependencies | None,
    ) -> None:
        self._session = session
        self._patient_service = patient_service
        self._graph_dependencies = graph_dependencies

    async def run_query(
        self,
        *,
        request_id: str,
        request: ClinicalQueryRequest,
        actor_user_id: uuid.UUID | None = None,
        actor_role: str | None = None,
    ) -> ClinicalQueryResponse:
        """Run one clinical query.

        ``actor_user_id``/``actor_role`` identify the authenticated
        caller (Phase 7); both default to ``None`` for anonymous or
        legacy callers. They are plain values, not an ``Actor`` - this
        service layer must not import the API layer
        (``app.api.deps`` imports this module). The actor is recorded
        on the persisted query row and carried into the graph state so
        the audit row is attributed to the caller."""

        patient_context: PatientContext | None = None

        if request.patient_id:
            try:
                patient_context = self._patient_service.get_patient(
                    request.patient_id
                )
            except PatientNotFoundError as exc:
                raise PatientNotFoundForQueryError(str(exc)) from exc

        if self._graph_dependencies is None:
            raise ClinicalRetrievalUnavailableError(
                "Clinical retrieval is not configured: no guideline "
                "documents have been ingested and/or no embedding "
                "provider is configured. Ingest at least one document "
                "via POST /documents/ingest first."
            )

        graph = build_clinical_graph(self._graph_dependencies)

        initial_state: ClinicalState = {
            "request_id": request_id,
            "question": request.question,
            "patient_context": patient_context,
            "actor_user_id": actor_user_id,
            "actor_role": actor_role,
            "warnings": [],
            "component_errors": [],
            "regeneration_count": 0,
            "terminal_reason": None,
            "stage_timings": {},
        }

        query_id = uuid.uuid4()

        try:
            result_state = await graph.ainvoke(initial_state)
        except Exception:
            # Section 7: an unexpected graph failure must still be
            # recorded when the database is available, so a failed
            # request never vanishes without a trace. The recording
            # itself is best-effort - if it also fails, the original
            # exception still propagates (the route maps it to a 500)
            # rather than being replaced by a persistence error.
            await self._record_failure(
                query_id=query_id,
                request_id=request_id,
                request=request,
                patient_context=patient_context,
                user_id=actor_user_id,
            )
            raise

        final_response = result_state["final_response"]
        status = result_state.get("terminal_reason") or "normal"

        try:
            await self._persist(
                query_id=query_id,
                request_id=request_id,
                request=request,
                status=status,
                result_state=result_state,
                patient_context=patient_context,
                user_id=actor_user_id,
            )
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise ClinicalQueryPersistenceError(
                "The clinical response was generated but could not be "
                "persisted."
            ) from exc

        retrieval_result = result_state.get("retrieval_result")

        return ClinicalQueryResponse(
            query_id=query_id,
            request_id=final_response.request_id,
            status=status,
            patient_summary=final_response.patient_summary,
            clinical_question=final_response.clinical_question,
            answer=final_response.answer,
            key_points=final_response.key_points,
            citations=final_response.citations,
            confidence=final_response.confidence,
            confidence_label=final_response.confidence_label,
            evidence_status=final_response.evidence_status,
            warnings=final_response.warnings,
            safety_flags=final_response.safety_flags,
            disclaimer=final_response.disclaimer,
            degraded=bool(
                retrieval_result is not None and retrieval_result.degraded
            ),
        )

    async def _record_failure(
        self,
        *,
        query_id: uuid.UUID,
        request_id: str,
        request: ClinicalQueryRequest,
        patient_context: PatientContext | None,
        user_id: uuid.UUID | None = None,
    ) -> None:
        """Best-effort minimal ``queries`` row for a graph that raised.

        Never raises: the caller re-raises the original graph
        exception, and a failure to record must not replace it.
        """

        try:
            await self._session.rollback()
            repository = ClinicalQueryRepository(self._session)
            await repository.ensure_patient_synced(patient_context)
            await repository.record_failed_query(
                query_id=query_id,
                request_id=request_id,
                patient_id=request.patient_id,
                question=request.question,
                user_id=user_id,
            )
            await self._session.commit()
        except Exception:
            await self._session.rollback()

    async def _persist(
        self,
        *,
        query_id: uuid.UUID,
        request_id: str,
        request: ClinicalQueryRequest,
        status: str,
        result_state: dict,
        patient_context: PatientContext | None,
        user_id: uuid.UUID | None = None,
    ) -> None:
        final_response = result_state["final_response"]
        question_category = result_state.get("question_category")
        retrieved_evidence = result_state.get("retrieved_evidence") or []
        citation_validation = result_state.get("citation_validation")
        total_elapsed = result_state.get("total_elapsed_seconds")

        # One row per (chunk, retrieval method) - see
        # app.models.query.RetrievalResultRecord: the unique key is
        # (query_id, chunk_id, retrieval_method) precisely so the
        # evidence-inspector view can show, for one chunk, its BM25
        # rank/score, its vector rank/score, its fused RRF rank/score,
        # and - when reranking actually ran - its reranker rank/score
        # side by side. Collapsing those into a single "fused or
        # reranked" row (as an earlier draft did) silently discards
        # the per-component provenance section 4 requires.
        retrieval_rows = _build_retrieval_rows(retrieved_evidence)

        model_name = (
            self._graph_dependencies.llm_provider.model
            if self._graph_dependencies is not None
            else None
        )

        data = PersistedQueryInput(
            query_id=query_id,
            request_id=request_id,
            patient_id=request.patient_id,
            question=request.question,
            question_category=(
                question_category.value if question_category else None
            ),
            status=status,
            final_response=final_response,
            model_name=model_name,
            latency_ms=(
                total_elapsed * 1000 if total_elapsed is not None else None
            ),
            retrieval_rows=retrieval_rows,
            citation_status_by_id=build_citation_status_map(citation_validation),
            user_id=user_id,
        )

        repository = ClinicalQueryRepository(self._session)
        # Must run (and be flushed) before persist_query when a
        # patient_id is set - see ClinicalQueryRepository's module
        # docstring: queries.patient_id is a real FK to
        # patients.patient_id, and synthetic patients are otherwise only
        # file-backed, never inserted into that table.
        await repository.ensure_patient_synced(patient_context)
        await repository.persist_query(data)


def _build_retrieval_rows(
    retrieved_evidence: list[RetrievedEvidence],
) -> list[RetrievedEvidenceInput]:
    """Expand fused evidence into one row per retrieval method.

    A chunk that both retrievers surfaced gets a ``bm25`` row (its
    keyword rank/score), a ``semantic`` row (its vector rank/score),
    and always a ``fused`` row (its RRF rank/score); a chunk the
    reranker actually scored additionally gets a ``reranked`` row
    carrying the reranker rank/score. A component that never touched
    the chunk contributes no row - its absence is itself the
    provenance (never a fabricated rank/score).
    """

    rows: list[RetrievedEvidenceInput] = []

    for evidence in retrieved_evidence:
        if (
            evidence.keyword_rank is not None
            and evidence.keyword_score is not None
        ):
            rows.append(
                RetrievedEvidenceInput(
                    chunk_id=evidence.chunk_id,
                    retrieval_method="bm25",
                    rank=evidence.keyword_rank,
                    score=evidence.keyword_score,
                )
            )
        if evidence.vector_rank is not None and evidence.vector_score is not None:
            rows.append(
                RetrievedEvidenceInput(
                    chunk_id=evidence.chunk_id,
                    retrieval_method="semantic",
                    rank=evidence.vector_rank,
                    score=evidence.vector_score,
                )
            )
        rows.append(
            RetrievedEvidenceInput(
                chunk_id=evidence.chunk_id,
                retrieval_method="fused",
                rank=evidence.fused_rank,
                score=evidence.rrf_score,
            )
        )
        if (
            evidence.reranker_rank is not None
            and evidence.reranker_score is not None
        ):
            rows.append(
                RetrievedEvidenceInput(
                    chunk_id=evidence.chunk_id,
                    retrieval_method="reranked",
                    rank=evidence.reranker_rank,
                    score=evidence.reranker_score,
                    reranker_score=evidence.reranker_score,
                )
            )

    return rows
