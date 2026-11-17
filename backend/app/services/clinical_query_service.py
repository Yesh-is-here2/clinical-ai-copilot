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
from app.retrieval.hybrid import RetrievalComponentStatus
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
        self, *, request_id: str, request: ClinicalQueryRequest
    ) -> ClinicalQueryResponse:
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
            "warnings": [],
            "component_errors": [],
            "regeneration_count": 0,
            "terminal_reason": None,
            "stage_timings": {},
        }

        result_state = await graph.ainvoke(initial_state)
        final_response = result_state["final_response"]
        status = result_state.get("terminal_reason") or "normal"

        query_id = uuid.uuid4()

        try:
            await self._persist(
                query_id=query_id,
                request_id=request_id,
                request=request,
                status=status,
                result_state=result_state,
                patient_context=patient_context,
            )
            await self._session.commit()
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise ClinicalQueryPersistenceError(
                "The clinical response was generated but could not be "
                "persisted."
            ) from exc

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
        )

    async def _persist(
        self,
        *,
        query_id: uuid.UUID,
        request_id: str,
        request: ClinicalQueryRequest,
        status: str,
        result_state: dict,
        patient_context: PatientContext | None,
    ) -> None:
        final_response = result_state["final_response"]
        question_category = result_state.get("question_category")
        retrieval_result = result_state.get("retrieval_result")
        retrieved_evidence = result_state.get("retrieved_evidence") or []
        citation_validation = result_state.get("citation_validation")
        total_elapsed = result_state.get("total_elapsed_seconds")

        method = (
            "reranked"
            if retrieval_result is not None
            and retrieval_result.reranker.status is RetrievalComponentStatus.SUCCESS
            and retrieval_result.reranked
            else "fused"
        )

        retrieval_rows = [
            RetrievedEvidenceInput(
                chunk_id=evidence.chunk_id,
                retrieval_method=method,
                rank=(
                    evidence.reranker_rank
                    if evidence.reranker_rank is not None
                    else evidence.fused_rank
                ),
                score=(
                    evidence.reranker_score
                    if evidence.reranker_score is not None
                    else evidence.rrf_score
                ),
                reranker_score=evidence.reranker_score,
            )
            for evidence in retrieved_evidence
        ]

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
        )

        repository = ClinicalQueryRepository(self._session)
        # Must run (and be flushed) before persist_query when a
        # patient_id is set - see ClinicalQueryRepository's module
        # docstring: queries.patient_id is a real FK to
        # patients.patient_id, and synthetic patients are otherwise only
        # file-backed, never inserted into that table.
        await repository.ensure_patient_synced(patient_context)
        await repository.persist_query(data)
