"""``GET /queries/{query_id}`` and ``GET /queries/{query_id}/evaluation``
(Milestone F, sections 8/9).

Both routes are pure reads over what
``app.services.clinical_query_service.ClinicalQueryService`` already
persisted - neither ever computes or fabricates anything. A query that
exists but has no evaluation rows yet reports an explicit
``status="not_run"``, never an invented metric.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_evaluation_repository, get_query_repository
from app.clinical.final_response import DISCLAIMER_TEXT
from app.db.repositories.evaluation_repository import EvaluationRepository
from app.db.repositories.query_repository import ClinicalQueryRepository
from app.schemas.clinical import (
    CitationRecordOut,
    EvaluationMetricOut,
    QueryDetailResponse,
    QueryEvaluationResponse,
)

router = APIRouter(prefix="/queries", tags=["Queries"])


@router.get(
    "/{query_id}",
    response_model=QueryDetailResponse,
    summary="Look up a previously submitted clinical query",
)
async def get_query(
    query_id: uuid.UUID,
    repository: Annotated[ClinicalQueryRepository, Depends(get_query_repository)],
) -> QueryDetailResponse:
    record = await repository.get_by_id(query_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' was not found.",
        )

    generated = record.generated_response

    return QueryDetailResponse(
        query_id=record.id,
        request_id=record.request_id,
        patient_id=record.patient_id,
        question=record.question,
        question_category=record.question_category,
        status=record.status,
        evidence_status=record.evidence_status,
        confidence=record.confidence,
        latency_ms=record.latency_ms,
        created_at=record.created_at,
        answer=generated.answer if generated else None,
        key_points=list(generated.key_points) if generated else [],
        warnings=list(generated.warnings) if generated else [],
        safety_flags=list(generated.safety_flags) if generated else [],
        citations=[
            CitationRecordOut(
                citation_label=citation.citation_label,
                chunk_id=citation.chunk_id,
                quoted_support=citation.quoted_support,
                support_classification=citation.support_classification,
            )
            for citation in (generated.citations if generated else [])
        ],
        disclaimer=DISCLAIMER_TEXT,
    )


@router.get(
    "/{query_id}/evaluation",
    response_model=QueryEvaluationResponse,
    summary="Look up evaluation metrics for a previously submitted query, if any exist",
    description=(
        "Phase 8 will build the full evaluation harness. For now, this "
        "returns whatever metrics were already computed and persisted "
        "for this query, or an explicit `not_run` status - it never "
        "runs or fabricates an evaluation on demand."
    ),
)
async def get_query_evaluation(
    query_id: uuid.UUID,
    query_repository: Annotated[ClinicalQueryRepository, Depends(get_query_repository)],
    evaluation_repository: Annotated[
        EvaluationRepository, Depends(get_evaluation_repository)
    ],
) -> QueryEvaluationResponse:
    if not await query_repository.exists(query_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' was not found.",
        )

    records = await evaluation_repository.get_metrics_for_query(query_id)

    if not records:
        return QueryEvaluationResponse(query_id=query_id, status="not_run", metrics=[])

    return QueryEvaluationResponse(
        query_id=query_id,
        status="available",
        metrics=[
            EvaluationMetricOut(
                metric_name=record.metric_name,
                metric_value=record.metric_value,
                run_id=record.run_id,
                chunking_config=record.chunking_config,
            )
            for record in records
        ],
    )
