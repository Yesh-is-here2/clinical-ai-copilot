"""``POST /clinical/query`` (Milestone F, section 2).

The route itself is deliberately thin: request validation is Pydantic's
job (``ClinicalQueryRequest``), all business logic lives in
``app.services.clinical_query_service.ClinicalQueryService``, and this
module's only real job is the HTTP error-mapping table from section 14 -
translating the service's typed exceptions into the right status code,
and translating any genuinely unexpected exception into a safe 500 with
no raw traceback/detail ever reaching the client.

Every *clinical-safety* decision (blocked injection, abstention, a
guarded/degraded answer) is already a normal, successful
``ClinicalQueryResponse`` by the time it reaches this route - the graph
resolves those internally (see app.clinical.graph's route table) and
none of them raise. Only genuine infrastructure failures (unknown
patient, retrieval not configured, DB unavailable, an unexpected bug)
turn into an HTTP error here.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.schemas.clinical import ClinicalQueryRequest, ClinicalQueryResponse
from app.services.clinical_query_service import (
    ClinicalQueryPersistenceError,
    ClinicalQueryService,
    ClinicalRetrievalUnavailableError,
    PatientNotFoundForQueryError,
)
from app.api.deps import get_clinical_query_service

logger = logging.getLogger("clinical_ai.clinical_api")

router = APIRouter(prefix="/clinical", tags=["Clinical"])


def _request_id(request: Request) -> str:
    # Set unconditionally by app.core.middleware.RequestContextMiddleware
    # for every request; the getattr default only guards test clients
    # that construct a route without that middleware installed.
    return getattr(request.state, "request_id", None) or str(uuid.uuid4())


@router.post(
    "/query",
    response_model=ClinicalQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a clinical question for evidence-grounded, safety-checked generation",
    description=(
        "Runs the synthetic-patient-aware, hybrid-retrieval-grounded "
        "clinical reasoning graph (prompt-injection defense, citation "
        "validation, hallucination detection, deterministic confidence "
        "scoring) and persists the full query lifecycle for audit. "
        "Educational and research prototype only - not a diagnostic "
        "tool and does not replace professional clinical judgment. "
        "Blocked, abstained, and guarded outcomes are all returned as "
        "normal 200 responses with an explanatory `status` and "
        "`warnings` - they are safety decisions, not errors."
    ),
)
async def submit_clinical_query(
    payload: ClinicalQueryRequest,
    request: Request,
    service: Annotated[ClinicalQueryService, Depends(get_clinical_query_service)],
) -> ClinicalQueryResponse:
    request_id = _request_id(request)

    try:
        return await service.run_query(request_id=request_id, request=payload)
    except PatientNotFoundForQueryError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except ClinicalRetrievalUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except ClinicalQueryPersistenceError as exc:
        logger.error(
            "Clinical query persistence failed",
            extra={"request_id": request_id},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The clinical response could not be recorded. Please retry.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 - final safety net, section 14
        logger.exception(
            "Unexpected error handling clinical query",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "An unexpected internal error occurred while processing "
                f"this request (request_id={request_id})."
            ),
        ) from exc
