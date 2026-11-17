from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_patient_service
from app.schemas.patient import PatientContext
from app.services.patient import PatientNotFoundError, PatientService


router = APIRouter(
    prefix="/patients",
    tags=["Patients"],
)


@router.get(
    "",
    response_model=list[PatientContext],
    summary="List synthetic patients",
)
async def list_patients(
    patient_service: Annotated[PatientService, Depends(get_patient_service)],
) -> list[PatientContext]:
    return patient_service.list_patients()


@router.get(
    "/{patient_id}",
    response_model=PatientContext,
    summary="Get synthetic patient context",
)
async def get_patient(
    patient_id: str,
    patient_service: Annotated[PatientService, Depends(get_patient_service)],
) -> PatientContext:
    try:
        return patient_service.get_patient(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
