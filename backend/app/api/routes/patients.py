from fastapi import APIRouter, HTTPException, status

from app.schemas.patient import PatientContext
from app.services.patient import (
    PatientNotFoundError,
    patient_service,
)


router = APIRouter(
    prefix="/patients",
    tags=["Patients"],
)


@router.get(
    "",
    response_model=list[PatientContext],
    summary="List synthetic patients",
)
async def list_patients() -> list[PatientContext]:
    return patient_service.list_patients()


@router.get(
    "/{patient_id}",
    response_model=PatientContext,
    summary="Get synthetic patient context",
)
async def get_patient(patient_id: str) -> PatientContext:
    try:
        return patient_service.get_patient(patient_id)
    except PatientNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
