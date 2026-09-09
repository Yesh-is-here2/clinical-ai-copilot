from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


FHIR_RESOURCE_TYPES = Literal[
    "Patient",
    "Observation",
    "Condition",
    "MedicationRequest",
    "AllergyIntolerance",
]


class Coding(BaseModel):
    system: str | None = None
    code: str | None = None
    display: str | None = None


class ConditionSummary(BaseModel):
    id: str
    code: str | None = None
    display: str
    coding_system: str | None = None
    clinical_status: str | None = None
    onset_date: date | None = None


class MedicationSummary(BaseModel):
    id: str
    code: str | None = None
    display: str
    coding_system: str | None = None
    status: str | None = None
    intent: str | None = None


class AllergySummary(BaseModel):
    id: str
    code: str | None = None
    display: str
    coding_system: str | None = None
    clinical_status: str | None = None
    criticality: str | None = None


class ObservationSummary(BaseModel):
    id: str
    code: str | None = None
    display: str
    coding_system: str | None = None
    value: float | str | None = None
    unit: str | None = None
    observed_at: datetime | None = None
    category: str | None = None


class BloodPressureSummary(BaseModel):
    systolic: float | None = None
    diastolic: float | None = None
    unit: str = "mmHg"
    observed_at: datetime | None = None


class PatientContext(BaseModel):
    patient_id: str

    full_name: str
    sex: str | None = None
    birth_date: date | None = None
    age: int | None = Field(default=None, ge=0, le=130)

    conditions: list[ConditionSummary] = Field(default_factory=list)
    medications: list[MedicationSummary] = Field(default_factory=list)
    allergies: list[AllergySummary] = Field(default_factory=list)
    observations: list[ObservationSummary] = Field(default_factory=list)

    blood_pressure: BloodPressureSummary | None = None

    source_resource_types: list[FHIR_RESOURCE_TYPES] = Field(default_factory=list)

    synthetic: bool = True

    model_config = {
        "extra": "forbid",
    }
