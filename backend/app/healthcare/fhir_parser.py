import json
from datetime import date
from pathlib import Path
from typing import Any

from app.schemas.patient import (
    AllergySummary,
    BloodPressureSummary,
    ConditionSummary,
    MedicationSummary,
    ObservationSummary,
    PatientContext,
)


class FHIRParseError(ValueError):
    """Raised when a FHIR bundle cannot be normalized safely."""


def _first_coding(codeable_concept: dict[str, Any] | None) -> dict[str, Any]:
    if not codeable_concept:
        return {}

    codings = codeable_concept.get("coding") or []

    if not codings:
        return {}

    return codings[0] or {}


def _display_text(codeable_concept: dict[str, Any] | None) -> str:
    if not codeable_concept:
        return "Unknown"

    text = codeable_concept.get("text")
    if text:
        return str(text)

    coding = _first_coding(codeable_concept)
    display = coding.get("display")

    return str(display) if display else "Unknown"


def _clinical_status(resource: dict[str, Any]) -> str | None:
    clinical_status = resource.get("clinicalStatus")
    coding = _first_coding(clinical_status)

    code = coding.get("code")
    return str(code) if code is not None else None


def _observation_category(resource: dict[str, Any]) -> str | None:
    categories = resource.get("category") or []

    if not categories:
        return None

    coding = _first_coding(categories[0])

    code = coding.get("code")
    return str(code) if code is not None else None


def _patient_name(patient: dict[str, Any]) -> str:
    names = patient.get("name") or []

    if not names:
        return "Unknown"

    name = names[0]

    given = name.get("given") or []
    family = name.get("family")

    parts = [str(part) for part in given if part]

    if family:
        parts.append(str(family))

    return " ".join(parts) if parts else "Unknown"


def _calculate_age(birth_date: date, reference_date: date | None = None) -> int:
    today = reference_date or date.today()

    age = today.year - birth_date.year

    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


def _parse_condition(resource: dict[str, Any]) -> ConditionSummary:
    coding = _first_coding(resource.get("code"))

    onset_date = None
    onset_raw = resource.get("onsetDateTime")

    if onset_raw:
        onset_date = date.fromisoformat(str(onset_raw)[:10])

    return ConditionSummary(
        id=str(resource.get("id", "unknown-condition")),
        code=coding.get("code"),
        display=_display_text(resource.get("code")),
        coding_system=coding.get("system"),
        clinical_status=_clinical_status(resource),
        onset_date=onset_date,
    )


def _parse_medication(resource: dict[str, Any]) -> MedicationSummary:
    concept = resource.get("medicationCodeableConcept")
    coding = _first_coding(concept)

    return MedicationSummary(
        id=str(resource.get("id", "unknown-medication")),
        code=coding.get("code"),
        display=_display_text(concept),
        coding_system=coding.get("system"),
        status=resource.get("status"),
        intent=resource.get("intent"),
    )


def _parse_allergy(resource: dict[str, Any]) -> AllergySummary:
    concept = resource.get("code")
    coding = _first_coding(concept)

    return AllergySummary(
        id=str(resource.get("id", "unknown-allergy")),
        code=coding.get("code"),
        display=_display_text(concept),
        coding_system=coding.get("system"),
        clinical_status=_clinical_status(resource),
        criticality=resource.get("criticality"),
    )


def _parse_quantity_observation(
    resource: dict[str, Any],
) -> ObservationSummary:
    concept = resource.get("code")
    coding = _first_coding(concept)
    quantity = resource.get("valueQuantity") or {}

    observed_at = resource.get("effectiveDateTime")

    return ObservationSummary(
        id=str(resource.get("id", "unknown-observation")),
        code=coding.get("code"),
        display=_display_text(concept),
        coding_system=coding.get("system"),
        value=quantity.get("value"),
        unit=quantity.get("unit"),
        observed_at=observed_at,
        category=_observation_category(resource),
    )


def _parse_blood_pressure(
    resource: dict[str, Any],
) -> BloodPressureSummary | None:
    concept_code = _first_coding(resource.get("code")).get("code")

    if concept_code != "85354-9":
        return None

    systolic = None
    diastolic = None
    unit = "mmHg"

    for component in resource.get("component") or []:
        component_code = _first_coding(component.get("code")).get("code")
        quantity = component.get("valueQuantity") or {}

        if component_code == "8480-6":
            systolic = quantity.get("value")
            unit = quantity.get("unit") or unit

        elif component_code == "8462-4":
            diastolic = quantity.get("value")
            unit = quantity.get("unit") or unit

    if systolic is None and diastolic is None:
        return None

    return BloodPressureSummary(
        systolic=systolic,
        diastolic=diastolic,
        unit=unit,
        observed_at=resource.get("effectiveDateTime"),
    )


def parse_fhir_bundle(
    bundle: dict[str, Any],
    reference_date: date | None = None,
) -> PatientContext:
    if bundle.get("resourceType") != "Bundle":
        raise FHIRParseError("Expected FHIR Bundle resource.")

    entries = bundle.get("entry")

    if not isinstance(entries, list) or not entries:
        raise FHIRParseError("FHIR Bundle contains no entries.")

    resources = [
        entry.get("resource")
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("resource"), dict)
    ]

    patient_resources = [
        resource
        for resource in resources
        if resource.get("resourceType") == "Patient"
    ]

    if len(patient_resources) != 1:
        raise FHIRParseError(
            "FHIR Bundle must contain exactly one Patient resource."
        )

    patient = patient_resources[0]

    patient_id = patient.get("id")

    if not patient_id:
        raise FHIRParseError("Patient resource is missing id.")

    birth_date = None
    age = None

    birth_date_raw = patient.get("birthDate")

    if birth_date_raw:
        try:
            birth_date = date.fromisoformat(str(birth_date_raw))
            age = _calculate_age(
                birth_date,
                reference_date=reference_date,
            )
        except ValueError as exc:
            raise FHIRParseError(
                "Patient birthDate is invalid."
            ) from exc

    conditions: list[ConditionSummary] = []
    medications: list[MedicationSummary] = []
    allergies: list[AllergySummary] = []
    observations: list[ObservationSummary] = []
    blood_pressure: BloodPressureSummary | None = None

    source_resource_types: list[str] = []

    for resource in resources:
        resource_type = resource.get("resourceType")

        if resource_type and resource_type not in source_resource_types:
            source_resource_types.append(resource_type)

        if resource_type == "Condition":
            conditions.append(_parse_condition(resource))

        elif resource_type == "MedicationRequest":
            medications.append(_parse_medication(resource))

        elif resource_type == "AllergyIntolerance":
            allergies.append(_parse_allergy(resource))

        elif resource_type == "Observation":
            parsed_bp = _parse_blood_pressure(resource)

            if parsed_bp is not None:
                blood_pressure = parsed_bp
            elif "valueQuantity" in resource:
                observations.append(
                    _parse_quantity_observation(resource)
                )

    return PatientContext(
        patient_id=str(patient_id),
        full_name=_patient_name(patient),
        sex=patient.get("gender"),
        birth_date=birth_date,
        age=age,
        conditions=conditions,
        medications=medications,
        allergies=allergies,
        observations=observations,
        blood_pressure=blood_pressure,
        source_resource_types=source_resource_types,
        synthetic=True,
    )


def load_fhir_bundle(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)

    if not file_path.exists():
        raise FHIRParseError(
            f"FHIR fixture does not exist: {file_path}"
        )

    try:
        return json.loads(
            file_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise FHIRParseError(
            f"FHIR fixture contains invalid JSON: {file_path}"
        ) from exc


def parse_fhir_file(
    path: str | Path,
    reference_date: date | None = None,
) -> PatientContext:
    bundle = load_fhir_bundle(path)

    return parse_fhir_bundle(
        bundle,
        reference_date=reference_date,
    )
