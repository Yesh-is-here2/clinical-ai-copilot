import json
from datetime import date
from pathlib import Path

import pytest

from app.healthcare.fhir_parser import (
    FHIRParseError,
    parse_fhir_bundle,
    parse_fhir_file,
)


DATA_DIR = Path("/app/data/fhir/patients")


def test_fhir_patient_parser() -> None:
    patient = parse_fhir_file(
        DATA_DIR / "patient-001.json",
        reference_date=date(2026, 9, 9),
    )

    assert patient.patient_id == "patient-001"
    assert patient.full_name == "Alex Morgan"
    assert patient.sex == "male"
    assert patient.birth_date == date(1972, 4, 18)
    assert patient.age == 54

    assert len(patient.conditions) == 1
    assert patient.conditions[0].code == "I10"

    assert len(patient.medications) == 1
    assert patient.medications[0].code == "314076"

    assert patient.blood_pressure is not None
    assert patient.blood_pressure.systolic == 148
    assert patient.blood_pressure.diastolic == 92

    assert len(patient.observations) == 1
    assert patient.observations[0].code == "2160-0"

    assert patient.synthetic is True


def test_all_ten_synthetic_patients_parse() -> None:
    files = sorted(DATA_DIR.glob("patient-*.json"))

    assert len(files) == 10

    contexts = [
        parse_fhir_file(
            file,
            reference_date=date(2026, 9, 9),
        )
        for file in files
    ]

    assert len(contexts) == 10
    assert len({context.patient_id for context in contexts}) == 10


def test_patient_with_allergy_parses() -> None:
    patient = parse_fhir_file(
        DATA_DIR / "patient-004.json",
        reference_date=date(2026, 9, 9),
    )

    assert len(patient.allergies) == 1
    assert patient.allergies[0].code == "91936005"
    assert patient.allergies[0].display == "Allergy to penicillin"


def test_complex_patient_parses_multiple_resources() -> None:
    patient = parse_fhir_file(
        DATA_DIR / "patient-010.json",
        reference_date=date(2026, 9, 9),
    )

    assert len(patient.conditions) == 3
    assert len(patient.medications) == 3
    assert len(patient.observations) == 3

    condition_codes = {
        condition.code for condition in patient.conditions
    }

    assert condition_codes == {
        "I10",
        "E11.9",
        "E78.5",
    }


def test_non_bundle_is_rejected() -> None:
    with pytest.raises(
        FHIRParseError,
        match="Expected FHIR Bundle",
    ):
        parse_fhir_bundle(
            {
                "resourceType": "Patient",
                "id": "bad-patient",
            }
        )


def test_bundle_without_patient_is_rejected() -> None:
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-1",
                }
            }
        ],
    }

    with pytest.raises(
        FHIRParseError,
        match="exactly one Patient",
    ):
        parse_fhir_bundle(bundle)


def test_invalid_birth_date_is_rejected() -> None:
    bundle_path = DATA_DIR / "patient-001.json"

    bundle = json.loads(
        bundle_path.read_text(encoding="utf-8")
    )

    for entry in bundle["entry"]:
        resource = entry["resource"]

        if resource.get("resourceType") == "Patient":
            resource["birthDate"] = "invalid-date"

    with pytest.raises(
        FHIRParseError,
        match="birthDate is invalid",
    ):
        parse_fhir_bundle(bundle)

