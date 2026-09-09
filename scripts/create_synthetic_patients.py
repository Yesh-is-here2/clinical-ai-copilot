import json
from pathlib import Path


OUTPUT_DIR = Path("data/fhir/patients")


def coding(system: str, code: str, display: str) -> dict:
    return {
        "coding": [
            {
                "system": system,
                "code": code,
                "display": display,
            }
        ],
        "text": display,
    }


def patient_resource(
    patient_id: str,
    given: str,
    family: str,
    gender: str,
    birth_date: str,
) -> dict:
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "meta": {
            "tag": [
                {
                    "system": "https://clinical-ai-copilot.example/tags",
                    "code": "synthetic",
                    "display": "Synthetic patient",
                }
            ]
        },
        "name": [
            {
                "use": "official",
                "family": family,
                "given": [given],
            }
        ],
        "gender": gender,
        "birthDate": birth_date,
    }


def condition_resource(
    resource_id: str,
    patient_id: str,
    code: str,
    display: str,
    onset_date: str,
) -> dict:
    return {
        "resourceType": "Condition",
        "id": resource_id,
        "clinicalStatus": coding(
            "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "active",
            "Active",
        ),
        "code": coding(
            "http://hl7.org/fhir/sid/icd-10-cm",
            code,
            display,
        ),
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "onsetDateTime": onset_date,
    }


def medication_resource(
    resource_id: str,
    patient_id: str,
    rxnorm_code: str,
    display: str,
) -> dict:
    return {
        "resourceType": "MedicationRequest",
        "id": resource_id,
        "status": "active",
        "intent": "order",
        "medicationCodeableConcept": coding(
            "http://www.nlm.nih.gov/research/umls/rxnorm",
            rxnorm_code,
            display,
        ),
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
    }


def allergy_resource(
    resource_id: str,
    patient_id: str,
    code: str,
    display: str,
    criticality: str = "low",
) -> dict:
    return {
        "resourceType": "AllergyIntolerance",
        "id": resource_id,
        "clinicalStatus": coding(
            "http://terminology.hl7.org/CodeSystem/allergyintolerance-clinical",
            "active",
            "Active",
        ),
        "criticality": criticality,
        "code": coding(
            "http://snomed.info/sct",
            code,
            display,
        ),
        "patient": {
            "reference": f"Patient/{patient_id}",
        },
    }


def quantity_observation(
    resource_id: str,
    patient_id: str,
    loinc_code: str,
    display: str,
    value: float,
    unit: str,
    observed_at: str,
    category: str = "laboratory",
) -> dict:
    return {
        "resourceType": "Observation",
        "id": resource_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": category,
                        "display": category.title(),
                    }
                ]
            }
        ],
        "code": coding(
            "http://loinc.org",
            loinc_code,
            display,
        ),
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "effectiveDateTime": observed_at,
        "valueQuantity": {
            "value": value,
            "unit": unit,
        },
    }


def blood_pressure_observation(
    resource_id: str,
    patient_id: str,
    systolic: float,
    diastolic: float,
    observed_at: str,
) -> dict:
    return {
        "resourceType": "Observation",
        "id": resource_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs",
                    }
                ]
            }
        ],
        "code": coding(
            "http://loinc.org",
            "85354-9",
            "Blood pressure panel",
        ),
        "subject": {
            "reference": f"Patient/{patient_id}",
        },
        "effectiveDateTime": observed_at,
        "component": [
            {
                "code": coding(
                    "http://loinc.org",
                    "8480-6",
                    "Systolic blood pressure",
                ),
                "valueQuantity": {
                    "value": systolic,
                    "unit": "mmHg",
                },
            },
            {
                "code": coding(
                    "http://loinc.org",
                    "8462-4",
                    "Diastolic blood pressure",
                ),
                "valueQuantity": {
                    "value": diastolic,
                    "unit": "mmHg",
                },
            },
        ],
    }


def make_bundle(patient: dict, resources: list[dict]) -> dict:
    all_resources = [patient, *resources]

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "id": f"{patient['id']}-bundle",
        "meta": {
            "tag": [
                {
                    "system": "https://clinical-ai-copilot.example/tags",
                    "code": "synthetic",
                    "display": "Synthetic educational fixture",
                }
            ]
        },
        "entry": [
            {
                "fullUrl": f"urn:uuid:{resource['id']}",
                "resource": resource,
            }
            for resource in all_resources
        ],
    }


PATIENTS = [
    {
        "id": "patient-001",
        "given": "Alex",
        "family": "Morgan",
        "gender": "male",
        "birth_date": "1972-04-18",
        "conditions": [
            ("I10", "Essential (primary) hypertension", "2017-06-12"),
        ],
        "medications": [
            ("314076", "Lisinopril 10 MG Oral Tablet"),
        ],
        "allergies": [],
        "bp": (148, 92),
        "labs": [
            ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", 1.0, "mg/dL"),
        ],
    },
    {
        "id": "patient-002",
        "given": "Jordan",
        "family": "Lee",
        "gender": "female",
        "birth_date": "1985-11-03",
        "conditions": [
            ("E11.9", "Type 2 diabetes mellitus without complications", "2020-02-15"),
        ],
        "medications": [
            ("860975", "Metformin hydrochloride 500 MG 24 HR Extended Release Oral Tablet"),
        ],
        "allergies": [],
        "bp": (132, 84),
        "labs": [
            ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 7.2, "%"),
        ],
    },
    {
        "id": "patient-003",
        "given": "Taylor",
        "family": "Reed",
        "gender": "male",
        "birth_date": "1959-07-22",
        "conditions": [
            ("I10", "Essential (primary) hypertension", "2012-05-10"),
            ("E78.5", "Hyperlipidemia, unspecified", "2015-09-01"),
        ],
        "medications": [
            ("314076", "Lisinopril 10 MG Oral Tablet"),
            ("617310", "Atorvastatin 20 MG Oral Tablet"),
        ],
        "allergies": [],
        "bp": (142, 86),
        "labs": [
            ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma", 118, "mg/dL"),
        ],
    },
    {
        "id": "patient-004",
        "given": "Morgan",
        "family": "Hayes",
        "gender": "female",
        "birth_date": "1994-01-16",
        "conditions": [],
        "medications": [],
        "allergies": [
            ("91936005", "Allergy to penicillin"),
        ],
        "bp": (116, 74),
        "labs": [
            ("718-7", "Hemoglobin [Mass/volume] in Blood", 13.1, "g/dL"),
        ],
    },
    {
        "id": "patient-005",
        "given": "Casey",
        "family": "Brooks",
        "gender": "male",
        "birth_date": "1966-09-28",
        "conditions": [
            ("E11.9", "Type 2 diabetes mellitus without complications", "2018-08-14"),
            ("I10", "Essential (primary) hypertension", "2019-01-20"),
        ],
        "medications": [
            ("860975", "Metformin hydrochloride 500 MG 24 HR Extended Release Oral Tablet"),
        ],
        "allergies": [],
        "bp": (154, 96),
        "labs": [
            ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 8.1, "%"),
            ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", 1.2, "mg/dL"),
        ],
    },
    {
        "id": "patient-006",
        "given": "Riley",
        "family": "Turner",
        "gender": "female",
        "birth_date": "1978-03-09",
        "conditions": [
            ("E78.5", "Hyperlipidemia, unspecified", "2021-04-04"),
        ],
        "medications": [
            ("617310", "Atorvastatin 20 MG Oral Tablet"),
        ],
        "allergies": [],
        "bp": (126, 80),
        "labs": [
            ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma", 94, "mg/dL"),
        ],
    },
    {
        "id": "patient-007",
        "given": "Avery",
        "family": "Mitchell",
        "gender": "male",
        "birth_date": "2001-12-11",
        "conditions": [],
        "medications": [],
        "allergies": [],
        "bp": (118, 72),
        "labs": [
            ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", 164, "mg/dL"),
        ],
    },
    {
        "id": "patient-008",
        "given": "Cameron",
        "family": "Diaz",
        "gender": "female",
        "birth_date": "1948-05-30",
        "conditions": [
            ("I10", "Essential (primary) hypertension", "2009-03-18"),
        ],
        "medications": [
            ("197361", "Amlodipine 5 MG Oral Tablet"),
        ],
        "allergies": [
            ("91936005", "Allergy to penicillin"),
        ],
        "bp": (138, 82),
        "labs": [
            ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", 1.3, "mg/dL"),
        ],
    },
    {
        "id": "patient-009",
        "given": "Parker",
        "family": "Evans",
        "gender": "male",
        "birth_date": "1989-06-05",
        "conditions": [
            ("R73.03", "Prediabetes", "2025-01-12"),
        ],
        "medications": [],
        "allergies": [],
        "bp": (128, 78),
        "labs": [
            ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 6.0, "%"),
        ],
    },
    {
        "id": "patient-010",
        "given": "Sydney",
        "family": "Bennett",
        "gender": "female",
        "birth_date": "1955-10-14",
        "conditions": [
            ("I10", "Essential (primary) hypertension", "2010-11-01"),
            ("E11.9", "Type 2 diabetes mellitus without complications", "2016-07-23"),
            ("E78.5", "Hyperlipidemia, unspecified", "2018-02-08"),
        ],
        "medications": [
            ("860975", "Metformin hydrochloride 500 MG 24 HR Extended Release Oral Tablet"),
            ("617310", "Atorvastatin 20 MG Oral Tablet"),
            ("197361", "Amlodipine 5 MG Oral Tablet"),
        ],
        "allergies": [],
        "bp": (146, 88),
        "labs": [
            ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", 7.5, "%"),
            ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma", 102, "mg/dL"),
            ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", 1.1, "mg/dL"),
        ],
    },
]


def build_patient_bundle(spec: dict) -> dict:
    patient_id = spec["id"]

    patient = patient_resource(
        patient_id=patient_id,
        given=spec["given"],
        family=spec["family"],
        gender=spec["gender"],
        birth_date=spec["birth_date"],
    )

    resources = []

    for index, condition in enumerate(spec["conditions"], start=1):
        code, display, onset_date = condition
        resources.append(
            condition_resource(
                resource_id=f"{patient_id}-condition-{index}",
                patient_id=patient_id,
                code=code,
                display=display,
                onset_date=onset_date,
            )
        )

    for index, medication in enumerate(spec["medications"], start=1):
        rxnorm_code, display = medication
        resources.append(
            medication_resource(
                resource_id=f"{patient_id}-medication-{index}",
                patient_id=patient_id,
                rxnorm_code=rxnorm_code,
                display=display,
            )
        )

    for index, allergy in enumerate(spec["allergies"], start=1):
        code, display = allergy
        resources.append(
            allergy_resource(
                resource_id=f"{patient_id}-allergy-{index}",
                patient_id=patient_id,
                code=code,
                display=display,
            )
        )

    systolic, diastolic = spec["bp"]

    resources.append(
        blood_pressure_observation(
            resource_id=f"{patient_id}-blood-pressure",
            patient_id=patient_id,
            systolic=systolic,
            diastolic=diastolic,
            observed_at="2026-08-20T09:00:00Z",
        )
    )

    for index, lab in enumerate(spec["labs"], start=1):
        loinc_code, display, value, unit = lab

        resources.append(
            quantity_observation(
                resource_id=f"{patient_id}-lab-{index}",
                patient_id=patient_id,
                loinc_code=loinc_code,
                display=display,
                value=value,
                unit=unit,
                observed_at="2026-08-20T09:15:00Z",
            )
        )

    return make_bundle(patient, resources)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for spec in PATIENTS:
        bundle = build_patient_bundle(spec)

        output_path = OUTPUT_DIR / f"{spec['id']}.json"

        output_path.write_text(
            json.dumps(bundle, indent=2),
            encoding="utf-8",
        )

        print(f"Created {output_path}")

    print(f"Generated {len(PATIENTS)} synthetic FHIR patient bundles.")


if __name__ == "__main__":
    main()


