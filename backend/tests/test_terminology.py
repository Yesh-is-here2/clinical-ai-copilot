from app.healthcare.terminology import (
    SYSTEM_URIS,
    TerminologySystem,
    is_known_code,
    list_concepts,
    lookup_concept,
)


def test_icd10_hypertension_lookup() -> None:
    concept = lookup_concept(
        TerminologySystem.ICD10_CM,
        "I10",
    )

    assert concept is not None
    assert concept.code == "I10"
    assert concept.display == "Essential (primary) hypertension"
    assert concept.system_uri == "http://hl7.org/fhir/sid/icd-10-cm"


def test_snomed_allergy_lookup() -> None:
    concept = lookup_concept(
        TerminologySystem.SNOMED_CT,
        "91936005",
    )

    assert concept is not None
    assert concept.display == "Allergy to penicillin"
    assert concept.system_uri == "http://snomed.info/sct"


def test_loinc_blood_pressure_codes() -> None:
    panel = lookup_concept(
        TerminologySystem.LOINC,
        "85354-9",
    )
    systolic = lookup_concept(
        TerminologySystem.LOINC,
        "8480-6",
    )
    diastolic = lookup_concept(
        TerminologySystem.LOINC,
        "8462-4",
    )

    assert panel is not None
    assert systolic is not None
    assert diastolic is not None

    assert systolic.display == "Systolic blood pressure"
    assert diastolic.display == "Diastolic blood pressure"


def test_rxnorm_medication_codes() -> None:
    lisinopril = lookup_concept(
        TerminologySystem.RXNORM,
        "314076",
    )
    atorvastatin = lookup_concept(
        TerminologySystem.RXNORM,
        "617310",
    )
    amlodipine = lookup_concept(
        TerminologySystem.RXNORM,
        "197361",
    )
    metformin = lookup_concept(
        TerminologySystem.RXNORM,
        "860975",
    )

    assert lisinopril is not None
    assert atorvastatin is not None
    assert amlodipine is not None
    assert metformin is not None

    assert lisinopril.display == "Lisinopril 10 MG Oral Tablet"
    assert atorvastatin.display == "Atorvastatin 20 MG Oral Tablet"
    assert amlodipine.display == "Amlodipine 5 MG Oral Tablet"

    assert "Extended Release" in metformin.display


def test_unknown_code_returns_none() -> None:
    concept = lookup_concept(
        TerminologySystem.LOINC,
        "NOT-A-REAL-CODE",
    )

    assert concept is None


def test_is_known_code() -> None:
    assert is_known_code(
        TerminologySystem.ICD10_CM,
        "I10",
    )

    assert not is_known_code(
        TerminologySystem.ICD10_CM,
        "UNKNOWN",
    )


def test_list_concepts() -> None:
    concepts = list_concepts(
        TerminologySystem.RXNORM,
    )

    codes = {
        concept.code
        for concept in concepts
    }

    assert {
        "314076",
        "860975",
        "617310",
        "197361",
    }.issubset(codes)


def test_all_system_uris_exist() -> None:
    assert set(SYSTEM_URIS) == {
        TerminologySystem.ICD10_CM,
        TerminologySystem.SNOMED_CT,
        TerminologySystem.LOINC,
        TerminologySystem.RXNORM,
    }
