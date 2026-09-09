from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class TerminologySystem(StrEnum):
    ICD10_CM = "ICD-10-CM"
    SNOMED_CT = "SNOMED CT"
    LOINC = "LOINC"
    RXNORM = "RxNorm"


class TerminologyConcept(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: TerminologySystem
    system_uri: str
    code: str
    display: str
    description: str | None = None


SYSTEM_URIS = {
    TerminologySystem.ICD10_CM: "http://hl7.org/fhir/sid/icd-10-cm",
    TerminologySystem.SNOMED_CT: "http://snomed.info/sct",
    TerminologySystem.LOINC: "http://loinc.org",
    TerminologySystem.RXNORM: "http://www.nlm.nih.gov/research/umls/rxnorm",
}


ICD10_CM_CONCEPTS = {
    "I10": TerminologyConcept(
        system=TerminologySystem.ICD10_CM,
        system_uri=SYSTEM_URIS[TerminologySystem.ICD10_CM],
        code="I10",
        display="Essential (primary) hypertension",
        description="Representative ICD-10-CM condition used in synthetic fixtures.",
    ),
    "E11.9": TerminologyConcept(
        system=TerminologySystem.ICD10_CM,
        system_uri=SYSTEM_URIS[TerminologySystem.ICD10_CM],
        code="E11.9",
        display="Type 2 diabetes mellitus without complications",
        description="Representative ICD-10-CM diabetes concept used in synthetic fixtures.",
    ),
    "E78.5": TerminologyConcept(
        system=TerminologySystem.ICD10_CM,
        system_uri=SYSTEM_URIS[TerminologySystem.ICD10_CM],
        code="E78.5",
        display="Hyperlipidemia, unspecified",
        description="Representative ICD-10-CM lipid disorder concept.",
    ),
    "R73.03": TerminologyConcept(
        system=TerminologySystem.ICD10_CM,
        system_uri=SYSTEM_URIS[TerminologySystem.ICD10_CM],
        code="R73.03",
        display="Prediabetes",
        description="Representative ICD-10-CM metabolic finding used in synthetic fixtures.",
    ),
}


SNOMED_CT_CONCEPTS = {
    "91936005": TerminologyConcept(
        system=TerminologySystem.SNOMED_CT,
        system_uri=SYSTEM_URIS[TerminologySystem.SNOMED_CT],
        code="91936005",
        display="Allergy to penicillin",
        description="Representative SNOMED CT allergy concept used in synthetic fixtures.",
    ),
}


LOINC_CONCEPTS = {
    "85354-9": TerminologyConcept(
        system=TerminologySystem.LOINC,
        system_uri=SYSTEM_URIS[TerminologySystem.LOINC],
        code="85354-9",
        display="Blood pressure panel with all children optional",
        description="Blood pressure panel represented by systolic and diastolic components.",
    ),
    "8480-6": TerminologyConcept(
        system=TerminologySystem.LOINC,
        system_uri=SYSTEM_URIS[TerminologySystem.LOINC],
        code="8480-6",
        display="Systolic blood pressure",
    ),
    "8462-4": TerminologyConcept(
        system=TerminologySystem.LOINC,
        system_uri=SYSTEM_URIS[TerminologySystem.LOINC],
        code="8462-4",
        display="Diastolic blood pressure",
    ),
    "4548-4": TerminologyConcept(
        system=TerminologySystem.LOINC,
        system_uri=SYSTEM_URIS[TerminologySystem.LOINC],
        code="4548-4",
        display="Hemoglobin A1c/Hemoglobin.total in Blood",
    ),
    "2160-0": TerminologyConcept(
        system=TerminologySystem.LOINC,
        system_uri=SYSTEM_URIS[TerminologySystem.LOINC],
        code="2160-0",
        display="Creatinine [Mass/volume] in Serum or Plasma",
    ),
    "13457-7": TerminologyConcept(
        system=TerminologySystem.LOINC,
        system_uri=SYSTEM_URIS[TerminologySystem.LOINC],
        code="13457-7",
        display="Cholesterol in LDL [Mass/volume] in Serum or Plasma",
    ),
}


RXNORM_CONCEPTS = {
    "314076": TerminologyConcept(
        system=TerminologySystem.RXNORM,
        system_uri=SYSTEM_URIS[TerminologySystem.RXNORM],
        code="314076",
        display="Lisinopril 10 MG Oral Tablet",
    ),
    "860975": TerminologyConcept(
        system=TerminologySystem.RXNORM,
        system_uri=SYSTEM_URIS[TerminologySystem.RXNORM],
        code="860975",
        display="Metformin hydrochloride 500 MG 24 HR Extended Release Oral Tablet",
    ),
    "617310": TerminologyConcept(
        system=TerminologySystem.RXNORM,
        system_uri=SYSTEM_URIS[TerminologySystem.RXNORM],
        code="617310",
        display="Atorvastatin 20 MG Oral Tablet",
    ),
    "197361": TerminologyConcept(
        system=TerminologySystem.RXNORM,
        system_uri=SYSTEM_URIS[TerminologySystem.RXNORM],
        code="197361",
        display="Amlodipine 5 MG Oral Tablet",
    ),
}


CONCEPTS_BY_SYSTEM = {
    TerminologySystem.ICD10_CM: ICD10_CM_CONCEPTS,
    TerminologySystem.SNOMED_CT: SNOMED_CT_CONCEPTS,
    TerminologySystem.LOINC: LOINC_CONCEPTS,
    TerminologySystem.RXNORM: RXNORM_CONCEPTS,
}


def lookup_concept(
    system: TerminologySystem,
    code: str,
) -> TerminologyConcept | None:
    normalized_code = code.strip()

    if not normalized_code:
        return None

    concepts = CONCEPTS_BY_SYSTEM.get(system, {})

    return concepts.get(normalized_code)


def is_known_code(
    system: TerminologySystem,
    code: str,
) -> bool:
    return lookup_concept(system, code) is not None


def list_concepts(
    system: TerminologySystem,
) -> list[TerminologyConcept]:
    concepts = CONCEPTS_BY_SYSTEM.get(system, {})

    return list(concepts.values())
