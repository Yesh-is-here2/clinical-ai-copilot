"""Unit tests for app.clinical.final_response (Milestone E, sections 15/16)."""

from __future__ import annotations

import pydantic
import pytest

from app.clinical.confidence import ConfidenceLabel
from app.clinical.evidence_sufficiency import EvidenceSufficiencyStatus
from app.clinical.final_response import DISCLAIMER_TEXT, FinalClinicalResponse


def test_disclaimer_never_claims_fda_hipaa_or_clinical_validation() -> None:
    lowered = DISCLAIMER_TEXT.lower()

    assert "fda" not in lowered
    assert "hipaa" not in lowered
    assert "not a diagnostic tool" in lowered
    assert "does not replace professional clinical judgment" in lowered


def test_minimal_valid_response_uses_the_default_disclaimer() -> None:
    response = FinalClinicalResponse(
        request_id="req-1",
        patient_summary="No patient context was provided.",
        clinical_question="What is hypertension?",
        answer="Guidelines describe hypertension as elevated blood pressure [C1].",
        confidence=0.9,
        confidence_label=ConfidenceLabel.HIGH,
        evidence_status=EvidenceSufficiencyStatus.SUFFICIENT,
    )

    assert response.disclaimer == DISCLAIMER_TEXT
    assert response.key_points == []
    assert response.citations == []
    assert response.warnings == []
    assert response.safety_flags == []


def test_forbids_extra_fields_so_chain_of_thought_cannot_leak_through() -> None:
    with pytest.raises(pydantic.ValidationError):
        FinalClinicalResponse(
            request_id="req-1",
            patient_summary="x",
            clinical_question="x",
            answer="x",
            confidence=0.5,
            confidence_label=ConfidenceLabel.LOW,
            evidence_status=EvidenceSufficiencyStatus.INSUFFICIENT,
            reasoning_trace="hidden chain of thought",
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_must_be_within_zero_and_one(confidence: float) -> None:
    with pytest.raises(pydantic.ValidationError):
        FinalClinicalResponse(
            request_id="req-1",
            patient_summary="x",
            clinical_question="x",
            answer="x",
            confidence=confidence,
            confidence_label=ConfidenceLabel.LOW,
            evidence_status=EvidenceSufficiencyStatus.INSUFFICIENT,
        )
