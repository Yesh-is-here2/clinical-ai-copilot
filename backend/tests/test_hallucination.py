"""Unit tests for app.clinical.hallucination (Milestone E, section 12)."""

from __future__ import annotations

import pytest

from app.clinical.citation_validation import (
    CitationValidationReport,
    CitationValidationStatus,
    ClaimCitationValidation,
)
from app.clinical.claims import Claim
from app.clinical.hallucination import (
    HallucinationClassification,
    evaluate_hallucinations,
)
from app.retrieval.fusion import RetrievedEvidence


def make_evidence(chunk_id: str, text: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title="Guideline",
        organization="Example Org",
        source_url="https://example.org/guideline",
        section_id="s1",
        section_heading="Overview",
        text=text,
        keyword_score=1.0,
        keyword_rank=1,
        vector_score=0.9,
        vector_rank=1,
        rrf_score=0.5,
        fused_rank=1,
    )


def make_validation(claim_id: str, status: CitationValidationStatus) -> ClaimCitationValidation:
    return ClaimCitationValidation(
        claim_id=claim_id,
        status=status,
        supporting_citation_ids=[],
        fabricated_citation_ids=[],
        reasoning="test",
        correctness_score=1.0 if status == CitationValidationStatus.SUPPORTED else 0.0,
    )


def test_supported_claim_is_classified_supported() -> None:
    claims = [Claim(claim_id="claim-1", text="A supported claim.")]
    report = CitationValidationReport(
        claim_validations=[make_validation("claim-1", CitationValidationStatus.SUPPORTED)]
    )

    evaluation = evaluate_hallucinations(claims, report, [])

    assert evaluation.claim_evaluations[0].classification == HallucinationClassification.SUPPORTED
    assert evaluation.supported == 1
    assert evaluation.hallucination_rate == 0.0


def test_partially_supported_claim_is_uncertain_not_hallucination() -> None:
    claims = [Claim(claim_id="claim-1", text="A partly supported claim.")]
    report = CitationValidationReport(
        claim_validations=[
            make_validation("claim-1", CitationValidationStatus.PARTIALLY_SUPPORTED)
        ]
    )

    evaluation = evaluate_hallucinations(claims, report, [])

    assert evaluation.claim_evaluations[0].classification == HallucinationClassification.UNCERTAIN
    assert evaluation.hallucination_rate == 0.0


def test_unsupported_claim_with_no_contradicting_evidence_is_unsupported() -> None:
    claims = [Claim(claim_id="claim-1", text="Metformin cures diabetes completely.")]
    report = CitationValidationReport(
        claim_validations=[make_validation("claim-1", CitationValidationStatus.UNSUPPORTED)]
    )
    evidence = [make_evidence("c1", "Unrelated evidence about blood pressure monitoring.")]

    evaluation = evaluate_hallucinations(claims, report, evidence)

    assert evaluation.claim_evaluations[0].classification == HallucinationClassification.UNSUPPORTED
    assert evaluation.hallucination_rate == 1.0


def test_unsupported_claim_contradicted_by_evidence_is_contradicted() -> None:
    claims = [
        Claim(
            claim_id="claim-1",
            text="Aspirin is recommended for all patients with hypertension routinely.",
        )
    ]
    report = CitationValidationReport(
        claim_validations=[make_validation("claim-1", CitationValidationStatus.UNSUPPORTED)]
    )
    evidence = [
        make_evidence(
            "c1",
            "Aspirin is not recommended for all patients with hypertension routinely.",
        )
    ]

    evaluation = evaluate_hallucinations(claims, report, evidence)

    assert evaluation.claim_evaluations[0].classification == HallucinationClassification.CONTRADICTED
    assert evaluation.contradicted == 1
    assert evaluation.hallucination_rate == 1.0


def test_hallucination_rate_defaults_to_zero_for_no_claims() -> None:
    report = CitationValidationReport()

    evaluation = evaluate_hallucinations([], report, [])

    assert evaluation.total_claims == 0
    assert evaluation.hallucination_rate == 0.0


def test_hallucination_rate_is_fraction_of_unsupported_and_contradicted() -> None:
    claims = [
        Claim(claim_id="claim-1", text="Supported claim."),
        Claim(claim_id="claim-2", text="Unrelated unsupported claim about something else."),
        Claim(claim_id="claim-3", text="Aspirin is recommended for all patients routinely."),
    ]
    report = CitationValidationReport(
        claim_validations=[
            make_validation("claim-1", CitationValidationStatus.SUPPORTED),
            make_validation("claim-2", CitationValidationStatus.UNSUPPORTED),
            make_validation("claim-3", CitationValidationStatus.UNSUPPORTED),
        ]
    )
    evidence = [
        make_evidence("c1", "Aspirin is not recommended for all patients routinely.")
    ]

    evaluation = evaluate_hallucinations(claims, report, evidence)

    assert evaluation.total_claims == 3
    assert evaluation.supported == 1
    assert evaluation.hallucination_rate == pytest.approx(2 / 3)


def test_missing_validation_entry_defaults_to_uncertain() -> None:
    claims = [Claim(claim_id="claim-1", text="A claim with no matching validation entry.")]
    report = CitationValidationReport(claim_validations=[])

    evaluation = evaluate_hallucinations(claims, report, [])

    assert evaluation.claim_evaluations[0].classification == HallucinationClassification.UNCERTAIN
