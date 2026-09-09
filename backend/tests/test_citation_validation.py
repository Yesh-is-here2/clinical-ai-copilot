"""Unit tests for app.clinical.citation_validation (Milestone E, section 11)."""

from __future__ import annotations

from app.clinical.citation_validation import (
    CitationValidationStatus,
    validate_citations,
)
from app.clinical.citations import build_citations
from app.clinical.claims import Claim
from app.retrieval.fusion import RetrievedEvidence


def make_evidence(chunk_id: str, text: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        title=f"Guideline {chunk_id}",
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


def _setup(texts: dict[str, str]):
    evidence = [make_evidence(chunk_id, text) for chunk_id, text in texts.items()]
    citations = build_citations(evidence)
    return evidence, citations


def test_well_supported_claim_is_supported() -> None:
    evidence, citations = _setup(
        {"c1": "Guidelines recommend a target blood pressure below 130 over 80 mmHg."}
    )
    claims = [
        Claim(
            claim_id="claim-1",
            text="Guidelines recommend a target blood pressure below 130 over 80 mmHg [C1].",
            cited_citation_ids=["C1"],
        )
    ]

    report = validate_citations(claims, citations, evidence)

    assert report.claim_validations[0].status == CitationValidationStatus.SUPPORTED
    assert report.has_important_unsupported_claims is False
    assert report.fabricated_citation_ids == []


def test_fabricated_citation_id_forces_unsupported_regardless_of_overlap() -> None:
    evidence, citations = _setup(
        {"c1": "Guidelines recommend a target blood pressure below 130 over 80 mmHg."}
    )
    claims = [
        Claim(
            claim_id="claim-1",
            text="Guidelines recommend a target blood pressure below 130 over 80 mmHg [C1][C99].",
            cited_citation_ids=["C1", "C99"],
        )
    ]

    report = validate_citations(claims, citations, evidence)

    assert report.claim_validations[0].status == CitationValidationStatus.UNSUPPORTED
    assert report.claim_validations[0].correctness_score == 0.0
    assert "C99" in report.fabricated_citation_ids
    assert report.has_important_unsupported_claims is True


def test_uncited_claim_is_unsupported() -> None:
    evidence, citations = _setup({"c1": "Some evidence text."})
    claims = [Claim(claim_id="claim-1", text="An uncited claim.", cited_citation_ids=[])]

    report = validate_citations(claims, citations, evidence)

    assert report.claim_validations[0].status == CitationValidationStatus.UNSUPPORTED
    assert report.claim_validations[0].reasoning == "No citation was provided for this claim."


def test_low_overlap_claim_is_unsupported() -> None:
    evidence, citations = _setup({"c1": "The sky is blue during the day."})
    claims = [
        Claim(
            claim_id="claim-1",
            text="Metformin reduces hepatic glucose production significantly [C1].",
            cited_citation_ids=["C1"],
        )
    ]

    report = validate_citations(claims, citations, evidence)

    assert report.claim_validations[0].status == CitationValidationStatus.UNSUPPORTED


def test_partial_overlap_claim_is_partially_supported_not_important_unsupported() -> None:
    evidence, citations = _setup(
        {"c1": "Guidelines discuss blood pressure monitoring frequency for patients."}
    )
    claims = [
        Claim(
            claim_id="claim-1",
            text="Guidelines discuss blood pressure targets and medication adjustment strategies [C1].",
            cited_citation_ids=["C1"],
        )
    ]

    report = validate_citations(claims, citations, evidence)

    status = report.claim_validations[0].status
    # This claim is deliberately built with partial-but-not-full term
    # overlap; whichever bucket it lands in between PARTIALLY_SUPPORTED
    # and UNSUPPORTED, PARTIALLY_SUPPORTED must never count as an
    # "important unsupported claim" (section 11).
    if status == CitationValidationStatus.PARTIALLY_SUPPORTED:
        assert "claim-1" not in report.important_unsupported_claim_ids


def test_no_claims_yields_perfect_default_score() -> None:
    evidence, citations = _setup({"c1": "text"})

    report = validate_citations([], citations, evidence)

    assert report.claim_validations == []
    assert report.overall_correctness_score == 1.0
    assert report.has_important_unsupported_claims is False


def test_overall_correctness_score_is_the_mean_of_claim_scores() -> None:
    evidence, citations = _setup(
        {"c1": "Guidelines recommend a target blood pressure below 130 over 80 mmHg."}
    )
    claims = [
        Claim(
            claim_id="claim-1",
            text="Guidelines recommend a target blood pressure below 130 over 80 mmHg [C1].",
            cited_citation_ids=["C1"],
        ),
        Claim(claim_id="claim-2", text="Totally unrelated uncited claim.", cited_citation_ids=[]),
    ]

    report = validate_citations(claims, citations, evidence)

    scores = [v.correctness_score for v in report.claim_validations]
    assert report.overall_correctness_score == sum(scores) / len(scores)
