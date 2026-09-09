"""Unit tests for app.clinical.evidence_sufficiency (Milestone E, section 6)."""

from __future__ import annotations

from datetime import date

import pytest

from app.clinical.evidence_sufficiency import (
    EvidenceSufficiencyConfig,
    EvidenceSufficiencyStatus,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    evaluate_evidence_sufficiency,
)
from app.retrieval.fusion import RetrievedEvidence


def make_evidence(
    chunk_id: str,
    *,
    document_id: str = "doc-1",
    vector_score: float | None = 0.9,
    rrf_score: float = 0.5,
    rank: int = 1,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk_id=chunk_id,
        document_id=document_id,
        title="Guideline",
        organization="Example Org",
        source_url="https://example.org/guideline",
        section_id="s1",
        section_heading="Overview",
        text=f"evidence text for {chunk_id}",
        keyword_score=1.0,
        keyword_rank=1,
        vector_score=vector_score,
        vector_rank=1,
        rrf_score=rrf_score,
        fused_rank=rank,
    )


def test_no_evidence_is_always_insufficient() -> None:
    result = evaluate_evidence_sufficiency([])

    assert result.status is EvidenceSufficiencyStatus.INSUFFICIENT
    assert result.score == 0.0
    assert result.evidence_count == 0
    assert result.independent_source_count == 0
    assert INSUFFICIENT_EVIDENCE_MESSAGE in result.reasons[0] or any(
        "No evidence" in r for r in result.reasons
    )


def test_strong_evidence_pool_is_sufficient() -> None:
    evidence = [
        make_evidence("c1", document_id="doc-1", vector_score=0.95),
        make_evidence("c2", document_id="doc-2", vector_score=0.9),
        make_evidence("c3", document_id="doc-3", vector_score=0.88),
    ]

    result = evaluate_evidence_sufficiency(evidence)

    assert result.status is EvidenceSufficiencyStatus.SUFFICIENT
    assert result.evidence_count == 3
    assert result.independent_source_count == 3
    assert result.top_score == 0.95


def test_weak_but_nonzero_evidence_pool_is_partial() -> None:
    # One chunk, one source, mediocre similarity - some signal, not
    # enough to be confidently sufficient.
    evidence = [make_evidence("c1", document_id="doc-1", vector_score=0.5)]

    result = evaluate_evidence_sufficiency(evidence)

    assert result.status is EvidenceSufficiencyStatus.PARTIAL


def test_very_weak_evidence_pool_is_insufficient() -> None:
    evidence = [make_evidence("c1", document_id="doc-1", vector_score=0.05)]

    result = evaluate_evidence_sufficiency(evidence)

    assert result.status is EvidenceSufficiencyStatus.INSUFFICIENT
    assert any(INSUFFICIENT_EVIDENCE_MESSAGE in reason for reason in result.reasons)


def test_missing_vector_score_falls_back_to_neutral_quality() -> None:
    evidence = [make_evidence("c1", vector_score=None)]

    result = evaluate_evidence_sufficiency(evidence)

    assert result.top_score is None
    assert any("no vector similarity" in reason for reason in result.reasons)


def test_independent_source_count_deduplicates_by_document_id() -> None:
    evidence = [
        make_evidence("c1", document_id="doc-1"),
        make_evidence("c2", document_id="doc-1"),
        make_evidence("c3", document_id="doc-2"),
    ]

    result = evaluate_evidence_sufficiency(evidence)

    assert result.evidence_count == 3
    assert result.independent_source_count == 2


def test_custom_config_thresholds_are_respected() -> None:
    config = EvidenceSufficiencyConfig(
        sufficient_threshold=0.9, partial_threshold=0.8
    )
    evidence = [make_evidence("c1", document_id="doc-1", vector_score=0.85)]

    result = evaluate_evidence_sufficiency(evidence, config=config)

    # Would be SUFFICIENT under default thresholds but not under this
    # stricter config.
    assert result.status is not EvidenceSufficiencyStatus.SUFFICIENT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"partial_threshold": 0.9, "sufficient_threshold": 0.5},
        {"target_evidence_count": 0},
        {"target_independent_source_count": -1},
        {"count_weight": 0.5, "source_weight": 0.5, "quality_weight": 0.5},
    ],
)
def test_invalid_config_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EvidenceSufficiencyConfig(**kwargs)
