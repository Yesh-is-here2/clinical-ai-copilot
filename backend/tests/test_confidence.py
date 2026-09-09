"""Unit tests for app.clinical.confidence (Milestone E, section 14)."""

from __future__ import annotations

import pytest

from app.clinical.citation_validation import CitationValidationReport
from app.clinical.confidence import (
    ConfidenceConfig,
    ConfidenceLabel,
    calculate_confidence,
)
from app.clinical.evidence_sufficiency import (
    EvidenceSufficiencyResult,
    EvidenceSufficiencyStatus,
)
from app.clinical.hallucination import HallucinationEvaluation
from app.clinical.safety import SafetyCheckResult, SafetySeverity


def _sufficiency(status: EvidenceSufficiencyStatus, score: float = 1.0) -> EvidenceSufficiencyResult:
    return EvidenceSufficiencyResult(status=status, score=score)


def _safety(*, guarded: bool = False, passed: bool = True) -> SafetyCheckResult:
    return SafetyCheckResult(
        passed=passed,
        severity=SafetySeverity.NONE if passed else SafetySeverity.HIGH,
        guarded_response_required=guarded,
    )


def _hallucination(rate: float = 0.0, total: int = 1, supported: int = 1) -> HallucinationEvaluation:
    return HallucinationEvaluation(
        total_claims=total, supported=supported, hallucination_rate=rate
    )


def test_ideal_inputs_yield_high_confidence() -> None:
    result = calculate_confidence(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.SUFFICIENT, score=1.0),
        citation_validation=CitationValidationReport(overall_correctness_score=1.0),
        hallucination_evaluation=_hallucination(rate=0.0, total=2, supported=2),
        safety_result=_safety(),
        retrieval_degraded=False,
    )

    assert result.label == ConfidenceLabel.HIGH
    assert result.score >= ConfidenceConfig().high_threshold


def test_insufficient_evidence_caps_below_high() -> None:
    result = calculate_confidence(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.INSUFFICIENT, score=1.0),
        citation_validation=CitationValidationReport(overall_correctness_score=1.0),
        hallucination_evaluation=_hallucination(rate=0.0, total=2, supported=2),
        safety_result=_safety(),
        retrieval_degraded=False,
    )

    assert result.label != ConfidenceLabel.HIGH
    assert any("Insufficient evidence" in reason for reason in result.reasons)


def test_guarded_safety_state_caps_below_high() -> None:
    result = calculate_confidence(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.SUFFICIENT, score=1.0),
        citation_validation=CitationValidationReport(overall_correctness_score=1.0),
        hallucination_evaluation=_hallucination(rate=0.0, total=2, supported=2),
        safety_result=_safety(guarded=True, passed=False),
        retrieval_degraded=False,
    )

    assert result.label != ConfidenceLabel.HIGH
    assert any("safety state caps" in reason for reason in result.reasons)


def test_severe_hallucination_rate_forces_low() -> None:
    result = calculate_confidence(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.SUFFICIENT, score=1.0),
        citation_validation=CitationValidationReport(overall_correctness_score=1.0),
        hallucination_evaluation=_hallucination(rate=0.5, total=2, supported=1),
        safety_result=_safety(),
        retrieval_degraded=False,
    )

    assert result.label == ConfidenceLabel.LOW
    assert any("forces LOW" in reason for reason in result.reasons)


def test_moderate_inputs_yield_moderate_label() -> None:
    config = ConfidenceConfig()
    result = calculate_confidence(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.PARTIAL, score=0.5),
        citation_validation=CitationValidationReport(overall_correctness_score=0.6),
        hallucination_evaluation=_hallucination(rate=0.1, total=2, supported=1),
        safety_result=_safety(passed=True),
        retrieval_degraded=False,
    )

    assert config.moderate_threshold <= result.score < config.high_threshold or (
        result.label == ConfidenceLabel.MODERATE
    )


def test_retrieval_degradation_lowers_score_relative_to_non_degraded() -> None:
    kwargs = dict(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.SUFFICIENT, score=0.9),
        citation_validation=CitationValidationReport(overall_correctness_score=0.9),
        hallucination_evaluation=_hallucination(rate=0.0, total=2, supported=2),
        safety_result=_safety(),
    )

    normal = calculate_confidence(retrieval_degraded=False, **kwargs)
    degraded = calculate_confidence(retrieval_degraded=True, **kwargs)

    assert degraded.score < normal.score


def test_score_is_always_within_zero_and_one() -> None:
    result = calculate_confidence(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.INSUFFICIENT, score=0.0),
        citation_validation=CitationValidationReport(overall_correctness_score=0.0),
        hallucination_evaluation=_hallucination(rate=1.0, total=2, supported=0),
        safety_result=_safety(guarded=True, passed=False),
        retrieval_degraded=True,
    )

    assert 0.0 <= result.score <= 1.0
    assert result.label == ConfidenceLabel.LOW


def test_components_dict_reports_all_six_signals() -> None:
    result = calculate_confidence(
        evidence_sufficiency=_sufficiency(EvidenceSufficiencyStatus.SUFFICIENT, score=1.0),
        citation_validation=CitationValidationReport(overall_correctness_score=1.0),
        hallucination_evaluation=_hallucination(rate=0.0, total=0, supported=0),
        safety_result=_safety(),
        retrieval_degraded=False,
    )

    assert set(result.components.keys()) == {
        "evidence_sufficiency",
        "citation_correctness",
        "grounded_claim_ratio",
        "hallucination_penalty",
        "safety",
        "retrieval_quality",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"moderate_threshold": 0.9, "high_threshold": 0.5},
        {"severe_hallucination_rate_threshold": 1.5},
        {
            "evidence_weight": 0.5,
            "citation_weight": 0.5,
            "grounded_ratio_weight": 0.5,
            "hallucination_weight": 0.0,
            "safety_weight": 0.0,
            "retrieval_weight": 0.0,
        },
    ],
)
def test_invalid_config_raises(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ConfidenceConfig(**kwargs)
