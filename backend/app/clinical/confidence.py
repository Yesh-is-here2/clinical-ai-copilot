"""Deterministic composite confidence scoring (Milestone E, section 14).

Never an LLM self-confidence score - a weighted composite of signals
already computed by earlier stages (evidence sufficiency, citation
correctness, grounded-claim ratio, hallucination rate, safety status,
retrieval degradation), with three hard overrides applied afterward:
insufficient evidence, a blocked/guarded safety state, and a severe
hallucination rate each cap the score below HIGH (the hallucination
override caps below MODERATE, i.e. forces LOW) regardless of how the
weighted components alone would have scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.clinical.citation_validation import CitationValidationReport
from app.clinical.evidence_sufficiency import (
    EvidenceSufficiencyResult,
    EvidenceSufficiencyStatus,
)
from app.clinical.hallucination import HallucinationEvaluation
from app.clinical.safety import SafetyCheckResult


class ConfidenceLabel(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


_EPSILON = 1e-6


@dataclass(frozen=True)
class ConfidenceConfig:
    high_threshold: float = 0.80
    moderate_threshold: float = 0.60
    # hallucination_rate at/above this forces the LOW override.
    severe_hallucination_rate_threshold: float = 0.30

    evidence_weight: float = 0.25
    citation_weight: float = 0.20
    grounded_ratio_weight: float = 0.20
    hallucination_weight: float = 0.15
    safety_weight: float = 0.15
    retrieval_weight: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 <= self.moderate_threshold <= self.high_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= moderate_threshold <= high_threshold <= 1."
            )
        if not 0.0 <= self.severe_hallucination_rate_threshold <= 1.0:
            raise ValueError(
                "severe_hallucination_rate_threshold must be between 0 and 1."
            )
        total_weight = (
            self.evidence_weight
            + self.citation_weight
            + self.grounded_ratio_weight
            + self.hallucination_weight
            + self.safety_weight
            + self.retrieval_weight
        )
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(f"Composite weights must sum to 1.0, got {total_weight}.")


@dataclass(frozen=True)
class ConfidenceResult:
    score: float
    label: ConfidenceLabel
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


_DEFAULT_CONFIG = ConfidenceConfig()


def _label_for_score(score: float, config: ConfidenceConfig) -> ConfidenceLabel:
    if score >= config.high_threshold:
        return ConfidenceLabel.HIGH
    if score >= config.moderate_threshold:
        return ConfidenceLabel.MODERATE
    return ConfidenceLabel.LOW


def calculate_confidence(
    *,
    evidence_sufficiency: EvidenceSufficiencyResult,
    citation_validation: CitationValidationReport,
    hallucination_evaluation: HallucinationEvaluation,
    safety_result: SafetyCheckResult,
    retrieval_degraded: bool,
    config: ConfidenceConfig = _DEFAULT_CONFIG,
) -> ConfidenceResult:
    grounded_ratio = (
        hallucination_evaluation.supported / hallucination_evaluation.total_claims
        if hallucination_evaluation.total_claims
        else 1.0
    )
    hallucination_penalty_component = 1.0 - hallucination_evaluation.hallucination_rate

    if safety_result.guarded_response_required:
        safety_component = 0.0
    elif not safety_result.passed:
        safety_component = 0.7  # flagged (e.g. missing disclaimer) but not guarded
    else:
        safety_component = 1.0

    retrieval_component = 0.7 if retrieval_degraded else 1.0

    components = {
        "evidence_sufficiency": evidence_sufficiency.score,
        "citation_correctness": citation_validation.overall_correctness_score,
        "grounded_claim_ratio": grounded_ratio,
        "hallucination_penalty": hallucination_penalty_component,
        "safety": safety_component,
        "retrieval_quality": retrieval_component,
    }

    score = (
        config.evidence_weight * components["evidence_sufficiency"]
        + config.citation_weight * components["citation_correctness"]
        + config.grounded_ratio_weight * components["grounded_claim_ratio"]
        + config.hallucination_weight * components["hallucination_penalty"]
        + config.safety_weight * components["safety"]
        + config.retrieval_weight * components["retrieval_quality"]
    )

    reasons = [
        f"Weighted composite score before overrides: {score:.2f}.",
    ]

    # --- Hard overrides (section 14) -----------------------------------
    if evidence_sufficiency.status is EvidenceSufficiencyStatus.INSUFFICIENT:
        capped = config.high_threshold - _EPSILON
        if score > capped:
            score = capped
        reasons.append("Insufficient evidence caps confidence below HIGH.")

    if safety_result.guarded_response_required:
        capped = config.high_threshold - _EPSILON
        if score > capped:
            score = capped
        reasons.append("A blocked/guarded safety state caps confidence below HIGH.")

    if hallucination_evaluation.hallucination_rate >= config.severe_hallucination_rate_threshold:
        capped = config.moderate_threshold - _EPSILON
        if score > capped:
            score = capped
        reasons.append(
            f"Hallucination rate {hallucination_evaluation.hallucination_rate:.0%} "
            f"is at/above the severe threshold "
            f"({config.severe_hallucination_rate_threshold:.0%}) and forces LOW."
        )

    score = max(0.0, min(1.0, score))
    label = _label_for_score(score, config)

    reasons.append(f"Final score {score:.2f} -> {label.value.upper()}.")

    return ConfidenceResult(score=score, label=label, components=components, reasons=reasons)
