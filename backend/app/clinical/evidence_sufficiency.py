"""Deterministic evidence sufficiency scoring (Milestone E, section 6).

Decides whether the retrieved/fused/reranked evidence pool is enough to
answer a clinical question at all, before generation is even attempted.
Every signal used is either a plain count (how many chunks, how many
distinct source documents) or the top result's vector similarity
(bounded ~0-1, semantically meaningful) when it's available - this
module deliberately does NOT weight raw BM25/RRF scores into the
composite, for the same reason app.retrieval.fusion keeps RRF
rank-based: those scores are not on a comparable, threshold-able scale.

Thresholds live in ``EvidenceSufficiencyConfig`` (explicit, constructor-
injected) rather than being read from global Settings here, so this
stays independently unit-testable with whatever thresholds a test wants
- see app.clinical.graph for where production Settings values are wired
into a config instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from app.retrieval.fusion import RetrievedEvidence


class EvidenceSufficiencyStatus(StrEnum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


INSUFFICIENT_EVIDENCE_MESSAGE = (
    "Insufficient supporting evidence was found in the indexed clinical "
    "guidelines to answer this question reliably."
)


@dataclass(frozen=True)
class EvidenceSufficiencyConfig:
    sufficient_threshold: float = 0.65
    partial_threshold: float = 0.35
    # "Target" counts used to normalize the count-based signals to
    # 0-1 - reaching the target maxes out that component, exceeding it
    # doesn't add further credit.
    target_evidence_count: int = 3
    target_independent_source_count: int = 2
    # Composite weights - must sum to 1.0 (checked in __post_init__).
    count_weight: float = 0.4
    source_weight: float = 0.3
    quality_weight: float = 0.3

    def __post_init__(self) -> None:
        if not 0.0 <= self.partial_threshold <= self.sufficient_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= partial_threshold <= sufficient_threshold <= 1."
            )
        if self.target_evidence_count <= 0 or self.target_independent_source_count <= 0:
            raise ValueError("target counts must be greater than zero.")
        total_weight = self.count_weight + self.source_weight + self.quality_weight
        if abs(total_weight - 1.0) > 1e-9:
            raise ValueError(f"Composite weights must sum to 1.0, got {total_weight}.")


@dataclass(frozen=True)
class EvidenceSufficiencyResult:
    status: EvidenceSufficiencyStatus
    score: float
    reasons: list[str] = field(default_factory=list)
    evidence_count: int = 0
    independent_source_count: int = 0
    top_score: float | None = None


_DEFAULT_CONFIG = EvidenceSufficiencyConfig()


def evaluate_evidence_sufficiency(
    evidence: Sequence[RetrievedEvidence],
    *,
    config: EvidenceSufficiencyConfig = _DEFAULT_CONFIG,
) -> EvidenceSufficiencyResult:
    evidence_count = len(evidence)

    if evidence_count == 0:
        return EvidenceSufficiencyResult(
            status=EvidenceSufficiencyStatus.INSUFFICIENT,
            score=0.0,
            reasons=["No evidence was retrieved for this question."],
            evidence_count=0,
            independent_source_count=0,
            top_score=None,
        )

    independent_source_count = len({item.document_id for item in evidence})

    # Prefer the top result's vector (cosine similarity) score as the
    # quality signal - it's bounded and semantically meaningful. When
    # vector search wasn't available/used (BM25-only), fall back to a
    # neutral 0.5 rather than penalizing or crediting quality we can't
    # actually measure on a comparable scale.
    top_vector_score = next(
        (item.vector_score for item in evidence if item.vector_score is not None),
        None,
    )
    quality_component = top_vector_score if top_vector_score is not None else 0.5

    count_component = min(evidence_count / config.target_evidence_count, 1.0)
    source_component = min(
        independent_source_count / config.target_independent_source_count, 1.0
    )

    score = (
        config.count_weight * count_component
        + config.source_weight * source_component
        + config.quality_weight * quality_component
    )

    if score >= config.sufficient_threshold:
        status = EvidenceSufficiencyStatus.SUFFICIENT
    elif score >= config.partial_threshold:
        status = EvidenceSufficiencyStatus.PARTIAL
    else:
        status = EvidenceSufficiencyStatus.INSUFFICIENT

    reasons = [
        f"{evidence_count} evidence chunk(s) retrieved "
        f"(target {config.target_evidence_count}).",
        f"{independent_source_count} independent source document(s) "
        f"(target {config.target_independent_source_count}).",
    ]
    if top_vector_score is not None:
        reasons.append(f"top vector similarity {top_vector_score:.2f}.")
    else:
        reasons.append(
            "no vector similarity available (lexical-only retrieval) - "
            "quality component defaulted to neutral."
        )

    if status is EvidenceSufficiencyStatus.INSUFFICIENT:
        reasons.append(INSUFFICIENT_EVIDENCE_MESSAGE)

    return EvidenceSufficiencyResult(
        status=status,
        score=score,
        reasons=reasons,
        evidence_count=evidence_count,
        independent_source_count=independent_source_count,
        top_score=top_vector_score,
    )
