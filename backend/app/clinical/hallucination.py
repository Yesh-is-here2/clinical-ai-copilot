"""Hallucination detection (Milestone E, section 12).

Deterministic - built directly from the citation-validation report
(app.clinical.citation_validation) rather than re-deriving support from
scratch, plus one extra check: whether an UNSUPPORTED claim's terms
actually appear in some evidence chunk *alongside a negation* (e.g. the
evidence says "not recommended" where the claim asserts it's
recommended) - that distinguishes CONTRADICTED (evidence exists and
disagrees) from plain UNSUPPORTED (no evidence found either way).
PARTIALLY_SUPPORTED claims are UNCERTAIN, never automatically
"hallucinated" - see section 12's explicit "do not label uncertainty as
hallucination automatically."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from app.clinical.citation_validation import CitationValidationReport, CitationValidationStatus
from app.clinical.claims import Claim
from app.retrieval.fusion import RetrievedEvidence


class HallucinationClassification(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class ClaimHallucinationEvaluation:
    claim_id: str
    classification: HallucinationClassification


@dataclass(frozen=True)
class HallucinationEvaluation:
    total_claims: int = 0
    supported: int = 0
    unsupported: int = 0
    contradicted: int = 0
    uncertain: int = 0
    # (unsupported + contradicted) / total_claims. Defined as 0.0 when
    # total_claims == 0 (no factual claims to have hallucinated) - a
    # deliberate, documented convention, not an accidental division
    # avoided via a try/except.
    hallucination_rate: float = 0.0
    claim_evaluations: list[ClaimHallucinationEvaluation] = field(default_factory=list)


_CITATION_MARKER_PATTERN = re.compile(r"\[C\d+\]")

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to",
    "in", "on", "for", "and", "or", "with", "as", "that", "this", "it", "by",
    "at", "from", "may", "can", "should", "not", "no", "have", "has", "had",
}

_NEGATION_TERMS = {
    "not", "no", "without", "contraindicated", "avoid", "never", "isn't",
    "don't", "doesn't", "shouldn't", "cannot", "can't", "against",
}


def _significant_tokens(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}


def _contains_negation(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & _NEGATION_TERMS)


def _is_contradicted(
    claim_text: str, evidence: Sequence[RetrievedEvidence], *, min_overlap: float = 0.5
) -> bool:
    claim_terms = _significant_tokens(_CITATION_MARKER_PATTERN.sub("", claim_text))

    if not claim_terms:
        return False

    for item in evidence:
        evidence_terms = _significant_tokens(item.text)

        if not evidence_terms:
            continue

        overlap = len(claim_terms & evidence_terms) / len(claim_terms)

        if overlap >= min_overlap and _contains_negation(item.text):
            return True

    return False


def evaluate_hallucinations(
    claims: Sequence[Claim],
    citation_validation: CitationValidationReport,
    evidence: Sequence[RetrievedEvidence],
) -> HallucinationEvaluation:
    validation_by_claim_id = {
        v.claim_id: v for v in citation_validation.claim_validations
    }

    claim_evaluations: list[ClaimHallucinationEvaluation] = []
    counts: dict[HallucinationClassification, int] = {
        classification: 0 for classification in HallucinationClassification
    }

    for claim in claims:
        validation = validation_by_claim_id.get(claim.claim_id)

        if validation is None:
            # Defensive: every claim passed in should have a matching
            # validation entry. If one is somehow missing, treat it as
            # UNCERTAIN (not SUPPORTED, not silently dropped) rather
            # than guessing either way.
            classification = HallucinationClassification.UNCERTAIN
        elif validation.status is CitationValidationStatus.SUPPORTED:
            classification = HallucinationClassification.SUPPORTED
        elif validation.status is CitationValidationStatus.PARTIALLY_SUPPORTED:
            classification = HallucinationClassification.UNCERTAIN
        elif _is_contradicted(claim.text, evidence):
            classification = HallucinationClassification.CONTRADICTED
        else:
            classification = HallucinationClassification.UNSUPPORTED

        counts[classification] += 1
        claim_evaluations.append(
            ClaimHallucinationEvaluation(
                claim_id=claim.claim_id, classification=classification
            )
        )

    total_claims = len(claims)
    hallucinated = (
        counts[HallucinationClassification.UNSUPPORTED]
        + counts[HallucinationClassification.CONTRADICTED]
    )
    hallucination_rate = hallucinated / total_claims if total_claims else 0.0

    return HallucinationEvaluation(
        total_claims=total_claims,
        supported=counts[HallucinationClassification.SUPPORTED],
        unsupported=counts[HallucinationClassification.UNSUPPORTED],
        contradicted=counts[HallucinationClassification.CONTRADICTED],
        uncertain=counts[HallucinationClassification.UNCERTAIN],
        hallucination_rate=hallucination_rate,
        claim_evaluations=claim_evaluations,
    )
