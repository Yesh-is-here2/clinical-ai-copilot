"""Citation validation (Milestone E, section 11).

Deterministic, term-overlap-based validation of each extracted claim
against the evidence its citation IDs actually resolve to - no LLM
involved, so this runs in every default test without a provider. A
fabricated citation ID (one that doesn't match anything
``app.clinical.citations.build_citations`` actually assigned) can never
result in a SUPPORTED claim, however much overlap the claim's other,
valid citations have - see ``filter_valid_citation_ids``.

Term overlap alone cannot tell "the evidence says this" apart from "the
evidence says the opposite of this": negation words are deliberately
excluded from the overlap vocabulary (see ``_STOPWORDS``), so a claim and
its exact negation can otherwise score as fully supported. This module
also checks negation *polarity* between a claim and its cited evidence
(``_contains_negation``) and never reports SUPPORTED/PARTIALLY_SUPPORTED
when polarity disagrees - it reports UNSUPPORTED instead, which is what
lets app.clinical.hallucination's own, separate negation-overlap check
correctly classify that same claim as CONTRADICTED rather than plain
UNSUPPORTED. The two modules are related but distinct: this one asks
"does the claim correctly cite supporting evidence?"; hallucination.py
asks "is the claim itself supported, unsupported, contradicted, or
uncertain relative to the evidence?".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Sequence

from app.clinical.citations import Citation, citations_by_id, filter_valid_citation_ids
from app.clinical.claims import Claim
from app.retrieval.fusion import RetrievedEvidence


class CitationValidationStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ClaimCitationValidation:
    claim_id: str
    status: CitationValidationStatus
    supporting_citation_ids: list[str]
    fabricated_citation_ids: list[str]
    reasoning: str
    correctness_score: float


@dataclass(frozen=True)
class CitationValidationReport:
    claim_validations: list[ClaimCitationValidation] = field(default_factory=list)
    overall_correctness_score: float = 1.0
    # Claims whose status is UNSUPPORTED - the trigger for one
    # regeneration attempt (section 11). Deliberately excludes
    # PARTIALLY_SUPPORTED, which is a real but non-fatal signal.
    important_unsupported_claim_ids: list[str] = field(default_factory=list)
    fabricated_citation_ids: list[str] = field(default_factory=list)

    @property
    def has_important_unsupported_claims(self) -> bool:
        return bool(self.important_unsupported_claim_ids)


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
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token not in _STOPWORDS and len(token) > 2}


def _contains_negation(text: str) -> bool:
    tokens = set(re.findall(r"[a-z']+", text.lower()))
    return bool(tokens & _NEGATION_TERMS)


def _support_score(claim_text: str, evidence_text: str) -> float:
    """Fraction of the claim's significant terms that also appear in the
    cited evidence text. 0.0 when the claim has no significant terms at
    all (nothing to check support for)."""

    claim_terms = _significant_tokens(claim_text)

    if not claim_terms:
        return 0.0

    evidence_terms = _significant_tokens(evidence_text)

    return len(claim_terms & evidence_terms) / len(claim_terms)


def validate_citations(
    claims: Sequence[Claim],
    citations: Sequence[Citation],
    evidence: Sequence[RetrievedEvidence],
    *,
    supported_threshold: float = 0.6,
    partial_threshold: float = 0.3,
) -> CitationValidationReport:
    known_citations = citations_by_id(citations)
    evidence_text_by_chunk_id = {item.chunk_id: item.text for item in evidence}

    claim_validations: list[ClaimCitationValidation] = []
    important_unsupported: list[str] = []
    all_fabricated: list[str] = []

    for claim in claims:
        valid_ids, fabricated_ids = filter_valid_citation_ids(
            claim.cited_citation_ids, known_citations
        )
        all_fabricated.extend(fabricated_ids)

        if fabricated_ids:
            status = CitationValidationStatus.UNSUPPORTED
            score = 0.0
            reasoning = (
                f"Claim cites fabricated citation ID(s) {fabricated_ids} "
                "that do not correspond to any retrieved evidence - rejected."
            )
        elif not valid_ids:
            status = CitationValidationStatus.UNSUPPORTED
            score = 0.0
            reasoning = "No citation was provided for this claim."
        else:
            claim_text = _CITATION_MARKER_PATTERN.sub("", claim.text)
            cited_evidence_text = " ".join(
                evidence_text_by_chunk_id[known_citations[cid].chunk_id]
                for cid in valid_ids
                if known_citations[cid].chunk_id in evidence_text_by_chunk_id
            )
            score = _support_score(claim_text, cited_evidence_text)
            polarity_mismatch = (
                _contains_negation(claim_text) != _contains_negation(cited_evidence_text)
            )

            if polarity_mismatch and score >= partial_threshold:
                # Meaningful term overlap, but the claim and its cited
                # evidence disagree on negation - e.g. the claim asserts
                # something the evidence explicitly says is not the case.
                # Never let that read as SUPPORTED or PARTIALLY_SUPPORTED;
                # app.clinical.hallucination re-examines this exact
                # claim/evidence pair to decide CONTRADICTED vs plain
                # UNSUPPORTED - this module's job stops at "not supported."
                status = CitationValidationStatus.UNSUPPORTED
                reasoning = (
                    f"Cited evidence shares {score:.0%} of the claim's key "
                    "terms but appears to state the opposite (negation "
                    "polarity mismatch) - not counted as support."
                )
            elif score >= supported_threshold:
                status = CitationValidationStatus.SUPPORTED
                reasoning = (
                    f"Cited evidence covers {score:.0%} of the claim's key terms."
                )
            elif score >= partial_threshold:
                status = CitationValidationStatus.PARTIALLY_SUPPORTED
                reasoning = (
                    f"Cited evidence covers only {score:.0%} of the claim's "
                    "key terms - partial support."
                )
            else:
                status = CitationValidationStatus.UNSUPPORTED
                reasoning = (
                    f"Cited evidence covers only {score:.0%} of the claim's "
                    "key terms - not enough overlap to consider it supported."
                )

        if status is CitationValidationStatus.UNSUPPORTED:
            important_unsupported.append(claim.claim_id)

        claim_validations.append(
            ClaimCitationValidation(
                claim_id=claim.claim_id,
                status=status,
                supporting_citation_ids=valid_ids,
                fabricated_citation_ids=fabricated_ids,
                reasoning=reasoning,
                correctness_score=score,
            )
        )

    # No claims at all -> nothing to be wrong about. Documented, not an
    # implicit/accidental default: see app.clinical.hallucination for the
    # matching zero-claims convention.
    overall_score = (
        sum(v.correctness_score for v in claim_validations) / len(claim_validations)
        if claim_validations
        else 1.0
    )

    return CitationValidationReport(
        claim_validations=claim_validations,
        overall_correctness_score=overall_score,
        important_unsupported_claim_ids=important_unsupported,
        fabricated_citation_ids=list(dict.fromkeys(all_fabricated)),
    )
