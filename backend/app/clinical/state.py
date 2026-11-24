"""Typed LangGraph state for the clinical query graph (Milestone E,
section 1).

A ``TypedDict`` (LangGraph's native state shape) whose values are the
strongly-typed dataclasses/Pydantic models from the rest of
``app.clinical`` and ``app.retrieval`` - never raw dicts standing in for
domain objects. ``total=False`` because the graph fills this in
progressively, node by node; every node reads fields with ``.get(...,
default)`` rather than assuming a key is present.
"""

from __future__ import annotations

import uuid
from typing import TypedDict

from app.clinical.citation_validation import CitationValidationReport
from app.clinical.citations import Citation
from app.clinical.classification import QuestionCategory
from app.clinical.claims import Claim
from app.clinical.confidence import ConfidenceResult
from app.clinical.evidence_sufficiency import EvidenceSufficiencyResult
from app.clinical.final_response import FinalClinicalResponse
from app.clinical.generation import GroundedAnswer
from app.clinical.hallucination import HallucinationEvaluation
from app.clinical.injection import InjectionDetectionResult
from app.clinical.safety import SafetyCheckResult
from app.retrieval.fusion import RetrievedEvidence
from app.retrieval.hybrid import HybridRetrievalResult
from app.schemas.patient import PatientContext


class ClinicalState(TypedDict, total=False):
    # Input
    request_id: str
    question: str
    patient_context: PatientContext | dict | None
    # Request actor (Phase 7): set by the route via run_query's actor
    # arguments; None for anonymous/legacy callers. Carried through
    # so write_audit_log_node can attribute the audit row.
    actor_user_id: uuid.UUID | None
    actor_role: str | None

    # validate_input / parse_patient_context / classify_question /
    # detect_prompt_injection
    question_category: QuestionCategory | None
    injection_result: InjectionDetectionResult | None

    # build_search_query / retrieve_evidence / rerank_evidence
    search_query: str | None
    retrieval_result: HybridRetrievalResult | None
    retrieved_evidence: list[RetrievedEvidence]
    citations: list[Citation]

    # evaluate_evidence_sufficiency
    evidence_sufficiency: EvidenceSufficiencyResult | None

    # generate_grounded_answer / extract_claims
    generated_response: GroundedAnswer | None
    extracted_claims: list[Claim]

    # validate_citations / detect_hallucinations / run_safety_checks /
    # calculate_confidence
    citation_validation: CitationValidationReport | None
    hallucination_evaluation: HallucinationEvaluation | None
    safety_evaluation: SafetyCheckResult | None
    confidence: ConfidenceResult | None

    # construct_final_response
    final_response: FinalClinicalResponse | None

    # Cross-cutting: warnings/errors/degradation, regeneration bookkeeping,
    # observability (section 19). Never contains chain-of-thought - only
    # short, structured strings and numbers.
    warnings: list[str]
    component_errors: list[str]
    regeneration_count: int
    terminal_reason: str | None
    stage_timings: dict[str, float]
    total_elapsed_seconds: float | None
