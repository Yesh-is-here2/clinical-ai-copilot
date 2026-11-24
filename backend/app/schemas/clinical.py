"""API request/response schemas for the clinical query and query-history
endpoints (Milestone F, sections 3/8/9).

Deliberately separate from ``app.clinical.final_response.FinalClinicalResponse``
(Milestone E's internal graph output) even though ``ClinicalQueryResponse``
below is shaped very close to it: this module is the *externally-facing*
HTTP contract, so it is where an explicit allowlist of fields lives, and
it is safe to evolve independently of the graph's internal state shape.
Nothing in this module ever exposes: the system prompt, chain-of-thought,
raw LLM responses, stack traces, secrets, or any ``ClinicalState``
internals - only plain identifiers, scores, status strings, and the
already-safe fields ``FinalClinicalResponse`` itself produces.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.clinical.citations import Citation
from app.clinical.confidence import ConfidenceLabel
from app.clinical.evidence_sufficiency import EvidenceSufficiencyStatus
from app.clinical.final_response import DISCLAIMER_TEXT


class ClinicalQueryRequest(BaseModel):
    """``POST /clinical/query`` request.

    ``patient_id`` is optional: a question may be asked without
    patient context (the graph then answers from guideline evidence
    alone, with a "no patient context" summary). When given it must
    name a known synthetic patient, otherwise the request is a 404.
    """

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    patient_id: str | None = Field(default=None, min_length=1, max_length=64)

    @field_validator("question")
    @classmethod
    def _reject_blank_question(cls, value: str) -> str:
        # min_length=1 alone still accepts a whitespace-only string;
        # the graph would then treat it as a blank question. Reject it
        # here so it is a clean 422 instead of a wasted graph run.
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be blank")
        return stripped

    @field_validator("patient_id")
    @classmethod
    def _normalise_patient_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class ClinicalQueryResponse(BaseModel):
    """The full, safe response for ``POST /clinical/query``.

    ``status`` mirrors the graph's ``terminal_reason`` (e.g.
    "normal", "blocked_injection", "abstained_insufficient_evidence",
    "guarded_citation_failure", "guarded_hallucination", "guarded_safety",
    "error_retrieval_failed", "error_generation_failed") - a successful,
    safe, structured response in every one of these cases, never an HTTP
    error (see section 14's error-mapping table / app.api.routes.clinical).
    """

    model_config = ConfigDict(extra="forbid")

    query_id: uuid.UUID
    request_id: str
    status: str
    patient_summary: str
    clinical_question: str
    answer: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_label: ConfidenceLabel
    evidence_status: EvidenceSufficiencyStatus
    warnings: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    disclaimer: str
    # Structured degradation/fallback metadata (section 3): True when
    # any configured retrieval component failed and the graph fell back
    # to the remaining components (see HybridRetrievalResult.degraded).
    # The human-readable detail stays in `warnings`.
    degraded: bool = False


class CitationRecordOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_label: str
    chunk_id: str
    quoted_support: str
    support_classification: str


class QueryDetailResponse(BaseModel):
    """``GET /queries/{query_id}`` - a safe, typed view of what was
    actually persisted for one query. Never fabricates a field that
    wasn't recorded; a query with no generated response (should not
    normally happen - every graph path reaches ``construct_final_
    response`` - but is defensively handled) reports ``answer: None``
    rather than inventing placeholder text.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: uuid.UUID
    request_id: str
    patient_id: str | None
    question: str
    question_category: str | None
    status: str
    evidence_status: str | None
    confidence: float | None
    latency_ms: float | None
    created_at: datetime
    answer: str | None
    key_points: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    citations: list[CitationRecordOut] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER_TEXT


class EvaluationMetricOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_name: str
    metric_value: float
    run_id: str
    chunking_config: str | None = None


class QueryEvaluationResponse(BaseModel):
    """``GET /queries/{query_id}/evaluation`` (section 9). ``status`` is
    explicitly "not_run" (not an empty-but-ambiguous list) when no
    ``EvaluationResultRecord`` rows exist for this query - never
    fabricates a metric value that was never actually computed.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: uuid.UUID
    status: Literal["available", "not_run"]
    metrics: list[EvaluationMetricOut] = Field(default_factory=list)


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    organization: str
    category: str
    version: str | None = None
    publication_date: date | None = None
    access_date: date
    chunk_count: int


class DocumentIngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        min_length=1,
        description=(
            "A registered guideline document ID from "
            "app.retrieval.sources.GUIDELINE_SOURCES - never an arbitrary "
            "URL. This is itself the primary SSRF defense for this "
            "endpoint: only pre-registered, allowlisted sources can ever "
            "be fetched."
        ),
    )


class DocumentIngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    status: str
    chunk_count: int
    section_count: int
    warnings: list[str] = Field(default_factory=list)
