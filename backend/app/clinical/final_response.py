"""Final structured response + disclaimer (Milestone E, sections 15/16).

Deliberately minimal: only structured decisions, scores, flags, and
citations - never the system prompt, never hidden reasoning, never
chain-of-thought. Richer (but still concise, audit-safe) reasoning - the
kind app.clinical.confidence/evidence_sufficiency produce as short
`reasons` lists - belongs on the audit log entry (app.clinical.audit),
not on this externally-facing schema.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.clinical.citations import Citation
from app.clinical.confidence import ConfidenceLabel
from app.clinical.evidence_sufficiency import EvidenceSufficiencyStatus

DISCLAIMER_TEXT = (
    "This prototype provides evidence-grounded clinical information for "
    "educational and research purposes using synthetic patient data. It "
    "is not a diagnostic tool and does not replace professional clinical "
    "judgment."
)


class FinalClinicalResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
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
    disclaimer: str = DISCLAIMER_TEXT
