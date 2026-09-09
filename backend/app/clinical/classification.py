"""Question classification (Milestone E, section 4).

A deterministic, keyword-based classifier is the default and the safe
fallback - it requires no LLM, no network, and always returns a
category. An optional ``LLMQuestionClassifier`` can refine this using
the structured LLM abstraction (app.clinical.llm), but it falls back to
the deterministic classifier on any ``LLMError`` rather than ever
failing the whole request over an unavailable/misbehaving LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from app.clinical.llm import LLMError, StructuredLLMProvider


class QuestionCategory(StrEnum):
    GUIDELINE = "guideline"
    MEDICATION = "medication"
    CONDITION = "condition"
    LAB_INTERPRETATION = "lab_interpretation"
    PREVENTION = "prevention"
    CONTRAINDICATION_SAFETY = "contraindication_safety"
    GENERAL_CLINICAL_INFORMATION = "general_clinical_information"
    UNSUPPORTED_OUT_OF_SCOPE = "unsupported_out_of_scope"


@dataclass(frozen=True)
class QuestionClassificationResult:
    category: QuestionCategory
    confidence: float
    method: str  # "deterministic" | "llm"
    matched_keywords: list[str] = field(default_factory=list)


@runtime_checkable
class QuestionClassifier(Protocol):
    async def classify(self, question: str) -> QuestionClassificationResult:
        ...


def _keyword_patterns(keywords: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE) for kw in keywords]


# Checked in this order - a question matching an earlier, more specific
# category (e.g. a drug-interaction/safety question that also mentions a
# medication) is classified there rather than the more generic category
# later in the list.
_CATEGORY_KEYWORDS: list[tuple[QuestionCategory, list[str]]] = [
    (
        QuestionCategory.CONTRAINDICATION_SAFETY,
        [
            "contraindicated",
            "contraindication",
            "safe to take",
            "drug interaction",
            "interact with",
            "allergic to",
            "allergy to",
            "should not take",
        ],
    ),
    (
        QuestionCategory.LAB_INTERPRETATION,
        [
            "lab result",
            "lab value",
            "a1c",
            "hba1c",
            "creatinine",
            "cholesterol level",
            "blood test",
            "ldl",
            "hdl",
            "elevated level",
        ],
    ),
    (
        QuestionCategory.MEDICATION,
        [
            "medication",
            "dosage",
            "prescription",
            "how much should i take",
            "starting dose",
        ],
    ),
    (
        QuestionCategory.PREVENTION,
        [
            "prevent",
            "prevention",
            "screening",
            "risk reduction",
            "vaccination",
            "vaccine",
        ],
    ),
    (
        QuestionCategory.CONDITION,
        [
            "what is",
            "what causes",
            "diagnosis of",
            "symptoms of",
            "condition",
        ],
    ),
    (
        QuestionCategory.GUIDELINE,
        [
            "guideline",
            "recommend",
            "recommendation",
            "standard of care",
            "according to",
        ],
    ),
]

_CATEGORY_PATTERNS: list[tuple[QuestionCategory, list[str], list[re.Pattern[str]]]] = [
    (category, keywords, _keyword_patterns(keywords))
    for category, keywords in _CATEGORY_KEYWORDS
]

# Broad clinical-domain signal used only to decide the fallback between
# GENERAL_CLINICAL_INFORMATION (some clinical content, no specific
# category matched) and UNSUPPORTED_OUT_OF_SCOPE (no clinical content at
# all - a question about the weather, code, sports, etc).
_CLINICAL_SIGNAL_TERMS = _keyword_patterns(
    [
        "patient",
        "symptom",
        "treatment",
        "diagnosis",
        "doctor",
        "physician",
        "health",
        "medical",
        "clinical",
        "disease",
        "therapy",
        "blood pressure",
        "hypertension",
        "diabetes",
    ]
)


class DeterministicQuestionClassifier:
    """Default, always-available classifier. No LLM, no network."""

    async def classify(self, question: str) -> QuestionClassificationResult:
        if not question or not question.strip():
            return QuestionClassificationResult(
                category=QuestionCategory.UNSUPPORTED_OUT_OF_SCOPE,
                confidence=1.0,
                method="deterministic",
            )

        for category, keywords, patterns in _CATEGORY_PATTERNS:
            matched = [
                keyword
                for keyword, pattern in zip(keywords, patterns)
                if pattern.search(question)
            ]

            if matched:
                return QuestionClassificationResult(
                    category=category,
                    confidence=0.7,
                    method="deterministic",
                    matched_keywords=matched,
                )

        if any(pattern.search(question) for pattern in _CLINICAL_SIGNAL_TERMS):
            return QuestionClassificationResult(
                category=QuestionCategory.GENERAL_CLINICAL_INFORMATION,
                confidence=0.4,
                method="deterministic",
            )

        return QuestionClassificationResult(
            category=QuestionCategory.UNSUPPORTED_OUT_OF_SCOPE,
            confidence=0.5,
            method="deterministic",
        )


class _LLMClassificationResponse(BaseModel):
    category: QuestionCategory
    confidence: float = Field(ge=0.0, le=1.0)


_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify a clinical question into exactly one category from a "
    "fixed set. Respond only with the requested JSON structure. Do not "
    "answer the question itself."
)


class LLMQuestionClassifier:
    """Optional refinement over the deterministic classifier.

    Falls back to the deterministic classifier - not an exception - on
    any ``LLMError`` (not configured, timeout, malformed response,
    provider failure), so an unavailable/misbehaving LLM never blocks
    classification.
    """

    def __init__(
        self,
        llm: StructuredLLMProvider,
        *,
        fallback: QuestionClassifier | None = None,
    ) -> None:
        self._llm = llm
        self._fallback = fallback or DeterministicQuestionClassifier()

    async def classify(self, question: str) -> QuestionClassificationResult:
        try:
            response = await self._llm.generate_structured(
                system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
                user_prompt=question,
                response_model=_LLMClassificationResponse,
            )
        except LLMError:
            return await self._fallback.classify(question)

        return QuestionClassificationResult(
            category=response.category,
            confidence=response.confidence,
            method="llm",
        )
