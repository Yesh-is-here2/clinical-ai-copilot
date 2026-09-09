"""Claim extraction (Milestone E, section 10).

The default extractor is deterministic (sentence-splitting + citation-
marker regex) - per the Phase 5 constraint "do not use an LLM for
deterministic tasks merely because an LLM abstraction exists," this is
not LLM-based, which also means the default pipeline never needs an
OpenAI key just to extract claims. A sentence with no [C#] markers still
becomes a claim (with an empty ``cited_citation_ids``) - that's
deliberate: an uncited factual sentence is exactly what citation
validation (app.clinical.citation_validation) and hallucination
detection (app.clinical.hallucination) need to be able to flag as
unsupported, not something claim extraction should quietly drop.
"""

from __future__ import annotations

import re
from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    text: str
    cited_citation_ids: list[str] = Field(default_factory=list)
    importance: float | None = None


@runtime_checkable
class ClaimExtractor(Protocol):
    async def extract(self, answer_text: str) -> list[Claim]:
        ...


_CITATION_MARKER_PATTERN = re.compile(r"\[C(\d+)\]")

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

# Disclaimer/meta/opinion sentences that should never be treated as
# factual clinical claims, even though they may appear right alongside
# real claims in the generated answer.
_NON_CLAIM_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"this (system|prototype|assistant|tool) (is|does not|cannot)",
        r"educational (and|or) research purposes",
        r"not (a )?diagnostic tool",
        r"does not replace",
        r"consult (a|your) (doctor|physician|clinician|healthcare provider)",
        r"for informational purposes",
        r"speak (with|to) (a|your) (doctor|physician|clinician)",
    ]
]


def _split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []

    return [s.strip() for s in _SENTENCE_SPLIT_PATTERN.split(text.strip()) if s.strip()]


def _is_non_claim_sentence(sentence: str) -> bool:
    return any(pattern.search(sentence) for pattern in _NON_CLAIM_PATTERNS)


def _extract_citation_ids(sentence: str) -> list[str]:
    # dict.fromkeys: dedupe while preserving first-seen order.
    return list(dict.fromkeys(f"C{number}" for number in _CITATION_MARKER_PATTERN.findall(sentence)))


class DeterministicClaimExtractor:
    """Default, always-available extractor. No LLM, no network."""

    async def extract(self, answer_text: str) -> list[Claim]:
        claims: list[Claim] = []

        for index, sentence in enumerate(_split_sentences(answer_text), start=1):
            if _is_non_claim_sentence(sentence):
                continue

            claims.append(
                Claim(
                    claim_id=f"claim-{index}",
                    text=sentence,
                    cited_citation_ids=_extract_citation_ids(sentence),
                )
            )

        return claims


class FakeClaimExtractor:
    """Test-only extractor returning a fixed, pre-built list of claims
    regardless of the answer text - lets tests exercise downstream
    citation-validation/hallucination logic without depending on the
    deterministic sentence-splitting heuristics."""

    def __init__(self, claims: Sequence[Claim]) -> None:
        self._claims = list(claims)

    async def extract(self, answer_text: str) -> list[Claim]:
        return list(self._claims)
