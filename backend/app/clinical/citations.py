"""Citation model and deterministic citation-ID assignment (Milestone E,
section 9).

Citation IDs (``C1``, ``C2``, ...) are assigned positionally from the
already-fused/reranked evidence order - never by the LLM, and never in a
way the LLM could influence. ``quoted_support`` is always a short,
verbatim excerpt of the actual retrieved chunk text (see ``_excerpt``);
nothing here ever asks an LLM to produce or paraphrase a quote.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import BaseModel, ConfigDict

from app.retrieval.fusion import RetrievedEvidence


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str
    chunk_id: str
    document_id: str
    document_title: str
    organization: str
    source_url: str
    section: str | None = None
    quoted_support: str
    retrieval_score: float | None = None
    reranker_score: float | None = None


_EXCERPT_MAX_CHARS = 240


def _excerpt(text: str, *, max_chars: int = _EXCERPT_MAX_CHARS) -> str:
    stripped = " ".join(text.split())

    if len(stripped) <= max_chars:
        return stripped

    truncated = stripped[:max_chars].rsplit(" ", 1)[0]

    return f"{truncated}..."


def build_citations(evidence: Sequence[RetrievedEvidence]) -> list[Citation]:
    """Assign deterministic ``C1..Cn`` citation IDs to fused/reranked
    evidence, in the order given (the caller is expected to have already
    put ``evidence`` in the order it should be cited)."""

    citations: list[Citation] = []

    for index, item in enumerate(evidence, start=1):
        # Prefer vector (cosine similarity, bounded ~0-1) as the
        # reported retrieval_score when available; otherwise fall back
        # to the raw keyword score so a BM25-only citation still carries
        # some retrieval-quality provenance.
        retrieval_score = (
            item.vector_score if item.vector_score is not None else item.keyword_score
        )

        citations.append(
            Citation(
                citation_id=f"C{index}",
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_title=item.title,
                organization=item.organization,
                source_url=item.source_url,
                section=item.section_heading or item.section_id,
                quoted_support=_excerpt(item.text),
                retrieval_score=retrieval_score,
                reranker_score=item.reranker_score,
            )
        )

    return citations


def citations_by_id(citations: Sequence[Citation]) -> dict[str, Citation]:
    return {citation.citation_id: citation for citation in citations}


def filter_valid_citation_ids(
    claimed_ids: Sequence[str], known_citations: dict[str, Citation]
) -> tuple[list[str], list[str]]:
    """Split citation IDs an LLM claims to have used into (valid,
    fabricated), preserving order and without duplicates.

    Never trust an LLM-reported citation_ids list at face value - it can
    only ever legitimately reference the fixed C1..Cn set this module
    itself assigned. Anything else is, by definition, fabricated and
    must never survive into the final response (section 9/11)."""

    valid: list[str] = []
    fabricated: list[str] = []
    seen: set[str] = set()

    for citation_id in claimed_ids:
        if citation_id in seen:
            continue
        seen.add(citation_id)

        if citation_id in known_citations:
            valid.append(citation_id)
        else:
            fabricated.append(citation_id)

    return valid, fabricated
