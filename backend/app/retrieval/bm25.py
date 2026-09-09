"""In-memory BM25 (Okapi) lexical retrieval over indexed document chunks.

Implemented directly rather than pulling in a third-party BM25 package:
the algorithm is small, well-specified, and this way there is no extra
runtime dependency for something this self-contained (the project spec
allows either - "a practical Python implementation is acceptable").

Each ``BM25Index`` instance owns its own corpus and term statistics.
There is no module-level/global mutable state, so multiple indices (for
example, one per chunking-config experiment during an MLflow comparison
run - see section 32) can be built and queried independently and safely,
including concurrently.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from app.schemas.document import DocumentChunk

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Deterministic, dependency-free tokenizer: lowercased alphanumeric
    words. Good enough for lexical matching on clinical guideline
    prose - it does not need to match the chunking tokenizer, which
    counts tokens for a completely different purpose (context-window
    sizing, not term matching)."""

    return _TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class BM25Config:
    """Standard Okapi BM25 smoothing parameters."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError("k1 must not be negative.")

        if not 0.0 <= self.b <= 1.0:
            raise ValueError("b must be between 0 and 1.")


@dataclass(frozen=True)
class BM25Result:
    chunk_id: str
    document_id: str
    text: str
    score: float
    rank: int


class BM25Index:
    """A BM25 index built once over a fixed set of chunks.

    Rebuilding (e.g. after re-ingestion) means constructing a new
    ``BM25Index`` - it is intentionally immutable-once-built rather than
    supporting incremental updates, which keeps the term-statistics math
    simple and correct.
    """

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        config: BM25Config | None = None,
    ) -> None:
        self._config = config or BM25Config()
        self._chunks_by_id: dict[str, DocumentChunk] = {}
        self._term_frequencies: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._document_frequency: dict[str, int] = {}

        self._build(chunks)

    def _build(self, chunks: Sequence[DocumentChunk]) -> None:
        total_length = 0

        for chunk in chunks:
            if chunk.chunk_id in self._chunks_by_id:
                # Defensive: chunk IDs are supposed to be unique (see
                # app.retrieval.chunking.build_chunk_id). Silently
                # overwriting would corrupt term-frequency accounting.
                raise ValueError(
                    f"Duplicate chunk_id '{chunk.chunk_id}' passed to BM25Index."
                )

            tokens = tokenize(chunk.text)
            term_counts: dict[str, int] = {}

            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            self._chunks_by_id[chunk.chunk_id] = chunk
            self._term_frequencies[chunk.chunk_id] = term_counts
            self._doc_lengths[chunk.chunk_id] = len(tokens)
            total_length += len(tokens)

            for term in term_counts:
                self._document_frequency[term] = (
                    self._document_frequency.get(term, 0) + 1
                )

        self._corpus_size = len(self._chunks_by_id)
        self._average_doc_length = (
            total_length / self._corpus_size if self._corpus_size else 0.0
        )

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    def _idf(self, term: str) -> float:
        document_frequency = self._document_frequency.get(term, 0)

        if document_frequency == 0:
            return 0.0

        # Standard Okapi BM25 IDF with +1 smoothing so common terms never
        # go negative for a moderately sized corpus.
        return math.log(
            1.0
            + (self._corpus_size - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )

    def _score(self, chunk_id: str, query_terms: Sequence[str]) -> float:
        term_counts = self._term_frequencies[chunk_id]
        doc_length = self._doc_lengths[chunk_id]
        k1 = self._config.k1
        b = self._config.b

        score = 0.0

        for term in query_terms:
            frequency = term_counts.get(term)

            if not frequency:
                continue

            idf = self._idf(term)

            if idf <= 0.0:
                continue

            numerator = frequency * (k1 + 1)
            denominator = frequency + k1 * (
                1 - b + b * (doc_length / self._average_doc_length)
                if self._average_doc_length
                else 1.0
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, *, top_k: int = 15) -> list[BM25Result]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if not query or not query.strip():
            return []

        if self._corpus_size == 0:
            return []

        query_terms = tokenize(query)

        if not query_terms:
            return []

        scored: list[tuple[float, str]] = []

        for chunk_id in self._chunks_by_id:
            score = self._score(chunk_id, query_terms)

            if score > 0.0:
                scored.append((score, chunk_id))

        # Stable, deterministic ordering: score descending, then
        # chunk_id ascending as a tie-break so repeated searches (and
        # searches across process restarts) always return chunks in the
        # same order.
        scored.sort(key=lambda item: (-item[0], item[1]))

        results: list[BM25Result] = []

        for rank, (score, chunk_id) in enumerate(scored[:top_k], start=1):
            chunk = self._chunks_by_id[chunk_id]
            results.append(
                BM25Result(
                    chunk_id=chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    rank=rank,
                )
            )

        return results
