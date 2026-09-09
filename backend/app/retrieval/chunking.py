"""Deterministic, token-aware chunking of ingested clinical guidelines.

Chunks are the retrieval unit: BM25 and pgvector search will operate over
``DocumentChunk`` rows, so this module is the boundary between "one big
ingested document" and "many independently retrievable, citable pieces of
evidence".

Everything here is deterministic on purpose (see section 9 of the project
spec): the same document + the same ``ChunkingConfig`` always produces the
same chunks with the same IDs, so re-running ingestion is idempotent and
evaluation results are reproducible. Chunk IDs are derived from stable
provenance (document id, section id, chunk number, content fingerprint)
rather than random UUIDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.retrieval.tokenizer import Tokenizer, get_default_tokenizer
from app.schemas.document import ClinicalDocument, ClinicalSection, DocumentChunk


class ChunkingConfigError(ValueError):
    """Raised when a chunking configuration is invalid."""


@dataclass(frozen=True)
class ChunkingConfig:
    """A named token-window configuration.

    ``chunk_overlap_tokens`` must be strictly less than
    ``chunk_size_tokens`` - equal or larger overlap would either stall
    the sliding window (no forward progress) or produce chunks that are
    entirely redundant with their predecessor.
    """

    name: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ChunkingConfigError(
                "chunk_size_tokens must be greater than zero."
            )

        if self.chunk_overlap_tokens < 0:
            raise ChunkingConfigError(
                "chunk_overlap_tokens must not be negative."
            )

        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ChunkingConfigError(
                "chunk_overlap_tokens must be smaller than "
                "chunk_size_tokens, or the sliding window never "
                "advances."
            )

    @property
    def stride_tokens(self) -> int:
        """How many tokens the window advances by on each step."""

        return self.chunk_size_tokens - self.chunk_overlap_tokens


# The three experiment configurations required by the project spec.
# Evaluation runs (MLflow) are expected to compare retrieval quality
# across these three.
CHUNKING_CONFIG_SMALL = ChunkingConfig("small_500_75", 500, 75)
CHUNKING_CONFIG_MEDIUM = ChunkingConfig("medium_750_100", 750, 100)
CHUNKING_CONFIG_LARGE = ChunkingConfig("large_1000_150", 1000, 150)

CHUNKING_CONFIGS: tuple[ChunkingConfig, ...] = (
    CHUNKING_CONFIG_SMALL,
    CHUNKING_CONFIG_MEDIUM,
    CHUNKING_CONFIG_LARGE,
)

DEFAULT_CHUNKING_CONFIG = CHUNKING_CONFIG_MEDIUM


def _fingerprint(text: str) -> str:
    """Short, stable content fingerprint used inside chunk IDs.

    Not a security control - just enough entropy that two chunks at the
    same (document, section, chunk_number) position with different text
    (e.g. after a re-ingestion picks up a corrected source) get visibly
    different IDs instead of silently colliding.
    """

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def build_chunk_id(
    *,
    document_id: str,
    section_id: str | None,
    chunk_number: int,
    text: str,
) -> str:
    section_component = section_id or "doc"

    return (
        f"{document_id}:{section_component}:"
        f"{chunk_number}:{_fingerprint(text)}"
    )


def _windows_for_tokens(
    token_count: int,
    config: ChunkingConfig,
) -> list[tuple[int, int]]:
    """Compute [start, end) token index windows covering ``token_count``
    tokens under ``config``. Pure integer math - no text involved - so
    it is trivial to unit test boundary behavior in isolation."""

    if token_count <= 0:
        return []

    if token_count <= config.chunk_size_tokens:
        return [(0, token_count)]

    windows: list[tuple[int, int]] = []
    start = 0

    while start < token_count:
        end = min(start + config.chunk_size_tokens, token_count)
        windows.append((start, end))

        if end >= token_count:
            break

        start += config.stride_tokens

    return windows


def chunk_text(
    text: str,
    config: ChunkingConfig,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[tuple[str, int]]:
    """Split ``text`` into ``(chunk_text, token_count)`` pairs.

    Returns an empty list for empty/whitespace-only input rather than
    producing an empty chunk.
    """

    if not text or not text.strip():
        return []

    active_tokenizer = tokenizer or get_default_tokenizer()
    tokens = active_tokenizer.encode(text)

    windows = _windows_for_tokens(len(tokens), config)

    pieces: list[tuple[str, int]] = []

    for start, end in windows:
        window_tokens = tokens[start:end]
        piece_text = active_tokenizer.decode(window_tokens).strip()

        if not piece_text:
            continue

        pieces.append((piece_text, len(window_tokens)))

    return pieces


def chunk_section(
    section: ClinicalSection,
    config: ChunkingConfig,
    *,
    starting_chunk_number: int = 0,
    tokenizer: Tokenizer | None = None,
) -> list[tuple[str, int]]:
    """Convenience wrapper kept separate from ``chunk_document`` so a
    single section can be re-chunked in isolation (e.g. for tests, or a
    future "re-chunk just this section" admin action)."""

    return chunk_text(section.text, config, tokenizer=tokenizer)


def chunk_document(
    document: ClinicalDocument,
    config: ChunkingConfig,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[DocumentChunk]:
    """Chunk every section of ``document`` into ``DocumentChunk`` rows.

    ``chunk_number`` is a single sequence across the whole document (not
    reset per section) so chunk IDs are trivially unique within a
    document while ``section_id``/``section_heading`` still preserve
    exactly which section each chunk came from.
    """

    active_tokenizer = tokenizer or get_default_tokenizer()
    metadata = document.metadata

    chunks: list[DocumentChunk] = []
    chunk_number = 0

    for section in document.sections:
        pieces = chunk_text(
            section.text,
            config,
            tokenizer=active_tokenizer,
        )

        for piece_text, token_count in pieces:
            chunk_id = build_chunk_id(
                document_id=metadata.document_id,
                section_id=section.section_id,
                chunk_number=chunk_number,
                text=piece_text,
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=metadata.document_id,
                    document_title=metadata.title,
                    organization=metadata.organization,
                    source_url=metadata.source_url,
                    category=metadata.category,
                    version=metadata.version,
                    publication_date=metadata.publication_date,
                    access_date=metadata.access_date,
                    section_id=section.section_id,
                    section_heading=section.heading,
                    chunk_number=chunk_number,
                    text=piece_text,
                    token_count=token_count,
                )
            )

            chunk_number += 1

    return chunks
