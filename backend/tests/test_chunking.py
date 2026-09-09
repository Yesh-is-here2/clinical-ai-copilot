from datetime import date

import pytest

from app.retrieval.chunking import (
    CHUNKING_CONFIGS,
    ChunkingConfig,
    ChunkingConfigError,
    build_chunk_id,
    chunk_document,
    chunk_text,
)
from app.retrieval.tokenizer import WhitespaceTokenizer
from app.schemas.document import (
    ClinicalDocument,
    ClinicalDocumentCategory,
    ClinicalDocumentMetadata,
    ClinicalSection,
)


TOKENIZER = WhitespaceTokenizer()


def make_document(sections: list[ClinicalSection]) -> ClinicalDocument:
    return ClinicalDocument(
        metadata=ClinicalDocumentMetadata(
            document_id="doc-1",
            title="Test Guideline",
            organization="Example Org",
            source_url="https://example.org/guideline",
            publication_date=date(2021, 8, 24),
            access_date=date(2026, 9, 9),
            category=ClinicalDocumentCategory.HYPERTENSION,
            version="1.0",
            license_or_usage_note="Public test fixture.",
        ),
        sections=sections,
        raw_text=" ".join(section.text for section in sections),
    )


def words(count: int, prefix: str = "word") -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


# ---------------------------------------------------------------------
# ChunkingConfig validation
# ---------------------------------------------------------------------


def test_config_rejects_zero_chunk_size() -> None:
    with pytest.raises(ChunkingConfigError):
        ChunkingConfig("bad", 0, 0)


def test_config_rejects_negative_overlap() -> None:
    with pytest.raises(ChunkingConfigError):
        ChunkingConfig("bad", 100, -1)


def test_config_rejects_overlap_equal_to_chunk_size() -> None:
    with pytest.raises(ChunkingConfigError):
        ChunkingConfig("bad", 100, 100)


def test_config_rejects_overlap_larger_than_chunk_size() -> None:
    with pytest.raises(ChunkingConfigError):
        ChunkingConfig("bad", 100, 150)


def test_required_experiment_configs_exist() -> None:
    sizes = {
        (config.chunk_size_tokens, config.chunk_overlap_tokens)
        for config in CHUNKING_CONFIGS
    }

    assert (500, 75) in sizes
    assert (750, 100) in sizes
    assert (1000, 150) in sizes


# ---------------------------------------------------------------------
# chunk_text: boundaries, overlap, empty/small/large input
# ---------------------------------------------------------------------


def test_chunk_text_empty_input_produces_no_chunks() -> None:
    config = ChunkingConfig("t", 10, 2)

    assert chunk_text("", config, tokenizer=TOKENIZER) == []
    assert chunk_text("   \n\t  ", config, tokenizer=TOKENIZER) == []


def test_chunk_text_small_input_produces_single_chunk() -> None:
    config = ChunkingConfig("t", 500, 75)
    text = words(10)

    pieces = chunk_text(text, config, tokenizer=TOKENIZER)

    assert len(pieces) == 1
    piece_text, token_count = pieces[0]
    assert token_count == 10
    assert piece_text == text


def test_chunk_text_exact_boundary_produces_single_chunk() -> None:
    config = ChunkingConfig("t", 10, 2)
    text = words(10)

    pieces = chunk_text(text, config, tokenizer=TOKENIZER)

    assert len(pieces) == 1
    assert pieces[0][1] == 10


def test_chunk_text_large_input_produces_overlapping_windows() -> None:
    config = ChunkingConfig("t", 10, 3)
    text = words(25)

    pieces = chunk_text(text, config, tokenizer=TOKENIZER)

    # stride = 10 - 3 = 7; windows: [0,10) [7,17) [14,24) [21,25)
    assert [count for _, count in pieces] == [10, 10, 10, 4]

    all_tokens = text.split()
    assert pieces[0][0] == " ".join(all_tokens[0:10])
    assert pieces[1][0] == " ".join(all_tokens[7:17])
    assert pieces[2][0] == " ".join(all_tokens[14:24])
    assert pieces[3][0] == " ".join(all_tokens[21:25])


def test_chunk_text_consecutive_windows_actually_overlap() -> None:
    config = ChunkingConfig("t", 10, 3)
    text = words(25)

    pieces = chunk_text(text, config, tokenizer=TOKENIZER)

    first_words = set(pieces[0][0].split())
    second_words = set(pieces[1][0].split())

    overlap = first_words & second_words
    assert len(overlap) == 3


def test_chunk_text_last_window_never_exceeds_input() -> None:
    config = ChunkingConfig("t", 9, 2)
    text = words(20)

    pieces = chunk_text(text, config, tokenizer=TOKENIZER)

    reconstructed_last = pieces[-1][0].split()
    assert reconstructed_last == text.split()[-len(reconstructed_last):]


def test_chunk_text_never_produces_empty_chunks() -> None:
    config = ChunkingConfig("t", 5, 1)
    text = words(37)

    pieces = chunk_text(text, config, tokenizer=TOKENIZER)

    assert all(piece_text.strip() for piece_text, _ in pieces)
    assert all(count > 0 for _, count in pieces)


def test_chunk_text_is_deterministic() -> None:
    config = ChunkingConfig("t", 8, 2)
    text = words(50)

    first_run = chunk_text(text, config, tokenizer=TOKENIZER)
    second_run = chunk_text(text, config, tokenizer=TOKENIZER)

    assert first_run == second_run


# ---------------------------------------------------------------------
# chunk_document: metadata propagation, section attribution, stable IDs
# ---------------------------------------------------------------------


def test_chunk_document_propagates_metadata_to_every_chunk() -> None:
    config = ChunkingConfig("t", 500, 75)
    section = ClinicalSection(
        section_id="s1",
        heading="Overview",
        text=words(20),
        order=0,
    )
    document = make_document([section])

    chunks = chunk_document(document, config, tokenizer=TOKENIZER)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.document_id == "doc-1"
    assert chunk.document_title == "Test Guideline"
    assert chunk.organization == "Example Org"
    assert str(chunk.source_url) == "https://example.org/guideline"
    assert chunk.category == ClinicalDocumentCategory.HYPERTENSION
    assert chunk.version == "1.0"
    assert chunk.publication_date == date(2021, 8, 24)
    assert chunk.access_date == date(2026, 9, 9)
    assert chunk.section_id == "s1"
    assert chunk.section_heading == "Overview"


def test_chunk_document_does_not_lose_section_attribution() -> None:
    config = ChunkingConfig("t", 10, 2)
    sections = [
        ClinicalSection(
            section_id="s1",
            heading="Overview",
            text=words(5),
            order=0,
        ),
        ClinicalSection(
            section_id="s2",
            heading="Monitoring",
            text=words(25),
            order=1,
        ),
    ]
    document = make_document(sections)

    chunks = chunk_document(document, config, tokenizer=TOKENIZER)

    s1_chunks = [c for c in chunks if c.section_id == "s1"]
    s2_chunks = [c for c in chunks if c.section_id == "s2"]

    assert len(s1_chunks) == 1
    assert all(c.section_heading == "Overview" for c in s1_chunks)
    assert len(s2_chunks) >= 2
    assert all(c.section_heading == "Monitoring" for c in s2_chunks)


def test_chunk_document_chunk_numbers_are_sequential_across_sections() -> None:
    config = ChunkingConfig("t", 10, 2)
    sections = [
        ClinicalSection(
            section_id="s1",
            heading="Overview",
            text=words(15),
            order=0,
        ),
        ClinicalSection(
            section_id="s2",
            heading="Monitoring",
            text=words(15),
            order=1,
        ),
    ]
    document = make_document(sections)

    chunks = chunk_document(document, config, tokenizer=TOKENIZER)

    assert [c.chunk_number for c in chunks] == list(range(len(chunks)))


def test_chunk_document_produces_stable_deterministic_ids() -> None:
    config = ChunkingConfig("t", 500, 75)
    section = ClinicalSection(
        section_id="s1",
        heading="Overview",
        text=words(20),
        order=0,
    )
    document = make_document([section])

    first_run = chunk_document(document, config, tokenizer=TOKENIZER)
    second_run = chunk_document(document, config, tokenizer=TOKENIZER)

    assert [c.chunk_id for c in first_run] == [
        c.chunk_id for c in second_run
    ]


def test_chunk_ids_are_not_random_uuids() -> None:
    chunk_id = build_chunk_id(
        document_id="doc-1",
        section_id="s1",
        chunk_number=0,
        text="hello world",
    )

    # Deterministic IDs are reproducible from the same inputs.
    again = build_chunk_id(
        document_id="doc-1",
        section_id="s1",
        chunk_number=0,
        text="hello world",
    )

    assert chunk_id == again
    assert chunk_id.startswith("doc-1:s1:0:")


def test_chunk_ids_differ_when_content_differs_at_same_position() -> None:
    first = build_chunk_id(
        document_id="doc-1",
        section_id="s1",
        chunk_number=0,
        text="original text",
    )
    second = build_chunk_id(
        document_id="doc-1",
        section_id="s1",
        chunk_number=0,
        text="corrected text",
    )

    assert first != second


def test_chunk_document_with_no_sections_produces_no_chunks() -> None:
    config = ChunkingConfig("t", 500, 75)
    document = make_document([])

    assert chunk_document(document, config, tokenizer=TOKENIZER) == []
