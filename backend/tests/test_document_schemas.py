from datetime import date

import pytest
from pydantic import ValidationError

from app.schemas.document import (
    ClinicalDocument,
    ClinicalDocumentCategory,
    ClinicalDocumentMetadata,
    ClinicalSection,
    DocumentChunk,
)


def test_clinical_document_metadata() -> None:
    metadata = ClinicalDocumentMetadata(
        document_id="test-guideline",
        title="Test Clinical Guideline",
        organization="Example Organization",
        source_url="https://example.org/guideline",
        publication_date=date(2026, 1, 1),
        access_date=date(2026, 9, 9),
        category=ClinicalDocumentCategory.HYPERTENSION,
        version="1.0",
        license_or_usage_note="Test fixture only.",
    )

    assert metadata.document_id == "test-guideline"
    assert metadata.category == ClinicalDocumentCategory.HYPERTENSION


def test_clinical_document_requires_valid_url() -> None:
    with pytest.raises(ValidationError):
        ClinicalDocumentMetadata(
            document_id="invalid",
            title="Invalid",
            organization="Example",
            source_url="not-a-url",
            access_date=date(2026, 9, 9),
            category=ClinicalDocumentCategory.GENERAL,
        )


def test_clinical_section() -> None:
    section = ClinicalSection(
        section_id="section-001",
        heading="Diagnosis",
        text="Representative test text.",
        order=0,
    )

    assert section.heading == "Diagnosis"
    assert section.order == 0


def test_clinical_document() -> None:
    metadata = ClinicalDocumentMetadata(
        document_id="document-001",
        title="Example",
        organization="Example Organization",
        source_url="https://example.org/document",
        access_date=date(2026, 9, 9),
        category=ClinicalDocumentCategory.GENERAL,
    )

    document = ClinicalDocument(
        metadata=metadata,
        sections=[
            ClinicalSection(
                section_id="section-001",
                heading="Overview",
                text="Example evidence text.",
                order=0,
            )
        ],
    )

    assert len(document.sections) == 1


def test_document_chunk_preserves_provenance() -> None:
    chunk = DocumentChunk(
        chunk_id="document-001-chunk-000",
        document_id="document-001",
        document_title="Example Guideline",
        organization="Example Organization",
        source_url="https://example.org/guideline",
        category=ClinicalDocumentCategory.HYPERTENSION,
        access_date=date(2026, 9, 9),
        section_id="section-001",
        section_heading="Diagnosis",
        chunk_number=0,
        text="Evidence text.",
        token_count=3,
    )

    assert chunk.document_id == "document-001"
    assert chunk.chunk_number == 0
    assert chunk.section_heading == "Diagnosis"


def test_document_chunk_rejects_negative_chunk_number() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(
            chunk_id="bad-chunk",
            document_id="document-001",
            document_title="Example",
            organization="Example",
            source_url="https://example.org",
            category=ClinicalDocumentCategory.GENERAL,
            access_date=date(2026, 9, 9),
            chunk_number=-1,
            text="Invalid chunk.",
        )
