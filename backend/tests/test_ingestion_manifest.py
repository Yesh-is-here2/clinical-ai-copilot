import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

from app.retrieval.manifest import (
    build_manifest_payload,
    write_ingestion_manifest,
)
from app.schemas.document import (
    ClinicalDocument,
    ClinicalDocumentCategory,
    ClinicalDocumentMetadata,
    ClinicalSection,
)
from app.schemas.ingestion import (
    IngestedGuideline,
    IngestionProvenance,
)


def make_ingested_guideline() -> IngestedGuideline:
    content = b"controlled-test-content"
    digest = hashlib.sha256(content).hexdigest()

    document = ClinicalDocument(
        metadata=ClinicalDocumentMetadata(
            document_id="test-guideline",
            title="Test Clinical Guideline",
            organization="Example Organization",
            source_url=(
                "https://example.org/guideline"
            ),
            publication_date=date(
                2026,
                1,
                15,
            ),
            access_date=date(
                2026,
                9,
                9,
            ),
            category=(
                ClinicalDocumentCategory.HYPERTENSION
            ),
            version="1.0",
            license_or_usage_note=(
                "Public test fixture."
            ),
        ),
        sections=[
            ClinicalSection(
                section_id="section-0001",
                heading="Overview",
                text="Clinical evidence.",
                order=0,
            ),
            ClinicalSection(
                section_id="section-0002",
                heading="Monitoring",
                text="Monitoring evidence.",
                order=1,
            ),
        ],
        raw_text=(
            "Clinical evidence.\n\n"
            "Monitoring evidence."
        ),
    )

    provenance = IngestionProvenance(
        document_id="test-guideline",
        source_url=(
            "https://example.org/guideline"
        ),
        final_url=(
            "https://example.org/guideline"
        ),
        content_type="text/html",
        content_length=len(content),
        sha256=digest,
        downloaded_at=datetime(
            2026,
            9,
            9,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        processed_at=datetime(
            2026,
            9,
            9,
            12,
            0,
            1,
            tzinfo=timezone.utc,
        ),
        section_count=2,
    )

    return IngestedGuideline(
        document=document,
        provenance=provenance,
    )


def test_manifest_payload_contains_provenance() -> None:
    result = make_ingested_guideline()

    payload = build_manifest_payload(
        result
    )

    assert payload["schema_version"] == "1.0"
    assert payload["status"] == "processed"

    assert (
        payload["document"]["document_id"]
        == "test-guideline"
    )

    assert (
        payload["provenance"]["sha256"]
        == result.provenance.sha256
    )

    assert (
        payload["provenance"]["section_count"]
        == 2
    )


def test_manifest_does_not_store_raw_text() -> None:
    payload = build_manifest_payload(
        make_ingested_guideline()
    )

    serialized = json.dumps(payload)

    assert "raw_text" not in serialized
    assert "Clinical evidence." not in serialized
    assert "Monitoring evidence." not in serialized


def test_manifest_records_section_identity() -> None:
    payload = build_manifest_payload(
        make_ingested_guideline()
    )

    assert payload["sections"] == [
        {
            "section_id": "section-0001",
            "heading": "Overview",
            "order": 0,
        },
        {
            "section_id": "section-0002",
            "heading": "Monitoring",
            "order": 1,
        },
    ]


def test_write_manifest(
    tmp_path: Path,
) -> None:
    result = make_ingested_guideline()

    manifest_path = write_ingestion_manifest(
        result,
        tmp_path,
    )

    assert manifest_path.exists()
    assert manifest_path.name == (
        "test-guideline.manifest.json"
    )

    payload = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        payload["document"]["document_id"]
        == "test-guideline"
    )

    assert (
        payload["provenance"]["sha256"]
        == result.provenance.sha256
    )


def test_manifest_write_is_deterministic_except_times(
    tmp_path: Path,
) -> None:
    result = make_ingested_guideline()

    first_path = write_ingestion_manifest(
        result,
        tmp_path,
    )

    first = first_path.read_text(
        encoding="utf-8"
    )

    second_path = write_ingestion_manifest(
        result,
        tmp_path,
    )

    second = second_path.read_text(
        encoding="utf-8"
    )

    assert first == second


def test_manifest_has_no_temporary_file_after_write(
    tmp_path: Path,
) -> None:
    result = make_ingested_guideline()

    write_ingestion_manifest(
        result,
        tmp_path,
    )

    temporary_files = list(
        tmp_path.glob("*.tmp")
    )

    assert temporary_files == []
