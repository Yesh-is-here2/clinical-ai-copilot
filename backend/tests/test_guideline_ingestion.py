import hashlib
from datetime import date, datetime, timezone

import httpx
import pytest

from app.retrieval.ingestion import (
    GuidelineIngestionError,
    build_clinical_document,
    ingest_guideline,
)
from app.retrieval.sources import GuidelineSource
from app.schemas.document import ClinicalDocumentCategory
from app.schemas.ingestion import DownloadedGuideline


def make_source() -> GuidelineSource:
    return GuidelineSource(
        document_id="test-guideline",
        title="Test Clinical Guideline",
        organization="Example Clinical Organization",
        source_url="https://example.org/guideline",
        publication_date=date(2026, 1, 15),
        category=ClinicalDocumentCategory.HYPERTENSION,
        version="1.0",
        license_or_usage_note="Public test fixture.",
        ingestion_method="html",
        allowed_hosts=("example.org",),
    )


def make_downloaded(
    *,
    document_id: str = "test-guideline",
) -> DownloadedGuideline:
    content = b"""
    <html>
      <body>
        <main>
          <h1>Overview</h1>
          <p>Clinical hypertension evidence.</p>

          <h2>Monitoring</h2>
          <p>Monitoring evidence.</p>
        </main>
      </body>
    </html>
    """

    return DownloadedGuideline(
        document_id=document_id,
        source_url="https://example.org/guideline",
        final_url="https://example.org/guideline",
        content_type="text/html",
        content_length=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        downloaded_at=datetime(
            2026,
            9,
            9,
            12,
            0,
            tzinfo=timezone.utc,
        ),
        content=content,
    )


def test_build_clinical_document() -> None:
    document = build_clinical_document(
        make_source(),
        make_downloaded(),
        access_date=date(2026, 9, 9),
    )

    assert document.metadata.document_id == (
        "test-guideline"
    )
    assert document.metadata.title == (
        "Test Clinical Guideline"
    )
    assert document.metadata.organization == (
        "Example Clinical Organization"
    )
    assert document.metadata.access_date == date(
        2026,
        9,
        9,
    )
    assert len(document.sections) == 2

    assert "Clinical hypertension evidence." in (
        document.raw_text
    )
    assert "Monitoring evidence." in (
        document.raw_text
    )


def test_build_document_rejects_id_mismatch() -> None:
    with pytest.raises(
        GuidelineIngestionError,
        match="does not match",
    ):
        build_clinical_document(
            make_source(),
            make_downloaded(
                document_id="different-guideline"
            ),
        )


@pytest.mark.anyio
async def test_ingest_guideline_end_to_end() -> None:
    content = b"""
    <html>
      <body>
        <main>
          <h1>Overview</h1>
          <p>Evidence from a controlled source.</p>
        </main>
      </body>
    </html>
    """

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "text/html",
                "content-length": str(len(content)),
            },
            content=content,
            request=request,
        )

    transport = httpx.MockTransport(
        handler
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        result = await ingest_guideline(
            make_source(),
            client=client,
            access_date=date(2026, 9, 9),
        )

    assert result.status.value == "processed"

    assert result.document.metadata.document_id == (
        "test-guideline"
    )

    assert len(
        result.document.sections
    ) == 1

    assert result.provenance.sha256 == (
        hashlib.sha256(content).hexdigest()
    )

    assert result.provenance.content_length == (
        len(content)
    )

    assert result.provenance.section_count == 1

    assert (
        result.provenance.downloaded_at
        <= result.provenance.processed_at
    )


@pytest.mark.anyio
async def test_ingestion_wraps_download_failure() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            503,
            request=request,
        )

    transport = httpx.MockTransport(
        handler
    )

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        with pytest.raises(
            GuidelineIngestionError,
            match="Download failed",
        ):
            await ingest_guideline(
                make_source(),
                client=client,
            )


def test_access_date_defaults_to_download_date() -> None:
    document = build_clinical_document(
        make_source(),
        make_downloaded(),
    )

    assert document.metadata.access_date == date(
        2026,
        9,
        9,
    )


def test_raw_text_preserves_section_order() -> None:
    document = build_clinical_document(
        make_source(),
        make_downloaded(),
    )

    first_position = document.raw_text.find(
        "Clinical hypertension evidence."
    )
    second_position = document.raw_text.find(
        "Monitoring evidence."
    )

    assert first_position >= 0
    assert second_position > first_position
