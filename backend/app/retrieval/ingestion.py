from datetime import date, datetime, timezone

import httpx

from app.retrieval.document_extractor import (
    DocumentExtractionError,
    extract_document_sections,
)
from app.retrieval.downloader import (
    GuidelineDownloadError,
    download_guideline,
)
from app.retrieval.sources import GuidelineSource
from app.schemas.document import (
    ClinicalDocument,
    ClinicalDocumentMetadata,
)
from app.schemas.ingestion import (
    IngestedGuideline,
    IngestionProvenance,
)


class GuidelineIngestionError(RuntimeError):
    pass


def build_clinical_document(
    source: GuidelineSource,
    downloaded,
    *,
    access_date: date | None = None,
) -> ClinicalDocument:
    if source.document_id != downloaded.document_id:
        raise GuidelineIngestionError(
            "Downloaded guideline document ID does not "
            "match the registered source."
        )

    try:
        sections = extract_document_sections(
            downloaded
        )
    except DocumentExtractionError as exc:
        raise GuidelineIngestionError(
            f"Extraction failed for "
            f"'{source.document_id}'."
        ) from exc

    resolved_access_date = (
        access_date
        or downloaded.downloaded_at.date()
    )

    raw_text = "\n\n".join(
        section.text
        for section in sections
    )

    return ClinicalDocument(
        metadata=ClinicalDocumentMetadata(
            document_id=source.document_id,
            title=source.title,
            organization=source.organization,
            source_url=source.source_url,
            publication_date=source.publication_date,
            access_date=resolved_access_date,
            category=source.category,
            version=source.version,
            license_or_usage_note=(
                source.license_or_usage_note
            ),
        ),
        sections=sections,
        raw_text=raw_text,
    )


async def ingest_guideline(
    source: GuidelineSource,
    *,
    client: httpx.AsyncClient | None = None,
    access_date: date | None = None,
) -> IngestedGuideline:
    try:
        downloaded = await download_guideline(
            source,
            client=client,
        )
    except GuidelineDownloadError as exc:
        raise GuidelineIngestionError(
            f"Download failed for "
            f"'{source.document_id}'."
        ) from exc

    document = build_clinical_document(
        source,
        downloaded,
        access_date=access_date,
    )

    processed_at = datetime.now(
        timezone.utc
    )

    provenance = IngestionProvenance(
        document_id=downloaded.document_id,
        source_url=downloaded.source_url,
        final_url=downloaded.final_url,
        content_type=downloaded.content_type,
        content_length=downloaded.content_length,
        sha256=downloaded.sha256,
        downloaded_at=downloaded.downloaded_at,
        processed_at=processed_at,
        section_count=len(document.sections),
    )

    return IngestedGuideline(
        document=document,
        provenance=provenance,
    )
