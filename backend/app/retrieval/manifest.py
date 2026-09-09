import json
from pathlib import Path

from app.schemas.ingestion import IngestedGuideline


class ManifestPersistenceError(RuntimeError):
    pass


def build_manifest_payload(
    result: IngestedGuideline,
) -> dict:
    return {
        "schema_version": "1.0",
        "status": result.status.value,
        "document": {
            "document_id": (
                result.document.metadata.document_id
            ),
            "title": result.document.metadata.title,
            "organization": (
                result.document.metadata.organization
            ),
            "source_url": str(
                result.document.metadata.source_url
            ),
            "publication_date": (
                result.document.metadata.publication_date.isoformat()
                if result.document.metadata.publication_date
                else None
            ),
            "access_date": (
                result.document.metadata.access_date.isoformat()
            ),
            "category": (
                result.document.metadata.category.value
            ),
            "version": result.document.metadata.version,
            "license_or_usage_note": (
                result.document.metadata.license_or_usage_note
            ),
        },
        "provenance": {
            "source_url": str(
                result.provenance.source_url
            ),
            "final_url": str(
                result.provenance.final_url
            ),
            "content_type": (
                result.provenance.content_type
            ),
            "content_length": (
                result.provenance.content_length
            ),
            "sha256": result.provenance.sha256,
            "downloaded_at": (
                result.provenance.downloaded_at.isoformat()
            ),
            "processed_at": (
                result.provenance.processed_at.isoformat()
            ),
            "section_count": (
                result.provenance.section_count
            ),
        },
        "sections": [
            {
                "section_id": section.section_id,
                "heading": section.heading,
                "order": section.order,
            }
            for section in result.document.sections
        ],
    }


def write_ingestion_manifest(
    result: IngestedGuideline,
    output_directory: Path,
) -> Path:
    document_id = (
        result.document.metadata.document_id
    )

    if not document_id:
        raise ManifestPersistenceError(
            "Cannot persist a manifest without "
            "a document ID."
        )

    try:
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        manifest_path = (
            output_directory
            / f"{document_id}.manifest.json"
        )

        payload = build_manifest_payload(
            result
        )

        temporary_path = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(
            manifest_path
        )

    except OSError as exc:
        raise ManifestPersistenceError(
            f"Failed to persist ingestion manifest "
            f"for '{document_id}'."
        ) from exc

    return manifest_path
