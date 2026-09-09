import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app.retrieval import cli
from app.retrieval.ingestion import GuidelineIngestionError
from app.retrieval.manifest import ManifestPersistenceError
from app.schemas.document import (
    ClinicalDocument,
    ClinicalDocumentCategory,
    ClinicalDocumentMetadata,
    ClinicalSection,
)
from app.schemas.ingestion import (
    IngestedGuideline,
    IngestionProvenance,
    IngestionStatus,
)


REGISTERED_DOCUMENT_ID = "who-hypertension-pharmacological-2021"


def make_ingested_guideline(
    document_id: str = REGISTERED_DOCUMENT_ID,
) -> IngestedGuideline:
    document = ClinicalDocument(
        metadata=ClinicalDocumentMetadata(
            document_id=document_id,
            title="Guideline for the pharmacological treatment of hypertension in adults",
            organization="World Health Organization",
            source_url="https://iris.who.int/server/api/core/bitstreams/f062769d-f075-4a00-87af-0a2106e0bd04/content",
            publication_date=date(2021, 8, 24),
            access_date=date(2026, 9, 9),
            category=ClinicalDocumentCategory.HYPERTENSION,
            version="2021",
            license_or_usage_note="WHO publication distributed under CC BY-NC-SA 3.0 IGO.",
        ),
        sections=[
            ClinicalSection(
                section_id="s1",
                heading="Overview",
                text="Pharmacological treatment evidence.",
                order=0,
            )
        ],
        raw_text="Pharmacological treatment evidence.",
    )

    provenance = IngestionProvenance(
        document_id=document_id,
        source_url=document.metadata.source_url,
        final_url=document.metadata.source_url,
        content_type="application/pdf",
        content_length=1234,
        sha256="a" * 64,
        downloaded_at=datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc),
        processed_at=datetime(2026, 9, 9, 12, 0, 1, tzinfo=timezone.utc),
        section_count=1,
    )

    return IngestedGuideline(
        status=IngestionStatus.PROCESSED,
        document=document,
        provenance=provenance,
    )


# ---------------------------------------------------------------------
# Unknown source
# ---------------------------------------------------------------------


def test_cli_rejects_unknown_document_id(tmp_path: Path, capsys) -> None:
    exit_code = cli.main(
        ["ingest", "not-a-registered-source", "--output-dir", str(tmp_path)]
    )

    assert exit_code == cli.EXIT_UNKNOWN_SOURCE

    captured = capsys.readouterr()
    assert "unknown guideline source" in captured.err
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------
# Download / ingestion failure
# ---------------------------------------------------------------------


def test_cli_reports_ingestion_failure(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failing_ingest(source, **kwargs):
        raise GuidelineIngestionError("Download failed for source.")

    monkeypatch.setattr(cli, "ingest_guideline", failing_ingest)

    exit_code = cli.main(
        ["ingest", REGISTERED_DOCUMENT_ID, "--output-dir", str(tmp_path)]
    )

    assert exit_code == cli.EXIT_INGESTION_FAILED

    captured = capsys.readouterr()
    assert "ingestion failed" in captured.err
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------
# Manifest persistence failure
# ---------------------------------------------------------------------


def test_cli_reports_manifest_persistence_failure(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_ingested_guideline()

    async def fake_ingest(source, **kwargs):
        return result

    def failing_write(result_arg, output_directory):
        raise ManifestPersistenceError("disk full")

    monkeypatch.setattr(cli, "ingest_guideline", fake_ingest)
    monkeypatch.setattr(cli, "write_ingestion_manifest", failing_write)

    exit_code = cli.main(
        ["ingest", REGISTERED_DOCUMENT_ID, "--output-dir", str(tmp_path)]
    )

    assert exit_code == cli.EXIT_MANIFEST_FAILED

    captured = capsys.readouterr()
    assert "failed to persist manifest" in captured.err


# ---------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------


def test_cli_success_writes_manifest_and_prints_summary(
    tmp_path: Path,
    capsys,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = make_ingested_guideline()

    async def fake_ingest(source, **kwargs):
        assert source.document_id == REGISTERED_DOCUMENT_ID
        return result

    monkeypatch.setattr(cli, "ingest_guideline", fake_ingest)

    exit_code = cli.main(
        ["ingest", REGISTERED_DOCUMENT_ID, "--output-dir", str(tmp_path)]
    )

    assert exit_code == cli.EXIT_OK

    manifest_path = tmp_path / f"{REGISTERED_DOCUMENT_ID}.manifest.json"
    assert manifest_path.exists()

    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert summary["document_id"] == REGISTERED_DOCUMENT_ID
    assert summary["status"] == "processed"
    assert summary["section_count"] == 1
    assert summary["sha256"] == "a" * 64
    assert summary["manifest_path"] == str(manifest_path)

    # No secrets/API keys should ever appear in CLI output.
    assert "OPENAI_API_KEY" not in captured.out
    assert "sk-" not in captured.out


# ---------------------------------------------------------------------
# Default output location
# ---------------------------------------------------------------------


def test_cli_default_output_directory_is_processed_dir() -> None:
    assert cli.DEFAULT_OUTPUT_DIR == Path("/app/data/guidelines/processed")


def test_cli_uses_default_output_directory_when_not_specified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_output_dir = {}

    result = make_ingested_guideline()

    async def fake_ingest(source, **kwargs):
        return result

    def capturing_write(result_arg, output_directory):
        captured_output_dir["value"] = output_directory
        return output_directory / f"{REGISTERED_DOCUMENT_ID}.manifest.json"

    monkeypatch.setattr(cli, "ingest_guideline", fake_ingest)
    monkeypatch.setattr(cli, "write_ingestion_manifest", capturing_write)

    exit_code = cli.main(["ingest", REGISTERED_DOCUMENT_ID])

    assert exit_code == cli.EXIT_OK
    assert captured_output_dir["value"] == cli.DEFAULT_OUTPUT_DIR
