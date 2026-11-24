"""Default (no DB/network) API tests for ``GET /documents`` and
``POST /documents/ingest`` (Milestone F, section 20).

The ingestion route is exercised through
``app.services.document_ingestion_service.DocumentIngestionService``
directly overridden as a dependency (rather than monkeypatching
``ingest_guideline``/httpx) - this keeps these tests fast, offline, and
focused on the route's own status-code mapping (section 14), while
``tests/test_guideline_ingestion.py`` already covers
``ingest_guideline`` itself end-to-end against a fake HTTP transport.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

import app.main as main_module
from app.api.deps import (
    Actor,
    get_current_actor,
    get_document_ingestion_service,
    get_document_repository,
)
from app.models.user import UserRole
from app.services.document_ingestion_service import (
    DocumentIngestionResult,
    DocumentPersistenceError,
    GuidelineIngestionFailedError,
    UnknownGuidelineSourceError,
)

_CLINICIAN = Actor(
    user_id=uuid.uuid4(), email="clinician@example.com", role=UserRole.CLINICIAN
)
_ADMIN = Actor(
    user_id=uuid.uuid4(), email="admin@example.com", role=UserRole.ADMIN
)


def _as_clinician() -> None:
    main_module.app.dependency_overrides[get_current_actor] = lambda: _CLINICIAN


def _as_admin() -> None:
    main_module.app.dependency_overrides[get_current_actor] = lambda: _ADMIN


@dataclass
class _FakeDocumentRow:
    document_id: str
    title: str
    organization: str
    category: str
    version: str | None
    publication_date = None
    access_date = None


class _FakeDocumentRepository:
    def __init__(self, rows: list[tuple[_FakeDocumentRow, int]]) -> None:
        self._rows = rows

    async def list_documents_with_chunk_counts(self):
        return self._rows


class _FakeIngestionService:
    def __init__(self, *, result=None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    async def ingest(self, document_id: str) -> DocumentIngestionResult:
        if self._error is not None:
            raise self._error
        return self._result


def teardown_function() -> None:
    main_module.app.dependency_overrides.clear()


def test_list_documents_returns_metadata() -> None:
    _as_clinician()
    row = _FakeDocumentRow(
        document_id="nice-ng136-hypertension",
        title="Hypertension in adults",
        organization="NICE",
        category="hypertension",
        version="NG136",
    )
    row.access_date = "2026-09-09"
    main_module.app.dependency_overrides[get_document_repository] = (
        lambda: _FakeDocumentRepository([(row, 12)])
    )

    with TestClient(main_module.app) as client:
        response = client.get("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body[0]["document_id"] == "nice-ng136-hypertension"
    assert body[0]["chunk_count"] == 12


def test_list_documents_empty_when_nothing_ingested() -> None:
    _as_clinician()
    main_module.app.dependency_overrides[get_document_repository] = (
        lambda: _FakeDocumentRepository([])
    )

    with TestClient(main_module.app) as client:
        response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_requires_authentication() -> None:
    main_module.app.dependency_overrides[get_document_repository] = (
        lambda: _FakeDocumentRepository([])
    )

    with TestClient(main_module.app) as client:
        response = client.get("/documents")

    assert response.status_code == 401


def test_ingest_unknown_source_returns_404() -> None:
    _as_admin()
    main_module.app.dependency_overrides[get_document_ingestion_service] = (
        lambda: _FakeIngestionService(
            error=UnknownGuidelineSourceError("'nope' is not registered.")
        )
    )

    with TestClient(main_module.app) as client:
        response = client.post("/documents/ingest", json={"document_id": "nope"})

    assert response.status_code == 404


def test_ingest_failure_maps_to_502() -> None:
    _as_admin()
    main_module.app.dependency_overrides[get_document_ingestion_service] = (
        lambda: _FakeIngestionService(
            error=GuidelineIngestionFailedError("download failed")
        )
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/documents/ingest",
            json={"document_id": "who-hypertension-pharmacological-2021"},
        )

    assert response.status_code == 502


def test_ingest_persistence_failure_maps_to_503() -> None:
    _as_admin()
    main_module.app.dependency_overrides[get_document_ingestion_service] = (
        lambda: _FakeIngestionService(
            error=DocumentPersistenceError("db unavailable")
        )
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/documents/ingest",
            json={"document_id": "who-hypertension-pharmacological-2021"},
        )

    assert response.status_code == 503


def test_ingest_success_returns_chunk_count() -> None:
    _as_admin()
    main_module.app.dependency_overrides[get_document_ingestion_service] = (
        lambda: _FakeIngestionService(
            result=DocumentIngestionResult(
                document_id="who-hypertension-pharmacological-2021",
                status="processed",
                chunk_count=42,
                section_count=7,
            )
        )
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/documents/ingest",
            json={"document_id": "who-hypertension-pharmacological-2021"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 42
    assert body["status"] == "processed"


def test_ingest_malformed_request_returns_422() -> None:
    _as_admin()
    with TestClient(main_module.app) as client:
        response = client.post("/documents/ingest", json={})

    assert response.status_code == 422


def test_ingest_requires_admin_role() -> None:
    # A clinician is authenticated but not an admin: 403, and the
    # ingestion service is never even called.
    _as_clinician()
    main_module.app.dependency_overrides[get_document_ingestion_service] = (
        lambda: _FakeIngestionService(
            result=DocumentIngestionResult(
                document_id="who-hypertension-pharmacological-2021",
                status="processed",
                chunk_count=42,
                section_count=7,
            )
        )
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/documents/ingest",
            json={"document_id": "who-hypertension-pharmacological-2021"},
        )

    assert response.status_code == 403


def test_ingest_requires_authentication() -> None:
    with TestClient(main_module.app) as client:
        response = client.post(
            "/documents/ingest",
            json={"document_id": "who-hypertension-pharmacological-2021"},
        )

    assert response.status_code == 401
