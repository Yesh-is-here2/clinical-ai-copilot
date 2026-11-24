"""Default (no DB) API tests for ``GET /queries/{query_id}`` and
``GET /queries/{query_id}/evaluation`` (Milestone F, section 20).

Overrides the repository dependencies directly with small, deterministic
fakes (plain ``SimpleNamespace``/dataclass stand-ins for the ORM rows) -
no real Postgres anywhere in this file. This exercises the route's own
response-shaping logic (section 8/9) independently of
``app.db.repositories.query_repository``'s SQL, which is covered by the
Postgres integration tests instead.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi.testclient import TestClient

import app.main as main_module
from app.api.deps import get_evaluation_repository, get_query_repository
from app.clinical.final_response import DISCLAIMER_TEXT


@dataclass
class _FakeCitation:
    citation_label: str
    chunk_id: str
    quoted_support: str
    support_classification: str


@dataclass
class _FakeGeneratedResponse:
    answer: str
    key_points: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    safety_flags: list[str] = field(default_factory=list)
    citations: list[_FakeCitation] = field(default_factory=list)


@dataclass
class _FakeQueryRecord:
    id: uuid.UUID
    request_id: str
    patient_id: str | None
    question: str
    question_category: str | None
    status: str
    evidence_status: str | None
    confidence: float | None
    latency_ms: float | None
    created_at: datetime
    generated_response: _FakeGeneratedResponse | None


class _FakeQueryRepository:
    def __init__(self, record: _FakeQueryRecord | None) -> None:
        self._record = record

    async def get_by_id(self, query_id: uuid.UUID):
        if self._record is not None and self._record.id == query_id:
            return self._record
        return None

    async def exists(self, query_id: uuid.UUID) -> bool:
        return self._record is not None and self._record.id == query_id


@dataclass
class _FakeEvaluationMetric:
    metric_name: str
    metric_value: float
    run_id: str
    chunking_config: str | None = None


class _FakeEvaluationRepository:
    def __init__(self, metrics: list[_FakeEvaluationMetric]) -> None:
        self._metrics = metrics

    async def get_metrics_for_query(self, query_id: uuid.UUID):
        return self._metrics


def _make_record(query_id: uuid.UUID) -> _FakeQueryRecord:
    return _FakeQueryRecord(
        id=query_id,
        request_id="req-abc",
        patient_id="patient-001",
        question="What is the target blood pressure?",
        question_category="guideline",
        status="normal",
        evidence_status="sufficient",
        confidence=0.87,
        latency_ms=452.1,
        created_at=datetime.now(timezone.utc),
        generated_response=_FakeGeneratedResponse(
            answer="Target blood pressure below 130/80 mmHg [C1].",
            key_points=["Target below 130/80 mmHg"],
            citations=[
                _FakeCitation(
                    citation_label="C1",
                    chunk_id="c1",
                    quoted_support="...guideline text...",
                    support_classification="supported",
                )
            ],
        ),
    )


def teardown_function() -> None:
    main_module.app.dependency_overrides.clear()


def test_get_query_found_returns_persisted_detail() -> None:
    query_id = uuid.uuid4()
    record = _make_record(query_id)
    main_module.app.dependency_overrides[get_query_repository] = (
        lambda: _FakeQueryRepository(record)
    )

    with TestClient(main_module.app) as client:
        response = client.get(f"/queries/{query_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["query_id"] == str(query_id)
    assert body["answer"].startswith("Target blood pressure")
    assert body["citations"][0]["citation_label"] == "C1"
    assert body["disclaimer"] == DISCLAIMER_TEXT


def test_get_query_not_found_returns_404() -> None:
    main_module.app.dependency_overrides[get_query_repository] = (
        lambda: _FakeQueryRepository(None)
    )

    with TestClient(main_module.app) as client:
        response = client.get(f"/queries/{uuid.uuid4()}")

    assert response.status_code == 404


def test_get_query_evaluation_not_run_when_no_metrics_exist() -> None:
    query_id = uuid.uuid4()
    main_module.app.dependency_overrides[get_query_repository] = (
        lambda: _FakeQueryRepository(_make_record(query_id))
    )
    main_module.app.dependency_overrides[get_evaluation_repository] = (
        lambda: _FakeEvaluationRepository([])
    )

    with TestClient(main_module.app) as client:
        response = client.get(f"/queries/{query_id}/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_run"
    assert body["metrics"] == []


def test_get_query_evaluation_returns_persisted_metrics() -> None:
    query_id = uuid.uuid4()
    main_module.app.dependency_overrides[get_query_repository] = (
        lambda: _FakeQueryRepository(_make_record(query_id))
    )
    main_module.app.dependency_overrides[get_evaluation_repository] = (
        lambda: _FakeEvaluationRepository(
            [_FakeEvaluationMetric(metric_name="faithfulness", metric_value=0.92, run_id="run-1")]
        )
    )

    with TestClient(main_module.app) as client:
        response = client.get(f"/queries/{query_id}/evaluation")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "available"
    assert body["metrics"][0]["metric_name"] == "faithfulness"


def test_get_query_evaluation_for_unknown_query_returns_404() -> None:
    main_module.app.dependency_overrides[get_query_repository] = (
        lambda: _FakeQueryRepository(None)
    )
    main_module.app.dependency_overrides[get_evaluation_repository] = (
        lambda: _FakeEvaluationRepository([])
    )

    with TestClient(main_module.app) as client:
        response = client.get(f"/queries/{uuid.uuid4()}/evaluation")

    assert response.status_code == 404
