"""Integration tests for Phase 7 authentication and RBAC, requiring a
live PostgreSQL + pgvector database.

Excluded from the default test run by backend/pytest.ini
(`addopts = -m "not integration"`). Run explicitly with:

    docker compose exec backend pytest -m integration -q

These are NOT run in the sandbox this code was developed in (no route to
a real Postgres instance there) - do not report them as passing, and do
not weaken/skip-with-a-mock any of the assertions below to make that
happen. See tests/integration/README.md and
tests/integration/test_query_persistence_integration.py for the
established NullPool/isolated-engine + `itest-` scoped-cleanup pattern
this file follows.

What this file proves against the real database (the part the unit
tests cannot): users persist and authenticate end to end over HTTP,
role gates hold on the real routes, query ownership is enforced on
real rows, and the actor is recorded on the real `queries` /
`audit_logs` rows.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

import app.main as main_module
from app.clinical.audit import AuditLogEntry
from app.clinical.confidence import ConfidenceLabel
from app.clinical.evidence_sufficiency import EvidenceSufficiencyStatus
from app.clinical.final_response import DISCLAIMER_TEXT, FinalClinicalResponse
from app.core.config import settings
from app.core.security import decode_access_token
from app.db.repositories.audit_repository import SqlAlchemyAuditLogWriter
from app.db.repositories.query_repository import (
    ClinicalQueryRepository,
    PersistedQueryInput,
)
from app.models.audit import AuditLogRecord
from app.models.document import ClinicalDocumentRecord
from app.models.patient import Patient
from app.models.query import QueryRecord
from app.models.user import User, UserRole
from app.services.auth_service import AuthService

pytestmark = [pytest.mark.integration, pytest.mark.anyio]

_PREFIX = "itest-"


async def _delete_test_owned_rows(session: AsyncSession) -> None:
    # Same order as test_query_persistence_integration.py (queries
    # first - citations.chunk_id is ondelete=RESTRICT), plus users
    # last: queries/audit_logs reference users.id with ondelete
    # SET NULL, so deleting users first would orphan the attribution
    # instead of removing the rows.
    await session.execute(
        delete(QueryRecord).where(QueryRecord.request_id.like(f"{_PREFIX}%"))
    )
    await session.execute(
        delete(ClinicalDocumentRecord).where(
            ClinicalDocumentRecord.document_id.like(f"{_PREFIX}%")
        )
    )
    await session.execute(
        delete(AuditLogRecord).where(AuditLogRecord.request_id.like(f"{_PREFIX}%"))
    )
    await session.execute(
        delete(Patient).where(Patient.patient_id.like(f"{_PREFIX}%"))
    )
    await session.execute(
        delete(User).where(User.email.like(f"{_PREFIX}%"))
    )
    await session.commit()


@pytest.fixture
async def db_session():
    test_engine = create_async_engine(
        settings.database_url, poolclass=NullPool
    )
    test_session_factory = async_sessionmaker(
        bind=test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with test_session_factory() as session:
        await _delete_test_owned_rows(session)
        try:
            yield session
        finally:
            await _delete_test_owned_rows(session)

    await test_engine.dispose()


def _email(role: str) -> str:
    return f"{_PREFIX}auth-{role}@example.com"


async def _create_user(
    session: AsyncSession, *, role: UserRole, password: str = "correct horse"
) -> User:
    """Create a real user through the real AuthService (bcrypt hash,
    real commit) and return the row."""

    return await AuthService(session, settings).create_user(
        email=_email(role.value),
        password=password,
        full_name=f"itest {role.value}",
        role=role,
    )


def _login(client: TestClient, *, email: str, password: str = "correct horse") -> str:
    response = client.post(
        "/auth/login", json={"email": email, "password": password}
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _make_final_response(request_id: str) -> FinalClinicalResponse:
    return FinalClinicalResponse(
        request_id=request_id,
        patient_summary="No patient context was provided for this question.",
        clinical_question="What is the target blood pressure?",
        answer="Target blood pressure below 130/80 mmHg [C1].",
        key_points=["Target below 130/80 mmHg"],
        citations=[],
        confidence=0.9,
        confidence_label=ConfidenceLabel.HIGH,
        evidence_status=EvidenceSufficiencyStatus.SUFFICIENT,
        warnings=[],
        safety_flags=[],
        disclaimer=DISCLAIMER_TEXT,
    )


class TestAuthEndToEnd:
    async def test_login_issues_token_and_patients_requires_it(
        self, db_session: AsyncSession
    ) -> None:
        user = await _create_user(db_session, role=UserRole.CLINICIAN)

        with TestClient(main_module.app) as client:
            assert client.get("/patients").status_code == 401

            token = _login(client, email=user.email)
            payload = decode_access_token(token, secret=settings.jwt_secret)
            assert payload.subject == str(user.id)
            assert payload.role == "clinician"

            response = client.get("/patients", headers=_auth_headers(token))
            assert response.status_code == 200
            assert len(response.json()) == 10

    async def test_admin_can_create_user_and_clinician_cannot(
        self, db_session: AsyncSession
    ) -> None:
        admin = await _create_user(db_session, role=UserRole.ADMIN)
        await _create_user(db_session, role=UserRole.CLINICIAN)

        with TestClient(main_module.app) as client:
            admin_token = _login(client, email=admin.email)
            created = client.post(
                "/auth/users",
                json={
                    "email": f"{_PREFIX}auth-new-reviewer@example.com",
                    "password": "correct horse",
                    "role": "reviewer",
                },
                headers=_auth_headers(admin_token),
            )
            assert created.status_code == 201, created.text
            assert created.json()["email"] == f"{_PREFIX}auth-new-reviewer@example.com"

            # The newly created reviewer can log in with the password
            # set by the admin.
            _login(
                client,
                email=f"{_PREFIX}auth-new-reviewer@example.com",
            )

            clinician_token = _login(client, email=_email("clinician"))
            forbidden = client.post(
                "/auth/users",
                json={"email": "x@example.com", "password": "correct horse"},
                headers=_auth_headers(clinician_token),
            )
            assert forbidden.status_code == 403

    async def test_query_ownership_enforced_on_real_rows(
        self, db_session: AsyncSession
    ) -> None:
        owner = await _create_user(db_session, role=UserRole.CLINICIAN)
        # A second clinician needs a distinct email from the owner's.
        other_email = f"{_PREFIX}auth-other@example.com"
        other = await AuthService(db_session, settings).create_user(
            email=other_email,
            password="correct horse",
            full_name="itest other",
            role=UserRole.CLINICIAN,
        )
        admin = await _create_user(db_session, role=UserRole.ADMIN)

        query_id = uuid.uuid4()
        db_session.add(
            QueryRecord(
                id=query_id,
                request_id=f"{_PREFIX}req-ownership",
                user_id=owner.id,
                question="What is the target blood pressure?",
                status="normal",
            )
        )
        await db_session.commit()

        with TestClient(main_module.app) as client:
            owner_token = _login(client, email=owner.email)
            assert (
                client.get(
                    f"/queries/{query_id}", headers=_auth_headers(owner_token)
                ).status_code
                == 200
            )

            other_token = _login(client, email=other.email)
            assert (
                client.get(
                    f"/queries/{query_id}", headers=_auth_headers(other_token)
                ).status_code
                == 404
            )

            admin_token = _login(client, email=admin.email)
            assert (
                client.get(
                    f"/queries/{query_id}", headers=_auth_headers(admin_token)
                ).status_code
                == 200
            )

    async def test_audit_requires_reviewer_or_admin(
        self, db_session: AsyncSession
    ) -> None:
        await _create_user(db_session, role=UserRole.CLINICIAN)
        await _create_user(db_session, role=UserRole.REVIEWER)
        await _create_user(db_session, role=UserRole.ADMIN)

        with TestClient(main_module.app) as client:
            clinician_token = _login(client, email=_email("clinician"))
            assert (
                client.get("/audit", headers=_auth_headers(clinician_token)).status_code
                == 403
            )

            reviewer_token = _login(client, email=_email("reviewer"))
            response = client.get("/audit", headers=_auth_headers(reviewer_token))
            assert response.status_code == 200
            assert response.json()["total"] == 0

            admin_token = _login(client, email=_email("admin"))
            assert (
                client.get("/audit", headers=_auth_headers(admin_token)).status_code
                == 200
            )

    async def test_clinical_query_auth_gate_holds_for_reviewer(
        self, db_session: AsyncSession
    ) -> None:
        await _create_user(db_session, role=UserRole.CLINICIAN)
        await _create_user(db_session, role=UserRole.REVIEWER)

        with TestClient(main_module.app) as client:
            reviewer_token = _login(client, email=_email("reviewer"))
            assert (
                client.post(
                    "/clinical/query",
                    json={"question": "What is the target blood pressure?"},
                    headers=_auth_headers(reviewer_token),
                ).status_code
                == 403
            )

            # A clinician passes the auth gate: the outcome is 503
            # (nothing ingested / no embedding provider) or a 200
            # guarded response, depending on what is ingested in this
            # database - either way it is NOT an auth rejection.
            clinician_token = _login(client, email=_email("clinician"))
            response = client.post(
                "/clinical/query",
                json={"question": "What is the target blood pressure?"},
                headers=_auth_headers(clinician_token),
            )
            assert response.status_code not in (401, 403), response.text

    async def test_persisted_query_and_audit_rows_carry_actor(
        self, db_session: AsyncSession
    ) -> None:
        user = await _create_user(db_session, role=UserRole.CLINICIAN)
        request_id = f"{_PREFIX}req-attribution"
        query_id = uuid.uuid4()

        await ClinicalQueryRepository(db_session).persist_query(
            PersistedQueryInput(
                query_id=query_id,
                request_id=request_id,
                patient_id=None,
                question="What is the target blood pressure?",
                question_category="guideline",
                status="normal",
                final_response=_make_final_response(request_id),
                model_name="test-model",
                latency_ms=123.4,
                user_id=user.id,
            )
        )
        await SqlAlchemyAuditLogWriter(db_session).write(
            AuditLogEntry(
                request_id=request_id,
                patient_id=None,
                question="What is the target blood pressure?",
                question_category="guideline",
                model="test-model",
                output_status="normal",
                actor_user_id=user.id,
                actor_role="clinician",
            )
        )
        await db_session.commit()

        query_row = (
            await db_session.execute(
                select(QueryRecord).where(QueryRecord.id == query_id)
            )
        ).scalar_one()
        assert query_row.user_id == user.id

        audit_row = (
            await db_session.execute(
                select(AuditLogRecord).where(
                    AuditLogRecord.request_id == request_id
                )
            )
        ).scalar_one()
        assert audit_row.user_id == user.id
        assert audit_row.role == "clinician"
