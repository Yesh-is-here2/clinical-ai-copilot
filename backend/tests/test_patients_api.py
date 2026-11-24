"""Default (no DB) API tests for ``GET /patients`` (Milestone F,
section 20; Phase 7).

Patients are readable by any authenticated user, so the tests run
with a clinician actor overridden in (``get_current_actor``) and
assert the unauthenticated case separately.
"""

import uuid

from fastapi.testclient import TestClient

import app.main as main_module
from app.api.deps import Actor, get_current_actor
from app.main import app
from app.models.user import UserRole

_CLINICIAN = Actor(
    user_id=uuid.uuid4(), email="clinician@example.com", role=UserRole.CLINICIAN
)


def teardown_function() -> None:
    main_module.app.dependency_overrides.clear()


def test_list_patients_endpoint() -> None:
    main_module.app.dependency_overrides[get_current_actor] = lambda: _CLINICIAN

    with TestClient(app) as client:
        response = client.get("/patients")

    assert response.status_code == 200

    payload = response.json()

    assert isinstance(payload, list)
    assert len(payload) == 10

    patient_ids = {
        patient["patient_id"]
        for patient in payload
    }

    assert "patient-001" in patient_ids
    assert "patient-010" in patient_ids


def test_get_patient_endpoint() -> None:
    main_module.app.dependency_overrides[get_current_actor] = lambda: _CLINICIAN

    with TestClient(app) as client:
        response = client.get("/patients/patient-001")

    assert response.status_code == 200

    payload = response.json()

    assert payload["patient_id"] == "patient-001"
    assert payload["full_name"] == "Alex Morgan"
    assert payload["conditions"][0]["code"] == "I10"

    assert payload["blood_pressure"] is not None
    assert payload["blood_pressure"]["systolic"] == 148
    assert payload["blood_pressure"]["diastolic"] == 92


def test_get_patient_not_found() -> None:
    main_module.app.dependency_overrides[get_current_actor] = lambda: _CLINICIAN

    with TestClient(app) as client:
        response = client.get("/patients/patient-999")

    assert response.status_code == 404

    payload = response.json()

    assert "not found" in payload["detail"].lower()


def test_list_patients_requires_authentication() -> None:
    with TestClient(app) as client:
        response = client.get("/patients")

    assert response.status_code == 401
