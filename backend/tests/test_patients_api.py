from fastapi.testclient import TestClient

from app.main import app


def test_list_patients_endpoint() -> None:
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
    with TestClient(app) as client:
        response = client.get("/patients/patient-999")

    assert response.status_code == 404

    payload = response.json()

    assert "not found" in payload["detail"].lower()
