from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    payload = response.json()

    assert payload["service"] == "Clinical AI Copilot"
    assert payload["health"] == "/health"
    assert payload["docs"] == "/docs"
