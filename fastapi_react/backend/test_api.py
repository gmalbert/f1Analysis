from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_meta_contains_parity_tabs():
    response = client.get("/api/meta")
    assert response.status_code == 200
    tabs = response.json()["tabs"]
    for expected in (
        "Data Explorer", "Analytics", "Current Season", "Next Race",
        "Predictive Models", "Raw Data", "Betting Research",
    ):
        assert expected in tabs
