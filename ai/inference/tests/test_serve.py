"""API-level tests for the inference service (FastAPI TestClient, no startup hooks)."""

import pytest
from fastapi.testclient import TestClient

import registry as reg
import serve


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(reg, "_latest_registered_version", lambda n: None)
    monkeypatch.setattr(reg, "SUPABASE_URL", "")
    monkeypatch.setattr(reg, "MODELS_ROOT", tmp_path)
    serve.registry._models.clear()
    serve.registry.load_all()
    return TestClient(serve.app)


def test_health_degraded_when_no_models_loaded(monkeypatch):
    serve.registry._models.clear()
    body = TestClient(serve.app).get("/health").json()
    assert body["status"] == "degraded" and body["models_total"] == 0


def test_health_ok_and_reports_build_version(client, monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "abc123")
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["details"]["_image_version"] == "abc123"
    assert body["models_total"] == len(reg.SUPPORTED_MODELS)


def test_health_version_defaults_to_dev(client, monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    assert client.get("/health").json()["details"]["_image_version"] == "dev"


def test_models_lists_all_supported(client):
    ids = {m["id"] for m in client.get("/models").json()}
    assert ids == set(reg.SUPPORTED_MODELS)


def test_models_untrained_reported_degraded(client):
    assert all(m["status"] == "degraded" and m["source"] == "mock" for m in client.get("/models").json())


def test_predict_valid(client):
    r = client.post("/models/finance_fraud_anomaly/predict", json={"features": {"amount": 9000.0}})
    assert r.status_code == 200
    body = r.json()
    assert body["model_id"] == "finance_fraud_anomaly" and body["domain"] == "finance"
    assert 0 <= body["score"] <= 1 and body["warnings"]


def test_predict_unknown_model_404(client):
    assert client.post("/models/nope/predict", json={"features": {}}).status_code == 404


def test_predict_rejects_non_numeric_features(client):
    r = client.post("/models/anomaly_detection/predict", json={"features": {"a": "high"}})
    assert r.status_code == 422


def test_predict_rejects_non_object_features(client):
    assert client.post("/models/anomaly_detection/predict", json={"features": [1, 2]}).status_code == 422


def test_predict_empty_body_defaults_to_no_features(client):
    assert client.post("/models/anomaly_detection/predict", json={}).status_code == 200


def test_predict_repeated_is_stable(client):
    outs = [client.post("/models/anomaly_detection/predict", json={"features": {"a": 1.0}}).json() for _ in range(25)]
    assert all(o == outs[0] for o in outs)


def test_reload_unknown_404(client):
    assert client.post("/models/nope/reload").status_code == 404


def test_reload_known_reports_state(client):
    r = client.post("/models/phishing_detection/reload").json()
    assert r["model_id"] == "phishing_detection" and r["source"] == "mock" and r["changed"] is False
