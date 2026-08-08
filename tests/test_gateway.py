"""
test_gateway.py
----------------
Run with:  pytest tests/ -v

Uses FastAPI's TestClient, so no live server / Docker / Celery / Redis is
required for these tests to pass (Redis/Celery are optional and the app
degrades gracefully without them).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from gateway.main import app

client = TestClient(app)
with client:
    pass  # trigger lifespan startup (creates DB, loads models) once, deterministically

VALID_PAYLOAD = {
    "age": 35,
    "education_num": 13,
    "hours_per_week": 45,
    "capital_gain": 0,
    "capital_loss": 0,
    "workclass": "Private",
    "marital_status": "Married",
    "occupation": "Tech",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_a" in body["models_loaded"]
    assert "model_b" in body["models_loaded"]


def test_models_endpoint():
    r = client.get("/models")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"model_a", "model_b"}


def test_predict_valid():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["model_used"] in ("model_a", "model_b")
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert "request_id" in body


def test_predict_missing_field():
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "occupation"}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_age_out_of_range():
    bad = dict(VALID_PAYLOAD)
    bad["age"] = 999
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_unknown_category_falls_back_gracefully():
    payload = dict(VALID_PAYLOAD)
    payload["occupation"] = "Astronaut"  # not in training data
    r = client.post("/predict", json=payload)
    assert r.status_code == 200  # should not crash; falls back to a known category


def test_metrics_endpoint_exposes_prometheus_format():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"ml_platform_requests_total" in r.content


def test_compare_endpoint_shape():
    r = client.get("/compare")
    assert r.status_code == 200
    assert "sample_size" in r.json()


def test_drift_endpoint_shape():
    r = client.get("/drift")
    assert r.status_code == 200
    body = r.json()
    assert "age" in body
    assert "ks_statistic" in body["age"]


def test_canary_admin_endpoint():
    r = client.post("/admin/canary", json={"percent": 25})
    assert r.status_code == 200
    assert r.json()["canary_traffic_percent"] == 25

    # reset back to 0 so other tests aren't affected
    r2 = client.post("/admin/canary", json={"percent": 0})
    assert r2.json()["canary_traffic_percent"] == 0


def test_canary_admin_rejects_out_of_range():
    r = client.post("/admin/canary", json={"percent": 150})
    assert r.status_code == 422
