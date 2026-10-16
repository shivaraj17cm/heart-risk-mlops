"""API contract tests using FastAPI's TestClient (in-process, no network)."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(trained):
    import os

    os.environ["MODEL_DIR"] = str(trained["model_dir"])
    from heartrisk.api.app import app

    with TestClient(app) as c:  # context manager runs the lifespan (model load)
        yield c


def test_health_and_ready(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_loaded"] is True
    assert client.get("/ready").status_code == 200


def test_predict_returns_prediction_and_confidence(client, sample_payload):
    r = client.post("/predict", json=sample_payload)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "label", "probability_disease", "confidence",
                         "model_version", "request_id"}
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert r.headers["x-request-id"] == body["request_id"]


def test_request_id_is_propagated(client, sample_payload):
    r = client.post("/predict", json=sample_payload, headers={"X-Request-ID": "abc123"})
    assert r.json()["request_id"] == "abc123"


def test_optional_ca_thal_can_be_null(client, sample_payload):
    r = client.post("/predict", json={**sample_payload, "ca": None, "thal": None})
    assert r.status_code == 200


@pytest.mark.parametrize("patch, field", [
    ({"sex": 5}, "sex"),
    ({"age": -3}, "age"),
    ({"chol": 9999}, "chol"),
    ({"cp": 0}, "cp"),
    ({"thalach": "fast"}, "thalach"),
    ({"thal": 5}, "thal"),
    ({"sex": 0.5}, "sex"),
])
def test_invalid_values_rejected_with_422(client, sample_payload, patch, field):
    r = client.post("/predict", json={**sample_payload, **patch})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "invalid_input"
    assert any(d["field"] == field for d in body["details"])


def test_missing_required_field_rejected(client, sample_payload):
    payload = {k: v for k, v in sample_payload.items() if k != "age"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422 and any(d["field"] == "age" for d in r.json()["details"])


def test_unknown_field_rejected(client, sample_payload):
    r = client.post("/predict", json={**sample_payload, "patient_name": "x"})
    assert r.status_code == 422


def test_malformed_json_rejected(client):
    r = client.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_metrics_endpoint_exposes_counters(client, sample_payload):
    client.post("/predict", json=sample_payload)
    text = client.get("/metrics").text
    assert "heartrisk_http_requests_total" in text
    assert "heartrisk_predictions_total" in text
    assert "heartrisk_http_request_duration_seconds_bucket" in text
    assert 'heartrisk_model_info{' in text


def test_structured_logs_have_request_fields_but_no_patient_features(client, sample_payload, caplog):
    import json
    import logging

    from heartrisk.api.observability import JsonFormatter

    caplog.set_level(logging.INFO, logger="heartrisk.api")
    client.post("/predict", json={**sample_payload, "chol": 287}, headers={"X-Request-ID": "log-test"})
    fmt = JsonFormatter()
    lines = [json.loads(fmt.format(r)) for r in caplog.records if r.name == "heartrisk.api"]
    access = next(x for x in lines if x["msg"] == "request" and x["request_id"] == "log-test")
    assert access["path"] == "/predict" and access["status"] == 200 and access["latency_ms"] >= 0
    pred = next(x for x in lines if x["msg"] == "prediction" and x["request_id"] == "log-test")
    assert {"prediction", "confidence", "model_version"} <= set(pred)
    assert "287" not in json.dumps(lines), "raw clinical values must not be logged"


def test_serving_modules_do_not_import_training_stack():
    """The Docker image installs only requirements-serve.txt: importing the API must not
    pull in matplotlib / mlflow / seaborn (regression test for a real startup crash)."""
    import subprocess
    import sys

    code = ("import sys, heartrisk.api.app; "
            "bad = [m for m in ('matplotlib', 'mlflow', 'seaborn', 'yaml') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_whole_number_floats_are_accepted(client, sample_payload):
    r = client.post("/predict", json={**sample_payload, "sex": 1.0, "cp": 4.0})
    assert r.status_code == 200
