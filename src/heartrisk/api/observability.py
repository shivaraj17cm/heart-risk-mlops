"""Structured JSON logging and Prometheus metrics for the API.

Privacy: request logs contain request metadata, the prediction and confidence
only - never the patient's clinical feature values.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter(
    "heartrisk_http_requests_total", "HTTP requests handled",
    ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "heartrisk_http_request_duration_seconds", "HTTP request latency (seconds)",
    ["method", "path"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter(
    "heartrisk_predictions_total", "Predictions served, by predicted class", ["prediction"])
PREDICTION_CONFIDENCE = Histogram(
    "heartrisk_prediction_confidence", "Confidence of served predictions",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))
PREDICTION_ERRORS = Counter(
    "heartrisk_prediction_errors_total", "Prediction failures", ["reason"])
MODEL_INFO = Gauge(
    "heartrisk_model_info", "Loaded model (value 1) labelled with its version", ["version", "family"])


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).disabled = True  # replaced by our structured access log
