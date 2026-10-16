"""FastAPI model-serving application.

Endpoints
  GET  /          service info
  GET  /health    liveness  (process up)
  GET  /ready     readiness (model loaded) - used by Kubernetes readinessProbe
  POST /predict   JSON patient features -> prediction + confidence
  GET  /metrics   Prometheus exposition format
  GET  /docs      OpenAPI UI (auto-generated)

Run locally:  uvicorn heartrisk.api.app:app --port 8000
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from heartrisk import __version__, paths
from heartrisk.api.observability import (
    HTTP_LATENCY,
    HTTP_REQUESTS,
    MODEL_INFO,
    PREDICTION_CONFIDENCE,
    PREDICTION_ERRORS,
    PREDICTIONS,
    configure_logging,
)
from heartrisk.api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from heartrisk.model_io import LoadedModel, load_model

configure_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("heartrisk.api")

STATE: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = os.getenv("MODEL_DIR", str(paths.MODEL_DIR))
    try:
        model: LoadedModel = load_model(model_dir)
        STATE["model"] = model
        MODEL_INFO.labels(version=model.version,
                          family=model.metadata.get("model_family", "unknown")).set(1)
        logger.info("model loaded", extra={"fields": {"model_dir": model_dir, "model_version": model.version}})
    except Exception:  # noqa: BLE001 - keep serving /health so the failure is observable
        logger.exception("model failed to load", extra={"fields": {"model_dir": model_dir}})
    yield
    STATE["model"] = None


app = FastAPI(
    title="Heart Disease Risk API",
    version=__version__,
    description="Academic MLOps demo (UCI Cleveland data). Not a medical device.",
    lifespan=lifespan,
)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")  # bounded label cardinality


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = _route_template(request)
        HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
        HTTP_LATENCY.labels(request.method, path).observe(elapsed)
        if path != "/metrics":
            logger.info("request", extra={"fields": {
                "request_id": request_id, "method": request.method, "path": path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    PREDICTION_ERRORS.labels("validation").inc()
    errors = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body", "message": e["msg"]}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={
        "error": "invalid_input", "details": errors,
        "request_id": getattr(request.state, "request_id", None)})


@app.get("/")
def root():
    return {"service": "heart-risk-api", "version": __version__,
            "endpoints": ["/health", "/ready", "/predict", "/metrics", "/docs"]}


@app.get("/health", response_model=HealthResponse)
def health():
    m = STATE["model"]
    return HealthResponse(status="ok", model_loaded=m is not None, model_version=m.version if m else None)


@app.get("/ready", response_model=HealthResponse)
def ready():
    m = STATE["model"]
    if m is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ready", model_loaded=True, model_version=m.version)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures, request: Request):
    m = STATE["model"]
    if m is None:
        PREDICTION_ERRORS.labels("model_unavailable").inc()
        raise HTTPException(status_code=503, detail="model not loaded")
    try:
        result = m.predict_records([features.model_dump()])[0]
    except Exception:  # noqa: BLE001
        PREDICTION_ERRORS.labels("inference").inc()
        logger.exception("inference failed", extra={"fields": {"request_id": request.state.request_id}})
        raise HTTPException(status_code=500, detail="inference failed")
    PREDICTIONS.labels(str(result["prediction"])).inc()
    PREDICTION_CONFIDENCE.observe(result["confidence"])
    logger.info("prediction", extra={"fields": {
        "request_id": request.state.request_id, "prediction": result["prediction"],
        "confidence": result["confidence"], "model_version": m.version}})
    return PredictionResponse(**result, model_version=m.version, request_id=request.state.request_id)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
